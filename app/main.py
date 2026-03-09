"""Home Node Daemon — entry point.

Lifecycle:
  1. If UPnP enabled, try UPnP/NAT-PMP port mapping
  2. Detect public IP (or use configured value)
  3. Register with Coordination API
  4. Start asyncio TCP server + optional UPnP lease renewal
  5. Wait for SIGTERM / SIGINT
  6. Cancel UPnP renewal + remove port mapping
  7. Deregister node (best-effort)
  8. Shutdown
"""

import asyncio
import functools
import logging
import signal
import sys

import httpx

from app.config import settings
from app.proxy_handler import handle_client
from app.registration import deregister_node, detect_public_ip, register_node
from app.tls import create_server_ssl_context, ensure_certificates

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _run(settings_override=None) -> None:  # noqa: ANN001
    s = settings_override or settings
    stop_event = asyncio.Event()

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)
    else:
        # Windows: loop.add_signal_handler() is not supported.
        # Use signal.signal() and schedule the event via call_soon_threadsafe
        # since signal handlers can interrupt the event loop.
        loop = asyncio.get_running_loop()

        def _handle_signal(signum, frame):  # noqa: ANN001
            loop.call_soon_threadsafe(stop_event.set)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    async with httpx.AsyncClient() as http_client:
        # 1. Try UPnP/NAT-PMP port mapping (if enabled)
        upnp_endpoint = None
        if s.UPNP_ENABLED:
            from app.upnp import setup_upnp_mapping

            upnp_endpoint = await setup_upnp_mapping(
                s.NODE_PORT, lease_duration=s.UPNP_LEASE_DURATION,
            )
            if upnp_endpoint:
                logger.info(
                    "UPnP mapping active: %s:%d",
                    upnp_endpoint[0], upnp_endpoint[1],
                )
            else:
                logger.warning(
                    "UPnP enabled but mapping failed — "
                    "falling back to direct public IP mode"
                )

        # 2. Detect public IP (always needed for metadata)
        if s.PUBLIC_IP:
            public_ip = s.PUBLIC_IP
            logger.info("Using configured public IP: %s", public_ip)
        else:
            try:
                public_ip = await detect_public_ip(http_client)
            except RuntimeError:
                logger.error("Cannot detect public IP — aborting")
                sys.exit(1)

        # 3. Register with Coordination API
        try:
            node_id = await register_node(
                http_client, s, public_ip, upnp_endpoint=upnp_endpoint,
            )
        except Exception:
            logger.exception("Failed to register with Coordination API — aborting")
            sys.exit(1)

        # 4. Start TLS server
        ensure_certificates(s.TLS_CERT_PATH, s.TLS_KEY_PATH)
        ssl_ctx = create_server_ssl_context(s.TLS_CERT_PATH, s.TLS_KEY_PATH)

        handler = functools.partial(handle_client, settings=s)
        server = await asyncio.start_server(
            handler, host="0.0.0.0", port=s.NODE_PORT, ssl=ssl_ctx,
        )
        logger.info(
            "Home Node listening on port %d with TLS (node_id=%s, upnp=%s)",
            s.NODE_PORT, node_id,
            f"{upnp_endpoint[0]}:{upnp_endpoint[1]}" if upnp_endpoint else "disabled",
        )

        # Start UPnP lease renewal if applicable
        renewal_task = None
        if upnp_endpoint and s.UPNP_LEASE_DURATION > 0:
            from app.upnp import renew_upnp_mapping

            async def _renew_loop() -> None:
                interval = max(s.UPNP_LEASE_DURATION // 2, 60)
                while True:
                    await asyncio.sleep(interval)
                    ok = await renew_upnp_mapping(
                        s.NODE_PORT, upnp_endpoint[1], s.UPNP_LEASE_DURATION,
                    )
                    if ok:
                        logger.debug("UPnP lease renewed")
                    else:
                        logger.warning("UPnP lease renewal failed")

            renewal_task = asyncio.create_task(_renew_loop())

        try:
            await stop_event.wait()
        finally:
            logger.info("Shutting down…")

            # 5. Stop accepting new connections
            server.close()
            await server.wait_closed()

            # 6. Cancel UPnP renewal + remove mapping
            if renewal_task is not None:
                renewal_task.cancel()
                try:
                    await renewal_task
                except asyncio.CancelledError:
                    pass

            if upnp_endpoint:
                from app.upnp import remove_upnp_mapping
                await remove_upnp_mapping(upnp_endpoint[1])

            # 7. Deregister (best-effort)
            await deregister_node(http_client, s, node_id)

    logger.info("Home Node shut down cleanly")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
