"""UPnP/NAT-PMP port mapping for the Home Node.

Automatically configures port forwarding on the user's router
so the Gateway can reach the Home Node without manual port forwarding.
"""

import asyncio
import logging
import socket

logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    """Detect the machine's LAN IP (the IP the router sees).

    Uses a UDP connect trick — no traffic is actually sent.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _do_upnp_mapping(
    internal_ip: str,
    internal_port: int,
    lease_duration: int,
) -> tuple[str, int] | None:
    """Synchronous UPnP mapping — runs in a thread."""
    try:
        import miniupnpc
    except ImportError:
        logger.warning("miniupnpc not installed — UPnP unavailable")
        return None

    u = miniupnpc.UPnP()
    u.discoverdelay = 2000  # ms

    try:
        devices = u.discover()
    except Exception as exc:
        logger.warning("UPnP discovery failed: %s", exc)
        return None

    if devices == 0:
        logger.info("No UPnP devices found on the network")
        return None

    try:
        u.selectigd()
    except Exception as exc:
        logger.warning("Failed to select UPnP IGD: %s", exc)
        return None

    external_ip = u.externalipaddress()
    if not external_ip:
        logger.warning("UPnP device did not report external IP")
        return None

    external_port = internal_port
    description = f"SpaceRouter Home Node ({internal_port}/TCP)"

    try:
        u.addportmapping(
            external_port, "TCP",
            internal_ip, internal_port,
            description, "", lease_duration,
        )
        logger.info(
            "UPnP port mapping created: %s:%d -> %s:%d (lease=%ds)",
            external_ip, external_port, internal_ip, internal_port, lease_duration,
        )
        return external_ip, external_port
    except Exception as exc:
        logger.warning("UPnP addportmapping failed: %s", exc)
        return None


def _do_upnp_removal(external_port: int) -> None:
    """Synchronous UPnP removal — runs in a thread."""
    try:
        import miniupnpc
    except ImportError:
        return

    u = miniupnpc.UPnP()
    u.discoverdelay = 2000

    if u.discover() == 0:
        return

    try:
        u.selectigd()
        u.deleteportmapping(external_port, "TCP")
        logger.info("UPnP port mapping removed: port %d/TCP", external_port)
    except Exception as exc:
        logger.warning("Failed to remove UPnP mapping: %s", exc)


async def setup_upnp_mapping(
    internal_port: int,
    lease_duration: int = 3600,
) -> tuple[str, int] | None:
    """Request a UPnP/NAT-PMP port mapping on the local router.

    Returns ``(external_ip, external_port)`` on success, or ``None``.
    """
    internal_ip = _get_local_ip()
    logger.info("Local IP for UPnP: %s", internal_ip)
    return await asyncio.to_thread(
        _do_upnp_mapping, internal_ip, internal_port, lease_duration,
    )


async def remove_upnp_mapping(external_port: int) -> None:
    """Remove a previously created port mapping. Best-effort."""
    await asyncio.to_thread(_do_upnp_removal, external_port)


async def renew_upnp_mapping(
    internal_port: int,
    external_port: int,
    lease_duration: int = 3600,
) -> bool:
    """Re-add the port mapping to refresh the lease. Returns True on success."""
    internal_ip = _get_local_ip()
    result = await asyncio.to_thread(
        _do_upnp_mapping, internal_ip, internal_port, lease_duration,
    )
    return result is not None
