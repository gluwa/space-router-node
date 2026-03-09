"""Core proxy logic for the Home Node.

The Home Node is the server-side counterpart to the Proxy Gateway's
_connect_to_node → handle_connect / handle_http_forward flow.  It receives
proxied traffic from the Proxy Gateway and forwards it to target servers
from its residential IP.
"""

import asyncio
import logging
from urllib.parse import urlparse

from app.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared protocol utilities (mirrored from proxy-gateway/app/proxy.py)
# ---------------------------------------------------------------------------

SPACEROUTER_HEADER_PREFIX = "x-spacerouter-"


def parse_headers(raw: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.split(b"\r\n"):
        if b":" in line:
            key, _, value = line.partition(b":")
            headers[key.decode("latin-1").strip()] = value.decode("latin-1").strip()
    return headers


async def _read_request_head(
    reader: asyncio.StreamReader,
    timeout: float = 30.0,
) -> tuple[bytes, str, str, str, dict[str, str]] | None:
    """Read and parse the HTTP request line + headers from *reader*.

    Returns (raw_head, method, target, version, headers) or None on error.
    """
    try:
        request_line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionResetError):
        return None

    parts = request_line.decode("latin-1").strip().split(" ", 2)
    if len(parts) != 3:
        return None

    method, target, version = parts

    header_data = b""
    try:
        while True:
            line = await asyncio.wait_for(reader.readuntil(b"\r\n"), timeout=10.0)
            if line == b"\r\n":
                break
            header_data += line
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionResetError):
        return None

    headers = parse_headers(header_data)
    raw_head = request_line + header_data + b"\r\n"
    return raw_head, method, target, version, headers


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    counter: list[int],
    buffer_size: int,
) -> None:
    try:
        while True:
            data = await reader.read(buffer_size)
            if not data:
                break
            writer.write(data)
            await writer.drain()
            counter[0] += len(data)
    except (ConnectionResetError, BrokenPipeError, OSError, asyncio.CancelledError):
        pass


async def relay_streams(
    reader_a: asyncio.StreamReader,
    writer_a: asyncio.StreamWriter,
    reader_b: asyncio.StreamReader,
    writer_b: asyncio.StreamWriter,
    buffer_size: int,
    timeout: float = 300.0,
) -> tuple[int, int]:
    bytes_a_to_b = [0]
    bytes_b_to_a = [0]

    task_a = asyncio.create_task(_pipe(reader_a, writer_b, bytes_a_to_b, buffer_size))
    task_b = asyncio.create_task(_pipe(reader_b, writer_a, bytes_b_to_a, buffer_size))

    try:
        await asyncio.wait_for(
            asyncio.gather(task_a, task_b, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        pass
    finally:
        task_a.cancel()
        task_b.cancel()
        try:
            await asyncio.gather(task_a, task_b, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    return bytes_a_to_b[0], bytes_b_to_a[0]


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------

def _error_response(status: int, reason: str, body: str) -> bytes:
    payload = body.encode()
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: text/plain\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + payload


def _bad_request(detail: str = "Bad Request") -> bytes:
    return _error_response(400, "Bad Request", detail)


def _bad_gateway(detail: str = "Bad Gateway") -> bytes:
    return _error_response(502, "Bad Gateway", detail)


def _gateway_timeout(detail: str = "Gateway Timeout") -> bytes:
    return _error_response(504, "Gateway Timeout", detail)


# ---------------------------------------------------------------------------
# Strip internal headers before forwarding to the target
# ---------------------------------------------------------------------------

def _strip_spacerouter_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove X-SpaceRouter-* and Proxy-Authorization headers."""
    return {
        k: v
        for k, v in headers.items()
        if not k.lower().startswith(SPACEROUTER_HEADER_PREFIX)
        and k.lower() != "proxy-authorization"
    }


# ---------------------------------------------------------------------------
# CONNECT handler — tunnel to target
# ---------------------------------------------------------------------------

async def handle_connect(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
    settings: Settings,
) -> None:
    """Open a TCP connection to *target_host:target_port*, reply 200, then
    relay bytes bidirectionally between the client (Proxy Gateway) and the
    target server.
    """
    try:
        target_reader, target_writer = await asyncio.wait_for(
            asyncio.open_connection(target_host, target_port),
            timeout=settings.REQUEST_TIMEOUT,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("CONNECT failed to %s:%s — %s", target_host, target_port, exc)
        client_writer.write(_bad_gateway(f"Cannot connect to {target_host}:{target_port}"))
        await client_writer.drain()
        return

    # Tell the Proxy Gateway the tunnel is established
    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await client_writer.drain()

    try:
        await relay_streams(
            client_reader,
            client_writer,
            target_reader,
            target_writer,
            settings.BUFFER_SIZE,
            settings.RELAY_TIMEOUT,
        )
    finally:
        try:
            target_writer.close()
            await target_writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP forward handler — plain-text HTTP proxy
# ---------------------------------------------------------------------------

async def handle_http_forward(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    method: str,
    target: str,
    version: str,
    headers: dict[str, str],
    settings: Settings,
) -> None:
    """Forward an HTTP request to the target server and stream the response
    back to the client (Proxy Gateway).

    The *target* is an absolute URI (``http://example.com/path``).  We parse
    it, connect to the origin, rewrite the request line to a relative path,
    and relay request + response.
    """
    parsed = urlparse(target)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    if not host:
        client_writer.write(_bad_request("Missing host in target URL"))
        await client_writer.drain()
        return

    # Connect to target
    try:
        target_reader, target_writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=settings.REQUEST_TIMEOUT,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("HTTP forward failed to %s:%s — %s", host, port, exc)
        client_writer.write(_bad_gateway(f"Cannot connect to {host}:{port}"))
        await client_writer.drain()
        return

    try:
        # Build the forwarded request — relative path, strip internal headers
        forward_headers = _strip_spacerouter_headers(headers)
        # Ensure Host header is correct
        forward_headers["Host"] = f"{host}:{port}" if port not in (80, 443) else host

        header_str = "".join(f"{k}: {v}\r\n" for k, v in forward_headers.items())
        request_head = f"{method} {path} {version}\r\n{header_str}\r\n".encode()
        target_writer.write(request_head)
        await target_writer.drain()

        # Forward request body if present
        content_length = int(headers.get("Content-Length", headers.get("content-length", "0")))
        if content_length > 0:
            remaining = content_length
            while remaining > 0:
                chunk = await client_reader.read(min(remaining, settings.BUFFER_SIZE))
                if not chunk:
                    break
                target_writer.write(chunk)
                await target_writer.drain()
                remaining -= len(chunk)

        # Read response from target
        try:
            response_line = await asyncio.wait_for(
                target_reader.readuntil(b"\r\n"),
                timeout=settings.REQUEST_TIMEOUT,
            )
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            client_writer.write(_gateway_timeout("Target server timed out"))
            await client_writer.drain()
            return

        # Read response headers
        resp_header_data = b""
        while True:
            try:
                line = await asyncio.wait_for(target_reader.readuntil(b"\r\n"), timeout=10.0)
                if line == b"\r\n":
                    break
                resp_header_data += line
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                break

        resp_headers = parse_headers(resp_header_data)

        # Forward response status + headers to the Proxy Gateway
        client_writer.write(response_line)
        client_writer.write(resp_header_data)
        client_writer.write(b"\r\n")
        await client_writer.drain()

        # Stream response body
        resp_content_length = resp_headers.get(
            "Content-Length", resp_headers.get("content-length")
        )
        transfer_encoding = resp_headers.get(
            "Transfer-Encoding", resp_headers.get("transfer-encoding", "")
        )

        if resp_content_length:
            remaining = int(resp_content_length)
            while remaining > 0:
                chunk = await target_reader.read(min(remaining, settings.BUFFER_SIZE))
                if not chunk:
                    break
                client_writer.write(chunk)
                await client_writer.drain()
                remaining -= len(chunk)
        elif "chunked" in transfer_encoding.lower():
            while True:
                try:
                    size_line = await asyncio.wait_for(
                        target_reader.readuntil(b"\r\n"),
                        timeout=settings.REQUEST_TIMEOUT,
                    )
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    break
                client_writer.write(size_line)
                await client_writer.drain()

                chunk_size = int(size_line.strip(), 16)
                if chunk_size == 0:
                    trailer = await target_reader.readuntil(b"\r\n")
                    client_writer.write(trailer)
                    await client_writer.drain()
                    break

                chunk_data = await target_reader.readexactly(chunk_size + 2)
                client_writer.write(chunk_data)
                await client_writer.drain()
        else:
            # No content-length or chunked: read until connection close
            while True:
                chunk = await target_reader.read(settings.BUFFER_SIZE)
                if not chunk:
                    break
                client_writer.write(chunk)
                await client_writer.drain()

    finally:
        try:
            target_writer.close()
            await target_writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Client dispatch
# ---------------------------------------------------------------------------

async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    settings: Settings,
) -> None:
    """Entry point for each inbound connection from the Proxy Gateway."""
    peer = writer.get_extra_info("peername")
    logger.debug("New connection from %s", peer)

    try:
        result = await _read_request_head(reader, settings.REQUEST_TIMEOUT)
        if result is None:
            writer.write(_bad_request("Malformed request"))
            await writer.drain()
            return

        _raw_head, method, target, version, headers = result

        if method.upper() == "CONNECT":
            host_port = target.split(":")
            target_host = host_port[0]
            target_port = int(host_port[1]) if len(host_port) > 1 else 443

            logger.info("CONNECT %s:%s", target_host, target_port)
            await handle_connect(reader, writer, target_host, target_port, settings)
        else:
            logger.info("%s %s", method, target)
            await handle_http_forward(reader, writer, method, target, version, headers, settings)

    except Exception:
        logger.exception("Unhandled error in client handler")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
