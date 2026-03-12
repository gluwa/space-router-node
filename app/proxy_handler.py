"""Core proxy logic for the Home Node.

The Home Node is the server-side counterpart to the Proxy Gateway's
_connect_to_node → handle_connect / handle_http_forward flow.  It receives
proxied traffic from the Proxy Gateway and forwards it to target servers
from its residential IP.
"""

import asyncio
import base64
import ipaddress
import logging
from urllib.parse import urlparse

from app.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

MAX_HEADER_SIZE = 64 * 1024          # 64 KB cap on total request headers
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB cap on request/response bodies
MAX_CHUNK_SIZE = 10 * 1024 * 1024     # 10 MB cap per chunked transfer chunk

# Private/reserved IP ranges — deny SSRF to internal networks
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local / cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),      # multicast
    ipaddress.ip_network("240.0.0.0/4"),      # reserved
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# Ports that should never be proxied (internal services)
_BLOCKED_PORTS = {22, 23, 25, 135, 136, 137, 138, 139, 445, 3306, 5432, 6379, 11211, 27017}


def _is_private_ip(host: str) -> bool:
    """Return True if *host* (an IP literal) is in a private/reserved range."""
    try:
        addr = ipaddress.ip_address(host)
        if addr.version == 6 and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        return any(addr in net for net in _BLOCKED_NETWORKS)
    except ValueError:
        return False


def _is_private_target(host: str, port: int) -> bool:
    """Return True if the target host resolves to a private/reserved IP or a blocked port."""
    if port in _BLOCKED_PORTS:
        return True
    if _is_private_ip(host):
        return True
    # hostname, not IP — block obvious patterns
    try:
        ipaddress.ip_address(host)
    except ValueError:
        lower = host.lower()
        if lower in ("localhost", "localhost.localdomain") or lower.endswith(".local"):
            return True
    return False


import socket


class _DNSRebindingError(Exception):
    """Raised when DNS resolution yields a private/reserved IP."""


async def _resolve_and_connect(
    host: str,
    port: int,
    timeout: float,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Resolve *host* via DNS, check the resolved IP against the blocklist,
    then connect to the resolved IP directly (preventing DNS rebinding).

    Raises ``ConnectionRefusedError`` if the resolved IP is private/reserved.
    Raises ``OSError`` / ``asyncio.TimeoutError`` on network failure.
    """
    loop = asyncio.get_running_loop()

    # Resolve hostname → IP(s)
    try:
        infos = await asyncio.wait_for(
            loop.getaddrinfo(host, port, type=socket.SOCK_STREAM),
            timeout=min(timeout, 10.0),
        )
    except (socket.gaierror, asyncio.TimeoutError) as exc:
        raise OSError(f"DNS resolution failed for {host}: {exc}") from exc

    if not infos:
        raise OSError(f"DNS resolution returned no results for {host}")

    # Check ALL resolved IPs — an attacker might have multiple A records
    # mixing public and private IPs.
    for family, _type, _proto, _canonname, sockaddr in infos:
        resolved_ip = sockaddr[0]
        if _is_private_ip(resolved_ip):
            raise _DNSRebindingError(
                f"DNS rebinding blocked: {host} resolved to private IP {resolved_ip}"
            )

    # Connect to the first resolved address
    # Using the resolved IP directly prevents TOCTOU race.
    family, _type, _proto, _canonname, sockaddr = infos[0]
    resolved_ip = sockaddr[0]

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(resolved_ip, port),
        timeout=timeout,
    )
    return reader, writer


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
            if len(header_data) > MAX_HEADER_SIZE:
                return None  # Header too large — reject
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionResetError):
        return None

    headers = parse_headers(header_data)
    raw_head = request_line + header_data + b"\r\n"
    return raw_head, method, target, version, headers


async def _pipe(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    counter: list[int],
    buffer_size: int,
    activity_event: asyncio.Event | None = None,
) -> None:
    try:
        while True:
            data = await reader.read(buffer_size)
            if not data:
                break
            writer.write(data)
            await writer.drain()
            counter[0] += len(data)
            if activity_event is not None:
                activity_event.set()
    except (ConnectionResetError, BrokenPipeError, OSError, asyncio.CancelledError):
        pass


# Maximum absolute wall-clock timeout for a single relay (safety cap).
MAX_RELAY_DURATION = 3600.0  # 1 hour


async def relay_streams(
    reader_a: asyncio.StreamReader,
    writer_a: asyncio.StreamWriter,
    reader_b: asyncio.StreamReader,
    writer_b: asyncio.StreamWriter,
    buffer_size: int,
    timeout: float = 300.0,
) -> tuple[int, int]:
    """Bidirectional byte relay with **idle** timeout.

    The *timeout* is reset every time data flows in either direction.
    A hard safety cap (``MAX_RELAY_DURATION``) ensures no relay lives
    forever even if there is continuous low-rate traffic.
    """
    bytes_a_to_b = [0]
    bytes_b_to_a = [0]
    activity = asyncio.Event()

    task_a = asyncio.create_task(
        _pipe(reader_a, writer_b, bytes_a_to_b, buffer_size, activity)
    )
    task_b = asyncio.create_task(
        _pipe(reader_b, writer_a, bytes_b_to_a, buffer_size, activity)
    )

    done = asyncio.gather(task_a, task_b, return_exceptions=True)
    deadline = asyncio.get_event_loop().time() + MAX_RELAY_DURATION

    try:
        while True:
            activity.clear()
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.debug("Relay hit absolute duration cap")
                break

            idle_limit = min(timeout, remaining)

            # Wait for either: both pipes finish, or activity, or idle timeout
            activity_task = asyncio.create_task(activity.wait())
            finished, pending = await asyncio.wait(
                [done, activity_task],
                timeout=idle_limit,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Clean up the activity waiter if it didn't fire
            if activity_task in pending:
                activity_task.cancel()
                try:
                    await activity_task
                except asyncio.CancelledError:
                    pass

            if done in finished:
                # Both pipes completed naturally
                break

            if not finished:
                # Neither activity nor completion → idle timeout
                logger.debug("Relay idle timeout (%ss)", timeout)
                break

            # Activity detected — loop again with fresh idle timer
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

def _error_response(
    status: int,
    reason: str,
    body: str,
    request_id: str | None = None,
) -> bytes:
    payload = body.encode()
    extra_headers = ""
    if request_id:
        extra_headers = f"X-SpaceRouter-Request-Id: {request_id}\r\n"
    return (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: text/plain\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"{extra_headers}"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + payload


def _bad_request(detail: str = "Bad Request", request_id: str | None = None) -> bytes:
    return _error_response(400, "Bad Request", detail, request_id)


def _forbidden(detail: str = "Forbidden", request_id: str | None = None) -> bytes:
    return _error_response(403, "Forbidden", detail, request_id)


def _bad_gateway(detail: str = "Bad Gateway", request_id: str | None = None) -> bytes:
    return _error_response(502, "Bad Gateway", detail, request_id)


def _gateway_timeout(detail: str = "Gateway Timeout", request_id: str | None = None) -> bytes:
    return _error_response(504, "Gateway Timeout", detail, request_id)


def _proxy_auth_required(request_id: str | None = None) -> bytes:
    return _error_response(407, "Proxy Authentication Required", "Proxy Authentication Required", request_id)


def _parse_basic_auth(auth_header: str) -> tuple[str, str] | None:
    """Parse a ``Proxy-Authorization: Basic <b64>`` header.

    Returns ``(username, password)`` on success, or ``None`` if the header is
    missing, malformed, or not Basic scheme.
    """
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(parts[1]).decode("utf-8")
    except Exception:
        return None
    if ":" not in decoded:
        return None
    username, _, password = decoded.partition(":")
    return username, password


# ---------------------------------------------------------------------------
# Strip internal headers before forwarding to the target
# ---------------------------------------------------------------------------

def _strip_spacerouter_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove X-SpaceRouter-*, Proxy-Authorization, and other sensitive headers."""
    # Standard headers to remove for anonymity and security
    stripped_headers = {
        "x-forwarded-for",
        "x-real-ip",
        "via",
        "forwarded",
        "proxy-authorization",
        "proxy-connection",
    }
    return {
        k: v
        for k, v in headers.items()
        if not k.lower().startswith(SPACEROUTER_HEADER_PREFIX)
        and k.lower() not in stripped_headers
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
    request_id: str | None = None,
) -> None:
    """Open a TCP connection to *target_host:target_port*, reply 200, then
    relay bytes bidirectionally between the client (Proxy Gateway) and the
    target server.
    """
    rid_tag = f" [request_id={request_id}]" if request_id else ""

    # SSRF protection — block private/reserved targets (static check)
    if _is_private_target(target_host, target_port):
        logger.warning("CONNECT blocked — private target %s:%s%s", target_host, target_port, rid_tag)
        client_writer.write(_forbidden("Target not allowed", request_id))
        await client_writer.drain()
        return

    # DNS-resolved SSRF check — prevent DNS rebinding attacks
    try:
        target_reader, target_writer = await _resolve_and_connect(
            target_host, target_port, settings.REQUEST_TIMEOUT,
        )
    except _DNSRebindingError as exc:
        logger.warning("CONNECT blocked (DNS rebinding) — %s%s", exc, rid_tag)
        client_writer.write(_forbidden("Target not allowed", request_id))
        await client_writer.drain()
        return
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("CONNECT failed to %s:%s — %s%s", target_host, target_port, exc, rid_tag)
        client_writer.write(_bad_gateway("Cannot connect to target", request_id))
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
    request_id: str | None = None,
) -> None:
    """Forward an HTTP request to the target server and stream the response
    back to the client (Proxy Gateway).

    The *target* is an absolute URI (``http://example.com/path``).  We parse
    it, connect to the origin, rewrite the request line to a relative path,
    and relay request + response.
    """
    rid_tag = f" [request_id={request_id}]" if request_id else ""

    parsed = urlparse(target)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    if not host:
        client_writer.write(_bad_request("Missing host in target URL", request_id))
        await client_writer.drain()
        return

    # SSRF protection — block private/reserved targets (static check)
    if _is_private_target(host, port):
        logger.warning("HTTP forward blocked — private target %s:%s%s", host, port, rid_tag)
        client_writer.write(_forbidden("Target not allowed", request_id))
        await client_writer.drain()
        return

    # DNS-resolved SSRF check — prevent DNS rebinding attacks
    try:
        target_reader, target_writer = await _resolve_and_connect(
            host, port, settings.REQUEST_TIMEOUT,
        )
    except _DNSRebindingError as exc:
        logger.warning("HTTP forward blocked (DNS rebinding) — %s%s", exc, rid_tag)
        client_writer.write(_forbidden("Target not allowed", request_id))
        await client_writer.drain()
        return
    except (OSError, asyncio.TimeoutError) as exc:
        logger.warning("HTTP forward failed to %s:%s — %s%s", host, port, exc, rid_tag)
        client_writer.write(_bad_gateway("Cannot connect to target", request_id))
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

        # Forward request body if present (with size cap)
        content_length = int(headers.get("Content-Length", headers.get("content-length", "0")))
        if content_length > MAX_CONTENT_LENGTH:
            client_writer.write(_bad_request("Request body too large", request_id))
            await client_writer.drain()
            return
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
            client_writer.write(_gateway_timeout("Target server timed out", request_id))
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
                if chunk_size > MAX_CHUNK_SIZE:
                    break  # Reject oversized chunks
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

# Global connection semaphore — initialized lazily on first use.
# The limit comes from Settings.MAX_CONNECTIONS.
_connection_semaphore: asyncio.Semaphore | None = None
_active_connections: int = 0


def _get_semaphore(max_connections: int) -> asyncio.Semaphore:
    """Return (or create) the global connection semaphore."""
    global _connection_semaphore
    if _connection_semaphore is None:
        _connection_semaphore = asyncio.Semaphore(max_connections)
    return _connection_semaphore


def _service_unavailable() -> bytes:
    body = b"503 Service Unavailable - connection limit reached"
    return (
        b"HTTP/1.1 503 Service Unavailable\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n"
        b"\r\n" + body
    )


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    settings: Settings,
) -> None:
    """Entry point for each inbound connection from the Proxy Gateway."""
    global _active_connections
    peer = writer.get_extra_info("peername")

    sem = _get_semaphore(settings.MAX_CONNECTIONS)
    if not sem._value:  # noqa: SLF001 — fast check without awaiting
        logger.warning(
            "Connection limit reached (%d) — rejecting %s",
            settings.MAX_CONNECTIONS, peer,
        )
        try:
            writer.write(_service_unavailable())
            await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        return

    async with sem:
        _active_connections += 1
        logger.debug("New connection from %s (active=%d)", peer, _active_connections)
        try:
            result = await _read_request_head(reader, settings.REQUEST_TIMEOUT)
            if result is None:
                writer.write(_bad_request("Malformed request"))
                await writer.drain()
                return

            _raw_head, method, target, version, headers = result

            # Extract X-SpaceRouter-Request-Id for log correlation
            request_id: str | None = (
                headers.get("X-SpaceRouter-Request-Id")
                or headers.get("x-spacerouter-request-id")
                or None
            )

            # Validate Proxy-Authorization if credentials are configured
            if settings.PROXY_AUTH_USERNAME and settings.PROXY_AUTH_PASSWORD:
                auth_header = (
                    headers.get("Proxy-Authorization")
                    or headers.get("proxy-authorization")
                    or ""
                )
                parsed_auth = _parse_basic_auth(auth_header)
                if (
                    parsed_auth is None
                    or parsed_auth[0] != settings.PROXY_AUTH_USERNAME
                    or parsed_auth[1] != settings.PROXY_AUTH_PASSWORD
                ):
                    logger.warning(
                        "Proxy auth failed from %s%s",
                        peer,
                        f" [request_id={request_id}]" if request_id else "",
                    )
                    writer.write(_proxy_auth_required(request_id))
                    await writer.drain()
                    return

            if method.upper() == "CONNECT":
                host_port = target.split(":")
                target_host = host_port[0]
                target_port = int(host_port[1]) if len(host_port) > 1 else 443

                logger.info(
                    "CONNECT %s:%s%s",
                    target_host,
                    target_port,
                    f" [request_id={request_id}]" if request_id else "",
                )
                await handle_connect(reader, writer, target_host, target_port, settings, request_id)
            else:
                logger.info(
                    "%s %s%s",
                    method,
                    target,
                    f" [request_id={request_id}]" if request_id else "",
                )
                await handle_http_forward(reader, writer, method, target, version, headers, settings, request_id)

        except Exception:
            logger.exception("Unhandled error in client handler")
        finally:
            _active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
