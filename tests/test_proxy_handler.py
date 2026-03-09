"""Tests for the Home Node proxy handler.

Each test starts a real asyncio TLS server (home-node) and a fake target
server, then sends requests through the home-node and asserts on the result.
"""

import asyncio
import functools
import ssl

import pytest

from app.proxy_handler import handle_client, parse_headers
from app.tls import create_server_ssl_context, ensure_certificates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_ssl_context():
    """Return an SSL context that trusts self-signed certs (for test clients)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _start_home_node(settings):
    """Start the home-node TLS server on a random port; return (server, port)."""
    ensure_certificates(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    ssl_ctx = create_server_ssl_context(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)

    handler = functools.partial(handle_client, settings=settings)
    server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _start_target_server(handler):
    """Start a fake target server (plain TCP); return (server, port)."""
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


# ---------------------------------------------------------------------------
# parse_headers
# ---------------------------------------------------------------------------

class TestParseHeaders:
    def test_basic(self):
        raw = b"Host: example.com\r\nContent-Type: text/html\r\n"
        h = parse_headers(raw)
        assert h["Host"] == "example.com"
        assert h["Content-Type"] == "text/html"

    def test_empty(self):
        assert parse_headers(b"") == {}


# ---------------------------------------------------------------------------
# TLS
# ---------------------------------------------------------------------------

class TestTLS:
    @pytest.mark.asyncio
    async def test_tls_handshake_succeeds(self, settings):
        """Client can complete a TLS handshake with the home-node."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            # Connection succeeded — send a minimal request to confirm it works
            writer.write(b"GET http://127.0.0.1:1/test HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            await writer.drain()

            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            # Should get some HTTP response (502 since target doesn't exist)
            assert len(resp) > 0

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_plain_tcp_rejected(self, settings):
        """A plain TCP (non-TLS) client cannot talk to the TLS server."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", home_port)
            writer.write(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
            await writer.drain()

            # The TLS server should reject/close the non-TLS connection
            resp = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            # Either empty (connection closed) or an error — no valid HTTP response
            assert b"200" not in resp

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()


# ---------------------------------------------------------------------------
# CONNECT tunnel (over TLS)
# ---------------------------------------------------------------------------

class TestConnectTunnel:
    @pytest.mark.asyncio
    async def test_connect_tunnel_echo(self, settings):
        """CONNECT through home-node (TLS) → target echoes data back."""

        async def echo_handler(reader, writer):
            try:
                while True:
                    data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
            except (asyncio.TimeoutError, ConnectionResetError):
                pass
            finally:
                writer.close()
                await writer.wait_closed()

        target, target_port = await _start_target_server(echo_handler)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )

            # Send CONNECT to home-node
            writer.write(
                f"CONNECT 127.0.0.1:{target_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()

            # Expect 200 Connection Established
            resp = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
            assert b"200 Connection Established" in resp

            # Send data through the tunnel
            writer.write(b"ping")
            await writer.drain()
            echoed = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            assert echoed == b"ping"

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_target_unreachable(self, settings):
        """CONNECT to a port with nothing listening → 502 Bad Gateway."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"\r\n"
            )
            await writer.drain()

            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"502" in resp
            assert b"Bad Gateway" in resp

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()


# ---------------------------------------------------------------------------
# HTTP forward (over TLS)
# ---------------------------------------------------------------------------

class TestHTTPForward:
    @pytest.mark.asyncio
    async def test_http_forward_get(self, settings):
        """HTTP GET through home-node (TLS) → target returns 200 + body."""

        async def target_handler(reader, writer):
            # Read the forwarded request
            await asyncio.wait_for(reader.read(4096), timeout=5.0)
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain\r\n"
                b"Content-Length: 2\r\n"
                b"\r\n"
                b"OK"
            )
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        target, target_port = await _start_target_server(target_handler)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                f"GET http://127.0.0.1:{target_port}/test HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()

            # Read until connection closes (TLS may split headers and body)
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            text = b"".join(chunks).decode("latin-1")
            assert "200 OK" in text
            assert "OK" in text

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()

    @pytest.mark.asyncio
    async def test_http_forward_post_with_body(self, settings):
        """HTTP POST with a body is forwarded correctly over TLS."""
        received_body = []

        async def target_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(8192), timeout=5.0)
            # Extract body after \r\n\r\n
            if b"\r\n\r\n" in data:
                body = data.split(b"\r\n\r\n", 1)[1]
                received_body.append(body)
            response = (
                b"HTTP/1.1 201 Created\r\n"
                b"Content-Length: 7\r\n"
                b"\r\n"
                b"created"
            )
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        target, target_port = await _start_target_server(target_handler)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            body = b'{"key": "value"}'
            writer.write(
                f"POST http://127.0.0.1:{target_port}/data HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n".encode() + body
            )
            await writer.drain()

            # Read until connection closes (TLS may split headers and body)
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            text = b"".join(chunks).decode("latin-1")
            assert "201 Created" in text
            assert "created" in text

            # Verify the target received the body
            assert len(received_body) == 1
            assert received_body[0] == body

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()

    @pytest.mark.asyncio
    async def test_http_forward_target_unreachable(self, settings):
        """HTTP forward to unreachable target → 502 Bad Gateway."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"GET http://127.0.0.1:1/nope HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"\r\n"
            )
            await writer.drain()

            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"502" in resp
            assert b"Bad Gateway" in resp

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()


# ---------------------------------------------------------------------------
# Malformed request
# ---------------------------------------------------------------------------

class TestMalformed:
    @pytest.mark.asyncio
    async def test_malformed_request(self, settings):
        """Sending garbage over TLS → 400 Bad Request."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(b"NOT_VALID\r\n\r\n")
            await writer.drain()

            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"400" in resp
            assert b"Bad Request" in resp

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()


# ---------------------------------------------------------------------------
# X-SpaceRouter header stripping
# ---------------------------------------------------------------------------

class TestHeaderStripping:
    @pytest.mark.asyncio
    async def test_spacerouter_headers_stripped(self, settings):
        """X-SpaceRouter-* and Proxy-Authorization headers must NOT reach the target."""
        received_headers = {}

        async def target_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            # Parse all headers from the forwarded request
            head = data.split(b"\r\n\r\n")[0]
            lines = head.split(b"\r\n")[1:]  # skip request line
            for line in lines:
                if b":" in line:
                    k, _, v = line.partition(b":")
                    received_headers[k.decode().strip().lower()] = v.decode().strip()

            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 2\r\n"
                b"\r\n"
                b"OK"
            )
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        target, target_port = await _start_target_server(target_handler)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                f"GET http://127.0.0.1:{target_port}/check HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"X-SpaceRouter-Request-Id: abc123\r\n"
                f"Proxy-Authorization: Basic dGVzdDp0ZXN0\r\n"
                f"\r\n".encode()
            )
            await writer.drain()

            await asyncio.wait_for(reader.read(4096), timeout=5.0)

            # These headers should have been stripped by the home-node
            assert "x-spacerouter-request-id" not in received_headers
            assert "proxy-authorization" not in received_headers
            # But Host should still be there
            assert "host" in received_headers

            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()
