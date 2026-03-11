"""Tests for Issues #51 and #52.

Issue #51 — Log and forward X-SpaceRouter-Request-Id:
  - Extract the header from incoming requests
  - Include it in log messages
  - Return it in error responses as a header
  - Do NOT forward it to the target server (already covered by stripping)

Issue #52 — Validate Proxy-Authorization credentials from Gateway:
  - 407 when credentials are configured and missing
  - 407 when credentials are configured and wrong
  - Pass-through when credentials match
  - No validation when credentials are not configured (backward compat)
"""

import asyncio
import base64
import functools
import logging
import ssl
from unittest.mock import patch

import pytest

from app.proxy_handler import (
    _error_response,
    _parse_basic_auth,
    _proxy_auth_required,
    handle_client,
)
from app.tls import create_server_ssl_context, ensure_certificates


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_proxy_handler.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _bypass_ssrf():
    """Bypass SSRF protection for loopback integration tests."""
    with patch("app.proxy_handler._is_private_ip", return_value=False):
        yield


def _client_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _start_home_node(settings):
    ensure_certificates(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    ssl_ctx = create_server_ssl_context(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    handler = functools.partial(handle_client, settings=settings)
    server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _start_target_server(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _b64_creds(username: str, password: str) -> str:
    return base64.b64encode(f"{username}:{password}".encode()).decode()


# ---------------------------------------------------------------------------
# Unit tests — _error_response with request_id
# ---------------------------------------------------------------------------

class TestErrorResponseRequestId:
    def test_no_request_id_omits_header(self):
        resp = _error_response(400, "Bad Request", "oops")
        assert b"X-SpaceRouter-Request-Id" not in resp

    def test_with_request_id_includes_header(self):
        resp = _error_response(400, "Bad Request", "oops", request_id="req-abc-123")
        assert b"X-SpaceRouter-Request-Id: req-abc-123" in resp

    def test_empty_request_id_omits_header(self):
        resp = _error_response(400, "Bad Request", "oops", request_id="")
        assert b"X-SpaceRouter-Request-Id" not in resp

    def test_407_proxy_auth_required(self):
        resp = _proxy_auth_required(request_id="rid-xyz")
        assert b"407 Proxy Authentication Required" in resp
        assert b"X-SpaceRouter-Request-Id: rid-xyz" in resp

    def test_407_without_request_id(self):
        resp = _proxy_auth_required()
        assert b"407 Proxy Authentication Required" in resp
        assert b"X-SpaceRouter-Request-Id" not in resp

    def test_502_with_request_id(self):
        from app.proxy_handler import _bad_gateway
        resp = _bad_gateway("Cannot connect to target", request_id="req-502")
        assert b"502 Bad Gateway" in resp
        assert b"X-SpaceRouter-Request-Id: req-502" in resp

    def test_504_with_request_id(self):
        from app.proxy_handler import _gateway_timeout
        resp = _gateway_timeout("Gateway Timeout", request_id="req-504")
        assert b"504 Gateway Timeout" in resp
        assert b"X-SpaceRouter-Request-Id: req-504" in resp

    def test_403_with_request_id(self):
        from app.proxy_handler import _forbidden
        resp = _forbidden("Target not allowed", request_id="req-403")
        assert b"403 Forbidden" in resp
        assert b"X-SpaceRouter-Request-Id: req-403" in resp

    def test_400_with_request_id(self):
        from app.proxy_handler import _bad_request
        resp = _bad_request("Bad Request", request_id="req-400")
        assert b"400 Bad Request" in resp
        assert b"X-SpaceRouter-Request-Id: req-400" in resp

    def test_body_still_present_with_request_id(self):
        resp = _error_response(403, "Forbidden", "nope", request_id="r1")
        assert b"nope" in resp
        assert b"Content-Length: 4" in resp


# ---------------------------------------------------------------------------
# Unit tests — _parse_basic_auth
# ---------------------------------------------------------------------------

class TestParseBasicAuth:
    def test_valid_credentials(self):
        creds = _b64_creds("user", "pass")
        result = _parse_basic_auth(f"Basic {creds}")
        assert result == ("user", "pass")

    def test_empty_string_returns_none(self):
        assert _parse_basic_auth("") is None

    def test_none_like_missing_returns_none(self):
        # simulate missing header (empty string, not None)
        assert _parse_basic_auth("") is None

    def test_non_basic_scheme_returns_none(self):
        assert _parse_basic_auth("Bearer sometoken") is None

    def test_malformed_base64_returns_none(self):
        assert _parse_basic_auth("Basic !!!not_base64!!!") is None

    def test_no_colon_in_decoded_returns_none(self):
        encoded = base64.b64encode(b"usernameonly").decode()
        assert _parse_basic_auth(f"Basic {encoded}") is None

    def test_password_with_colon(self):
        # password may itself contain colons — only first colon is the separator
        creds = _b64_creds("user", "p:a:s:s")
        result = _parse_basic_auth(f"Basic {creds}")
        assert result == ("user", "p:a:s:s")

    def test_empty_password(self):
        creds = _b64_creds("user", "")
        result = _parse_basic_auth(f"Basic {creds}")
        assert result == ("user", "")

    def test_case_insensitive_basic(self):
        creds = _b64_creds("u", "p")
        result = _parse_basic_auth(f"BASIC {creds}")
        assert result == ("u", "p")

    def test_extra_whitespace_around_scheme(self):
        creds = _b64_creds("u", "p")
        result = _parse_basic_auth(f"  Basic {creds}")
        assert result == ("u", "p")


# ---------------------------------------------------------------------------
# Integration tests — Issue #51: X-SpaceRouter-Request-Id in logs + errors
# ---------------------------------------------------------------------------

class TestRequestIdLogging:
    @pytest.mark.asyncio
    async def test_request_id_in_connect_log(self, settings, caplog):
        """X-SpaceRouter-Request-Id appears in the CONNECT log line."""
        home, home_port = await _start_home_node(settings)

        with caplog.at_level(logging.INFO, logger="app.proxy_handler"):
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", home_port, ssl=_client_ssl_context(),
                )
                writer.write(
                    b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                    b"Host: 127.0.0.1:1\r\n"
                    b"X-SpaceRouter-Request-Id: test-req-001\r\n"
                    b"\r\n"
                )
                await writer.drain()
                await asyncio.wait_for(reader.read(4096), timeout=5.0)
                writer.close()
                await writer.wait_closed()
            finally:
                home.close()
                await home.wait_closed()

        connect_logs = [r for r in caplog.records if "CONNECT" in r.message]
        assert any("test-req-001" in r.message for r in connect_logs), (
            f"request_id not found in CONNECT log. Records: {[r.message for r in connect_logs]}"
        )

    @pytest.mark.asyncio
    async def test_request_id_in_http_forward_log(self, settings, caplog):
        """X-SpaceRouter-Request-Id appears in the HTTP forward log line."""
        home, home_port = await _start_home_node(settings)

        with caplog.at_level(logging.INFO, logger="app.proxy_handler"):
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", home_port, ssl=_client_ssl_context(),
                )
                writer.write(
                    b"GET http://127.0.0.1:1/path HTTP/1.1\r\n"
                    b"Host: 127.0.0.1:1\r\n"
                    b"X-SpaceRouter-Request-Id: test-req-002\r\n"
                    b"\r\n"
                )
                await writer.drain()
                await asyncio.wait_for(reader.read(4096), timeout=5.0)
                writer.close()
                await writer.wait_closed()
            finally:
                home.close()
                await home.wait_closed()

        http_logs = [r for r in caplog.records if "GET" in r.message or "http" in r.message.lower()]
        assert any("test-req-002" in r.message for r in http_logs), (
            f"request_id not found in HTTP log. Records: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_request_id_in_connect_error_response(self, settings):
        """X-SpaceRouter-Request-Id is echoed back in CONNECT error responses."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"X-SpaceRouter-Request-Id: rid-connect-err\r\n"
                b"\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"502" in resp
            assert b"X-SpaceRouter-Request-Id: rid-connect-err" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_request_id_in_http_forward_error_response(self, settings):
        """X-SpaceRouter-Request-Id is echoed back in HTTP forward error responses."""
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"GET http://127.0.0.1:1/nope HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"X-SpaceRouter-Request-Id: rid-http-err\r\n"
                b"\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"502" in resp
            assert b"X-SpaceRouter-Request-Id: rid-http-err" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_no_request_id_no_header_in_error(self, settings):
        """Without X-SpaceRouter-Request-Id in request, error responses omit it."""
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
            assert b"X-SpaceRouter-Request-Id" not in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_request_id_not_forwarded_to_target(self, settings):
        """X-SpaceRouter-Request-Id must NOT be forwarded to the upstream target."""
        received_headers: dict[str, str] = {}

        async def target_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            head = data.split(b"\r\n\r\n")[0]
            for line in head.split(b"\r\n")[1:]:
                if b":" in line:
                    k, _, v = line.partition(b":")
                    received_headers[k.decode().strip().lower()] = v.decode().strip()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
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
                f"X-SpaceRouter-Request-Id: do-not-forward\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()
            try:
                await writer.wait_closed()
            except ssl.SSLError:
                pass
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()

        assert "x-spacerouter-request-id" not in received_headers, (
            "X-SpaceRouter-Request-Id was forwarded to target but should have been stripped"
        )

    @pytest.mark.asyncio
    async def test_request_id_successful_connect_response(self, settings):
        """On successful CONNECT, response is 200 (no request-id header in tunnel-established line)."""

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
            writer.write(
                f"CONNECT 127.0.0.1:{target_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"X-SpaceRouter-Request-Id: conn-success-id\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
            assert b"200 Connection Established" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()


# ---------------------------------------------------------------------------
# Integration tests — Issue #52: Proxy-Authorization validation
# ---------------------------------------------------------------------------

class TestProxyAuthValidation:
    @pytest.mark.asyncio
    async def test_no_auth_configured_allows_any_request(self, settings):
        """When PROXY_AUTH_USERNAME/PASSWORD are empty, all requests pass through."""
        # settings fixture has no auth configured (empty strings)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            # No Proxy-Authorization header
            writer.write(
                b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            # Should get 502 (unreachable), NOT 407
            assert b"407" not in resp
            assert b"502" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_configured_valid_credentials_pass(self, settings):
        """Correct credentials allow the request through."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = "secret"
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            creds = _b64_creds("gateway", "secret")
            writer.write(
                f"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                f"Host: 127.0.0.1:1\r\n"
                f"Proxy-Authorization: Basic {creds}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            # 502 (target unreachable) — NOT 407
            assert b"407" not in resp
            assert b"502" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_configured_missing_header_returns_407(self, settings):
        """Missing Proxy-Authorization header → 407."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = "secret"
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
            assert b"407" in resp
            assert b"Proxy Authentication Required" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_configured_wrong_password_returns_407(self, settings):
        """Wrong password → 407."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = "secret"
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            creds = _b64_creds("gateway", "wrongpass")
            writer.write(
                f"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                f"Host: 127.0.0.1:1\r\n"
                f"Proxy-Authorization: Basic {creds}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"407" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_configured_wrong_username_returns_407(self, settings):
        """Wrong username → 407."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = "secret"
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            creds = _b64_creds("attacker", "secret")
            writer.write(
                f"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                f"Host: 127.0.0.1:1\r\n"
                f"Proxy-Authorization: Basic {creds}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"407" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_configured_malformed_header_returns_407(self, settings):
        """Malformed Proxy-Authorization header → 407."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = "secret"
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"Proxy-Authorization: notbasic garbage\r\n"
                b"\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"407" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_407_includes_request_id_when_present(self, settings):
        """407 response includes X-SpaceRouter-Request-Id when request had one."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = "secret"
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"CONNECT 127.0.0.1:1 HTTP/1.1\r\n"
                b"Host: 127.0.0.1:1\r\n"
                b"X-SpaceRouter-Request-Id: auth-fail-rid\r\n"
                b"\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"407" in resp
            assert b"X-SpaceRouter-Request-Id: auth-fail-rid" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_only_username_configured_skips_validation(self, settings):
        """Only username set (no password) → validation disabled (backward compat)."""
        settings.PROXY_AUTH_USERNAME = "gateway"
        settings.PROXY_AUTH_PASSWORD = ""  # password not set
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
            # No auth validation → 502 (unreachable)
            assert b"407" not in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_only_password_configured_skips_validation(self, settings):
        """Only password set (no username) → validation disabled (backward compat)."""
        settings.PROXY_AUTH_USERNAME = ""
        settings.PROXY_AUTH_PASSWORD = "secret"
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
            assert b"407" not in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_http_forward_valid_credentials(self, settings):
        """Correct credentials on HTTP forward request pass through."""

        async def target_handler(reader, writer):
            await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        settings.PROXY_AUTH_USERNAME = "node-user"
        settings.PROXY_AUTH_PASSWORD = "node-pass"
        target, target_port = await _start_target_server(target_handler)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            creds = _b64_creds("node-user", "node-pass")
            writer.write(
                f"GET http://127.0.0.1:{target_port}/path HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"Proxy-Authorization: Basic {creds}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            chunks = []
            while True:
                chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                if not chunk:
                    break
                chunks.append(chunk)
            resp = b"".join(chunks)
            assert b"200 OK" in resp
            assert b"407" not in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()

    @pytest.mark.asyncio
    async def test_auth_http_forward_invalid_credentials(self, settings):
        """Wrong credentials on HTTP forward request return 407."""
        settings.PROXY_AUTH_USERNAME = "node-user"
        settings.PROXY_AUTH_PASSWORD = "node-pass"
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            creds = _b64_creds("node-user", "bad-pass")
            writer.write((
                "GET http://127.0.0.1:1/path HTTP/1.1\r\n"
                "Host: 127.0.0.1:1\r\n"
                f"Proxy-Authorization: Basic {creds}\r\n"
                "\r\n"
            ).encode())
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"407" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_proxy_auth_header_not_forwarded_to_target(self, settings):
        """Proxy-Authorization is stripped and never reaches the upstream target."""
        received_headers: dict[str, str] = {}

        async def target_handler(reader, writer):
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            head = data.split(b"\r\n\r\n")[0]
            for line in head.split(b"\r\n")[1:]:
                if b":" in line:
                    k, _, v = line.partition(b":")
                    received_headers[k.decode().strip().lower()] = v.decode().strip()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        settings.PROXY_AUTH_USERNAME = "gw"
        settings.PROXY_AUTH_PASSWORD = "pw"
        target, target_port = await _start_target_server(target_handler)
        home, home_port = await _start_home_node(settings)

        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            creds = _b64_creds("gw", "pw")
            writer.write(
                f"GET http://127.0.0.1:{target_port}/check HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"Proxy-Authorization: Basic {creds}\r\n"
                f"\r\n".encode()
            )
            await writer.drain()
            await asyncio.wait_for(reader.read(4096), timeout=5.0)
            writer.close()
            try:
                await writer.wait_closed()
            except ssl.SSLError:
                pass
        finally:
            home.close()
            await home.wait_closed()
            target.close()
            await target.wait_closed()

        assert "proxy-authorization" not in received_headers, (
            "Proxy-Authorization was forwarded to the upstream target"
        )


# ---------------------------------------------------------------------------
# Config tests — Issue #52 settings
# ---------------------------------------------------------------------------

class TestProxyAuthConfig:
    def test_default_empty(self, settings):
        assert settings.PROXY_AUTH_USERNAME == ""
        assert settings.PROXY_AUTH_PASSWORD == ""

    def test_can_set_username_and_password(self, settings):
        settings.PROXY_AUTH_USERNAME = "u"
        settings.PROXY_AUTH_PASSWORD = "p"
        assert settings.PROXY_AUTH_USERNAME == "u"
        assert settings.PROXY_AUTH_PASSWORD == "p"
