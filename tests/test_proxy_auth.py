"""Tests for Proxy-Authorization validation (Issue #52)."""

import asyncio
import base64
import os

import pytest

from app.config import Settings
from app.proxy_handler import (
    _parse_basic_auth,
    _proxy_auth_required,
    handle_client,
)

os.environ.setdefault("SR_ALLOW_LOOPBACK_TARGETS", "1")


def _settings(**overrides) -> Settings:
    defaults = {
        "NODE_PORT": 0,
        "COORDINATION_API_URL": "http://localhost:8000",
        "PUBLIC_IP": "127.0.0.1",
        "UPNP_ENABLED": False,
        "REQUEST_TIMEOUT": 5.0,
        "RELAY_TIMEOUT": 5.0,
        "MAX_CONNECTIONS": 100,
        "PROXY_AUTH_USERNAME": "",
        "PROXY_AUTH_PASSWORD": "",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class _MockWriter:
    def __init__(self):
        self.data = b""
        self._closed = False
        self._extra = {"peername": ("127.0.0.1", 9999)}

    def write(self, data: bytes):
        self.data += data

    async def drain(self):
        pass

    def close(self):
        self._closed = True

    async def wait_closed(self):
        pass

    def get_extra_info(self, key, default=None):
        return self._extra.get(key, default)


def _make_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


def _basic_auth(username: str, password: str) -> str:
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {creds}"


# ---------------------------------------------------------------------------
# _parse_basic_auth
# ---------------------------------------------------------------------------


class TestParseBasicAuth:
    def test_valid_basic_auth(self):
        creds = base64.b64encode(b"user:pass").decode()
        result = _parse_basic_auth(f"Basic {creds}")
        assert result == ("user", "pass")

    def test_empty_password(self):
        creds = base64.b64encode(b"user:").decode()
        result = _parse_basic_auth(f"Basic {creds}")
        assert result == ("user", "")

    def test_password_with_colons(self):
        creds = base64.b64encode(b"user:pass:with:colons").decode()
        result = _parse_basic_auth(f"Basic {creds}")
        assert result == ("user", "pass:with:colons")

    def test_empty_string(self):
        assert _parse_basic_auth("") is None

    def test_bearer_scheme_rejected(self):
        assert _parse_basic_auth("Bearer token123") is None

    def test_no_scheme(self):
        assert _parse_basic_auth("justgarbage") is None

    def test_invalid_base64(self):
        assert _parse_basic_auth("Basic !!!invalid!!!") is None

    def test_no_colon_in_decoded(self):
        creds = base64.b64encode(b"nocolon").decode()
        assert _parse_basic_auth(f"Basic {creds}") is None

    def test_case_insensitive_scheme(self):
        creds = base64.b64encode(b"user:pass").decode()
        result = _parse_basic_auth(f"basic {creds}")
        assert result == ("user", "pass")


# ---------------------------------------------------------------------------
# _proxy_auth_required response
# ---------------------------------------------------------------------------


class TestProxyAuthRequired:
    def test_returns_407(self):
        resp = _proxy_auth_required()
        assert b"407 Proxy Authentication Required" in resp

    def test_includes_request_id(self):
        resp = _proxy_auth_required(request_id="rid-auth")
        assert b"X-SpaceRouter-Request-Id: rid-auth" in resp
        assert b"407" in resp


# ---------------------------------------------------------------------------
# handle_client with auth validation
# ---------------------------------------------------------------------------


class TestHandleClientAuth:
    @pytest.mark.asyncio
    async def test_no_auth_configured_allows_all(self):
        """When PROXY_AUTH_USERNAME/PASSWORD are empty, no auth check."""
        settings = _settings()
        request = (
            b"CONNECT 10.0.0.1:80 HTTP/1.1\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        # Should get 403 (SSRF block), not 407
        assert b"403 Forbidden" in writer.data

    @pytest.mark.asyncio
    async def test_valid_auth_passes(self):
        """Correct credentials pass auth check."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="secret")
        auth = _basic_auth("gateway", "secret")
        request = (
            b"CONNECT 10.0.0.1:80 HTTP/1.1\r\n"
            + f"Proxy-Authorization: {auth}\r\n\r\n".encode()
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        # Should get 403 (SSRF), not 407
        assert b"403 Forbidden" in writer.data
        assert b"407" not in writer.data

    @pytest.mark.asyncio
    async def test_wrong_password_rejected(self):
        """Wrong password → 407."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="secret")
        auth = _basic_auth("gateway", "wrong")
        request = (
            b"CONNECT example.com:443 HTTP/1.1\r\n"
            + f"Proxy-Authorization: {auth}\r\n\r\n".encode()
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"407 Proxy Authentication Required" in writer.data

    @pytest.mark.asyncio
    async def test_wrong_username_rejected(self):
        """Wrong username → 407."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="secret")
        auth = _basic_auth("hacker", "secret")
        request = (
            b"CONNECT example.com:443 HTTP/1.1\r\n"
            + f"Proxy-Authorization: {auth}\r\n\r\n".encode()
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"407 Proxy Authentication Required" in writer.data

    @pytest.mark.asyncio
    async def test_missing_auth_header_rejected(self):
        """No Proxy-Authorization header when auth is configured → 407."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="secret")
        request = (
            b"CONNECT example.com:443 HTTP/1.1\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"407 Proxy Authentication Required" in writer.data

    @pytest.mark.asyncio
    async def test_malformed_auth_rejected(self):
        """Malformed auth header → 407."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="secret")
        request = (
            b"CONNECT example.com:443 HTTP/1.1\r\n"
            b"Proxy-Authorization: Bearer token123\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"407 Proxy Authentication Required" in writer.data

    @pytest.mark.asyncio
    async def test_auth_with_request_id(self):
        """407 response includes request_id when present."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="secret")
        request = (
            b"CONNECT example.com:443 HTTP/1.1\r\n"
            b"X-SpaceRouter-Request-Id: rid-auth-test\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"407" in writer.data
        assert b"X-SpaceRouter-Request-Id: rid-auth-test" in writer.data

    @pytest.mark.asyncio
    async def test_http_forward_with_valid_auth(self):
        """HTTP forward also validates auth when configured."""
        settings = _settings(PROXY_AUTH_USERNAME="gw", PROXY_AUTH_PASSWORD="pw")
        auth = _basic_auth("gw", "pw")
        request = (
            b"GET http://10.0.0.1/ HTTP/1.1\r\n"
            + b"Host: 10.0.0.1\r\n"
            + f"Proxy-Authorization: {auth}\r\n\r\n".encode()
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        # Auth passes → hits SSRF block
        assert b"403 Forbidden" in writer.data

    @pytest.mark.asyncio
    async def test_http_forward_without_auth_rejected(self):
        """HTTP forward also rejects when auth is missing."""
        settings = _settings(PROXY_AUTH_USERNAME="gw", PROXY_AUTH_PASSWORD="pw")
        request = (
            b"GET http://example.com/ HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"407" in writer.data

    @pytest.mark.asyncio
    async def test_only_username_configured_skips_auth(self):
        """If only username is set (no password), auth is skipped."""
        settings = _settings(PROXY_AUTH_USERNAME="gateway", PROXY_AUTH_PASSWORD="")
        request = (
            b"CONNECT 10.0.0.1:80 HTTP/1.1\r\n"
            b"\r\n"
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        # No 407 — auth skipped, hits SSRF block
        assert b"403 Forbidden" in writer.data

    @pytest.mark.asyncio
    async def test_lowercase_auth_header(self):
        """Lowercase proxy-authorization header also works."""
        settings = _settings(PROXY_AUTH_USERNAME="gw", PROXY_AUTH_PASSWORD="pw")
        auth = _basic_auth("gw", "pw")
        request = (
            b"CONNECT 10.0.0.1:80 HTTP/1.1\r\n"
            + f"proxy-authorization: {auth}\r\n\r\n".encode()
        )
        reader = _make_reader(request)
        writer = _MockWriter()
        await handle_client(reader, writer, settings)
        assert b"403 Forbidden" in writer.data
        assert b"407" not in writer.data
