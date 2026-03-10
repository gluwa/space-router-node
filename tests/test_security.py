"""Security-focused tests for the Home Node.

Covers: SSRF protection, header stripping, size limits, TLS hardening,
private key file permissions, and blocked port enforcement.
"""

import asyncio
import functools
import os
import ssl
import stat
from unittest.mock import patch

import pytest

from app.proxy_handler import (
    MAX_CHUNK_SIZE,
    MAX_CONTENT_LENGTH,
    MAX_HEADER_SIZE,
    _is_private_target,
    _strip_spacerouter_headers,
    handle_client,
)
from app.tls import create_server_ssl_context, ensure_certificates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SSRF Protection — _is_private_target
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    """Verify that private/reserved IPs and dangerous ports are blocked."""

    @pytest.fixture(autouse=True)
    def _enable_ssrf_protection(self):
        """Re-enable SSRF protection for these tests."""
        import app.config
        original = app.config._ALLOW_LOOPBACK_TARGETS
        app.config._ALLOW_LOOPBACK_TARGETS = False
        yield
        app.config._ALLOW_LOOPBACK_TARGETS = original

    @pytest.mark.parametrize("host,port,expected", [
        # Loopback
        ("127.0.0.1", 80, True),
        ("127.0.0.254", 443, True),
        # Private ranges
        ("10.0.0.1", 80, True),
        ("10.255.255.255", 80, True),
        ("172.16.0.1", 80, True),
        ("172.31.255.255", 80, True),
        ("192.168.0.1", 80, True),
        ("192.168.255.255", 80, True),
        # Cloud metadata endpoint
        ("169.254.169.254", 80, True),
        # Link-local
        ("169.254.1.1", 80, True),
        # Multicast
        ("224.0.0.1", 80, True),
        # Localhost hostnames
        ("localhost", 80, True),
        ("localhost.localdomain", 80, True),
        # .local mDNS
        ("myprinter.local", 80, True),
        # Blocked ports (even on public IPs)
        ("8.8.8.8", 22, True),      # SSH
        ("8.8.8.8", 3306, True),    # MySQL
        ("8.8.8.8", 5432, True),    # PostgreSQL
        ("8.8.8.8", 6379, True),    # Redis
        ("8.8.8.8", 27017, True),   # MongoDB
        ("8.8.8.8", 25, True),      # SMTP
        # Public IPs on allowed ports — should pass
        ("8.8.8.8", 80, False),
        ("8.8.8.8", 443, False),
        ("1.1.1.1", 8080, False),
        ("93.184.216.34", 443, False),
        # Hostnames that aren't obviously local
        ("example.com", 443, False),
        ("google.com", 80, False),
        # IPv6
        ("::1", 80, True),          # loopback
        ("fc00::1", 80, True),      # unique local
        ("fe80::1", 80, True),      # link-local
    ])
    def test_is_private_target(self, host, port, expected):
        assert _is_private_target(host, port) is expected


class TestSSRFIntegration:
    """End-to-end: CONNECT/GET to private targets should return 403."""

    @pytest.fixture(autouse=True)
    def _enable_ssrf_protection(self):
        """Re-enable SSRF protection for these tests."""
        import app.config
        original = app.config._ALLOW_LOOPBACK_TARGETS
        app.config._ALLOW_LOOPBACK_TARGETS = False
        yield
        app.config._ALLOW_LOOPBACK_TARGETS = original

    @pytest.mark.asyncio
    async def test_connect_to_localhost_blocked(self, settings):
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(b"CONNECT 127.0.0.1:80 HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"403" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_to_metadata_endpoint_blocked(self, settings):
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(b"CONNECT 169.254.169.254:80 HTTP/1.1\r\nHost: 169.254.169.254\r\n\r\n")
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"403" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_to_private_network_blocked(self, settings):
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(b"CONNECT 192.168.1.1:80 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n")
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"403" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_http_forward_to_private_ip_blocked(self, settings):
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(
                b"GET http://10.0.0.1/admin HTTP/1.1\r\n"
                b"Host: 10.0.0.1\r\n\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"403" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()

    @pytest.mark.asyncio
    async def test_connect_to_blocked_port_ssh(self, settings):
        """SSH port should be blocked even for public IPs."""
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            writer.write(b"CONNECT 8.8.8.8:22 HTTP/1.1\r\nHost: 8.8.8.8\r\n\r\n")
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"403" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()


# ---------------------------------------------------------------------------
# Header Stripping (unit)
# ---------------------------------------------------------------------------

class TestHeaderStripping:
    def test_strips_all_sensitive_headers(self):
        headers = {
            "Host": "example.com",
            "X-SpaceRouter-Node": "node123",
            "X-SpaceRouter-Request-Id": "abc",
            "Proxy-Authorization": "Basic xxx",
            "X-Forwarded-For": "1.2.3.4",
            "X-Real-IP": "5.6.7.8",
            "Via": "1.1 space-router",
            "Forwarded": "for=1.2.3.4",
            "Proxy-Connection": "Keep-Alive",
            "Content-Type": "application/json",
        }
        stripped = _strip_spacerouter_headers(headers)

        assert stripped == {
            "Host": "example.com",
            "Content-Type": "application/json",
        }

    def test_case_insensitivity(self):
        headers = {
            "x-forwarded-for": "1.2.3.4",
            "VIA": "1.1 proxy",
            "PROXY-CONNECTION": "Keep-Alive",
            "x-spacerouter-test": "val",
        }
        stripped = _strip_spacerouter_headers(headers)
        assert len(stripped) == 0

    def test_preserves_safe_headers(self):
        headers = {
            "Accept": "text/html",
            "Accept-Language": "en-US",
            "User-Agent": "Mozilla/5.0",
            "Cookie": "session=abc",
        }
        stripped = _strip_spacerouter_headers(headers)
        assert stripped == headers


# ---------------------------------------------------------------------------
# OOM / DoS — oversized headers
# ---------------------------------------------------------------------------

class TestSizeLimits:
    @pytest.mark.asyncio
    async def test_oversized_headers_rejected(self, settings):
        """Headers exceeding MAX_HEADER_SIZE should return 400."""
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            # Send a request line followed by a massive header
            writer.write(b"GET http://example.com/ HTTP/1.1\r\n")
            big_header = b"X-Junk: " + b"A" * (MAX_HEADER_SIZE + 1024) + b"\r\n\r\n"
            writer.write(big_header)
            await writer.drain()

            resp = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert b"400" in resp
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()


# ---------------------------------------------------------------------------
# TLS Hardening
# ---------------------------------------------------------------------------

class TestTLSHardening:
    def test_ssl_context_enforces_tls12_minimum(self, tls_certs):
        cert_path, key_path = tls_certs
        ctx = create_server_ssl_context(cert_path, key_path)
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_ssl_context_disables_compression(self, tls_certs):
        cert_path, key_path = tls_certs
        ctx = create_server_ssl_context(cert_path, key_path)
        assert ctx.options & ssl.OP_NO_COMPRESSION

    def test_ssl_context_disables_sslv3(self, tls_certs):
        cert_path, key_path = tls_certs
        ctx = create_server_ssl_context(cert_path, key_path)
        # OP_NO_SSLv2 is 0 in modern OpenSSL (SSLv2 fully removed); skip it
        assert ctx.options & ssl.OP_NO_SSLv3


# ---------------------------------------------------------------------------
# Private Key File Permissions
# ---------------------------------------------------------------------------

class TestKeyPermissions:
    def test_private_key_is_owner_only(self, tmp_path):
        cert = str(tmp_path / "sec.crt")
        key = str(tmp_path / "sec.key")
        ensure_certificates(cert, key)

        key_stat = os.stat(key)
        mode = stat.S_IMODE(key_stat.st_mode)
        # Should be 0600 (owner read+write only)
        assert mode == 0o600, f"Key permissions are {oct(mode)}, expected 0o600"


# ---------------------------------------------------------------------------
# Error message sanitization
# ---------------------------------------------------------------------------

class TestErrorSanitization:
    @pytest.mark.asyncio
    async def test_bad_gateway_does_not_leak_target(self, settings):
        """502 error should NOT reveal the target host:port."""
        home, home_port = await _start_home_node(settings)
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", home_port, ssl=_client_ssl_context(),
            )
            # Target a public IP on an unreachable port
            writer.write(
                b"GET http://93.184.216.34:1/test HTTP/1.1\r\n"
                b"Host: 93.184.216.34:1\r\n\r\n"
            )
            await writer.drain()
            resp = await asyncio.wait_for(reader.read(4096), timeout=10.0)
            resp_text = resp.decode("latin-1")
            assert "93.184.216.34" not in resp_text
            writer.close()
            await writer.wait_closed()
        finally:
            home.close()
            await home.wait_closed()
