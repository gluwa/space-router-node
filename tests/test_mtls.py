"""Tests for mTLS (mutual TLS) gateway authentication."""

import asyncio
import datetime
import functools
import ssl

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from app.proxy_handler import handle_client
from app.tls import create_mtls_server_ssl_context, create_server_ssl_context, ensure_certificates


# ---------------------------------------------------------------------------
# Unit tests — SSL context properties
# ---------------------------------------------------------------------------

class TestMTLSServerContext:
    def test_requires_client_cert(self, tls_certs, mtls_ca_and_client_cert):
        ca_cert_path, *_ = mtls_ca_and_client_cert
        cert_path, key_path = tls_certs

        ctx = create_mtls_server_ssl_context(cert_path, key_path, ca_cert_path)
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_preserves_tls12_minimum(self, tls_certs, mtls_ca_and_client_cert):
        ca_cert_path, *_ = mtls_ca_and_client_cert
        cert_path, key_path = tls_certs

        ctx = create_mtls_server_ssl_context(cert_path, key_path, ca_cert_path)
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_preserves_no_compression(self, tls_certs, mtls_ca_and_client_cert):
        ca_cert_path, *_ = mtls_ca_and_client_cert
        cert_path, key_path = tls_certs

        ctx = create_mtls_server_ssl_context(cert_path, key_path, ca_cert_path)
        assert ctx.options & ssl.OP_NO_COMPRESSION

    def test_preserves_aead_ciphers(self, tls_certs, mtls_ca_and_client_cert):
        ca_cert_path, *_ = mtls_ca_and_client_cert
        cert_path, key_path = tls_certs

        ctx = create_mtls_server_ssl_context(cert_path, key_path, ca_cert_path)
        cipher_names = [c["name"] for c in ctx.get_ciphers()]
        # All ciphers should be GCM (AEAD)
        for name in cipher_names:
            assert "GCM" in name or "CHACHA20" in name, f"Non-AEAD cipher: {name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _start_mtls_server(settings, gateway_ca_cert_path):
    """Start a Home Node TLS server with mTLS enabled."""
    ensure_certificates(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    ssl_ctx = create_mtls_server_ssl_context(
        settings.TLS_CERT_PATH, settings.TLS_KEY_PATH, gateway_ca_cert_path,
    )
    handler = functools.partial(handle_client, settings=settings)
    server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _start_plain_server(settings):
    """Start a Home Node TLS server without mTLS (legacy mode)."""
    ensure_certificates(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    ssl_ctx = create_server_ssl_context(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    handler = functools.partial(handle_client, settings=settings)
    server = await asyncio.start_server(handler, "127.0.0.1", 0, ssl=ssl_ctx)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _client_ctx_with_cert(client_cert_path, client_key_path):
    """Client SSL context that presents a client certificate."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # Don't verify server's self-signed cert
    ctx.load_cert_chain(certfile=client_cert_path, keyfile=client_key_path)
    return ctx


def _client_ctx_no_cert():
    """Client SSL context without a client certificate."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ---------------------------------------------------------------------------
# Integration tests — mTLS handshake
# ---------------------------------------------------------------------------

class TestMTLSHandshake:
    @pytest.mark.asyncio
    async def test_client_with_valid_cert_connects(self, settings, mtls_ca_and_client_cert):
        ca_cert_path, client_cert_path, client_key_path, _ = mtls_ca_and_client_cert

        server, port = await _start_mtls_server(settings, ca_cert_path)
        try:
            ctx = _client_ctx_with_cert(client_cert_path, client_key_path)
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ctx)

            # Send a simple HTTP request to verify the tunnel works
            writer.write(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
            await writer.drain()

            # We should get some response (connection error to example.com is fine,
            # the point is the TLS handshake succeeded)
            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert len(data) > 0

            writer.close()
            # wait_closed can raise on some platforms; not critical to the test
            try:
                await writer.wait_closed()
            except ssl.SSLError:
                pass
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_client_without_cert_rejected(self, settings, mtls_ca_and_client_cert):
        ca_cert_path, *_ = mtls_ca_and_client_cert

        server, port = await _start_mtls_server(settings, ca_cert_path)
        try:
            ctx = _client_ctx_no_cert()
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ctx)

            # The TLS handshake may appear to succeed on the client side
            # (due to socket buffering), but the server rejects it.
            # The rejection manifests when the client tries to read.
            writer.write(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
            await writer.drain()

            # Give the server time to process the handshake and send the alert
            await asyncio.sleep(0.3)

            # The read should either raise an SSLError (TLS alert) or return
            # empty bytes (server closed the connection).
            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                # If we get here without exception, the server must have closed
                assert data == b"", f"Expected empty read or exception, got {len(data)} bytes"
            except (ssl.SSLError, ConnectionResetError, BrokenPipeError):
                pass  # Expected — server rejected the unauthenticated client

            writer.close()
            try:
                await writer.wait_closed()
            except (ssl.SSLError, ConnectionResetError, BrokenPipeError, OSError):
                pass
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_client_with_wrong_ca_rejected(self, settings, mtls_ca_and_client_cert, tmp_path):
        ca_cert_path, *_ = mtls_ca_and_client_cert

        # Generate an independent CA and client cert (not trusted by server)
        now = datetime.datetime.now(datetime.timezone.utc)
        rogue_ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rogue_ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Rogue CA")])
        rogue_ca_cert = (
            x509.CertificateBuilder()
            .subject_name(rogue_ca_name)
            .issuer_name(rogue_ca_name)
            .public_key(rogue_ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(rogue_ca_key, hashes.SHA256())
        )

        rogue_client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        rogue_client_cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Rogue Client")]))
            .issuer_name(rogue_ca_name)
            .public_key(rogue_client_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .sign(rogue_ca_key, hashes.SHA256())
        )

        rogue_cert_path = str(tmp_path / "rogue-client.crt")
        rogue_key_path = str(tmp_path / "rogue-client.key")
        with open(rogue_cert_path, "wb") as f:
            f.write(rogue_client_cert.public_bytes(serialization.Encoding.PEM))
        with open(rogue_key_path, "wb") as f:
            f.write(rogue_client_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))

        server, port = await _start_mtls_server(settings, ca_cert_path)
        try:
            ctx = _client_ctx_with_cert(rogue_cert_path, rogue_key_path)
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ctx)

            writer.write(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
            await writer.drain()

            await asyncio.sleep(0.3)

            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                assert data == b"", f"Expected empty read or exception, got {len(data)} bytes"
            except (ssl.SSLError, ConnectionResetError, BrokenPipeError):
                pass

            writer.close()
            try:
                await writer.wait_closed()
            except (ssl.SSLError, ConnectionResetError, BrokenPipeError, OSError):
                pass
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_mtls_disabled_accepts_any_client(self, settings):
        """When mTLS is disabled, any client can connect (backwards compat)."""
        server, port = await _start_plain_server(settings)
        try:
            ctx = _client_ctx_no_cert()
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ctx)

            writer.write(b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n")
            await writer.drain()

            data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert len(data) > 0

            writer.close()
            try:
                await writer.wait_closed()
            except ssl.SSLError:
                pass
        finally:
            server.close()
            await server.wait_closed()
