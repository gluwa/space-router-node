"""Self-signed TLS certificate generation for Home Nodes.

On first startup the Home Node auto-generates a certificate + key pair so
the Gateway ↔ Home Node link is encrypted.  Files are reused on subsequent
startups unless deleted.
"""

import datetime
import logging
import os
import ssl

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


def ensure_certificates(cert_path: str, key_path: str) -> None:
    """Create a self-signed cert + key if they don't already exist."""
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        logger.info("TLS certificates found at %s", cert_path)
        return

    logger.info("Generating self-signed TLS certificate …")

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(cert_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)

    # Generate RSA private key (4096-bit for long-term security)
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

    # Build self-signed certificate (valid 365 days)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "SpaceRouter Home Node"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=365)
        )
        .sign(key, hashes.SHA256())
    )

    # Write key with restrictive permissions (0600)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(
            fd,
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ),
        )
    finally:
        os.close(fd)

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    os.chmod(cert_path, 0o644)

    logger.info("TLS certificate written to %s / %s", cert_path, key_path)


def create_server_ssl_context(cert_path: str, key_path: str) -> ssl.SSLContext:
    """Return an SSL context configured for the Home Node server."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

    # Security Hardening: Enforce TLS 1.2 minimum and secure ciphers
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.options |= ssl.OP_NO_SSLv3
    ctx.options |= ssl.OP_NO_COMPRESSION
    ctx.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384')

    return ctx


def create_mtls_server_ssl_context(
    cert_path: str,
    key_path: str,
    gateway_ca_cert_path: str,
) -> ssl.SSLContext:
    """Return an SSL context that requires client certs signed by the gateway CA.

    Builds on :func:`create_server_ssl_context` and additionally requires a
    valid client certificate during the TLS handshake.  Connections without a
    cert (or with a cert signed by an untrusted CA) are rejected at the
    transport layer before any application code runs.
    """
    ctx = create_server_ssl_context(cert_path, key_path)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cafile=gateway_ca_cert_path)
    logger.info(
        "mTLS enabled — requiring client certs signed by CA at %s",
        gateway_ca_cert_path,
    )
    return ctx
