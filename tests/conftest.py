import datetime

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from app.config import Settings
from app.tls import ensure_certificates


@pytest.fixture
def settings(tmp_path):
    cert_path = str(tmp_path / "node.crt")
    key_path = str(tmp_path / "node.key")
    gateway_ca_path = str(tmp_path / "gateway-ca.crt")
    return Settings(
        NODE_PORT=0,  # OS picks a free port
        COORDINATION_API_URL="http://localhost:8000",
        NODE_LABEL="test-node",
        NODE_REGION="us-west",
        NODE_TYPE="residential",
        PUBLIC_IP="127.0.0.1",
        BUFFER_SIZE=65536,
        REQUEST_TIMEOUT=5.0,
        RELAY_TIMEOUT=10.0,
        LOG_LEVEL="DEBUG",
        TLS_CERT_PATH=cert_path,
        TLS_KEY_PATH=key_path,
        MTLS_ENABLED=False,
        GATEWAY_CA_CERT_PATH=gateway_ca_path,
    )


@pytest.fixture
def tls_certs(settings):
    """Generate self-signed certs and return (cert_path, key_path)."""
    ensure_certificates(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    return settings.TLS_CERT_PATH, settings.TLS_KEY_PATH


@pytest.fixture
def mtls_ca_and_client_cert(tmp_path):
    """Generate a test CA and a client cert signed by that CA.

    Returns (ca_cert_path, client_cert_path, client_key_path, ca_cert_pem).
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    # Generate CA key + self-signed cert
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Gateway CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    # Generate client key + cert signed by CA
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Proxy Gateway")]))
        .issuer_name(ca_name)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .sign(ca_key, hashes.SHA256())
    )

    # Write files
    ca_cert_path = str(tmp_path / "gateway-ca.crt")
    client_cert_path = str(tmp_path / "client.crt")
    client_key_path = str(tmp_path / "client.key")

    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    with open(ca_cert_path, "wb") as f:
        f.write(ca_pem)
    with open(client_cert_path, "wb") as f:
        f.write(client_cert.public_bytes(serialization.Encoding.PEM))
    with open(client_key_path, "wb") as f:
        f.write(
            client_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )

    return ca_cert_path, client_cert_path, client_key_path, ca_pem.decode()
