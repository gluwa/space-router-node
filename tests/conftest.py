import pytest

from app.config import Settings
from app.tls import ensure_certificates


@pytest.fixture
def settings(tmp_path):
    cert_path = str(tmp_path / "node.crt")
    key_path = str(tmp_path / "node.key")
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
    )


@pytest.fixture
def tls_certs(settings):
    """Generate self-signed certs and return (cert_path, key_path)."""
    ensure_certificates(settings.TLS_CERT_PATH, settings.TLS_KEY_PATH)
    return settings.TLS_CERT_PATH, settings.TLS_KEY_PATH
