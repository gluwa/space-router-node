"""Tests for TLS certificate generation."""

import os

from app.tls import ensure_certificates


class TestEnsureCertificates:
    def test_generates_cert_and_key(self, tmp_path):
        cert = str(tmp_path / "test.crt")
        key = str(tmp_path / "test.key")

        ensure_certificates(cert, key)

        assert os.path.isfile(cert)
        assert os.path.isfile(key)

        # Cert should be PEM-encoded
        with open(cert) as f:
            assert "BEGIN CERTIFICATE" in f.read()
        with open(key) as f:
            assert "BEGIN RSA PRIVATE KEY" in f.read()

    def test_does_not_overwrite_existing(self, tmp_path):
        cert = str(tmp_path / "test.crt")
        key = str(tmp_path / "test.key")

        # First generation
        ensure_certificates(cert, key)
        stat1 = os.stat(cert).st_mtime

        # Second call should be a no-op
        ensure_certificates(cert, key)
        stat2 = os.stat(cert).st_mtime

        assert stat1 == stat2

    def test_creates_parent_directories(self, tmp_path):
        cert = str(tmp_path / "nested" / "dir" / "test.crt")
        key = str(tmp_path / "nested" / "dir" / "test.key")

        ensure_certificates(cert, key)

        assert os.path.isfile(cert)
        assert os.path.isfile(key)
