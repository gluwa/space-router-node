import pytest
from app.proxy_handler import _strip_spacerouter_headers

def test_strip_spacerouter_headers_standard():
    headers = {
        "Host": "example.com",
        "X-SpaceRouter-Node": "node123",
        "Proxy-Authorization": "Basic xxx",
        "X-Forwarded-For": "1.2.3.4",
        "X-Real-IP": "5.6.7.8",
        "Via": "1.1 space-router",
        "Content-Type": "application/json"
    }
    stripped = _strip_spacerouter_headers(headers)
    
    # Check what should be kept
    assert stripped["Host"] == "example.com"
    assert stripped["Content-Type"] == "application/json"
    
    # Check what should be removed
    assert "X-SpaceRouter-Node" not in stripped
    assert "Proxy-Authorization" not in stripped
    assert "X-Forwarded-For" not in stripped
    assert "X-Real-IP" not in stripped
    assert "Via" not in stripped

def test_strip_spacerouter_headers_case_insensitivity():
    headers = {
        "x-forwarded-for": "1.2.3.4",
        "VIA": "1.1 proxy",
        "Proxy-Connection": "Keep-Alive"
    }
    stripped = _strip_spacerouter_headers(headers)
    assert "x-forwarded-for" not in stripped
    assert "VIA" not in stripped
    assert "Proxy-Connection" not in stripped
