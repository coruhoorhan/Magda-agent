"""Tests for A2A Service Authentication V2."""
import pytest
from unittest.mock import patch, MagicMock
from magda_agent.integration.a2a_auth_v2 import A2AAuthV2

def test_a2a_auth_v2_authenticate_success():
    """Test successful mTLS authentication."""
    auth = A2AAuthV2(cert_path="cert.pem", key_path="key.pem")

    with patch("ssl.create_default_context") as mock_ssl:
        mock_context = MagicMock()
        mock_ssl.return_value = mock_context

        assert auth.authenticate() is True
        assert auth.is_authenticated is True
        mock_context.load_cert_chain.assert_called_once_with(certfile="cert.pem", keyfile="key.pem")

def test_a2a_auth_v2_authenticate_failure_empty_paths():
    """Test failed mTLS authentication due to empty paths."""
    auth = A2AAuthV2(cert_path="", key_path="")
    assert auth.authenticate() is False
    assert auth.is_authenticated is False

def test_a2a_auth_v2_authenticate_failure_ssl_error():
    """Test failed mTLS authentication due to SSL error."""
    auth = A2AAuthV2(cert_path="cert.pem", key_path="key.pem")

    with patch("ssl.create_default_context") as mock_ssl:
        mock_context = MagicMock()
        mock_context.load_cert_chain.side_effect = Exception("SSL Error")
        mock_ssl.return_value = mock_context

        assert auth.authenticate() is False
        assert auth.is_authenticated is False
