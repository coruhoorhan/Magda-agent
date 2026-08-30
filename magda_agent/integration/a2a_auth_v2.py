"""A2A Service Authentication V2."""

import ssl
from typing import Optional

class A2AAuthV2:
    """Handles mTLS authentication for A2A communication."""
    def __init__(self, cert_path: str, key_path: str):
        """Initialize the A2AAuthV2 instance."""
        self.cert_path = cert_path
        self.key_path = key_path
        self.is_authenticated = False
        self.ssl_context: Optional[ssl.SSLContext] = None

    def authenticate(self) -> bool:
        """Authenticate using mTLS."""
        if not self.cert_path or not self.key_path:
            self.is_authenticated = False
            return False

        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(certfile=self.cert_path, keyfile=self.key_path)
            self.is_authenticated = True
            return True
        except Exception:
            self.is_authenticated = False
            return False
