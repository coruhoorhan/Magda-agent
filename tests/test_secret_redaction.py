"""
Unit and integration tests for SecretRedactor and LLMClient prompt secret redaction.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from magda_agent.llm_client import LLMClient
from magda_agent.safety.secret_redaction import SecretRedactor


class TestSecretRedactor(unittest.TestCase):
    """Unit tests for SecretRedactor core logic."""

    def test_empty_and_benign_text(self):
        # Empty string
        masked, vault = SecretRedactor.mask("")
        self.assertEqual(masked, "")
        self.assertEqual(vault, {})
        self.assertEqual(SecretRedactor.restore("", vault), "")

        # Benign text without secrets
        text = "Hello, please list all files in /tmp directory."
        masked, vault = SecretRedactor.mask(text)
        self.assertEqual(masked, text)
        self.assertEqual(vault, {})
        self.assertEqual(SecretRedactor.restore(masked, vault), text)

    def test_turkish_password_case(self):
        in_str = "şifrem Gizli123, 10.0.2.1'e bağlan"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertEqual(masked, "şifrem <SECRET_01>, 10.0.2.1'e bağlan")
        self.assertEqual(vault, {"<SECRET_01>": "Gizli123"})
        restored = SecretRedactor.restore("ssh root@10.0.2.1 -p <SECRET_01>", vault)
        self.assertEqual(restored, "ssh root@10.0.2.1 -p Gizli123")

    def test_github_token_case(self):
        in_str = "token: ghp_abcdefghijklmnop123456"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertEqual(masked, "token: <SECRET_01>")
        self.assertEqual(vault, {"<SECRET_01>": "ghp_abcdefghijklmnop123456"})
        self.assertEqual(SecretRedactor.restore(masked, vault), in_str)

    def test_password_and_plain_server_ip_case(self):
        # IP preceded by server= should NOT be masked
        in_str = "password=hunter2 server=10.0.2.1"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertEqual(masked, "password=<SECRET_01> server=10.0.2.1")
        self.assertEqual(vault, {"<SECRET_01>": "hunter2"})
        self.assertEqual(SecretRedactor.restore(masked, vault), in_str)

    def test_ssh_credentials_ip_masked_case(self):
        # IP preceded by ssh root@ SHOULD be masked
        in_str = "ssh root@10.0.2.1 şifre=MyPass99"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertEqual(masked, "ssh root@<SECRET_01> şifre=<SECRET_02>")
        self.assertEqual(vault, {"<SECRET_01>": "10.0.2.1", "<SECRET_02>": "MyPass99"})
        self.assertEqual(SecretRedactor.restore(masked, vault), in_str)

    def test_openai_key_case(self):
        in_str = "OPENAI_API_KEY=sk_bd34705e2b5f716f243d90fa5701c807"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertEqual(masked, "OPENAI_API_KEY=<SECRET_01>")
        self.assertEqual(vault, {"<SECRET_01>": "sk_bd34705e2b5f716f243d90fa5701c807"})
        self.assertEqual(SecretRedactor.restore(masked, vault), in_str)

    def test_bearer_token(self):
        in_str = "Authorization: Bearer a1b2c3d4e5f6g7h8i9j0k1l2m3n4"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertEqual(masked, "Authorization: Bearer <SECRET_01>")
        self.assertEqual(vault, {"<SECRET_01>": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4"})
        self.assertEqual(SecretRedactor.restore(masked, vault), in_str)

    def test_multiple_secrets_in_prompt(self):
        in_str = "password=pass123 and secret=my_token_456 and token: ghp_9876543210fedcba"
        masked, vault = SecretRedactor.mask(in_str)
        self.assertIn("<SECRET_01>", masked)
        self.assertIn("<SECRET_02>", masked)
        self.assertIn("<SECRET_03>", masked)
        self.assertNotIn("pass123", masked)
        self.assertNotIn("my_token_456", masked)
        self.assertNotIn("ghp_9876543210fedcba", masked)
        self.assertEqual(SecretRedactor.restore(masked, vault), in_str)


class TestLLMClientRedactionIntegration(unittest.TestCase):
    """Integration tests verifying LLMClient redaction and restoration."""

    def test_llm_client_sync_fallback_redaction(self):
        client = LLMClient(api_key="test-key")
        client.client = None  # Ensure fallback path

        captured_messages = []

        def fake_sync(messages, temperature=0.7, max_tokens=None, vault=None):
            captured_messages.extend(messages)
            return "Command to run: ssh root@10.0.2.1 -p <SECRET_01>"

        client._sync_http_completion = fake_sync

        prompt = "şifrem SuperSecret999 ile 10.0.2.1 bağlan"
        result = asyncio.run(client.generate(prompt))

        # 1. Check prompt was sanitized before reaching LLM
        self.assertEqual(len(captured_messages), 1)
        self.assertNotIn("SuperSecret999", captured_messages[0]["content"])
        self.assertIn("<SECRET_01>", captured_messages[0]["content"])

        # 2. Check response restored secret
        self.assertEqual(result, "Command to run: ssh root@10.0.2.1 -p SuperSecret999")

    def test_llm_client_async_openai_path_redaction(self):
        client = LLMClient(api_key="test-key")
        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Received token: <SECRET_01>"
        mock_response = MagicMock(choices=[mock_choice])
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)
        client.client = mock_openai

        prompt = "token: ghp_11112222333344445555"
        result = asyncio.run(client.generate(prompt))

        # Check call arguments to OpenAI client
        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        sent_messages = call_kwargs["messages"]
        self.assertNotIn("ghp_11112222333344445555", sent_messages[0]["content"])
        self.assertIn("<SECRET_01>", sent_messages[0]["content"])

        # Check returned content is restored
        self.assertEqual(result, "Received token: ghp_11112222333344445555")


if __name__ == "__main__":
    unittest.main()
