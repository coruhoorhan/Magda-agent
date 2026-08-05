import pytest
from magda_agent.integration.routing import MultiPlatformRouter, NormalizedEvent

def test_telegram_routing():
    router = MultiPlatformRouter()
    payload = {
        "message": {
            "from": {"id": 12345},
            "chat": {"id": 67890},
            "text": "/start"
        }
    }

    event = router.normalize_event("telegram", payload)

    assert isinstance(event, NormalizedEvent)
    assert event.platform == "telegram"
    assert event.user_id == "12345"
    assert event.text == "/start"
    assert event.metadata["chat_id"] == 67890
    assert event.raw_payload == payload

def test_discord_routing():
    router = MultiPlatformRouter()
    payload = {
        "author": {"id": "987654321"},
        "channel_id": "112233",
        "content": "!ping"
    }

    event = router.normalize_event("discord", payload)

    assert event.platform == "discord"
    assert event.user_id == "987654321"
    assert event.text == "!ping"
    assert event.metadata["channel_id"] == "112233"

def test_cli_routing():
    router = MultiPlatformRouter()
    payload = {
        "user": "test_user",
        "input": "hello world"
    }

    event = router.normalize_event("cli", payload)

    assert event.platform == "cli"
    assert event.user_id == "test_user"
    assert event.text == "hello world"

def test_unsupported_platform():
    router = MultiPlatformRouter()

    with pytest.raises(ValueError, match="Unsupported platform: slack"):
        router.normalize_event("slack", {"text": "hello"})

def test_invalid_telegram_payload():
    router = MultiPlatformRouter()

    with pytest.raises(ValueError, match="Invalid Telegram payload: missing user id"):
        router.normalize_event("telegram", {"message": {"text": "hello"}})

def test_invalid_discord_payload():
    router = MultiPlatformRouter()

    with pytest.raises(ValueError, match="Invalid Discord payload: missing author id"):
        router.normalize_event("discord", {"content": "hello"})

def test_invalid_cli_payload():
    router = MultiPlatformRouter()

    with pytest.raises(ValueError, match="Invalid CLI payload: missing input text"):
        router.normalize_event("cli", {"user": "bob"})
