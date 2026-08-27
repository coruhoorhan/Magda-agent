import pytest
from unittest.mock import AsyncMock
from magda_agent.gateway.router import GatewayRouter, UnifiedMessage
from magda_agent.gateway.channels_v4 import DiscordChannel, WhatsAppChannel, CLIChannel

@pytest.mark.asyncio
async def test_discord_channel_v4():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = DiscordChannel()
    router.register_channel("discord", channel)

    raw_event = {"content": "hello discord", "author": {"id": "111"}}
    await channel.process(raw_event, router)

    handler.assert_called_once()
    msg: UnifiedMessage = handler.call_args[0][0]
    assert msg.channel == "discord"
    assert msg.text == "hello discord"
    assert msg.user_id == "111"
    assert msg.metadata == {"raw_event": raw_event}


@pytest.mark.asyncio
async def test_whatsapp_channel_v4():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = WhatsAppChannel()
    router.register_channel("whatsapp", channel)

    raw_payload = {"messages": [{"text": {"body": "hello whatsapp"}, "from": "222"}]}
    await channel.process(raw_payload, router)

    handler.assert_called_once()
    msg: UnifiedMessage = handler.call_args[0][0]
    assert msg.channel == "whatsapp"
    assert msg.text == "hello whatsapp"
    assert msg.user_id == "222"
    assert msg.metadata == {"raw_payload": raw_payload}


@pytest.mark.asyncio
async def test_whatsapp_channel_v4_empty():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = WhatsAppChannel()

    raw_payload = {"messages": []}
    result = await channel.process(raw_payload, router)

    handler.assert_not_called()
    assert result is None


@pytest.mark.asyncio
async def test_cli_channel_v4():
    router = GatewayRouter()
    handler = AsyncMock()
    router.set_message_handler(handler)
    channel = CLIChannel()
    router.register_channel("cli", channel)

    raw_event = {"input": "hello cli", "user": "local_dev"}
    await channel.process(raw_event, router)

    handler.assert_called_once()
    msg: UnifiedMessage = handler.call_args[0][0]
    assert msg.channel == "cli"
    assert msg.text == "hello cli"
    assert msg.user_id == "local_dev"
    assert msg.metadata == {"raw_cli_event": raw_event}
