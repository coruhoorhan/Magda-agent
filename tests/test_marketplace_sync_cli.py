"""Tests for the marketplace sync CLI module."""

import pytest
from unittest.mock import patch, AsyncMock

from magda_agent.skills.marketplace_sync_cli import main, parse_args


@pytest.mark.asyncio
@patch('magda_agent.skills.marketplace_sync_cli.MarketplaceSyncRoutineV4')
@patch('magda_agent.skills.marketplace_sync_cli.SkillRegistry')
async def test_marketplace_sync_cli_success(mock_registry_class, mock_sync_routine_class):
    """Test the CLI script returns 0 on successful sync."""
    mock_sync_routine_instance = mock_sync_routine_class.return_value
    mock_sync_routine_instance.run_sync_cycle = AsyncMock(return_value=5)

    argv = ["--url", "http://test.url", "--cache", ".test_cache.json"]
    result = await main(argv)

    assert result == 0
    mock_sync_routine_class.assert_called_once_with(
        registry=mock_registry_class.return_value,
        marketplace_url="http://test.url",
        cache_path=".test_cache.json"
    )
    mock_sync_routine_instance.run_sync_cycle.assert_called_once()


@pytest.mark.asyncio
@patch('magda_agent.skills.marketplace_sync_cli.MarketplaceSyncRoutineV4')
@patch('magda_agent.skills.marketplace_sync_cli.SkillRegistry')
async def test_marketplace_sync_cli_failure(mock_registry_class, mock_sync_routine_class):
    """Test the CLI script returns 1 when no skills are synced."""
    mock_sync_routine_instance = mock_sync_routine_class.return_value
    mock_sync_routine_instance.run_sync_cycle = AsyncMock(return_value=0)

    argv = []
    result = await main(argv)

    assert result == 1
    mock_sync_routine_class.assert_called_once_with(
        registry=mock_registry_class.return_value,
        marketplace_url="https://agentskills.io/api/skills",
        cache_path=".skill_cache_v4.json"
    )
    mock_sync_routine_instance.run_sync_cycle.assert_called_once()


def test_parse_args():
    """Test that command-line arguments are parsed correctly."""
    args = parse_args(["--url", "test_url", "--cache", "test_cache"])
    assert args.url == "test_url"
    assert args.cache == "test_cache"

    args = parse_args([])
    assert args.url == "https://agentskills.io/api/skills"
    assert args.cache == ".skill_cache_v4.json"
