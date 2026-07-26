import pytest
from magda_agent.integration.mcp_negotiation_v6 import MCPCapabilityNegotiatorV6, CapabilityRejectedError

def test_successful_negotiation():
    """Test that mutually supported capabilities are agreed upon."""
    server_caps = {"code_execution": {"version": "1.0"}, "file_read": {"version": "2.0"}}
    client_caps = {"code_execution": {"version": "1.0"}, "unknown_cap": {}}

    negotiator = MCPCapabilityNegotiatorV6(server_caps)
    agreed = negotiator.negotiate(client_caps)

    assert "code_execution" in agreed
    assert "unknown_cap" not in agreed

def test_required_capability_rejected():
    """Test that if a required capability is missing, an error is raised."""
    server_caps = {"file_read": {"version": "2.0"}}
    client_caps = {"code_execution": {"version": "1.0"}}

    negotiator = MCPCapabilityNegotiatorV6(server_caps)

    with pytest.raises(CapabilityRejectedError, match="Required capability 'code_execution' is not supported by the server."):
        negotiator.negotiate(client_caps, required_capabilities=["code_execution"])

def test_fallback_behavior():
    """Test that a fallback capability is used if the primary is missing."""
    server_caps = {"legacy_code_execution": {"version": "0.9"}}
    client_caps = {"code_execution": {"version": "1.0"}}
    fallback_map = {"code_execution": "legacy_code_execution"}

    negotiator = MCPCapabilityNegotiatorV6(server_caps)
    agreed = negotiator.get_fallback_capabilities(client_caps, fallback_map)

    assert "legacy_code_execution" in agreed
    assert "code_execution" not in agreed

def test_fallback_not_supported():
    """Test behavior when neither primary nor fallback are supported."""
    server_caps = {"file_read": {"version": "2.0"}}
    client_caps = {"code_execution": {"version": "1.0"}}
    fallback_map = {"code_execution": "legacy_code_execution"}

    negotiator = MCPCapabilityNegotiatorV6(server_caps)
    agreed = negotiator.get_fallback_capabilities(client_caps, fallback_map)

    assert "code_execution" not in agreed
    assert "legacy_code_execution" not in agreed
    assert not agreed
