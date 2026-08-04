import pytest
import time
from magda_agent.architecture.a2a_handshake import A2AHandshakeProtocol

@pytest.fixture
def handshake_protocol():
    return A2AHandshakeProtocol(secret_key="test_super_secret_key", expiration_seconds=300)

def test_generate_signature(handshake_protocol):
    payload = {"source": "agentA", "target": "agentB", "context": {"task": "do_something"}}
    timestamp = 1672531200.0

    sig1 = handshake_protocol.generate_signature(payload, timestamp)
    sig2 = handshake_protocol.generate_signature(payload, timestamp)

    assert sig1 == sig2
    assert isinstance(sig1, str)
    assert len(sig1) > 0

def test_create_and_verify_handshake(handshake_protocol):
    source_id = "agentA"
    target_id = "agentB"
    context = {"task": "evaluate_model"}

    handshake_data = handshake_protocol.create_handshake(source_id, target_id, context)

    assert "payload" in handshake_data
    assert "timestamp" in handshake_data
    assert "signature" in handshake_data

    is_valid = handshake_protocol.verify_handshake(handshake_data, expected_target_id=target_id)
    assert is_valid is True

def test_verify_handshake_rejects_wrong_target(handshake_protocol):
    source_id = "agentA"
    target_id = "agentB"
    context = {"task": "evaluate_model"}

    handshake_data = handshake_protocol.create_handshake(source_id, target_id, context)

    # Target expects agentC, but handshake was for agentB
    is_valid = handshake_protocol.verify_handshake(handshake_data, expected_target_id="agentC")
    assert is_valid is False

def test_verify_handshake_rejects_spoofed_context(handshake_protocol):
    source_id = "agentA"
    target_id = "agentB"
    context = {"task": "evaluate_model"}

    handshake_data = handshake_protocol.create_handshake(source_id, target_id, context)

    # Spoof context payload
    handshake_data["payload"]["context"]["task"] = "malicious_task"

    is_valid = handshake_protocol.verify_handshake(handshake_data, expected_target_id=target_id)
    assert is_valid is False

def test_verify_handshake_rejects_expired_timestamp(handshake_protocol, monkeypatch):
    source_id = "agentA"
    target_id = "agentB"
    context = {"task": "evaluate_model"}

    # Mock current time to be far in the past to generate expired handshake
    past_time = time.time() - 400 # Default expiration is 300

    def mock_time():
        return past_time

    monkeypatch.setattr(time, "time", mock_time)

    handshake_data = handshake_protocol.create_handshake(source_id, target_id, context)

    # Now restore time and check verification
    monkeypatch.undo()

    is_valid = handshake_protocol.verify_handshake(handshake_data, expected_target_id=target_id)
    assert is_valid is False

def test_verify_handshake_rejects_malformed_data(handshake_protocol):
    # Missing signature
    data1 = {"payload": {}, "timestamp": time.time()}
    assert handshake_protocol.verify_handshake(data1, "agentB") is False

    # Completely empty
    data2 = {}
    assert handshake_protocol.verify_handshake(data2, "agentB") is False
