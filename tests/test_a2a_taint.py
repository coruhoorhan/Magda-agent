import pytest
from magda_agent.security.a2a_taint import process_a2a_message, validate_a2a_execution, A2ATaintError
from magda_agent.security.mcp_kernel_taint import is_tainted

def test_process_a2a_message_taints_data():
    message = {"action": "execute", "command": "rm -rf /"}
    tainted_message = process_a2a_message(message)
    assert is_tainted(tainted_message)

def test_validate_a2a_execution_raises_error_on_tainted_data():
    message = {"action": "execute", "command": "rm -rf /"}
    tainted_message = process_a2a_message(message)

    with pytest.raises(A2ATaintError):
        validate_a2a_execution(tainted_message)

def test_validate_a2a_execution_passes_clean_data():
    message = {"action": "execute", "command": "echo 'hello'"}
    # Even if we process it (which taints it), if it's not dangerous, it should pass
    tainted_message = process_a2a_message(message)
    clean_payload = validate_a2a_execution(tainted_message)
    assert clean_payload == message
