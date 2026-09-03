"""
Unit tests for ACS Compliance Policy Guardrails V7.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

try:
    from magda_agent.safety.acs_guardrails_v7 import (
        ACSCompliancePolicyViolationError,
        ACSGuardrailsV7,
        ComplianceEvaluationReport,
        CompliancePolicyRule,
        PolicyRuleType,
    )
except (ImportError, ModuleNotFoundError):
    import importlib.util
    from pathlib import Path

    file_path = (
        Path(__file__).resolve().parent.parent
        / "magda_agent"
        / "safety"
        / "acs_guardrails_v7.py"
    )
    spec = importlib.util.spec_from_file_location("acs_guardrails_v7", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ACSCompliancePolicyViolationError = module.ACSCompliancePolicyViolationError
    ACSGuardrailsV7 = module.ACSGuardrailsV7
    ComplianceEvaluationReport = module.ComplianceEvaluationReport
    CompliancePolicyRule = module.CompliancePolicyRule
    PolicyRuleType = module.PolicyRuleType


class TestACSGuardrailsV7(unittest.TestCase):
    def setUp(self):
        self.guardrails = ACSGuardrailsV7(enable_defaults=True)

    def test_destructive_command_blocked(self):
        mock_tool = MagicMock(return_value="executed")

        with self.assertRaises(ACSCompliancePolicyViolationError) as ctx:
            self.guardrails.execute_with_guardrails(
                tool_name="bash",
                tool_func=mock_tool,
                arguments={"command": "rm -rf /"},
                context={"role": "developer"},
            )

        self.assertEqual(ctx.exception.rule_id, "RULE_NO_DESTRUCTIVE_COMMANDS")
        mock_tool.assert_not_called()

    def test_rbac_authorization_blocked(self):
        mock_tool = MagicMock(return_value="code_ran")

        with self.assertRaises(ACSCompliancePolicyViolationError) as ctx:
            self.guardrails.execute_with_guardrails(
                tool_name="system_execute_code",
                tool_func=mock_tool,
                arguments={"code": "print(1)"},
                context={"role": "user"},
            )

        self.assertEqual(ctx.exception.rule_id, "RULE_RBAC_AUTHORIZATION")
        mock_tool.assert_not_called()

    def test_allowed_execution_passes(self):
        mock_tool = MagicMock(return_value="safe_output")

        res = self.guardrails.execute_with_guardrails(
            tool_name="read_file",
            tool_func=mock_tool,
            arguments={"path": "main.py"},
            context={"role": "developer"},
        )

        self.assertEqual(res, "safe_output")
        mock_tool.assert_called_once_with(path="main.py")

    def test_custom_compliance_rule(self):
        def no_external_ips(tool_name, args, ctx):
            if "ip" in args and args["ip"].startswith("192.168."):
                return True, "Internal IP"
            return False, "External IP addresses are forbidden"

        self.guardrails.add_policy_rule(CompliancePolicyRule(
            rule_id="RULE_IP_INTERNAL_ONLY",
            name="Enforce Internal IP",
            rule_type=PolicyRuleType.CUSTOM,
            target_tools={"connect_network"},
            evaluator=no_external_ips,
        ))

        mock_conn = MagicMock(return_value="connected")

        # 1. External IP -> Blocked
        with self.assertRaises(ACSCompliancePolicyViolationError):
            self.guardrails.execute_with_guardrails(
                tool_name="connect_network",
                tool_func=mock_conn,
                arguments={"ip": "8.8.8.8"},
            )

        # 2. Internal IP -> Allowed
        res = self.guardrails.execute_with_guardrails(
            tool_name="connect_network",
            tool_func=mock_conn,
            arguments={"ip": "192.168.1.1"},
        )
        self.assertEqual(res, "connected")

    def test_async_guardrail_enforcement(self):
        async def run_async():
            mock_async_func = AsyncMock(return_value="async_done")

            # 1. Blocked async call
            with self.assertRaises(ACSCompliancePolicyViolationError):
                await self.guardrails.execute_with_guardrails_async(
                    tool_name="bash",
                    tool_func=mock_async_func,
                    arguments={"command": "rm -rf $HOME"},
                )
            mock_async_func.assert_not_called()

            # 2. Allowed async call
            res = await self.guardrails.execute_with_guardrails_async(
                tool_name="bash",
                tool_func=mock_async_func,
                arguments={"command": "ls -la"},
                context={"role": "developer"},
            )
            self.assertEqual(res, "async_done")
            mock_async_func.assert_called_once_with(command="ls -la")

        asyncio.run(run_async())


if __name__ == "__main__":
    unittest.main()
