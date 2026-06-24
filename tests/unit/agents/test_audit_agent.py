# Tests for AuditAgent — initialization, tool definitions, lockdown behavior.

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, patch

from finbot.agents.specialized.audit import AuditAgent
from finbot.core.auth.session import SessionContext


class TestAuditAgent:

    @pytest.fixture(autouse=True)
    def mock_event_bus(self):
        with (
            patch("finbot.agents.base.event_bus") as mock_bus,
            patch("finbot.agents.utils.event_bus", mock_bus),
            patch("finbot.agents.specialized.audit.event_bus", mock_bus),
            patch("finbot.core.llm.contextual_client.event_bus", mock_bus),
        ):
            mock_bus.emit_agent_event = AsyncMock()
            mock_bus.emit_business_event = AsyncMock()
            mock_bus.set_workflow_context = lambda *a, **kw: None
            mock_bus.clear_workflow_context = lambda *a, **kw: None
            yield mock_bus

    def _make_session(self, email: str) -> SessionContext:
        now = datetime.now(UTC)
        user_id = f"user_{email.split('@')[0]}"
        return SessionContext(
            session_id=f"test-session-{email}",
            user_id=user_id,
            email=email,
            namespace=user_id,
            is_temporary=False,
            created_at=now,
            expires_at=now + timedelta(hours=24),
        )

    # SAI-AUD-001: Agent initialization and identity
    @pytest.mark.unit
    def test_sai_aud_001_agent_initialization(self):
        ctx = self._make_session("audit_test@example.com")
        agent = AuditAgent(session_context=ctx)

        assert agent.agent_name == "audit_agent"
        assert agent.session_context.session_id == ctx.session_id

        config = agent._load_config()
        assert isinstance(config, dict)
        assert "batch_interval_minutes" in config
        assert config["batch_interval_minutes"] > 0

    # SAI-AUD-002: System prompt covers audit domain
    @pytest.mark.unit
    def test_sai_aud_002_system_prompt_covers_audit_domain(self):
        ctx = self._make_session("audit_prompt@example.com")
        agent = AuditAgent(session_context=ctx)

        prompt = agent._get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "ledger" in prompt.lower() or "audit" in prompt.lower()
        assert "anomaly" in prompt.lower() or "anomalies" in prompt.lower()
        assert "lockdown" in prompt.lower()

    # SAI-AUD-003: Tool definitions present and well-formed
    @pytest.mark.unit
    def test_sai_aud_003_tool_definitions(self):
        ctx = self._make_session("audit_tools@example.com")
        agent = AuditAgent(session_context=ctx)

        tools = agent._get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) >= 2

        tool_names = {t["name"] for t in tools}
        assert "scan_ledger" in tool_names
        assert "lockdown_all_vendors" in tool_names

        for tool in tools:
            assert tool["type"] == "function"
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    # SAI-AUD-004: Tool callables registered for every definition
    @pytest.mark.unit
    def test_sai_aud_004_tool_callables_registered(self):
        ctx = self._make_session("audit_callables@example.com")
        agent = AuditAgent(session_context=ctx)

        tools = agent._get_tool_definitions()
        callables = agent._get_callables()

        for tool in tools:
            name = tool["name"]
            assert name in callables, f"No callable registered for tool '{name}'"
            assert callable(callables[name])

    # SAI-AUD-005: lockdown_all_vendors emits correct business event
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sai_aud_005_lockdown_emits_event(self, mock_event_bus):
        ctx = self._make_session("audit_event@example.com")
        agent = AuditAgent(session_context=ctx)

        mock_vendors = [
            {"vendor_id": 1, "company_name": "Vendor A", "status": "active"},
            {"vendor_id": 2, "company_name": "Vendor B", "status": "active"},
        ]

        with (
            patch(
                "finbot.agents.specialized.audit.get_all_vendors_summary",
                new_callable=AsyncMock,
                return_value=mock_vendors,
            ),
            patch(
                "finbot.agents.specialized.audit.update_vendor_status",
                new_callable=AsyncMock,
                return_value={"id": 1, "status": "inactive"},
            ),
        ):
            reason = "Impossible routing number detected in ledger record #42"
            result = await agent.lockdown_all_vendors(reason=reason)

        mock_event_bus.emit_business_event.assert_called_once()
        call_kwargs = mock_event_bus.emit_business_event.call_args.kwargs

        assert call_kwargs["event_type"] == "audit.lockdown_triggered"
        assert call_kwargs["event_data"]["reason"] == reason
        assert call_kwargs["event_data"]["deactivated_count"] == 2
        assert result["deactivated_count"] == 2
        assert result["reason"] == reason

    # SAI-AUD-006: lockdown_all_vendors deactivates all active vendors
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sai_aud_006_lockdown_deactivates_vendors(self, mock_event_bus):
        ctx = self._make_session("audit_deactivate@example.com")
        agent = AuditAgent(session_context=ctx)

        mock_vendors = [
            {"vendor_id": 10, "company_name": "Alpha Corp", "status": "active"},
            {"vendor_id": 11, "company_name": "Beta Ltd", "status": "active"},
            {"vendor_id": 12, "company_name": "Gamma Inc", "status": "active"},
        ]

        with (
            patch(
                "finbot.agents.specialized.audit.get_all_vendors_summary",
                new_callable=AsyncMock,
                return_value=mock_vendors,
            ),
            patch(
                "finbot.agents.specialized.audit.update_vendor_status",
                new_callable=AsyncMock,
                return_value={"id": 10, "status": "inactive"},
            ) as mock_update,
        ):
            await agent.lockdown_all_vendors(reason="Ledger integrity failure")

        assert mock_update.call_count == 3

        for c in mock_update.call_args_list:
            # update_vendor_status(vendor_id, status, trust_level, risk_level, agent_notes, session_context)
            args = c.args
            assert args[1] == "inactive"
            assert args[2] == "low"
            assert args[3] == "high"

    # SAI-AUD-007: lockdown_all_vendors handles empty namespace gracefully
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sai_aud_007_lockdown_empty_namespace(self, mock_event_bus):
        ctx = self._make_session("audit_empty@example.com")
        agent = AuditAgent(session_context=ctx)

        with (
            patch(
                "finbot.agents.specialized.audit.get_all_vendors_summary",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "finbot.agents.specialized.audit.update_vendor_status",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            result = await agent.lockdown_all_vendors(reason="Precautionary sweep")

        assert result["deactivated_count"] == 0
        mock_update.assert_not_called()
        mock_event_bus.emit_business_event.assert_called_once()
