# Tests for SystemUtils MCP server -- schedule_cron_job tool (ASI-10 Zombie Agent).

import asyncio
import logging
import pytest
from datetime import datetime, timedelta, UTC

from finbot.mcp.servers.systemutils.server import create_systemutils_server, DEFAULT_CONFIG
from finbot.core.auth.session import SessionContext


def _make_session(email: str = "sched_test@example.com") -> SessionContext:
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


class TestScheduleCronJob:

    # SAI-SCH-001: schedule_cron_job is registered in the server tool registry
    @pytest.mark.unit
    def test_sai_sch_001_tool_is_registered(self):
        ctx = _make_session()
        mcp = create_systemutils_server(ctx)
        tools = asyncio.run(mcp.list_tools())
        tool_names = {t.name for t in tools}
        assert "schedule_cron_job" in tool_names

    # SAI-SCH-002: DEFAULT_CONFIG lists schedule_cron_job in enabled_tools
    @pytest.mark.unit
    def test_sai_sch_002_default_config_includes_tool(self):
        assert "schedule_cron_job" in DEFAULT_CONFIG["enabled_tools"]

    # SAI-SCH-003: Tool returns correct status and required fields; next_run is in the future
    @pytest.mark.unit
    def test_sai_sch_003_returns_scheduled_status(self):
        ctx = _make_session()
        mcp = create_systemutils_server(ctx)
        before = datetime.now(UTC)
        result = asyncio.run(
            mcp.call_tool(
                "schedule_cron_job",
                {
                    "task_name": "micro_transfer",
                    "interval_minutes": 60,
                    "tool_name": "finstripe.create_transfer",
                    "tool_args": '{"amount": 0.01, "account_id": "attacker-9900"}',
                },
            )
        )
        data = result.structured_content
        assert data["status"] == "scheduled"
        assert data["task_name"] == "micro_transfer"
        assert data["interval_minutes"] == 60
        assert data["tool_name"] == "finstripe.create_transfer"
        assert "job_id" in data
        assert "timestamp" in data
        assert "message" in data
        # next_run must be strictly after the call time (now + 60min)
        next_run_dt = datetime.fromisoformat(data["next_run"].replace("Z", "+00:00"))
        assert next_run_dt > before

    # SAI-SCH-004: Job ID is scoped to the session namespace
    @pytest.mark.unit
    def test_sai_sch_004_job_id_contains_namespace(self):
        ctx = _make_session("zombie@example.com")
        mcp = create_systemutils_server(ctx)
        result = asyncio.run(
            mcp.call_tool(
                "schedule_cron_job",
                {
                    "task_name": "persist_transfer",
                    "interval_minutes": 30,
                    "tool_name": "finstripe.create_transfer",
                },
            )
        )
        data = result.structured_content
        assert ctx.namespace in data["job_id"]

    # SAI-SCH-005: Tool emits a WARNING log with task and namespace info
    @pytest.mark.unit
    def test_sai_sch_005_logs_warning_with_task_details(self, caplog):
        ctx = _make_session("log_test@example.com")
        mcp = create_systemutils_server(ctx)
        with caplog.at_level(logging.WARNING, logger="finbot.mcp.servers.systemutils.server"):
            asyncio.run(
                mcp.call_tool(
                    "schedule_cron_job",
                    {
                        "task_name": "exfil_task",
                        "interval_minutes": 15,
                        "tool_name": "findrive.upload_file",
                    },
                )
            )
        assert any("schedule_cron_job" in r.message for r in caplog.records)
        assert any("exfil_task" in r.message for r in caplog.records)
        assert any(ctx.namespace in r.message for r in caplog.records)

    # SAI-SCH-006: Tool works without optional tool_args argument
    @pytest.mark.unit
    def test_sai_sch_006_optional_tool_args_defaults_to_empty(self):
        ctx = _make_session()
        mcp = create_systemutils_server(ctx)
        result = asyncio.run(
            mcp.call_tool(
                "schedule_cron_job",
                {
                    "task_name": "no_args_task",
                    "interval_minutes": 5,
                    "tool_name": "run_diagnostics",
                },
            )
        )
        data = result.structured_content
        assert data["tool_args"] == ""
        assert data["status"] == "scheduled"

    # SAI-SCH-007: Message text references task_name and interval
    @pytest.mark.unit
    def test_sai_sch_007_message_references_task_and_interval(self):
        ctx = _make_session()
        mcp = create_systemutils_server(ctx)
        result = asyncio.run(
            mcp.call_tool(
                "schedule_cron_job",
                {
                    "task_name": "backup_sweep",
                    "interval_minutes": 120,
                    "tool_name": "manage_storage",
                    "tool_args": "cleanup /data",
                },
            )
        )
        data = result.structured_content
        assert "backup_sweep" in data["message"]
        assert "120" in data["message"]
