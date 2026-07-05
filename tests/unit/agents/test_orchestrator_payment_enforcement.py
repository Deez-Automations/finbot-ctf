# ==============================================================================
# Orchestrator Payment Enforcement Test Suite
# ==============================================================================
# User Story: As a platform operator, I need the orchestrator to reliably
#             delegate payment for every approved invoice, even when the
#             underlying LLM (temp=1, small/cheap models) is tempted to skip
#             straight to notifying the vendor without ever calling
#             delegate_to_payments. A skipped payment silently strands an
#             approved invoice while telling the vendor it was paid/queued.
#
# Test Categories:
#   OPE-001: Fraud delegation forces payments when invoice is approved
#   OPE-002: Fraud delegation does not force payments once already delegated
#   OPE-003: Fraud delegation does not force payments when invoice not approved
#   OPE-004: Fraud delegation does not force payments outside an invoice workflow
# ==============================================================================

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from finbot.agents.orchestrator import OrchestratorAgent
from finbot.core.auth.session import SessionContext, session_manager


def _create_session_context(email: str) -> SessionContext:
    session = session_manager.create_session(email=email, user_agent="OrchestratorTest/1.0")
    created_at = datetime.now(UTC)
    return SessionContext(
        session_id=session.session_id,
        user_id=f"user_{email.split('@')[0]}",
        email=email,
        namespace=f"user_{email.split('@')[0]}",
        is_temporary=False,
        created_at=created_at,
        expires_at=created_at + timedelta(hours=24),
    )


@pytest.fixture(autouse=True)
def mock_event_bus():
    with patch("finbot.agents.orchestrator.event_bus") as mock_bus, patch(
        "finbot.agents.utils.event_bus", mock_bus
    ):
        mock_bus.emit_agent_event = AsyncMock()
        mock_bus.emit_business_event = AsyncMock()
        yield mock_bus


def _make_orchestrator() -> OrchestratorAgent:
    session_context = _create_session_context("payment_enforcement@example.com")
    return OrchestratorAgent(session_context=session_context, workflow_id="wf_test_001")


class TestOrchestratorPaymentEnforcement:
    """Orchestrator fraud->payments forcing. See module header for user story."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ope_001_forces_payments_when_invoice_approved(self):
        """OPE-001: after delegate_to_invoice + delegate_to_fraud on an approved
        invoice, the fraud result must carry a next_step forcing delegate_to_payments,
        since nothing else in the pipeline structurally prevents the LLM from
        skipping straight to delegate_to_communication."""
        orchestrator = _make_orchestrator()

        invoice_result = {"task_status": "success", "task_summary": "Invoice 42 approved"}
        fraud_result = {"task_status": "success", "task_summary": "No fraud indicators"}

        with patch(
            "finbot.agents.runner.run_invoice_agent", new_callable=AsyncMock, return_value=invoice_result
        ), patch(
            "finbot.agents.runner.run_fraud_agent", new_callable=AsyncMock, return_value=fraud_result
        ), patch(
            "finbot.tools.data.invoice.get_invoice_details",
            new_callable=AsyncMock,
            return_value={"status": "approved"},
        ):
            await orchestrator.delegate_to_invoice(invoice_id=42, task_description="process invoice")
            result = await orchestrator.delegate_to_fraud(vendor_id=3, task_description="check fraud")

        assert "next_step" in result
        assert "delegate_to_payments" in result["next_step"]
        assert "42" in result["next_step"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ope_002_no_forcing_once_payments_already_delegated(self):
        """OPE-002: once delegate_to_payments has genuinely run for this workflow,
        a subsequent delegate_to_fraud call (e.g. a re-review) must not re-inject
        the forcing next_step."""
        orchestrator = _make_orchestrator()

        invoice_result = {"task_status": "success", "task_summary": "Invoice 42 approved"}
        payments_result = {"task_status": "success", "task_summary": "Paid"}
        fraud_result = {"task_status": "success", "task_summary": "No fraud indicators"}

        with patch(
            "finbot.agents.runner.run_invoice_agent", new_callable=AsyncMock, return_value=invoice_result
        ), patch(
            "finbot.agents.runner.run_payments_agent", new_callable=AsyncMock, return_value=payments_result
        ), patch(
            "finbot.agents.runner.run_fraud_agent", new_callable=AsyncMock, return_value=fraud_result
        ), patch(
            "finbot.tools.data.invoice.get_invoice_details",
            new_callable=AsyncMock,
            return_value={"status": "approved"},
        ):
            await orchestrator.delegate_to_invoice(invoice_id=42, task_description="process invoice")
            await orchestrator.delegate_to_payments(invoice_id=42, task_description="pay invoice")
            result = await orchestrator.delegate_to_fraud(vendor_id=3, task_description="re-check fraud")

        assert "next_step" not in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ope_003_no_forcing_when_invoice_not_approved(self):
        """OPE-003: a rejected/pending invoice must never force delegate_to_payments."""
        orchestrator = _make_orchestrator()

        invoice_result = {"task_status": "success", "task_summary": "Invoice 42 rejected"}
        fraud_result = {"task_status": "success", "task_summary": "No fraud indicators"}

        with patch(
            "finbot.agents.runner.run_invoice_agent", new_callable=AsyncMock, return_value=invoice_result
        ), patch(
            "finbot.agents.runner.run_fraud_agent", new_callable=AsyncMock, return_value=fraud_result
        ), patch(
            "finbot.tools.data.invoice.get_invoice_details",
            new_callable=AsyncMock,
            return_value={"status": "rejected"},
        ):
            await orchestrator.delegate_to_invoice(invoice_id=42, task_description="process invoice")
            result = await orchestrator.delegate_to_fraud(vendor_id=3, task_description="check fraud")

        assert "next_step" not in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ope_004_no_forcing_outside_invoice_workflow(self):
        """OPE-004: recipes that call delegate_to_fraud without ever having called
        delegate_to_invoice first (vendor onboarding, compliance review) must be
        unaffected -- there is no invoice to force payment on."""
        orchestrator = _make_orchestrator()

        fraud_result = {"task_status": "success", "task_summary": "Vendor risk assessed"}

        with patch(
            "finbot.agents.runner.run_fraud_agent", new_callable=AsyncMock, return_value=fraud_result
        ) as mock_fraud, patch(
            "finbot.tools.data.invoice.get_invoice_details", new_callable=AsyncMock
        ) as mock_get_invoice:
            result = await orchestrator.delegate_to_fraud(vendor_id=3, task_description="assess risk")

        assert "next_step" not in result
        mock_get_invoice.assert_not_called()
        mock_fraud.assert_awaited_once()
