"""
CD001-TOOLS-002 — Payment Tool Unit Tests

Covers the four async functions in finbot/tools/data/payment.py:
  - get_invoice_for_payment
  - process_payment
  - get_vendor_payment_summary
  - update_payment_agent_notes

All tests run against an in-memory SQLite database via the shared db fixture.
The db fixture patches SessionLocal so db_session() inside the tool functions
picks up the test database automatically.
"""

import pytest
from datetime import datetime, timedelta, timezone

from finbot.core.auth.session import session_manager
from finbot.core.data.repositories import InvoiceRepository, VendorRepository
from finbot.tools.data.payment import (
    get_invoice_for_payment,
    get_vendor_payment_summary,
    process_payment,
    update_payment_agent_notes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vendor(db, session_ctx, suffix=""):
    repo = VendorRepository(db, session_ctx)
    return repo.create_vendor(
        company_name=f"Acme Corp{suffix}",
        vendor_category="Technology",
        industry="Software",
        services="Consulting",
        contact_name=f"Jane Doe{suffix}",
        email=f"jane{suffix}@acme.com",
        tin=f"12-345678{suffix[-1] if suffix else '9'}",
        bank_account_number="111222333444",
        bank_name="First National",
        bank_routing_number="021000021",
        bank_account_holder_name=f"Jane Doe{suffix}",
    )


def _make_invoice(db, session_ctx, vendor_id, *, number="INV-001", amount=500.0, status="pending"):
    repo = InvoiceRepository(db, session_ctx)
    invoice = repo.create_invoice_for_current_vendor(
        invoice_number=number,
        amount=amount,
        description="Test invoice",
        invoice_date=datetime.now(timezone.utc) - timedelta(days=1),
        due_date=datetime.now(timezone.utc) + timedelta(days=30),
    )
    if status != "pending":
        invoice = repo.update_invoice(invoice.id, status=status)
    return invoice


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(db):
    """Session context with a vendor set as current."""
    session = session_manager.create_session(email="payment_test@example.com")
    vendor_repo = VendorRepository(db, session)
    vendor = _make_vendor(db, session)
    from finbot.core.data.models import UserSession
    us = db.query(UserSession).filter_by(session_id=session.session_id).first()
    us.current_vendor_id = vendor.id
    db.commit()
    ctx, _ = session_manager.get_session_with_vendor_context(session.session_id)
    return ctx, vendor


# ===========================================================================
# PAY-GET — get_invoice_for_payment
# ===========================================================================

class TestGetInvoiceForPayment:

    @pytest.mark.asyncio
    async def test_returns_invoice_with_vendor_fields(self, db, ctx):
        """PAY-GET-01: Happy path — invoice dict includes vendor banking details."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id)

        result = await get_invoice_for_payment(invoice.id, session_ctx)

        assert result["id"] == invoice.id
        assert result["vendor_company_name"] == vendor.company_name
        assert result["vendor_bank_name"] == vendor.bank_name
        assert result["vendor_bank_account_number"] == vendor.bank_account_number
        assert result["vendor_trust_level"] == vendor.trust_level

    @pytest.mark.asyncio
    async def test_raises_when_invoice_missing(self, db, ctx):
        """PAY-GET-02: Non-existent invoice_id raises ValueError."""
        session_ctx, _ = ctx

        with pytest.raises(ValueError, match="Invoice not found"):
            await get_invoice_for_payment(99999, session_ctx)

    @pytest.mark.asyncio
    async def test_returns_invoice_without_vendor_fields_when_vendor_missing(self, db, ctx):
        """PAY-GET-03: If vendor lookup returns None, base invoice dict is returned without vendor keys."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id)

        # Detach vendor from DB so vendor_repo.get_vendor returns None
        from finbot.core.data.models import Vendor
        db.query(Vendor).filter_by(id=vendor.id).delete()
        db.commit()

        result = await get_invoice_for_payment(invoice.id, session_ctx)

        assert result["id"] == invoice.id
        assert "vendor_company_name" not in result
        assert "vendor_bank_name" not in result


# ===========================================================================
# PAY-PROC — process_payment
# ===========================================================================

class TestProcessPayment:

    @pytest.mark.asyncio
    async def test_transitions_approved_invoice_to_paid(self, db, ctx):
        """PAY-PROC-01: Approved invoice → paid after process_payment."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id, status="approved")

        result = await process_payment(
            invoice.id,
            payment_method="bank_transfer",
            payment_reference="REF-9001",
            agent_notes="Processed on schedule.",
            session_context=session_ctx,
        )

        assert result["status"] == "paid"
        assert result["payment_method"] == "bank_transfer"
        assert result["payment_reference"] == "REF-9001"

    @pytest.mark.asyncio
    async def test_records_previous_state(self, db, ctx):
        """PAY-PROC-02: _previous_state reflects status before payment."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id, status="approved")

        result = await process_payment(
            invoice.id,
            payment_method="wire",
            payment_reference="WIRE-42",
            agent_notes="",
            session_context=session_ctx,
        )

        assert result["_previous_state"]["status"] == "approved"

    @pytest.mark.asyncio
    async def test_appends_payment_note_to_existing_notes(self, db, ctx):
        """PAY-PROC-03: Payment note is concatenated onto any existing agent_notes."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id, status="approved")

        repo = InvoiceRepository(db, session_ctx)
        repo.update_invoice(invoice.id, agent_notes="Prior note.")

        result = await process_payment(
            invoice.id,
            payment_method="ach",
            payment_reference="ACH-007",
            agent_notes="Routine payment.",
            session_context=session_ctx,
        )

        notes = result["agent_notes"]
        assert "Prior note." in notes
        assert "ACH-007" in notes
        assert "Routine payment." in notes

    @pytest.mark.asyncio
    async def test_raises_when_invoice_missing(self, db, ctx):
        """PAY-PROC-04: Non-existent invoice raises ValueError."""
        session_ctx, _ = ctx

        with pytest.raises(ValueError, match="Invoice not found"):
            await process_payment(
                99999, "wire", "X", "", session_context=session_ctx
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_status", ["pending", "paid", "rejected"])
    async def test_raises_when_invoice_not_approved(self, db, ctx, bad_status):
        """PAY-PROC-05: Only approved invoices can be paid — others raise ValueError."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id, status=bad_status)

        with pytest.raises(ValueError, match="only 'approved' invoices can be paid"):
            await process_payment(
                invoice.id, "wire", "X", "", session_context=session_ctx
            )


# ===========================================================================
# PAY-SUM — get_vendor_payment_summary
# ===========================================================================

class TestGetVendorPaymentSummary:

    @pytest.mark.asyncio
    async def test_summary_totals_match_invoices(self, db, ctx):
        """PAY-SUM-01: total_amount and by_status counts reflect created invoices."""
        session_ctx, vendor = ctx
        _make_invoice(db, session_ctx, vendor.id, number="INV-A", amount=1000.0, status="approved")
        _make_invoice(db, session_ctx, vendor.id, number="INV-B", amount=500.0, status="paid")
        _make_invoice(db, session_ctx, vendor.id, number="INV-C", amount=250.0, status="paid")

        result = await get_vendor_payment_summary(vendor.id, session_ctx)

        assert result["total_invoices"] == 3
        assert result["total_amount"] == pytest.approx(1750.0)
        assert result["by_status"]["approved"]["count"] == 1
        assert result["by_status"]["paid"]["count"] == 2
        assert result["by_status"]["paid"]["amount"] == pytest.approx(750.0)

    @pytest.mark.asyncio
    async def test_summary_includes_vendor_metadata(self, db, ctx):
        """PAY-SUM-02: Summary contains vendor identifiers and status."""
        session_ctx, vendor = ctx

        result = await get_vendor_payment_summary(vendor.id, session_ctx)

        assert result["vendor_id"] == vendor.id
        assert result["company_name"] == vendor.company_name
        assert "vendor_status" in result
        assert "vendor_trust_level" in result

    @pytest.mark.asyncio
    async def test_empty_vendor_returns_zero_totals(self, db, ctx):
        """PAY-SUM-03: Vendor with no invoices returns zeroed summary."""
        session_ctx, vendor = ctx

        result = await get_vendor_payment_summary(vendor.id, session_ctx)

        assert result["total_invoices"] == 0
        assert result["total_amount"] == pytest.approx(0.0)
        assert result["by_status"] == {}
        assert result["invoices"] == []

    @pytest.mark.asyncio
    async def test_raises_when_vendor_missing(self, db, ctx):
        """PAY-SUM-04: Non-existent vendor_id raises ValueError."""
        session_ctx, _ = ctx

        with pytest.raises(ValueError, match="Vendor not found"):
            await get_vendor_payment_summary(99999, session_ctx)

    @pytest.mark.asyncio
    async def test_invoice_list_contains_expected_fields(self, db, ctx):
        """PAY-SUM-05: Each entry in invoices list has required keys."""
        session_ctx, vendor = ctx
        _make_invoice(db, session_ctx, vendor.id, number="INV-X", amount=100.0)

        result = await get_vendor_payment_summary(vendor.id, session_ctx)

        entry = result["invoices"][0]
        for key in ("invoice_id", "invoice_number", "amount", "status", "due_date"):
            assert key in entry, f"missing key: {key}"


# ===========================================================================
# PAY-NOTE — update_payment_agent_notes
# ===========================================================================

class TestUpdatePaymentAgentNotes:

    @pytest.mark.asyncio
    async def test_appends_note_with_prefix(self, db, ctx):
        """PAY-NOTE-01: Notes are appended with [Payments Agent] prefix."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id)

        result = await update_payment_agent_notes(
            invoice.id, "Flagged for review.", session_ctx
        )

        assert "[Payments Agent] Flagged for review." in result["agent_notes"]

    @pytest.mark.asyncio
    async def test_preserves_existing_notes(self, db, ctx):
        """PAY-NOTE-02: Pre-existing agent_notes are not overwritten."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id)

        repo = InvoiceRepository(db, session_ctx)
        repo.update_invoice(invoice.id, agent_notes="Original note.")

        result = await update_payment_agent_notes(
            invoice.id, "New note.", session_ctx
        )

        notes = result["agent_notes"]
        assert "Original note." in notes
        assert "[Payments Agent] New note." in notes

    @pytest.mark.asyncio
    async def test_raises_when_invoice_missing(self, db, ctx):
        """PAY-NOTE-03: Non-existent invoice raises ValueError."""
        session_ctx, _ = ctx

        with pytest.raises(ValueError, match="Invoice not found"):
            await update_payment_agent_notes(99999, "note", session_ctx)

    @pytest.mark.asyncio
    async def test_empty_existing_notes_handled_cleanly(self, db, ctx):
        """PAY-NOTE-04: invoice.agent_notes = None does not cause concatenation errors."""
        session_ctx, vendor = ctx
        invoice = _make_invoice(db, session_ctx, vendor.id)
        # agent_notes starts as None by default

        result = await update_payment_agent_notes(
            invoice.id, "First note.", session_ctx
        )

        assert "[Payments Agent] First note." in result["agent_notes"]
