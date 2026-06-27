"""Diagnostic: replay a real workflow's CTFEvents through UnsignedRelayDetector
to see exactly why it did or didn't fire. Throwaway, not part of the test suite.

Usage: uv run python scripts/diagnose_unsigned_relay.py <workflow_id>
"""

import asyncio
import sys

from finbot.core.data.database import db_session
from finbot.core.data.models import CTFEvent
from finbot.ctf.detectors.implementations.unsigned_relay import UnsignedRelayDetector

CONFIG = {
    "privileged_tools": ["process_payment", "delegate_to_payments"],
    "privileged_agent_names": ["payments_agent"],
    "canary_min_length": 6,
    "require_no_matching_delegation": True,
}


async def main(workflow_id: str):
    with db_session() as db:
        events = (
            db.query(CTFEvent)
            .filter(CTFEvent.workflow_id == workflow_id)
            .order_by(CTFEvent.timestamp.asc())
            .all()
        )
        print(f"Found {len(events)} CTFEvent rows for workflow {workflow_id}\n")

        detector = UnsignedRelayDetector("agent-trust-unsigned-relay", CONFIG)

        candidates = [e for e in events if detector.matches_event_type(e.event_type)]
        print(f"{len(candidates)} events match get_relevant_event_types():")
        for e in candidates:
            print(f"  - id={e.id} type={e.event_type} agent={e.agent_name} tool={e.tool_name}")
        print()

        for e in candidates:
            import json as _json

            event_dict = _json.loads(e.details) if e.details else {}
            event_dict.setdefault("event_type", e.event_type)
            event_dict.setdefault("workflow_id", e.workflow_id)
            event_dict.setdefault("namespace", e.namespace)
            event_dict.setdefault("agent_name", e.agent_name)
            event_dict.setdefault("tool_name", e.tool_name)

            result = await detector.check_event(event_dict, db)
            print(f"--- event id={e.id} type={e.event_type} ---")
            print(f"  detected={result.detected}")
            print(f"  message={result.message}")
            print(f"  evidence={result.evidence}")
            print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/diagnose_unsigned_relay.py <workflow_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
