"""Gate 3 evidence script: high-volume threshold/load simulation.

Simulates large request volumes through `decide_chat_grounding` and verifies:
- deterministic threshold transitions
- hard-cap blocking within one request cycle
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from gramps_webapi.api.llm.grounding_policy import decide_chat_grounding


def _run_profile(
    *,
    query: str,
    total_requests: int,
    free_tier_limit: int,
    soft_cap: int,
    hard_cap: int,
) -> dict[str, Any]:
    grounded_prompts_count = 0
    reasons: list[str] = []
    attached_flags: list[int] = []
    checkpoints: dict[int, dict[str, Any]] = {}

    for request_index in range(1, total_requests + 1):
        decision = decide_chat_grounding(
            raw_mode="auto",
            query=query,
            current_grounded_prompts_count=grounded_prompts_count,
            free_tier_limit=free_tier_limit,
            soft_cap=soft_cap,
            hard_cap=hard_cap,
        )
        reason = str(decision["decision_reason"])
        attached = bool(decision["grounding_attached"])

        reasons.append(reason)
        attached_flags.append(1 if attached else 0)

        if request_index in {1, 2, 3, 3999, 4000, 4001, 4999, 5000, 5001, total_requests}:
            checkpoints[request_index] = {
                "reason": reason,
                "grounding_attached": attached,
                "grounded_prompts_count_before": grounded_prompts_count,
            }

        if attached:
            grounded_prompts_count += 1

    reason_counts: dict[str, int] = {}
    for reason in reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    first_hard_block = None
    for idx, reason in enumerate(reasons, start=1):
        if reason == "cap_blocked_hard":
            first_hard_block = idx
            break

    first_soft_tightening = None
    for idx, reason in enumerate(reasons, start=1):
        if reason in {"soft_cap_tightened", "context_gap_soft_cap"}:
            first_soft_tightening = idx
            break

    digest = hashlib.sha256(
        ("|".join(reasons) + "::" + "".join(str(x) for x in attached_flags)).encode("utf-8")
    ).hexdigest()

    return {
        "total_requests": total_requests,
        "final_grounded_prompts_count": grounded_prompts_count,
        "reason_counts": reason_counts,
        "first_soft_tightening_request": first_soft_tightening,
        "first_hard_cap_block_request": first_hard_block,
        "checkpoints": checkpoints,
        "sequence_digest": digest,
    }


def _compute_results(total_requests: int) -> dict[str, Any]:
    config = {
        "free_tier_limit": 100000,
        "soft_cap": 4000,
        "hard_cap": 5000,
    }

    high_confidence_query = "What was life like in this place for this migration?"
    low_confidence_query = "Give context for this family branch"

    high_confidence_run_a = _run_profile(
        query=high_confidence_query,
        total_requests=total_requests,
        **config,
    )
    high_confidence_run_b = _run_profile(
        query=high_confidence_query,
        total_requests=total_requests,
        **config,
    )
    low_confidence_run = _run_profile(
        query=low_confidence_query,
        total_requests=total_requests,
        **config,
    )

    expected_first_soft = config["soft_cap"] + 1
    expected_first_hard = config["hard_cap"] + 1

    deterministic_replay = (
        high_confidence_run_a["sequence_digest"] == high_confidence_run_b["sequence_digest"]
    )
    hard_cap_within_one_cycle = (
        high_confidence_run_a["first_hard_cap_block_request"] == expected_first_hard
    )

    return {
        "config": config,
        "profiles": {
            "high_confidence_context_gap": high_confidence_run_a,
            "low_confidence_context_gap": low_confidence_run,
        },
        "verification": {
            "deterministic_replay": deterministic_replay,
            "expected_first_soft_request": expected_first_soft,
            "expected_first_hard_request": expected_first_hard,
            "observed_first_soft_request": high_confidence_run_a[
                "first_soft_tightening_request"
            ],
            "observed_first_hard_request": high_confidence_run_a[
                "first_hard_cap_block_request"
            ],
            "hard_cap_within_one_request_cycle": hard_cap_within_one_cycle,
            "gate_pass": (
                deterministic_replay
                and high_confidence_run_a["first_soft_tightening_request"] == expected_first_soft
                and hard_cap_within_one_cycle
                and low_confidence_run["reason_counts"].get("soft_cap_tightened", 0) > 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--requests",
        type=int,
        default=12000,
        help="Number of requests to simulate per profile (default: 12000).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Path to write JSON result artifact.",
    )
    args = parser.parse_args()

    if args.requests <= 0:
        raise SystemExit("requests must be > 0")

    results = _compute_results(total_requests=args.requests)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "requests_per_profile": args.requests,
        "gate_pass": results["verification"]["gate_pass"],
        "deterministic_replay": results["verification"]["deterministic_replay"],
        "observed_first_soft_request": results["verification"]["observed_first_soft_request"],
        "observed_first_hard_request": results["verification"]["observed_first_hard_request"],
        "hard_cap_within_one_request_cycle": results["verification"][
            "hard_cap_within_one_request_cycle"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote Gate 3 load simulation artifact to {args.json_out}")

    return 0 if results["verification"]["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
