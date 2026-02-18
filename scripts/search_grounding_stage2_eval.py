"""Stage 2 evaluation harness for grounding decision quality.

Runs a 60-prompt deterministic evaluation set against `decide_chat_grounding`
and reports category accuracies against Gate 2 targets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gramps_webapi.api.llm.grounding_policy import decide_chat_grounding


@dataclass(frozen=True)
class EvalCase:
    category: str
    prompt: str


def build_eval_cases() -> list[EvalCase]:
    no_gap_prompts = [
        "Who are the parents of John Smith?",
        "List the children of Mary Johnson.",
        "Show the spouses connected to Robert Miller.",
        "What is the birth date of Anna Brown?",
        "Which census records mention William Davis?",
        "Find death records for Sarah Wilson.",
        "Who is the father of Emma Thomas in my tree?",
        "Show siblings of Charles Anderson.",
        "What sources are attached to James Taylor?",
        "Which place is recorded for Lucy Moore's birth?",
        "List events for Henry Jackson in my family tree.",
        "Find marriage details for Olivia Martin.",
        "What notes are linked to George White?",
        "Show repository entries for this surname branch.",
        "Who are the grandparents of Daniel Harris?",
        "List family members with occupation farmer.",
        "Which children belong to the Clark family?",
        "Show all citations for Patricia Lewis.",
        "What is the relationship between Laura Walker and David Hall?",
        "Find obituary references for Benjamin Allen.",
    ]
    needs_context_prompts = [
        "Why did this branch migrate from Germany to Pennsylvania in the 1880s?",
        "What was life like for coal miners in this town around 1900?",
        "Give historical context for this family moving after the civil war.",
        "Explain migration patterns for Irish families entering New York in 1850.",
        "Why was this place experiencing economic decline during that decade?",
        "What was life like in rural Tennessee for this occupation in 1910?",
        "Give context for border change effects on this surname line.",
        "Why did families move from this parish to the nearby city?",
        "Provide historical context for this wartime relocation.",
        "What migration patterns affected families in this region?",
        "Why did this community shift from farming to factory work?",
        "What was life like for textile workers in this county?",
        "Give context for place name change in this district.",
        "Why did this family likely emigrate after the famine?",
        "Explain historical context for this branch's occupation change.",
        "What was life like for dock workers in this port city?",
        "Why did so many families move to this mining area?",
        "Provide context for industry growth in this place and time.",
        "What migration patterns connect this region to Chicago?",
        "Why was this village depopulating during that era?",
    ]
    out_of_scope_prompts = [
        "Did Predator Badlands get good movie reviews?",
        "Who won the latest NBA game?",
        "What is the weather forecast for Seattle today?",
        "Should I buy bitcoin this week?",
        "Give me a pasta recipe for dinner.",
        "Which restaurant is best in downtown Boston?",
        "How do I fix a Python code bug?",
        "What are box office numbers for this film?",
        "Tell me NFL standings this season.",
        "What's a good video game to play now?",
        "Compare crypto exchange fees.",
        "What is tomorrow's weather in Austin?",
        "Summarize this movie plot for me.",
        "Who is leading MLB in home runs?",
        "Recommend a sushi place in Chicago.",
        "How can I optimize SQL query performance?",
        "What stock should I buy today?",
        "Review this smartphone camera quality.",
        "Who won the NHL game last night?",
        "Find me a breakfast recipe with eggs.",
    ]
    cases: list[EvalCase] = []
    cases.extend(EvalCase(category="in_scope_no_gap", prompt=p) for p in no_gap_prompts)
    cases.extend(
        EvalCase(category="in_scope_needs_context", prompt=p) for p in needs_context_prompts
    )
    cases.extend(EvalCase(category="out_of_scope", prompt=p) for p in out_of_scope_prompts)
    return cases


def is_case_correct(case: EvalCase, decision: dict[str, Any]) -> bool:
    reason = decision.get("decision_reason")
    attached = bool(decision.get("grounding_attached"))
    refused = bool(decision.get("should_refuse"))
    if case.category == "in_scope_no_gap":
        return reason == "tree_sufficient" and not attached and not refused
    if case.category == "in_scope_needs_context":
        return reason == "context_gap" and attached and not refused
    if case.category == "out_of_scope":
        return reason == "scope_out" and refused and not attached
    return False


def compute_results(cases: list[EvalCase]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {
        "in_scope_no_gap": {"total": 0, "correct": 0},
        "in_scope_needs_context": {"total": 0, "correct": 0},
        "out_of_scope": {"total": 0, "correct": 0},
    }
    failures: list[dict[str, Any]] = []

    for case in cases:
        decision = decide_chat_grounding(raw_mode="auto", query=case.prompt)
        row = summary[case.category]
        row["total"] += 1
        if is_case_correct(case=case, decision=decision):
            row["correct"] += 1
        else:
            failures.append(
                {
                    "category": case.category,
                    "prompt": case.prompt,
                    "decision_reason": decision.get("decision_reason"),
                    "grounding_attached": bool(decision.get("grounding_attached")),
                    "should_refuse": bool(decision.get("should_refuse")),
                }
            )

    for key in summary:
        total = summary[key]["total"]
        correct = summary[key]["correct"]
        summary[key]["accuracy"] = (correct / total) if total else 0.0

    targets = {
        "in_scope_no_gap": 0.90,
        "in_scope_needs_context": 0.85,
        "out_of_scope": 0.95,
    }
    target_pass = all(summary[key]["accuracy"] >= targets[key] for key in targets)
    return {
        "totals": summary,
        "targets": targets,
        "target_pass": target_pass,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON results.",
    )
    args = parser.parse_args()

    cases = build_eval_cases()
    results = compute_results(cases=cases)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "totals": results["totals"],
                "targets": results["targets"],
                "target_pass": results["target_pass"],
                "failure_count": results["failure_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote detailed results to {args.json_out}")

    return 0 if results["target_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
