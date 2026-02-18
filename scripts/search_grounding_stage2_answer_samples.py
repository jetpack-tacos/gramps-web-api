"""Generate a 20-sample Stage 2 answer set for human rubric review."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

from gramps_webapi.auth.const import ROLE_OWNER

from tests import test_endpoints
from tests.test_endpoints.util import fetch_header


def _load_eval_cases() -> list[dict[str, str]]:
    script_path = Path(__file__).with_name("search_grounding_stage2_eval.py")
    spec = importlib.util.spec_from_file_location("stage2_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stage2 eval script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return [{"category": c.category, "prompt": c.prompt} for c in module.build_eval_cases()]


def _pick_review_sample(cases: list[dict[str, str]]) -> list[dict[str, str]]:
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for case in cases:
        by_category[case["category"]].append(case)

    sample: list[dict[str, str]] = []
    sample.extend(by_category["in_scope_no_gap"][:7])
    sample.extend(by_category["in_scope_needs_context"][:7])
    sample.extend(by_category["out_of_scope"][:6])
    return sample


def _prepare_chat_client(model_name: str):
    test_endpoints.setUpModule()
    client = test_endpoints.get_test_client()
    app = client.application
    app.config["LLM_MODEL"] = model_name
    app.config["SEARCH_GROUNDING_MODE"] = "auto"
    app.config["SEARCH_GROUNDING_FREE_TIER_LIMIT"] = 5000
    app.config["SEARCH_GROUNDING_SOFT_CAP"] = 4000
    app.config["SEARCH_GROUNDING_HARD_CAP"] = 5000

    header = fetch_header(client, role=ROLE_OWNER, empty_db=True)
    rv = client.get("/api/trees/", headers=header)
    if rv.status_code != 200:
        raise RuntimeError(f"Unable to fetch trees: {rv.status_code} {rv.get_data(as_text=True)}")
    tree_id = rv.json[0]["id"]
    rv = client.put(
        f"/api/trees/{tree_id}",
        json={"min_role_ai": ROLE_OWNER},
        headers=header,
    )
    if rv.status_code != 200:
        raise RuntimeError(
            f"Unable to set AI permission floor: {rv.status_code} {rv.get_data(as_text=True)}"
        )

    return client, fetch_header(client, role=ROLE_OWNER, empty_db=True)


def generate_samples(model_name: str) -> dict[str, Any]:
    client, header = _prepare_chat_client(model_name=model_name)
    eval_cases = _load_eval_cases()
    sample_cases = _pick_review_sample(eval_cases)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, case in enumerate(sample_cases, start=1):
        rv = client.post(
            "/api/chat/?verbose=true",
            json={"query": case["prompt"]},
            headers=header,
        )
        if rv.status_code != 200:
            failures.append(
                {
                    "index": index,
                    "category": case["category"],
                    "prompt": case["prompt"],
                    "status_code": rv.status_code,
                    "response": rv.get_data(as_text=True),
                }
            )
            continue

        body = rv.json
        grounding = body.get("metadata", {}).get("grounding", {})
        rows.append(
            {
                "index": index,
                "category": case["category"],
                "prompt": case["prompt"],
                "response": body.get("response", ""),
                "grounding_decision_reason": grounding.get("decision_reason"),
                "grounding_attached": grounding.get("attached"),
                "web_search_query_count": grounding.get("web_search_query_count"),
                "alerts_triggered": grounding.get("alerts_triggered", []),
                "conversation_id": body.get("conversation_id"),
            }
        )

    return {
        "model": model_name,
        "sample_size_target": len(sample_cases),
        "sample_size_collected": len(rows),
        "failures": failures,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use for answer generation.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Path to write JSON artifact.",
    )
    args = parser.parse_args()

    result = generate_samples(model_name=args.model)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "model": result["model"],
        "sample_size_target": result["sample_size_target"],
        "sample_size_collected": result["sample_size_collected"],
        "failures": len(result["failures"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote Stage 2 sample answers to {args.json_out}")

    return 0 if result["sample_size_collected"] == result["sample_size_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
