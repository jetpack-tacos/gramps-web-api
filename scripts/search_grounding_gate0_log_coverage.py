"""Gate 0 evidence script: decision-reason log coverage sampling.

Runs sampled chat requests through the backend test client and verifies that
`chat_grounding_usage` log entries include `decision_reason` for >=99% of
sampled requests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from gramps_webapi.auth.const import ROLE_OWNER

from tests import test_endpoints
from tests.test_endpoints.util import fetch_header


def _load_stage2_prompts() -> list[str]:
    script_path = Path(__file__).with_name("search_grounding_stage2_eval.py")
    spec = importlib.util.spec_from_file_location("stage2_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stage2 eval script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return [case.prompt for case in module.build_eval_cases()]


def _make_mock_response(text: str, web_search_queries: list[str]) -> SimpleNamespace:
    text_part = SimpleNamespace(text=text, function_call=None)
    content = SimpleNamespace(parts=[text_part])
    candidate = SimpleNamespace(content=content)
    if web_search_queries:
        candidate.grounding_metadata = SimpleNamespace(
            web_search_queries=[SimpleNamespace(text=query) for query in web_search_queries]
        )
    usage_metadata = SimpleNamespace(
        prompt_token_count=20,
        candidates_token_count=10,
        total_token_count=30,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage_metadata)


def _mock_run_agent(
    prompt: str,
    deps: Any,
    model_name: str,
    system_prompt_override: str | None = None,
    history: list[Any] | None = None,
    grounding_enabled: bool = True,
):
    del deps, model_name, system_prompt_override, history
    queries = [f"grounding: {prompt[:48]}"] if grounding_enabled else []
    return _make_mock_response(
        text="Genealogy response stub for gate0 log coverage sampling.",
        web_search_queries=queries,
    )


class _GroundingUsageLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.raw_messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("chat_grounding_usage "):
            self.raw_messages.append(message)


def _repeat_to_size(prompts: list[str], sample_size: int) -> list[str]:
    if not prompts:
        raise ValueError("Prompt list is empty")
    repeated: list[str] = []
    while len(repeated) < sample_size:
        repeated.extend(prompts)
    return repeated[:sample_size]


def run_sampling(sample_size: int) -> dict[str, Any]:
    test_endpoints.setUpModule()
    client = test_endpoints.get_test_client()

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

    header = fetch_header(client, role=ROLE_OWNER, empty_db=True)
    prompts = _repeat_to_size(_load_stage2_prompts(), sample_size=sample_size)

    logger = client.application.logger
    capture_handler = _GroundingUsageLogHandler()
    original_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(capture_handler)

    request_failures: list[dict[str, Any]] = []

    try:
        with patch("gramps_webapi.api.llm.run_agent", side_effect=_mock_run_agent):
            for index, prompt in enumerate(prompts):
                rv = client.post("/api/chat/", json={"query": prompt}, headers=header)
                if rv.status_code != 200:
                    request_failures.append(
                        {
                            "index": index,
                            "status_code": rv.status_code,
                            "prompt": prompt,
                            "response": rv.get_data(as_text=True),
                        }
                    )
    finally:
        logger.removeHandler(capture_handler)
        logger.setLevel(original_level)

    parsed_logs: list[dict[str, Any]] = []
    malformed_logs: list[str] = []
    for message in capture_handler.raw_messages:
        payload_text = message.split("chat_grounding_usage ", 1)[1]
        try:
            parsed_logs.append(json.loads(payload_text))
        except json.JSONDecodeError:
            malformed_logs.append(message)

    decision_reason_present = sum(
        1 for payload in parsed_logs if str(payload.get("decision_reason", "")).strip()
    )
    decision_reason_coverage = (
        decision_reason_present / sample_size if sample_size else 0.0
    )
    log_emission_coverage = len(parsed_logs) / sample_size if sample_size else 0.0

    reason_counts: dict[str, int] = {}
    for payload in parsed_logs:
        reason = str(payload.get("decision_reason", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "sampled_requests": sample_size,
        "http_failures": request_failures,
        "captured_log_entries": len(parsed_logs),
        "malformed_log_entries": len(malformed_logs),
        "decision_reason_present_entries": decision_reason_present,
        "decision_reason_coverage": round(decision_reason_coverage, 6),
        "log_emission_coverage": round(log_emission_coverage, 6),
        "reason_counts": reason_counts,
        "gate_threshold": 0.99,
        "gate_pass": (
            not request_failures
            and not malformed_logs
            and decision_reason_coverage >= 0.99
            and log_emission_coverage >= 0.99
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120,
        help="Number of sampled requests to execute (default: 120).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Path to write JSON result artifact.",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise SystemExit("sample-size must be > 0")

    results = run_sampling(sample_size=args.sample_size)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "sampled_requests": results["sampled_requests"],
                "captured_log_entries": results["captured_log_entries"],
                "decision_reason_coverage": results["decision_reason_coverage"],
                "log_emission_coverage": results["log_emission_coverage"],
                "gate_pass": results["gate_pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote Gate 0 log coverage artifact to {args.json_out}")

    return 0 if results["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
