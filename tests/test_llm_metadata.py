"""Tests for LLM metadata extraction helpers."""

from types import SimpleNamespace
import unittest

from gramps_webapi.api.llm import (
    extract_grounding_stats_from_result,
    extract_metadata_from_result,
)


def _make_response(web_search_queries=None):
    """Create a minimal Gemini-like response object for metadata tests."""
    text_part = SimpleNamespace(text="ok", function_call=None)
    content = SimpleNamespace(parts=[text_part])
    candidate = SimpleNamespace(content=content)

    if web_search_queries is not None:
        candidate.grounding_metadata = SimpleNamespace(
            web_search_queries=[
                SimpleNamespace(text=query) for query in web_search_queries
            ]
        )

    usage_metadata = SimpleNamespace(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage_metadata)


class TestLlmMetadata(unittest.TestCase):
    """Test cases for metadata extraction."""

    def test_extract_metadata_without_grounding_metadata(self):
        """No grounding metadata should produce zero query count."""
        response = _make_response(web_search_queries=None)

        metadata = extract_metadata_from_result(response)

        self.assertIn("usage", metadata)
        self.assertIn("grounding", metadata)
        self.assertEqual(metadata["grounding"]["web_search_query_count"], 0)
        self.assertNotIn("web_search_queries", metadata["grounding"])

    def test_extract_metadata_with_empty_query_list(self):
        """Empty grounding queries should produce zero query count."""
        response = _make_response(web_search_queries=[])

        metadata = extract_metadata_from_result(response)

        self.assertEqual(metadata["grounding"]["web_search_query_count"], 0)
        self.assertNotIn("web_search_queries", metadata["grounding"])

    def test_extract_metadata_with_multiple_queries(self):
        """Multiple web search queries should be counted and returned."""
        response = _make_response(
            web_search_queries=["mass migration to pennsylvania 1880s", "coal mining boom"]
        )

        metadata = extract_metadata_from_result(response)
        grounding = metadata["grounding"]

        self.assertEqual(grounding["web_search_query_count"], 2)
        self.assertEqual(
            grounding["web_search_queries"],
            ["mass migration to pennsylvania 1880s", "coal mining boom"],
        )

    def test_extract_grounding_stats_without_candidates(self):
        """Missing candidates should return zero query count."""
        response = SimpleNamespace(candidates=[], usage_metadata=None)

        stats = extract_grounding_stats_from_result(response)

        self.assertEqual(stats["web_search_query_count"], 0)
        self.assertEqual(stats["web_search_queries"], [])
