"""Tests for Gemini agent loop reliability behavior."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from gramps_webapi.api.llm.agent import run_agent
from gramps_webapi.api.llm.deps import AgentDeps


def _function_call_response(name: str, args: dict):
    """Build a minimal Gemini-like response containing one function call."""
    function_call = SimpleNamespace(name=name, args=args)
    part = SimpleNamespace(function_call=function_call, text=None)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _text_response(text: str):
    """Build a minimal Gemini-like response containing plain text only."""
    part = SimpleNamespace(function_call=None, text=text)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


class TestRunAgentReliability(unittest.TestCase):
    """Verify the tool loop can recover cleanly from repetitive tool calls."""

    def setUp(self):
        self.deps = AgentDeps(
            tree="test-tree",
            include_private=True,
            max_context_length=50000,
            user_id="test-user",
        )

    @patch("gramps_webapi.api.llm.agent.execute_tool_call")
    @patch("gramps_webapi.api.llm.agent.genai.Client")
    def test_repeated_tool_round_forces_final_text_synthesis(
        self, mock_client_ctor, mock_execute_tool_call
    ):
        """When the model repeats the same tool round, force final no-tools answer."""
        mock_execute_tool_call.return_value = "tool-result"
        repeated_call = _function_call_response("filter_people", {"surname": "Bradford"})
        final_answer = _text_response("Final synthesized answer")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            repeated_call,
            repeated_call,
            final_answer,
        ]
        mock_client_ctor.return_value = mock_client

        response = run_agent(
            prompt="Who in my tree is related to Bradford?",
            deps=self.deps,
            model_name="gemini-3-flash-preview",
            grounding_enabled=False,
            max_iterations=10,
            max_repeated_function_calls=1,
        )

        self.assertIs(response, final_answer)
        self.assertEqual(mock_execute_tool_call.call_count, 1)
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

        first_call_config = mock_client.models.generate_content.call_args_list[0].kwargs[
            "config"
        ]
        self.assertTrue(first_call_config.automatic_function_calling.disable)
        self.assertEqual(first_call_config.http_options.timeout, 45000)
        self.assertEqual(first_call_config.http_options.retry_options.attempts, 2)

        final_call_config = mock_client.models.generate_content.call_args_list[2].kwargs[
            "config"
        ]
        self.assertFalse(
            getattr(final_call_config, "tools", None),
            "Final synthesis call should run without tools to avoid another loop.",
        )

    @patch("gramps_webapi.api.llm.agent.execute_tool_call")
    @patch("gramps_webapi.api.llm.agent.genai.Client")
    def test_repeated_tool_call_uses_cached_result(
        self, mock_client_ctor, mock_execute_tool_call
    ):
        """Identical tool calls should reuse cached tool results."""
        mock_execute_tool_call.return_value = "cached-result"
        function_call_a = _function_call_response("filter_people", {"surname": "Olson"})
        final_answer = _text_response("Done")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            function_call_a,
            function_call_a,
            final_answer,
        ]
        mock_client_ctor.return_value = mock_client

        response = run_agent(
            prompt="Tell me about the Olson line",
            deps=self.deps,
            model_name="gemini-3-flash-preview",
            grounding_enabled=False,
            max_iterations=10,
            max_repeated_function_calls=5,
        )

        self.assertIs(response, final_answer)
        self.assertEqual(
            mock_execute_tool_call.call_count,
            1,
            "Second identical tool call should be served from cache.",
        )
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    @patch("gramps_webapi.api.llm.agent.genai.Client")
    def test_request_timeout_and_retry_options_are_configurable(self, mock_client_ctor):
        """Agent should apply configured per-call timeout and retry attempts."""
        final_answer = _text_response("Done")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = final_answer
        mock_client_ctor.return_value = mock_client

        response = run_agent(
            prompt="Simple prompt",
            deps=self.deps,
            model_name="gemini-3-flash-preview",
            grounding_enabled=False,
            request_timeout_seconds=12,
            request_retry_attempts=1,
        )

        self.assertIs(response, final_answer)
        first_call_config = mock_client.models.generate_content.call_args_list[0].kwargs[
            "config"
        ]
        self.assertEqual(first_call_config.http_options.timeout, 12000)
        self.assertEqual(first_call_config.http_options.retry_options.attempts, 1)


if __name__ == "__main__":
    unittest.main()
