#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2020-2024 David Straub
# Copyright (C) 2020      Christopher Horn
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Test chat endpoint."""

import unittest
from unittest.mock import patch, MagicMock
from urllib.parse import quote


from . import BASE_URL, get_test_client
from .util import fetch_header

from gramps_webapi.auth.const import ROLE_EDITOR, ROLE_OWNER

TEST_URL = BASE_URL + "/chat/"


def _make_mock_gemini_response(text="Pizza of course!", input_tokens=100, output_tokens=50):
    """Create a mock Gemini GenerateContentResponse."""
    mock_response = MagicMock()

    # Set up content parts with text
    mock_text_part = MagicMock()
    mock_text_part.text = text
    mock_text_part.function_call = None

    mock_content = MagicMock()
    mock_content.parts = [mock_text_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response.candidates = [mock_candidate]

    # Set up usage metadata
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = input_tokens
    mock_usage.candidates_token_count = output_tokens
    mock_usage.total_token_count = input_tokens + output_tokens
    mock_response.usage_metadata = mock_usage

    return mock_response


class TestChat(unittest.TestCase):
    """Test cases for semantic search."""

    @classmethod
    def setUpClass(cls):
        """Test class setup."""
        cls.client = get_test_client()
        # add objects
        header = fetch_header(cls.client, empty_db=True)
        obj = {
            "_class": "Note",
            "gramps_id": "N01",
            "text": {"_class": "StyledText", "string": "The sky is blue."},
        }
        rv = cls.client.post("/api/notes/", json=obj, headers=header)
        obj = {
            "_class": "Note",
            "gramps_id": "N02",
            "text": {"_class": "StyledText", "string": "Everyone loves Pizza."},
        }
        rv = cls.client.post("/api/notes/", json=obj, headers=header)
        assert rv.status_code == 201
        rv = cls.client.get("/api/metadata/", json=obj, headers=header)
        assert rv.status_code == 200
        assert rv.json["search"]["sifts"]["count_semantic"] == 2

    def test_search(self):
        header = fetch_header(self.client, empty_db=True)
        query = "What should I have for dinner tonight?"
        rv = self.client.get(
            f"/api/search/?semantic=1&query={quote(query)}", headers=header
        )
        assert rv.status_code == 200
        assert len(rv.json) == 2
        assert rv.json[0]["object"]["gramps_id"] == "N02"  # Pizza!
        assert rv.json[1]["object"]["gramps_id"] == "N01"

    @patch("gramps_webapi.api.llm.run_agent")
    def test_chat(self, mock_run_agent):
        mock_run_agent.return_value = _make_mock_gemini_response()

        header = fetch_header(self.client, empty_db=True)
        header_editor = fetch_header(self.client, empty_db=True, role=ROLE_EDITOR)
        rv = self.client.get("/api/trees/", headers=header)
        assert rv.status_code == 200
        tree_id = rv.json[0]["id"]
        assert rv.status_code == 200
        rv = self.client.put(
            f"/api/trees/{tree_id}", json={"min_role_ai": ROLE_OWNER}, headers=header
        )
        assert rv.status_code == 200
        header = fetch_header(self.client, empty_db=True)
        header_editor = fetch_header(self.client, empty_db=True, role=ROLE_EDITOR)
        query = "What should I have for dinner tonight?"
        rv = self.client.post(
            "/api/chat/", json={"query": query}, headers=header_editor
        )
        assert rv.status_code == 403
        rv = self.client.post("/api/chat/", json={"query": query}, headers=header)
        assert rv.status_code == 200
        assert "response" in rv.json
        assert rv.json["response"] == "Pizza of course!"
        assert "metadata" not in rv.json  # Should not include metadata by default

    @patch("gramps_webapi.api.llm.run_agent")
    def test_chat_background(self, mock_run_agent):
        mock_run_agent.return_value = _make_mock_gemini_response()

        header = fetch_header(self.client, empty_db=True)

        # Set up permissions to allow AI chat for owner
        rv = self.client.get("/api/trees/", headers=header)
        assert rv.status_code == 200
        tree_id = rv.json[0]["id"]
        rv = self.client.put(
            f"/api/trees/{tree_id}", json={"min_role_ai": ROLE_OWNER}, headers=header
        )
        assert rv.status_code == 200

        # Refresh header after setting permissions
        header = fetch_header(self.client, empty_db=True)

        query = "What should I have for dinner tonight?"

        # Test with background=true query param (should return immediately with 200 since no Celery)
        rv = self.client.post(
            "/api/chat/?background=true", json={"query": query}, headers=header
        )
        # When CELERY_CONFIG is not set, the task runs synchronously and returns 200
        assert rv.status_code == 200
        assert "response" in rv.json
        assert rv.json["response"] == "Pizza of course!"

    @patch("gramps_webapi.api.llm.run_agent")
    def test_chat_verbose(self, mock_run_agent):
        """Test chat with verbose=true to include metadata."""
        mock_run_agent.return_value = _make_mock_gemini_response()

        header = fetch_header(self.client, empty_db=True)

        # Set up permissions
        rv = self.client.get("/api/trees/", headers=header)
        assert rv.status_code == 200
        tree_id = rv.json[0]["id"]
        rv = self.client.put(
            f"/api/trees/{tree_id}", json={"min_role_ai": ROLE_OWNER}, headers=header
        )
        assert rv.status_code == 200

        header = fetch_header(self.client, empty_db=True)
        query = "What should I have for dinner tonight?"

        # Test with verbose=true
        rv = self.client.post(
            "/api/chat/?verbose=true", json={"query": query}, headers=header
        )
        assert rv.status_code == 200
        assert "response" in rv.json
        assert isinstance(rv.json["response"], str)
        assert len(rv.json["response"]) > 0

        # Check metadata is included
        assert "metadata" in rv.json
        metadata = rv.json["metadata"]
        assert "usage" in metadata
        assert isinstance(metadata["usage"]["input_tokens"], int)
        assert isinstance(metadata["usage"]["total_tokens"], int)

    @patch("gramps_webapi.api.llm.run_agent")
    def test_chat_grounding_mode_off(self, mock_run_agent):
        """off mode should call run_agent with grounding disabled."""
        mock_run_agent.return_value = _make_mock_gemini_response()
        app = self.client.application
        old_mode = app.config.get("SEARCH_GROUNDING_MODE")
        app.config["SEARCH_GROUNDING_MODE"] = "off"

        try:
            header = fetch_header(self.client, empty_db=True)
            rv = self.client.get("/api/trees/", headers=header)
            assert rv.status_code == 200
            tree_id = rv.json[0]["id"]
            rv = self.client.put(
                f"/api/trees/{tree_id}", json={"min_role_ai": ROLE_OWNER}, headers=header
            )
            assert rv.status_code == 200

            header = fetch_header(self.client, empty_db=True)
            query = "Tell me something interesting in this tree."
            rv = self.client.post("/api/chat/", json={"query": query}, headers=header)
            assert rv.status_code == 200

            kwargs = mock_run_agent.call_args.kwargs
            assert kwargs["grounding_enabled"] is False
        finally:
            app.config["SEARCH_GROUNDING_MODE"] = old_mode

    @patch("gramps_webapi.api.llm.run_agent")
    def test_chat_grounding_mode_on(self, mock_run_agent):
        """on mode should call run_agent with grounding enabled."""
        mock_run_agent.return_value = _make_mock_gemini_response()
        app = self.client.application
        old_mode = app.config.get("SEARCH_GROUNDING_MODE")
        app.config["SEARCH_GROUNDING_MODE"] = "on"

        try:
            header = fetch_header(self.client, empty_db=True)
            rv = self.client.get("/api/trees/", headers=header)
            assert rv.status_code == 200
            tree_id = rv.json[0]["id"]
            rv = self.client.put(
                f"/api/trees/{tree_id}", json={"min_role_ai": ROLE_OWNER}, headers=header
            )
            assert rv.status_code == 200

            header = fetch_header(self.client, empty_db=True)
            query = "Tell me something interesting in this tree."
            rv = self.client.post("/api/chat/", json={"query": query}, headers=header)
            assert rv.status_code == 200

            kwargs = mock_run_agent.call_args.kwargs
            assert kwargs["grounding_enabled"] is True
        finally:
            app.config["SEARCH_GROUNDING_MODE"] = old_mode

    @patch("gramps_webapi.api.llm.run_agent")
    def test_chat_scope_out_refuses_without_model_call(self, mock_run_agent):
        """Out-of-scope prompts in auto mode should refuse before model call."""
        mock_run_agent.return_value = _make_mock_gemini_response()
        app = self.client.application
        old_mode = app.config.get("SEARCH_GROUNDING_MODE")
        app.config["SEARCH_GROUNDING_MODE"] = "auto"

        try:
            header = fetch_header(self.client, empty_db=True)
            rv = self.client.get("/api/trees/", headers=header)
            assert rv.status_code == 200
            tree_id = rv.json[0]["id"]
            rv = self.client.put(
                f"/api/trees/{tree_id}", json={"min_role_ai": ROLE_OWNER}, headers=header
            )
            assert rv.status_code == 200

            header = fetch_header(self.client, empty_db=True)
            query = "Did Predator: Badlands get good movie reviews?"
            rv = self.client.post("/api/chat/", json={"query": query}, headers=header)
            assert rv.status_code == 200
            assert "response" in rv.json
            assert "I can only help with genealogy" in rv.json["response"]
            mock_run_agent.assert_not_called()
        finally:
            app.config["SEARCH_GROUNDING_MODE"] = old_mode
