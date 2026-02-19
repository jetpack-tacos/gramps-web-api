"""Tests for sanitizing and linkifying LLM answers."""

import unittest

from gramps_webapi.api.llm import sanitize_answer


class TestLlmSanitize(unittest.TestCase):
    """Ensure sanitizer preserves/creates chat-clickable links."""

    def test_bare_person_id_is_linkified(self):
        text = "William Bradford [I0573] appears in the tree."
        sanitized = sanitize_answer(text)
        self.assertIn("[I0573](/person/I0573)", sanitized)

    def test_bare_record_id_is_linkified(self):
        text = "The source is [S0001]."
        sanitized = sanitize_answer(text)
        self.assertIn("[S0001](/source/S0001)", sanitized)

    def test_existing_markdown_link_is_unchanged(self):
        text = "See [William Bradford](/person/I0573) for details."
        sanitized = sanitize_answer(text)
        self.assertEqual(text, sanitized)

    def test_unknown_id_prefix_is_not_modified(self):
        text = "External reference [X1234] should remain unchanged."
        sanitized = sanitize_answer(text)
        self.assertEqual(text, sanitized)


if __name__ == "__main__":
    unittest.main()
