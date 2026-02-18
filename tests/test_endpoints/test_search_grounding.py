#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      Gramps Web contributors
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#

"""Tests for search grounding admin endpoints."""

import os
import unittest
from unittest.mock import patch

from gramps.cli.clidbman import CLIDbManager
from gramps.gen.dbstate import DbState

from gramps_webapi.api.llm.grounding_usage import get_utc_month_start
from gramps_webapi.app import create_app
from gramps_webapi.auth import SearchGroundingUsageMonthly, add_user, user_db
from gramps_webapi.auth.const import ROLE_ADMIN, ROLE_MEMBER, ROLE_OWNER
from gramps_webapi.const import ENV_CONFIG_FILE, TEST_AUTH_CONFIG

from . import BASE_URL


class TestSearchGroundingEndpoints(unittest.TestCase):
    """Test permissions and payloads for search grounding endpoints."""

    def setUp(self):
        self.name = "Test Web API"
        self.dbman = CLIDbManager(DbState())
        dirpath, _name = self.dbman.create_new_db_cli(self.name, dbid="sqlite")
        tree = os.path.basename(dirpath)
        with patch.dict("os.environ", {ENV_CONFIG_FILE: TEST_AUTH_CONFIG}):
            self.app = create_app(
                config={"TESTING": True, "RATELIMIT_ENABLED": False},
                config_from_env=False,
            )
        self.client = self.app.test_client()
        with self.app.app_context():
            user_db.create_all()
            add_user(
                name="member",
                password="123",
                email="member@example.com",
                role=ROLE_MEMBER,
                tree=tree,
            )
            add_user(
                name="owner",
                password="123",
                email="owner@example.com",
                role=ROLE_OWNER,
                tree=tree,
            )
            add_user(
                name="admin",
                password="123",
                email="admin@example.com",
                role=ROLE_ADMIN,
                tree=tree,
            )
        self.ctx = self.app.test_request_context()
        self.ctx.push()
        rv = self.client.post(
            BASE_URL + "/token/", json={"username": "member", "password": "123"}
        )
        self.header_member = {"Authorization": f"Bearer {rv.json['access_token']}"}
        rv = self.client.post(
            BASE_URL + "/token/", json={"username": "owner", "password": "123"}
        )
        self.header_owner = {"Authorization": f"Bearer {rv.json['access_token']}"}
        rv = self.client.post(
            BASE_URL + "/token/", json={"username": "admin", "password": "123"}
        )
        self.header_admin = {"Authorization": f"Bearer {rv.json['access_token']}"}

    def tearDown(self):
        self.ctx.pop()
        self.dbman.remove_database(self.name)

    def test_settings_get_permissions(self):
        rv = self.client.get(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_member,
        )
        assert rv.status_code == 403
        rv = self.client.get(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_owner,
        )
        assert rv.status_code == 200
        rv = self.client.get(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_admin,
        )
        assert rv.status_code == 200

    def test_settings_put_requires_admin(self):
        payload = {"mode": "on", "soft_cap": 4000, "hard_cap": 5000}
        rv = self.client.put(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_member,
            json=payload,
        )
        assert rv.status_code == 403
        rv = self.client.put(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_owner,
            json=payload,
        )
        assert rv.status_code == 403

    def test_settings_put_updates_effective_values(self):
        payload = {"mode": "on", "soft_cap": 4100, "hard_cap": 4500}
        rv = self.client.put(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_admin,
            json=payload,
        )
        assert rv.status_code == 200
        assert rv.json["effective_mode"] == "on"
        assert rv.json["caps"]["soft_cap"] == 4100
        assert rv.json["caps"]["hard_cap"] == 4500

        rv = self.client.get(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_admin,
        )
        assert rv.status_code == 200
        assert rv.json["effective_mode"] == "on"
        assert rv.json["caps"]["soft_cap"] == 4100
        assert rv.json["caps"]["hard_cap"] == 4500

    def test_settings_put_rejects_soft_cap_above_hard_cap(self):
        rv = self.client.put(
            f"{BASE_URL}/search-grounding/settings/",
            headers=self.header_admin,
            json={"mode": "auto", "soft_cap": 5001, "hard_cap": 5000},
        )
        assert rv.status_code == 400

    def test_free_tier_boundary_states(self):
        with self.app.app_context():
            usage = SearchGroundingUsageMonthly(
                period_start=get_utc_month_start(),
                grounded_prompts_count=10,
                web_search_queries_count=4999,
            )
            user_db.session.add(usage)
            user_db.session.commit()

            rv = self.client.get(
                f"{BASE_URL}/search-grounding/settings/",
                headers=self.header_admin,
            )
            assert rv.status_code == 200
            assert rv.json["usage"]["free_tier_used"] == 4999
            assert rv.json["usage"]["free_tier_remaining"] == 1
            assert rv.json["usage"]["free_tier_status"] == "near_limit"

            usage.web_search_queries_count = 5000
            user_db.session.add(usage)
            user_db.session.commit()

            rv = self.client.get(
                f"{BASE_URL}/search-grounding/settings/",
                headers=self.header_admin,
            )
            assert rv.status_code == 200
            assert rv.json["usage"]["free_tier_used"] == 5000
            assert rv.json["usage"]["free_tier_remaining"] == 0
            assert rv.json["usage"]["free_tier_status"] == "exhausted"

    def test_diagnostics_schema_payload(self):
        rv = self.client.get(
            f"{BASE_URL}/search-grounding/diagnostics/",
            headers=self.header_admin,
        )
        assert rv.status_code == 200
        assert "metadata_schema_expectations" in rv.json
        assert rv.json["metadata_schema_expectations"]["path"] == "metadata.grounding"
