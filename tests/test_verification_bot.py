#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import unittest
from unittest import mock

from tools import verification_bot as bot


class VerificationBotTests(unittest.TestCase):
    def test_configured_maintainer_bypasses_benchmark_gate(self) -> None:
        pull = {
            "number": 17,
            "head": {"sha": "a" * 40},
            "user": {"login": "TaimurAyaz"},
            "labels": [],
        }
        with (
            mock.patch.dict(
                os.environ,
                {bot.BYPASS_LOGINS_ENV: "other, taimurayaz"},
                clear=False,
            ),
            mock.patch.object(bot, "publish_maintainer_bypass") as publish,
            mock.patch.object(bot, "process_pull_request") as process,
        ):
            result = bot.process({"action": "opened", "pull_request": pull})
        self.assertEqual(result, {"processed": True, "maintainer_bypass": True})
        publish.assert_called_once_with(pull)
        process.assert_not_called()

    def test_maintainer_bypass_check_does_not_claim_qualification(self) -> None:
        pull = {
            "head": {"sha": "a" * 40},
            "user": {"login": "TaimurAyaz"},
        }
        with mock.patch.object(bot, "api", return_value={}) as api:
            bot.publish_maintainer_bypass(pull)
        value = api.call_args.kwargs["value"]
        self.assertEqual(value["conclusion"], "success")
        self.assertIn("does not create benchmark consensus", value["output"]["summary"])

    def test_pending_check_remains_in_progress(self) -> None:
        calls: list[dict] = []

        def fake_api(_endpoint: str, **kwargs):
            calls.append(kwargs["value"])
            return {}

        consensus = {
            "consensus_id": "c" * 64,
            "qualification": {
                "passed": False,
                "safety_passed": True,
                "agreement_passed": True,
                "independent_verifiers": 2,
                "required_verifiers": 3,
            },
        }
        with mock.patch.object(bot, "api", side_effect=fake_api):
            bot.update_check({"head": {"sha": "a" * 40}}, consensus, "https://example")
        self.assertEqual(calls[0]["status"], "in_progress")
        self.assertNotIn("conclusion", calls[0])

    def test_unchanged_content_does_not_create_a_commit(self) -> None:
        data = b"same\n"
        existing = {
            "encoding": "base64",
            "content": base64.b64encode(data).decode("ascii"),
            "sha": "b" * 40,
        }
        with (
            mock.patch.object(bot, "_content", return_value=existing),
            mock.patch.object(bot, "_branch_head", return_value="c" * 40),
            mock.patch.object(bot, "api") as api,
        ):
            result = bot.put_content("manifest.json", data, branch="branch", message="x")
        self.assertEqual(result, "c" * 40)
        api.assert_not_called()

    def test_proposal_head_survives_bot_only_commits(self) -> None:
        release = {
            "provenance": {
                "repository": bot.REPOSITORY,
                "pull_request": 17,
                "execution_sha256": "d" * 64,
                "proposal_head_sha": "e" * 40,
            }
        }
        content = {
            "encoding": "base64",
            "content": base64.b64encode(
                json.dumps(release).encode("utf-8")
            ).decode("ascii"),
        }
        pr = {
            "number": 17,
            "head": {"ref": "proposal", "sha": "f" * 40},
        }
        with mock.patch.object(bot, "_content", return_value=content):
            result = bot.proposal_head(pr, "runtime", {"execution_sha256": "d" * 64})
        self.assertEqual(result, "e" * 40)

    def test_pull_request_waits_for_benchmark_ready_without_api_work(self) -> None:
        event = {
            "action": "synchronize",
            "pull_request": {"number": 17, "labels": []},
        }
        with mock.patch.object(bot, "process_pull_request") as process:
            result = bot.process(event)
        self.assertFalse(result["processed"])
        process.assert_not_called()

    def test_closed_pull_request_cancels_the_check(self) -> None:
        pull = {"number": 17, "head": {"sha": "a" * 40}}
        with mock.patch.object(bot, "cancel_check") as cancel:
            result = bot.process({"action": "closed", "pull_request": pull})
        self.assertTrue(result["closed"])
        cancel.assert_called_once_with(pull)

    def test_invalid_submission_is_rejected_without_blocking_later_votes(self) -> None:
        invalid = {
            "id": 1,
            "html_url": "https://example.invalid/1",
            "body": bot.SUBMISSION_MARKER,
        }
        valid = {
            "id": 2,
            "html_url": "https://example.invalid/2",
            "body": bot.SUBMISSION_MARKER,
        }
        accepted = {
            "record": {
                "verification_id": "a" * 64,
                "subject": {"execution_sha256": "b" * 64},
            }
        }
        with (
            mock.patch.object(bot, "api", return_value=[[invalid, valid]]),
            mock.patch.object(
                bot.community_verification,
                "accepted_comment_value",
                side_effect=[ValueError("bad signature"), accepted],
            ),
            mock.patch.object(bot, "_post_comment", return_value={"body": "rejected"}) as post,
        ):
            result = bot.accepted_submissions(
                17, subject={"execution_sha256": "b" * 64}
            )
        self.assertEqual(result, [accepted])
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
