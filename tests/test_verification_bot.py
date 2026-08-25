#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from tools import verification_bot as bot


class VerificationBotTests(unittest.TestCase):
    def test_scored_release_revokes_prior_unscored_bypass_identity(self) -> None:
        candidate = "engine--owner--model--target"
        runtime_digest = "sha256:" + "1" * 64
        consensus_sha = "2" * 64
        previous = {
            "models": {
                "model": {
                    "targets": {
                        "target": {
                            "candidates": {
                                candidate: {
                                    "releases": {
                                        "1.0.0": {
                                            "source": f"ghcr.io/letsinferlabs/runtime-artifacts@{runtime_digest}",
                                            "benchmark": None,
                                            "verification": {
                                                "method": "allowlisted-maintainer-bypass-v1",
                                                "consensus_sha256": consensus_sha,
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "revocations.json"
            path.write_text(
                json.dumps(
                    {
                        "generated_at_unix": 0,
                        "revocations": [],
                        "schema_version": 1,
                        "sequence": 0,
                    }
                )
            )
            with mock.patch.object(bot, "put_content", return_value="a" * 40) as put:
                changed = bot.revoke_superseded_unscored_releases(
                    root,
                    candidate=candidate,
                    current_version="1.0.1",
                    previous=previous,
                    branch="runtime/topic",
                    generated_at_unix=123,
                )
            ledger = json.loads(path.read_text())
        self.assertTrue(changed)
        self.assertEqual(ledger["sequence"], 1)
        self.assertEqual(ledger["generated_at_unix"], 123)
        self.assertEqual(
            ledger["revocations"],
            [
                {
                    "consensus_sha256": consensus_sha,
                    "runtime_oci_digest": runtime_digest,
                }
            ],
        )
        put.assert_called_once()

    def test_core_pull_request_contract_binds_base_and_head_by_name(self) -> None:
        class PullRequest:
            def __init__(self, **values):
                self.__dict__.update(values)

        contract_module = types.SimpleNamespace(
            GitHubIdentity=lambda login, numeric_id, kind: (login, numeric_id, kind),
            PullRequest=PullRequest,
        )
        pull = {
            "number": 17,
            "html_url": "https://github.com/letsinferlabs/runtimes/pull/17",
            "base": {"sha": "b" * 40},
            "head": {"sha": "a" * 40},
            "user": {"login": "Author", "id": 41, "type": "User"},
            "labels": [{"name": "benchmark-ready"}],
        }
        with mock.patch.object(bot, "_core", return_value=(contract_module, None)):
            contract = bot._pr_contract(pull, ["candidate/runtime.json"])
        self.assertEqual(contract.base_sha, "b" * 40)
        self.assertEqual(contract.head_sha, "a" * 40)
        self.assertEqual(contract.files, ("candidate/runtime.json",))

    def test_runtime_candidate_names_use_changed_paths(self) -> None:
        current = "sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark"
        new = "engine--owner--model--target"
        self.assertEqual(
            bot.runtime_candidate_names([
                {"filename": "README.md"},
                {"filename": f"{current}/engine/config.json"},
            ]),
            [current],
        )
        self.assertEqual(
            bot.runtime_candidate_names([
                {"filename": f"{new}/runtime.json"},
                {"filename": f"{new}/README.md"},
            ]),
            [new],
        )

    def test_non_runtime_pr_satisfies_verification_check(self) -> None:
        pull = {
            "number": 17,
            "head": {"sha": "a" * 40},
            "labels": [],
        }
        with (
            mock.patch.object(bot, "changed_runtime_candidates", return_value=[]),
            mock.patch.object(bot, "sync_runtime_label") as sync,
            mock.patch.object(bot, "publish_classification_check") as publish,
            mock.patch.object(bot, "process_pull_request") as process,
        ):
            result = bot.process({"action": "opened", "pull_request": pull})
        self.assertEqual(result, {"processed": True, "runtime_proposal": False})
        sync.assert_called_once_with(pull, runtime=False)
        publish.assert_called_once_with(pull, runtime_pending=False)
        process.assert_not_called()

    def test_exact_release_promotion_skips_runtime_verification(self) -> None:
        pull = {
            "number": 17,
            "base": {"ref": "release", "sha": "b" * 40},
            "head": {
                "sha": "a" * 40,
                "repo": {"full_name": "letsinferlabs/runtimes"},
            },
            "labels": [{"name": "runtime"}],
        }
        head_commit = {"parents": [{"sha": "b" * 40}], "tree": {"sha": "c" * 40}}
        main = {"commit": {"sha": "d" * 40}}
        main_commit = {"tree": {"sha": "c" * 40}}
        with mock.patch.object(bot, "api", side_effect=[head_commit, main, main_commit]):
            self.assertTrue(bot.exact_release_promotion(pull))
        with (
            mock.patch.object(bot, "exact_release_promotion", return_value=True),
            mock.patch.object(bot, "changed_runtime_candidates") as changed,
            mock.patch.object(bot, "sync_runtime_label") as sync,
            mock.patch.object(bot, "publish_classification_check") as publish,
        ):
            result = bot.process({"action": "synchronize", "pull_request": pull})
        self.assertEqual(result, {"processed": True, "release_promotion": True})
        changed.assert_not_called()
        sync.assert_called_once_with(pull, runtime=False)
        publish.assert_called_once_with(pull, runtime_pending=False)

    def test_non_runtime_check_does_not_claim_qualification(self) -> None:
        pull = {
            "head": {"sha": "a" * 40},
        }
        with mock.patch.object(bot, "api", return_value={}) as api:
            bot.publish_classification_check(pull, runtime_pending=False)
        value = api.call_args.kwargs["value"]
        self.assertEqual(value["conclusion"], "success")
        self.assertIn("No runtime candidate changed", value["output"]["summary"])

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
                "blocking_failures": [],
                "independent_verifiers": 1,
                "required_verifiers": 2,
            },
        }
        with mock.patch.object(bot, "api", side_effect=fake_api):
            bot.update_check({"head": {"sha": "a" * 40}}, consensus, "https://example")
        self.assertEqual(calls[0]["status"], "in_progress")
        self.assertNotIn("conclusion", calls[0])

    def test_empty_consensus_uses_two_independent_passes(self) -> None:
        document = bot.empty_consensus(
            candidate="engine--owner--model--target",
            version="1.2.3",
            subject={"execution_sha256": "a" * 64},
            proposal="b" * 40,
        )
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["qualification"]["required_verifiers"], 2)
        self.assertNotIn("agreement_passed", document["qualification"])

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
            "pull_request": {
                "number": 17,
                "head": {"sha": "a" * 40},
                "labels": [],
            },
        }
        with (
            mock.patch.object(
                bot, "changed_runtime_candidates", return_value=["candidate"]
            ),
            mock.patch.object(bot, "sync_runtime_label") as sync,
            mock.patch.object(bot, "publish_classification_check") as publish,
            mock.patch.object(bot, "process_pull_request") as process,
        ):
            result = bot.process(event)
        self.assertFalse(result["processed"])
        sync.assert_called_once_with(event["pull_request"], runtime=True)
        publish.assert_called_once_with(
            event["pull_request"], runtime_pending=True
        )
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
