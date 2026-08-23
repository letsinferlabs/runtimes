#!/usr/bin/env python3

from __future__ import annotations

import unittest

from tools import community_verification as consensus


class CommunityConsensusTests(unittest.TestCase):
    candidate = "sglang--owner--model--dgx-spark"
    version = "1.2.3"
    pr_url = "https://github.com/letsinferlabs/runtimes/pull/17"
    subject = {
        "candidate_id": candidate,
        "runtime_version": version,
        "execution_sha256": "1" * 64,
    }
    author = {"github_login": "Author", "github_id": 10, "github_type": "User"}

    def vote(
        self,
        numeric_id: int,
        *,
        multiplier: float = 1.0,
        device: int | None = None,
        safe: bool = True,
    ) -> dict:
        verifier = {
            "github_login": f"User{numeric_id}",
            "github_id": numeric_id,
            "github_type": "User",
        }
        rows = [
            {
                "workload": "pp32768,tg128,c1",
                "prompt_domain": domain,
                "is_prefix_cached": False,
                "aggregate_tps": base * multiplier,
                "decode_tps": (base + 10) * multiplier,
                "ttft_seconds": (100 / base) / multiplier,
            }
            for domain, base in (("code", 30.0), ("prose", 60.0))
        ]
        record = {
            "pull_request": 17,
            "pull_request_url": self.pr_url,
            "observed_head_sha": "2" * 40,
            "submitted_at_unix": 1000 + numeric_id,
            "verifier": verifier,
            "device_id": f"{device if device is not None else numeric_id:064x}",
            "subject": self.subject,
            "candidate": {"id": f"{numeric_id:064x}", "results": rows},
            "baseline": {"id": "f" * 64, "results": rows},
            "run_order": ["baseline", "candidate"],
            "correctness": {"passed": safe},
            "safety": {"passed": safe},
            "restoration": {"passed": safe},
            "failure": (
                None
                if safe
                else {
                    "category": "output_validation",
                    "phase": "benchmark:candidate",
                    "message": "output validation failed",
                }
            ),
            "counts_toward_consensus": numeric_id != self.author["github_id"],
            "run_score": (
                {
                    "overall": {
                        "aggregate_tps_geomean": 42.0 * multiplier,
                        "change_percent": 0.0,
                    }
                }
                if safe
                else None
            ),
            "verification_id": f"{numeric_id:064x}",
        }
        return {
            "record": record,
            "comment_id": numeric_id,
            "comment_url": self.pr_url + f"#issuecomment-{numeric_id}",
            "device_public_key_pem": "test-key",
        }

    def build(
        self, votes: list[dict], runtime_authors: list[dict] | None = None
    ) -> dict:
        return consensus.build_consensus(
            candidate_id=self.candidate,
            runtime_version=self.version,
            pull_request=17,
            pull_request_url=self.pr_url,
            proposal_head_sha="2" * 40,
            author=self.author,
            runtime_authors=runtime_authors or [],
            accepted_comments=votes,
        )

    def test_three_independent_agreeing_votes_qualify(self) -> None:
        document = self.build([self.vote(11), self.vote(12, multiplier=1.01), self.vote(13)])
        self.assertTrue(document["qualification"]["passed"])
        self.assertEqual(document["qualification"]["required_verifiers"], 3)
        self.assertEqual(len(document["verifiers"]), 3)

    def test_author_and_duplicate_device_do_not_supply_votes(self) -> None:
        document = self.build(
            [self.vote(10), self.vote(11), self.vote(12, device=11), self.vote(13)]
        )
        self.assertFalse(document["qualification"]["passed"])
        self.assertEqual(document["qualification"]["independent_verifiers"], 2)

    def test_declared_runtime_author_does_not_supply_a_vote(self) -> None:
        runtime_author = {
            "github_login": "User11",
            "github_id": 11,
            "github_type": "User",
        }
        vote = self.vote(11)
        vote["record"]["counts_toward_consensus"] = False
        document = self.build(
            [vote, self.vote(12), self.vote(13)], [runtime_author]
        )
        self.assertFalse(document["qualification"]["passed"])
        self.assertEqual(document["qualification"]["independent_verifiers"], 2)

    def test_disagreement_expands_to_five_and_remains_unqualified(self) -> None:
        document = self.build(
            [self.vote(number, multiplier=1.0 if number < 14 else 1.5) for number in range(11, 16)]
        )
        self.assertFalse(document["qualification"]["passed"])
        self.assertEqual(document["qualification"]["required_verifiers"], 5)
        self.assertFalse(document["qualification"]["agreement_passed"])

    def test_any_safety_failure_blocks_qualification(self) -> None:
        failed = self.vote(13, safe=False)
        document = self.build([self.vote(11), self.vote(12), failed])
        self.assertFalse(document["qualification"]["passed"])
        self.assertFalse(document["qualification"]["safety_passed"])
        failed["original_body"] = "<!-- letsinfer-verification:v1\n{}\n-->\n"
        rendered = consensus.canonical_accepted_comment(failed)
        self.assertIn("blocking evidence", rendered)
        self.assertIn("output_validation", rendered)

    def test_tally_is_deterministic_and_links_structured_verifiers(self) -> None:
        document = self.build([self.vote(11), self.vote(12), self.vote(13)])
        rendered = consensus.tally_comment(document)
        self.assertIn("3 / 3 independent verifications", rendered)
        self.assertIn("[@User11]", rendered)
        self.assertIn(document["consensus_id"], rendered)


if __name__ == "__main__":
    unittest.main()
