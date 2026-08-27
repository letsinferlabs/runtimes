# Contributing a runtime

Install the official [Let’s Infer agent skills](https://github.com/letsinferlabs/skills)
with `npx skills add letsinferlabs/skills`. Start every candidate with
[`letsinfer-runtime-authoring`](https://github.com/letsinferlabs/skills/blob/main/skills/letsinfer-runtime-authoring/SKILL.md).
When Engine executable inputs change, also use
[`letsinfer-engine-authoring`](https://github.com/letsinferlabs/skills/blob/main/skills/letsinfer-engine-authoring/SKILL.md).
They standardize the candidate directory, immutable model and Engine pins,
deterministic recipe, license/source closure, README install block, and local
checks before review.

## Choose the smallest source form

- Reuse an Engine when only the model, target, recipe arguments, or runtime
  behavior changes. Pin its public manifest and configuration digests and omit
  Engine source directories.
- Include `adapter/`, `image/`, and the complete applicable `engine/`,
  `kernels/`, `patches/`, and `scripts/` closure when changing Engine internals
  or introducing an Engine Let's Infer does not yet publish.

Do not commit container layers, build directories, model weights, archives,
credentials, private keys, benchmark caches, or generated package inventories.
The runtime pack retains the exact reviewed source; Git is the readable history,
while GHCR stores deployable immutable objects.

## Local loop

```bash
python3 tools/readme_onboarding.py --candidate <candidate> --write
python3 tools/candidate_policy.py audit --candidate <candidate> --mode <reuse-engine|build-engine>
python3 tools/generate_manifest.py --validate-only
python3 -m unittest discover -s tests -p 'test_*.py'
letsinfer pack <candidate> --output /tmp/runtime.letsinfer
```

For `build-engine`, use the same canonical builder CI uses:

```bash
python3 tools/build_engine.py --candidate <candidate> --output /tmp/engine.oci.tar --pin
```

It pins the transport manifest/configuration and stable execution payload in
`runtime.json`. CI builds once with the same pinned BuildKit contract and
fails with a direct correction message if the authored identity differs. CI
never commits generated pins to your branch.

Local installation is allowed before qualification and exercises the same
managed core placement and lifecycle as a catalog release. It does not make a
candidate qualified, publish an artifact, or change the evidence required for
catalog recommendation; do not add qualification or activation flags to
`runtime.json` to make local testing work.

## Pull request lifecycle

1. Open one PR changing one candidate.
2. A no-code PR sentinel triggers a read-only, secretless default-branch
   builder for the exact head. The canonical builder produces one Engine
   output, computes its normalized payload identity, and retains only the
   local overlay layers plus an immutable public-base reference.
3. A separate trusted default-branch finalizer reclassifies, re-audits,
   re-packs, creates SBOM/provenance, and publishes the exact verifier artifact
   without executing proposal code.
4. Maintainers complete source, security, license, and supply-chain review and
   apply the `benchmark-ready` label.
5. Two independent users run `letsinfer benchmark verify <PR URL>`. One user
   occupies one slot regardless of reruns; any blocking failure is terminal for
   that execution subject.
6. Repository maintainers process publication only after approval and green
   checks. Trusted automation verifies every payload's finalizer attestation,
   publishes the exact verified artifacts, verifies public pulls, records its
   receipt, and merges the exact head.
