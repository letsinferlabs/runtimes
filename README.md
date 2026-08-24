# Let's Infer runtimes

This repository contains reproducible deployment candidates. A candidate binds
one logical model to an exact Hugging Face artifact, an immutable Engine OCI,
and one hardware target. Let's Infer core remains engine- and model-agnostic.

## Features

- **Install by model name** — `letsinfer install qwen3.8-27b` resolves the best
  qualified candidate for your hardware; users never type an engine or target
  path.
- **Exact model provenance** — every candidate links its Hugging Face model
  and pins the immutable revision, required files, sizes, and hashes.
- **Engine freedom** — each Engine OCI carries one engine version and its
  matching Let's Infer protocol adapter, independent of core releases.
- **Target-specific performance** — candidates may include custom kernels,
  patches, sidecars, caches, and configuration for one exact hardware target.
- **Replica-ready by default** — core can run compatible single groups across
  main and child nodes while every node keeps its fastest qualified runtime.
- **Explicit parallel runtimes** — TP/PP candidates declare and qualify their
  complete multi-device topology, generic launch tasks, interconnect, and
  Engine recipe while roles and ranks stay private to the runtime.
- **Complete reproducibility** — runtime packs and Engine images are built from
  reviewed source, pinned dependencies, and deterministic recipes.
- **Benchmark-backed selection** — the signed catalog recommends only a
  qualified model/target candidate and binds that decision to machine-readable
  benchmark evidence.
- **Forkable by design** — change any model artifact, kernel, engine source, or
  serving parameter, then qualify and publish the result as a new candidate.

## Layout

Every candidate is a single top-level directory:

```text
<engine>--<hf-owner>--<hf-model>--<target>/
├── runtime.json
├── release.json  # structured authors, SPDX license, bot-owned provenance
├── README.md      # Let's Infer install block, model links, reproduction
├── adapter/       # present only when this PR builds a changed Engine OCI
├── engine/        # optional pinned engine source for radical Engine changes
├── image/         # required deterministic recipe when building an Engine OCI
├── kernels/       # optional runtime-specific kernels
├── patches/       # optional auditable source patches
├── scripts/       # optional build or qualification helpers
├── tests/         # candidate-specific gates
└── benchmark.consensus.json # bot-generated after independent verification
```

The directory name and `runtime.id` must be exactly:

```text
<engine>--<lowercase-hf-owner>--<lowercase-hf-model>--<target>
```

Nested model/engine/target hierarchies are forbidden. The generated root
`manifest.json` is the append-only release projection used to produce the
signed public catalog. It keeps every qualified version, immutable runtime,
structured verifiers, score, and consensus digest; it is never hand-edited.

Every candidate README begins with a link to
[Let's Infer](https://letsinfer.ai/), the canonical installer command, and
`letsinfer install LOGICAL_MODEL` derived from `runtime.json`. Add the block
without replacing an existing README:

```bash
python3 tools/readme_onboarding.py --candidate <candidate> --write
```

Pull-request validation requires the block whenever a new or existing
candidate is changed.

## Runtime contract

`release.json` declares one or more structured GitHub authors and the SPDX
license. It begins with `provenance: null`; qualification automation owns the
eventual PR/execution/consensus provenance. Those identities are versioned with
every catalog release and shown by `letsinfer list`. `runtime.json` declares:

- logical model alias;
- exact `hf://owner/repository` identity and immutable 40-hex revision;
- exact model files when the artifact is a single file;
- digest-pinned acquisition image;
- digest-pinned Engine OCI and image configuration identity;
- Engine protocol version;
- target capabilities;
- an optional bounded, engine-neutral orchestration contract for parallel
  targets;
- opaque engine arguments and environment;
- container, cache, serving, and benchmark contracts.

For an independent target, declare `target.placement.strategy` as `single`.
Let's Infer core may replicate that runtime without changing its bytes. Use
`parallel` only for a runtime that owns and qualifies the exact TP/PP topology;
core allocates its declared devices but never invents parallelism from a
single-device candidate.

A parallel candidate declares one generic task per required node, one endpoint
owner, phased startup order, shell-free commands, ports, environment, and
readiness. Core supplies authenticated node IDs, GPU UUIDs, addresses, ports,
credentials, and verified connection facts. Your Engine OCI privately maps
each `task-N` to its ranks, stages, collectives, rendezvous, and engine flags.
Changing those private details does not require a core release.

When you install a runtime, Let's Infer downloads every declared model
artifact. You do not need to preinstall weights, choose an engine, or supply a
target path.

## Engine boundary

The Engine OCI contains the inference engine and its matching adapter. The
adapter implements Engine protocol v2: lifecycle, health, normalized telemetry,
exact token counting, and authenticated inference proxying. Core does not carry
per-engine registries, flags, tokenizers, cache plugins, or version shims.

Inside your candidate, you may change engine configuration, kernels, patches,
sidecars, and model artifacts. A changed Engine OCI identity always invalidates
prior qualification evidence.

## Validate a candidate

```bash
python3 tools/generate_manifest.py --validate-only
python3 -m unittest discover -s tests -p 'test_*.py'
python3 <candidate>/adapter/engine-adapter verify --protocol 2 # build-engine only
letsinfer pack <candidate> --output /tmp/runtime.letsinfer
```

Pull requests run the same checks, build the runtime pack twice, require
byte-identical archives, and verify the deterministic OCI manifest identity.
No separate public `letsinfer dev` command family is required: authoring uses
the reviewed candidate directory, ordinary Docker/buildx commands, and the
existing deterministic `letsinfer pack` boundary.

A candidate that only targets an already-published Engine omits `adapter/`,
`engine/`, and `image/`; CI classifies it as `reuse-engine` and verifies the
exact digest in `runtime.json`. Adding or changing any Engine input directory
classifies the candidate as `build-engine` and includes the complete submitted
Engine source in the runtime pack.

For a smaller checkout, fetch only the candidate and repository tools. Git
downloads omitted Engine history blobs only if you later select them:

```bash
git clone --filter=blob:none --sparse https://github.com/letsinferlabs/runtimes.git
cd runtimes
git sparse-checkout set tools <candidate>
```

## Publication

Every runtime PR first runs a no-code sentinel. A default-branch `workflow_run`
builder—not the contributor-editable PR workflow—then checks out the exact
proposal as untrusted input. It has read-only permissions and no secrets. It
audits candidate size and contents, builds the runtime twice, and, for
`build-engine`, independently builds the Engine OCI twice plus its package
inventory. It cannot publish a package and retains only one byte-verified OCI
layout.

A second default-branch `workflow_run` finalizer treats the proposal and raw
outputs as untrusted data. It independently reclassifies the base-to-head diff,
audits and repacks the candidate, verifies both Engine identities, creates
runtime and Engine SPDX documents plus provenance and checksums, attests the
payloads, and uploads one immutable artifact named
`verification-bundle-pr-NUMBER-HEAD`. It never executes proposal code.

For a changed Engine, `runtime.json` must already pin the future production
manifest and configuration digests. If local authoring did not pin them, the
finalizer uploads a deterministic `engine-pin-pr-*` patch and refuses to make
the head benchmarkable. Apply that patch and push; the next head receives a
new bundle. No registry write is needed during local development or PR build.

Every runtime proposal must first pass source and supply-chain review. The bot
automatically adds the `runtime` label when a PR directly changes a runtime
candidate. Only those PRs enter community benchmark verification; shared
tooling, workflow, and documentation PRs receive a not-applicable check and
continue through ordinary validation and review.

After a runtime PR is labeled `benchmark-ready`, independent users run:

```bash
letsinfer benchmark verify https://github.com/letsinferlabs/runtimes/pull/123
```

Let's Infer runs a paired baseline/candidate benchmark, restores the verifier's
runtime, and posts complete signed evidence. It downloads only the trusted,
head-bound verifier bundle—not a PR source archive—and locally loads the exact
bundled OCI layout when the PR changes Engine internals. Core validates every
OCI descriptor and rootfs diff ID, converts it to a temporary Docker-load
archive locally, and removes both the temporary archive and any image it
introduced. Two successful non-author users
on distinct account and device identities qualify the exact execution
subject. One user occupies one slot even after a rerun. A correctness, safety,
or restoration failure blocks that subject; performance differences remain
visible evidence but do not expand the quorum or become a vote. A trusted
GitHub App validates comments, posts canonical copies and a sticky tally,
updates the required check, and owns `benchmark.consensus.json`, provenance,
and the generated manifest projection. Merge is the qualification boundary;
recommendation remains a separate calculated choice.

After review and qualification, a repository maintainer comments exactly:

```text
/shipit
```

The trusted publisher rechecks the current head, maintainer approval, all
blocking checks, exact artifact/workflow identities, every payload's
trusted-finalizer attestation, consensus, and bot-only qualification commits.
It promotes the exact bundled Engine when needed,
publishes the exact runtime pack, verifies both objects through anonymous
digest pulls, posts a publication receipt, and merges only that checked head.
It is the only workflow allowed to publish an Engine OCI; ordinary runtime
releases never rebuild or overwrite Engine images.

Configured maintainers may waive only the second verifier after one successful
independent verification and no correctness, safety, or restoration failure:

```text
/shipit --bypass-verifiers
Reason: concise auditable justification
```

The actor must have live `maintain` or `admin` permission and their immutable
numeric GitHub ID must appear in the trusted comma-separated repository
variable `LETSINFER_VERIFIER_BYPASS_GITHUB_IDS`. The waiver, reason, actor ID,
comment, and time are retained in consensus and the publication receipt. It
does not bypass review, failed checks, source audits, digest matching, public
pull verification, or exact-head merge protection.

After qualification, merging the exact revision to the `release` branch:

1. rebuilds and verifies deterministic runtime packs;
2. checks that every advertised OCI digest matches the planned artifact;
3. anonymously verifies every already-published runtime and Engine object;
4. preserves immutable qualified releases and calculates the best release for
   every model/target using the catalog's versioned scoring policy;
5. verifies that every model/target has a qualified recommendation;
6. signs the generated schema-v6 catalog and separate revocation ledger;
7. verifies both signatures with the public key shipped by core;
8. publishes the catalog, ledger, signatures, public key, and hash-bound
   metadata record as the latest immutable GitHub Release in this repository;
   and
9. emits build-provenance attestations for runtime packs and signed selection
   metadata.

Complete community evidence is not copied into an OCI. Canonical bot comments
and `benchmark.consensus.json` retain every accepted full record. The signed
catalog carries a compact verifier/score/digest projection. Post-release
invalidations enter `revocations.json`; immutable releases never acquire a
`revoked` or `disputed` status field.

The repository owner configures the verifier once with a GitHub App installed
only on `letsinferlabs/runtimes`. Its minimum repository permissions are
Contents, Issues, Pull requests, and Checks (read/write), plus Metadata (read).
The `runtime-verification-bot` environment stores
`LETSINFER_VERIFICATION_APP_PRIVATE_KEY`; repository variables store the App ID
and bot login as `LETSINFER_VERIFICATION_APP_ID` and
`LETSINFER_VERIFICATION_BOT_LOGIN`. The comma-separated
`LETSINFER_VERIFIER_BYPASS_GITHUB_IDS` variable contains the immutable numeric
IDs of maintainers authorized to use the verifier-only waiver. All workflows
also require `LETSINFER_VERIFICATION_CORE_SHA`, pinned to the exact 40-character
commit of the released core verification contract. The workflow mints a short-lived
token explicitly limited to this repository and those permissions.

GHCR needs one maintainer setup because a new package is private by default.
Pre-provision `ghcr.io/letsinferlabs/runtime-artifacts` and
`ghcr.io/letsinferlabs/engine-images`, link both to this public repository, and
make both public. New candidates publish into those shared packages by
immutable digest; existing candidates keep their already-public package.
`/shipit` always performs an anonymous digest pull before merge, so missing or
private package setup fails closed.

The default public runners are `ubuntu-24.04-arm` for isolated builds and
`ubuntu-24.04` for finalization. Repositories with large Engine images should
set `LETSINFER_VERIFIER_RUNNER` and `LETSINFER_FINALIZER_RUNNER` to isolated
GitHub-hosted larger-runner labels with adequate disk. Raw outputs expire after
one day and finalized verifier bundles after 30 days; an expired bundle is
rebuilt only from the same exact head and must reproduce every recorded digest.

The release fails closed when an Engine digest, model revision, benchmark
identity, catalog source, signature key, or recommendation is inconsistent.

A deliberate no-compatibility schema migration may supersede one prior
release without rerunning its performance matrix only through the reviewed
`runtime-contract-migration-v1` ledger. The generator independently verifies
the sealed benchmark identity and digest, prior version and OCI source, exact
model revision, Engine image, benchmark contract, and a canonical hash of the
engine-visible execution contract. Any behavior-bearing change makes the
migration fail; ordinary releases still require fresh community consensus.

## Installation

Install the model you want to run:

```bash
letsinfer install qwen3.8-27b
```

Let’s Infer detects your target and selects the best qualified candidate from
the signed catalog. You may pin an exact candidate with `--runtime`; you never
need to select an engine.

Discover every qualified candidate compatible with the current machine:

```bash
letsinfer list
letsinfer list qwen3.8-27b --versions
```

## License

AGPL-3.0-only unless a candidate's provenance file states additional licenses.
