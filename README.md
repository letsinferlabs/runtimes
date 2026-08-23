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
├── release.json  # runtime authors and SPDX license
├── README.md      # model links and exact reproduction command
├── adapter/       # Engine protocol frontend included in the Engine OCI
├── engine/        # pinned engine or kernel source used to build the Engine OCI
├── image/         # deterministic Engine OCI recipe
├── kernels/       # optional runtime-specific kernels
├── patches/       # optional auditable source patches
├── scripts/       # optional build or qualification helpers
├── tests/         # candidate-specific gates
└── benchmark.json # present only when qualified
```

The directory name and `runtime.id` must be exactly:

```text
<engine>--<lowercase-hf-owner>--<lowercase-hf-model>--<target>
```

Nested model/engine/target hierarchies are forbidden. The generated root
`manifest.json` is the append-only release projection used to produce the
signed public catalog. It keeps every qualified version and its immutable
runtime and benchmark references; it is never hand-edited.

## Runtime contract

`release.json` declares the runtime's one or more authors and SPDX license.
Those identities are versioned with every catalog release and shown by
`letsinfer list`. `runtime.json` declares:

- logical model alias;
- exact `hf://owner/repository` identity and immutable 40-hex revision;
- exact model files when the artifact is a single file;
- digest-pinned acquisition image;
- digest-pinned Engine OCI and image configuration identity;
- Engine protocol version;
- target capabilities;
- opaque engine arguments and environment;
- container, cache, serving, and benchmark contracts;
- an exact benchmark record when qualified.

When you install a runtime, Let's Infer downloads every declared model
artifact. You do not need to preinstall weights, choose an engine, or supply a
target path.

## Engine boundary

The Engine OCI contains the inference engine and its matching adapter. The
adapter implements Engine protocol v1: lifecycle, health, normalized telemetry,
exact token counting, and authenticated inference proxying. Core does not carry
per-engine registries, flags, tokenizers, cache plugins, or version shims.

Inside your candidate, you may change engine configuration, kernels, patches,
sidecars, and model artifacts. A changed Engine OCI identity always invalidates
prior qualification evidence.

## Validate a candidate

```bash
python3 tools/generate_manifest.py --validate-only
python3 -m unittest discover -s tests -p 'test_*.py'
python3 <candidate>/adapter/engine-adapter verify --protocol 1
letsinfer pack <candidate> --output /tmp/runtime.letsinfer
```

Pull requests run the same checks, build the runtime pack twice, require
byte-identical archives, and verify the deterministic OCI manifest identity.

## Publication

Engine sources, adapter, or image changes automatically start the Engine OCI
workflow after merge. It normalizes exported layer timestamps, publishes the
image by digest, exports a deterministic Debian/Python package inventory,
binds that inventory to the exact image and configuration identities as SPDX,
and attaches the SBOM attestation to the OCI. It then emits a deterministic
pin review patch; applying that patch resets the candidate to unqualified.

Every runtime merged to `main` must already have benchmark evidence verified by
a Let's Infer maintainer. The merge gate therefore makes every published
runtime qualified; recommendation is a separate, automatically calculated
decision. After target qualification is committed, merging the exact revision
to the `release` branch:

1. rebuilds and verifies deterministic runtime packs;
2. checks that every advertised OCI digest matches the planned artifact;
3. publishes the immutable runtime-pack OCI artifacts;
4. publishes each complete `benchmark.json` as an immutable, runtime-bound OCI
   evidence artifact;
5. preserves previous qualified releases and calculates the best release for
   every model/target using the catalog's versioned scoring policy;
6. verifies that every model/target has a qualified recommendation;
7. signs the generated schema-v5 catalog;
8. verifies the signature with the public key shipped by core;
9. publishes `catalog.json`, its signature, the public key, and a hash-bound
   metadata record as the latest immutable GitHub Release in this repository;
   and
10. emits build-provenance attestations for runtime packs, benchmark evidence,
    and the signed catalog.

The release fails closed when an Engine digest, model revision, benchmark
identity, catalog source, signature key, or recommendation is inconsistent.

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
