# Let's Infer runtimes

This repository contains reproducible deployment candidates. A candidate binds
one logical model to an exact Hugging Face artifact, an immutable Engine OCI,
and one hardware target. Let's Infer core remains engine- and model-agnostic.

## Layout

Every candidate is a single top-level directory:

```text
<engine>--<hf-owner>--<hf-model>--<target>/
├── runtime.json
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
`manifest.json` is the candidate projection consumed by the trusted catalog
release. It is never edited as an independent source of truth.

## Runtime contract

`runtime.json` declares:

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
and attaches the SBOM attestation to the OCI. It then opens a pinning PR; the
pinning change resets the candidate to unqualified.

After target qualification is committed, merging the exact revision to the
`release` branch:

1. rebuilds and verifies deterministic runtime packs;
2. checks that every advertised OCI digest matches the planned artifact;
3. publishes the immutable runtime-pack OCI artifacts;
4. verifies that every model/target has a qualified recommendation;
5. signs the generated schema-v4 catalog;
6. verifies the signature with the public key shipped by core;
7. publishes `catalog.json`, its signature, the public key, and a hash-bound
   metadata record as an immutable prerelease in this repository;
8. lets the catalog repository independently download, verify, and promote
   those exact bytes through its own review branch; and
9. emits build-provenance attestations.

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

## License

AGPL-3.0-only unless a candidate's provenance file states additional licenses.
