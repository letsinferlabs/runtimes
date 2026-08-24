## Runtime proposal

- [ ] This PR changes exactly one flat runtime candidate.
- [ ] I chose `reuse-engine` when no Engine source is needed, or included the
      complete Engine/adapter/image source closure for `build-engine`.
- [ ] `runtime.json` pins exact model revisions, target, Engine manifest, Engine
      configuration, and protocol v2.
- [ ] The candidate README begins with the canonical Let's Infer installation
      block and links every Hugging Face artifact.
- [ ] Build inputs and Docker `FROM` references are immutable; licenses and
      provenance cover all included source.
- [ ] I did not commit model weights, OCI layers, archives, credentials,
      generated builds, caches, or benchmark output.
- [ ] Local schema, unit, source audit, and deterministic pack checks pass.
- [ ] For a changed Engine, the locally calculated future production digest is
      pinned; CI will independently rebuild it before verification.

After the trusted verifier bundle is ready, maintainers perform security and
supply-chain review before applying `benchmark-ready`. Do not request `/shipit`
until community evidence, approval, and all blocking checks are complete.
