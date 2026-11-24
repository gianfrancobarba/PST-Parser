# Downstream runs

The layout a downstream run directory follows when it is copied here, `results/downstream/<run_id>/`,
and the one the `thesis-tables` command reads (`make thesis-tables DS_RUN=<run directory>`):

- `run-identity.json` and a copy of `protocol-freeze.json`;
- the variant files given to the generator;
- the generation journals (`generations.jsonl`) with their manifests;
- the oracle outputs: `outcomes.jsonl`, `details.jsonl`, `attempts.jsonl`, calibration files and manifests;
- a per-program summary of the CodeQL results;
- analysis designs, reports and manifests.

The runs of the empirical study are **not** copied here. Their journals and oracle outputs are large,
they hold the texts of BigCodeBench and LLMSecEval under those datasets' licences, and they carry paths
of the machine they ran on. Their identities, hashes and every estimate drawn from them are recorded
with the study; the tracked results are described in `results/README.md`.

Never versioned: model weights and caches, Docker images (only their digests), CodeQL databases, full
SARIF files, and logs with personal paths or session identifiers.

The historical parser run at the root of `results/` is not part of this folder and is never changed.
