# Downstream runs

One folder per run, `results/downstream/<run_id>/` (`ds-pilot-01`, `ds-core-01`, ...), copied from the run tree once the run is evaluated.

Versioned here:

- `run-identity.json` and a copy of `protocol-freeze.json`;
- the variant files given to the generator;
- the generation journals (`generations.jsonl`) with their manifests;
- the oracle outputs: `outcomes.jsonl`, `details.jsonl`, `attempts.jsonl`, calibration files and manifests;
- a per-program summary of the CodeQL results;
- analysis designs, reports and manifests.

Not versioned: model weights and caches, Docker images (only their digests), CodeQL databases, full SARIF files, logs with personal paths or session identifiers, and the texts of SecurityEval (only identifiers, hashes and outcomes).

The historical parser runs at the root of `results/` are not part of this folder and are never changed.
