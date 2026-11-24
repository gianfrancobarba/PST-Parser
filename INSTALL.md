# Installation and checks

Requirements are in [REQUIREMENTS.md](REQUIREMENTS.md). Run everything from the project root.
`outputs/new-run/` is an example: choose a new name for every preparation, and use the same overrides
in train and generate. Never overwrite historical splits or results.

The empirical study is closed and its protocol is frozen in commit `394c893`
(`protocol/protocol-freeze.json`). The supervisors' approval of the choices made after the proposal
is not recorded in the protocol; this is declared in `protocol/deviations.md`. Commands with purpose
`final` verify the freeze and run on an extraction of `git -c core.autocrlf=false archive 394c893`,
not on a Windows working tree, whose text files may be CRLF.

## Preparation and scoring on a CPU

```bash
uv sync --no-dev
uv run pstparser validate-config --config configs/experiments/baseline.yaml
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

Without `--run-dir`, score creates a `score-...` directory under `training.output_dir` (`outputs/` by
default), never next to the predictions, with `results.json`, `eval_details.jsonl` and
`score_manifest.json`; align does the same with `align-...`. It recomputes historical predictions with
the current metrics; it is not a new training. Collisions with existing outputs are refused.

## Training and generation

```bash
uv sync --extra cu128 --extra train --extra unsloth
```

For development on a CPU with the model stack and the tests:

```bash
uv sync --extra cpu --extra train --extra dev
```

The `cpu` and `cu128` extras are incompatible. The optimised backend may compile kernels when it is
imported: a successful installation does not mean a validated training. `--set model.backend=hf`
selects the alternative path, without guaranteeing numerical identity or the same requirements. Check
the actual model and precision with a smoke test.

```bash
uv run pstparser prepare-data --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --run-dir outputs/new-run/training
uv run pstparser generate --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --adapter outputs/new-run/training/adapter --run-dir outputs/new-run/generation
uv run pstparser score --config configs/experiments/baseline.yaml --predictions outputs/new-run/generation/predictions.jsonl --run-dir outputs/new-run/scoring
```

On the audited corpus there are 1,123 records before deduplication (975 from the spreadsheet and 148
from YAML), 17 corrections, 18 duplicates and the 905/100/100 split. These describe that snapshot and
are to be checked on the new report; they are not invariants for future corpora. The quality gate
checks placement and coverage with the configured thresholds and normalisations: it certifies neither
semantics nor a strict 100% coverage.

`prepare-data` needs fresh directories for records and splits. New manifests are separate:
`prepare_manifest.json`, `train_manifest.json`, `generate_manifest.json`, `score_manifest.json`. The
path overrides must go with train and generate so that they select the same data.

## Corpus audit and downstream modules

They are invoked as Python modules, without dedicated entry points; none needs a GPU except
`empirical generate`, which loads the model.

```bash
uv run python -m pstparser.data.audit --config configs/experiments/baseline.yaml --output outputs/new-run/corpus-audit.json
uv run python -m pstparser.empirical --help
```

The audit is read-only and refuses an existing output file; the directory must exist. The `render`,
`generate`, `check` and `analyze` subcommands of `pstparser.empirical` need explicit inputs
(canonical annotations, frozen configuration, oracles, analysis design). `check` needs Docker and a
Python image already present locally, pinned by digest (`name@sha256:...`): it pulls no image and runs
no generated code on the host.

## Containers

```bash
docker build -f docker/Dockerfile --target eval -t pstparser:eval .
docker compose -f docker/compose.yaml --profile eval run --rm score score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

The eval image does not need the model stack. Its size and compatibility depend on the build. For a
GPU, check the NVIDIA Container Toolkit and the runtime on the host:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
docker build -f docker/Dockerfile --target train -t pstparser:train .
docker compose -f docker/compose.yaml --profile train run --rm train prepare-data --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-container-run/processed --set data.split.output_dir=outputs/new-container-run/splits
docker compose -f docker/compose.yaml --profile train run --rm train train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-container-run/processed --set data.split.output_dir=outputs/new-container-run/splits --run-dir outputs/new-container-run/training
```

Compose mounts data, outputs and results, and a model cache. These images are not by themselves the
downstream sandbox for untrusted code.

## Tests

```bash
uv run pytest -m "not slow and not gpu"
uv run pytest -m slow
```

The fast suite needs the `cpu` (or `cu128`), `train` and `dev` extras: some tests import `torch`,
`transformers` and `tokenizers`. Two tests need a git checkout and are skipped in an archive
extraction. The slow tests may download a model and need the network and the model stack; their
duration and GPU support depend on the environment. If creating temporary files is not allowed, pass
`--basetemp outputs/test-temp-new` with a new dedicated directory: pytest manages it as temporary,
so never pick a path holding data to keep.

On Windows `make` needs a separate installation; the CLI commands above do not depend on it.
