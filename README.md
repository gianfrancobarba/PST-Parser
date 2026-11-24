# PST-Parser

Supervised fine-tuning and evaluation pipeline for the **structural segmentation of prompts**
according to the Prompt Syntax Tree (PST) framework, and the code of an empirical study that uses
the PST as a prompt format for code generation.

Given a natural-language prompt, the model extracts its text segments and assigns each of them to a
node of a hierarchical taxonomy, producing a JSON object:

```
prompt
├── main_instruction
├── context
│   ├── format          positive instructions on how the output is presented
│   ├── constraints     restrictions on the generation
│   ├── data            raw content to process
│   └── role            definition of the persona
├── examples            input-output pairs for few-shot learning
└── reasoning
    ├── influence           reasoning triggers
    ├── reasoning_examples  examples with explicit intermediate steps
    └── paths               alternative reasoning branches
```

Every leaf is a **list of segments**: a category can occur in several places of the prompt, and
keeping its pieces apart is what allows the original text to be rebuilt. How the discontinuity is
marked depends on the format the corpus is annotated in: in a spreadsheet with the `<sep>` token,
because a cell holds a single string; in a text file by writing one segment per list item, because a
sequence needs no token to say where it splits.

The PST and the original parser prototype come from the master's thesis of Giovanni Scorziello
(University of Salerno). This repository is a verifiable reimplementation and extension of that
parser, not an exact replica of its numbers: the extended dataset, the adapter and the predictions of
the original work were not available.

## Status of the project and of the empirical study

The PST pipeline and the downstream modules are separate: `src/pstparser/empirical/` holds the
contracts, rendering, generation, oracles and analysis that compare the prompt conditions.

**The empirical study is closed.** It asks whether presenting a programming task to an LLM as a PST,
with the same source text, instead of as the original text changes the correctness and the security
of the generated code.

- Conditions:
  - T: original text;
  - H-gold: PST of the reviewed gold;
  - E: the text inside `{"prompt": ...}`;
  - F: flat JSON;
  - H-auto: PST of the parser.
- Generator: Qwen2.5-Coder-7B-Instruct.
- Correctness: BigCodeBench-Instruct, with the official harness.
- Security (exploratory): LLMSecEval, with CodeQL.

- **Protocol.** Everything the study depends on is in `protocol/`: samples, configurations,
  variants, oracles, security cards, analysis designs and the analysis plan
  (`protocol/analysis/plan.md`). The gold annotations are in `data/annotated/`.
- **Freeze.** The protocol was frozen once, in commit `394c893` (`protocol/protocol-freeze.json`),
  and that commit was pushed to the public remote before the first final generation; the push is the
  preregistration. Final commands verify the freeze and run on an extraction of
  `git -c core.autocrlf=false archive 394c893`.
- **Deviations.** Every later change is a dated row of `protocol/deviations.md`. That includes the
  exploratory analyses decided after the main results (G, R, S) and the clean-up of comments, tests
  and documentation after the study was closed at commit `26e9362`. From that clean-up on, the
  working tree no longer verifies the freeze; the freeze commit does.
- **Gold.** The gold leaves of the judgement units were proposed by an LLM; an AI-assisted step
  drafted a decision and a reason for every unit, which the student checked, corrected in part and
  reviewed further. The gold is never described as written by hand, and an unassisted re-annotation
  of 30 + 30 prompts is the independent check (`protocol/reannotation/`, `results/reannotation/`).
- **Results.** The generation journals, oracle outputs and analysis reports are not versioned (size,
  dataset licences and paths of the machine). `results/` holds the tracked results of the study:
  the retrained parser r4, the re-annotation agreement, the draft-to-gold diff, the near-duplicate
  groups and the Colab notebooks of the exploratory runs (see `results/README.md`).

**The supervisor's decisions** the design rests on:

- D1: first the reviewed PST (H-gold), even on a subset of prompts; then the PST produced by the
  parser (H-auto);
- D2: BigCodeBench-Instruct for correctness;
- D3: CodeQL for security, with its unreliability declared as a threat to validity;
- D4: a second reviewer, for the annotations and for the security judgments, is optional.

**The parser run `parser-2026-11-12-r3`** is the one H-auto used; its identity is pinned in
`protocol/hauto/parser.json` and `configs/experiments/hauto-parser.yaml`. Two provenance caveats
hold:

- **Unsloth.** The optimised backend loads its own prequantised repository instead of the requested
  revision, so the model revision of that run is not verified (`revision_verified: false`). The
  downstream generation uses the `hf` backend, which verifies the revision and refuses final runs
  without verification.
- **CRLF.** The source archive of that run was created from a Windows checkout with
  `core.autocrlf=true`: the hashes of some text files in its manifest are those of the CRLF copies,
  not of the git blobs. The model still received LF text, because the files are read with
  `read_text`. New archives are created with `make source-archive`, which uses
  `git -c core.autocrlf=false archive` and verifies the tree.

The artefacts at the root of `results/` are historical: they do not attest that the current
configurations were trained and evaluated. Recomputations go into new directories and never replace
that evidence.

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) to manage the environment
- For training: an NVIDIA GPU and a compatible CUDA stack. 8 GB need a smoke test of memory and time
  and do not guarantee that the base configuration fits.
- A Hugging Face token with read permission, if the base model is gated

Evaluation alone needs no GPU.

## Installation

```bash
uv sync --extra cu128 --extra train --extra unsloth
```

On a machine without a GPU, enough to prepare the data, compute the metrics and run the test gate:

```bash
uv sync --extra cpu --extra train --extra dev
```

The test gate needs the `train` extra too: some tests import `transformers` and `tokenizers`.

Copy `.env.example` to `.env` and fill in the values you need.

Full instructions, including the containerised path, are in [INSTALL.md](INSTALL.md); system
requirements are in [REQUIREMENTS.md](REQUIREMENTS.md).

## Usage

The examples use `outputs/new-run/`: choose a new name. Preparation refuses records or splits that
already exist; train and generate must receive the same path overrides. The training examples are
alternatives, not to be run all in the same directory.

```bash
# Check that a configuration is well formed, without running anything
uv run pstparser validate-config --config configs/experiments/baseline.yaml
```

```bash
# Turn the annotated corpus into records and freeze its partition
uv run pstparser prepare-data --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits
```

```bash
# Train the adapters on the prepared corpus
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --run-dir outputs/new-run/training
```

```bash
# Generate the predictions for the evaluation split
uv run pstparser generate --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --adapter outputs/new-run/training/adapter --run-dir outputs/new-run/generation
```

```bash
# Without an adapter the base model answers, untrained, on the same test set
uv run pstparser generate --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --run-dir outputs/new-run/zero_shot
```

```bash
# Compute the metrics. Needs neither a GPU nor a model.
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

```bash
# Place every generated phrase in the prompt and inject its position. No GPU either.
uv run pstparser align --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

Every command accepts `--set nested.key=value` to override one value of the configuration, for
example `--set training.max_steps=50`. The overrides are recorded in the resolved configuration that
goes with every run.

Values are read as YAML, where `no`, `yes`, `on` and `off` are booleans. To pass one of those words
as text, quote it: `--set 'training.eval_strategy="no"'`. A wrong type is still refused by the
validation before the run starts.

On a GPU with 8 GB of VRAM, halving the batch and doubling the accumulation keeps the nominal number
of examples per update, but **does not guarantee the same optimisation trajectory**: padding,
numerical reductions, batch order and implementation can all matter.

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8 --run-dir outputs/new-run/training
```

For a quick test without a GPU, with a toy model in place of the real one:

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --set model.backend=hf --set model.name=hf-internal-testing/tiny-random-LlamaForCausalLM --set model.load_in_4bit=false --set model.max_seq_length=512 --set training.max_steps=2 --set training.warmup_steps=0 --set training.optim=adamw_torch --set inference.max_new_tokens=32 --run-dir outputs/new-run/training
```

## Running in a container

Two images, built from the same `Dockerfile` with different targets.

| Target | Base | Content |
|---|---|---|
| `train` | Python 3.11 slim | Full fine-tuning stack, needs a GPU |
| `eval` | Python 3.11 slim | Only what computing the metrics needs |

The image sizes depend on the build and on the actual dependencies.

```bash
docker build -f docker/Dockerfile --target eval -t pstparser:eval .
```

```bash
docker compose -f docker/compose.yaml --profile eval run --rm score score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

The evaluation image **contains neither PyTorch nor any model library**. This is not about size: it
keeps the metrics recomputable even when the training environment cannot be reproduced, and lets the
results be checked on any machine.

The image is self-contained: configurations and system prompt are inside it, and the only thing to
provide is the predictions file.

```bash
docker run --rm -v "$PWD/results:/work" pstparser:eval score --config configs/experiments/baseline.yaml --predictions /work/predictions.jsonl
```

Training needs the NVIDIA Container Toolkit:

```bash
docker compose -f docker/compose.yaml --profile train run --rm train train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits
```

The base model weights live in a named volume, so they never enter the image and survive rebuilds.
`data/`, `outputs/` and `results/` are mounted from the host.

## Artefacts of a run

Every `train` run writes a directory under `outputs/`:

| File | Content |
|---|---|
| `adapter/` | LoRA weights and tokenizer |
| `train_manifest.json` | v2 manifest of the stage: configuration, hashes of the recorded inputs and of the code snapshot, seed, versions and available training metadata |
| `training_metrics.jsonl` | Metrics emitted by the trainer, one per line |

New manifests are separate per stage: `prepare_manifest.json`, `train_manifest.json`,
`generate_manifest.json`, `score_manifest.json` and `align_manifest.json`. Preparation saves its
manifest in the directory of the prepared data. `run_manifest.json` is a name found in historical
artefacts, not the one of new runs. Hashes and configuration improve traceability, but they do not
guarantee bit-for-bit reproducibility, nor do they prove on their own that every external artefact
can be recovered. A null digest marks an input that was not available.

`generate`, `score` and `align` in turn write `predictions.jsonl`, `results.json`,
`eval_details.jsonl` and `alignments.jsonl`. Without `--run-dir`, `score` and `align` create a new
`score-...` or `align-...` subdirectory under `training.output_dir` (`outputs/` by default), never
next to the predictions, so that a recomputation never writes into `results/`. Collisions with
artefacts of the same stage are refused: choose a new directory instead of deleting or overwriting
historical runs.

## Evaluation of the parser

Generation is separate from computing the metrics. `generate` loads the model and stores the
predictions; `score` reads that file without loading any model. Time and memory depend on the data.
The historical predictions in `results/` allow recomputations on a CPU, keeping track of the version
of the metrics.

The content metrics are computed on the predictions that are well formed:

| Metric | Meaning | Expected value |
|---|---|---|
| JSON validity | share of predictions that parse **and** follow the schema | high |
| Per-field F1 | token agreement with the reference, per leaf, with its support | high |
| Confusion matrix | which node every prompt token was assigned to, against the right one | diagonal |
| Tree edit distance | how many edits separate the predicted structure from the expected one | low |
| Hallucination rate | share of predicted tokens absent from the prompt | zero |
| Coverage | share of prompt tokens placed in a node | one |
| Normalised legacy reconstruction | reconstruction of the aligned subset with normalisation; does not prove full fidelity | high |
| All leaves accounted for (v2) | share of predictions with no extracted leaf left unplaced | one |
| Strict full reconstruction (v2) | all leaves accounted for and exact textual reconstruction | one |
| Placed phrases | share of generated phrases found in the prompt | one |

None of these numbers says how much the fine-tuning added: that needs a point of comparison.
`generate` without `--adapter` makes the base model answer on the same test set with the same system
prompt, and it is the cheapest comparison available, since it needs no training.

The parser gate of the empirical study uses its own macro F1: per prediction and leaf, the distinct
tokens assigned to the leaf; overlap, predicted and gold token counts summed over the valid
predictions; the F1 of each leaf averaged over the leaves with gold support
(`MACRO_F1_DEFINITION` in `src/pstparser/empirical/auto.py`).

### Reconstruction and positions

The model generates only the text of the leaves. A second stage, deterministic and model-free, places
every phrase back in the prompt it was extracted from and derives its position: every phrase claims a
character interval, two intervals cannot overlap, and the position is the rank of the interval. From
there come the artefact with `{"position": ..., "phrase": ...}` leaves and the reconstruction check.
The old field `exact_reconstruction_rate` keeps its legacy meaning: it normalises whitespace and can
ignore unplaced leaves. The new v2 reports add `aligned_subset_reconstruction_normalized_rate`,
`all_leaves_accounted_for_rate` and `strict_full_reconstruction_rate`. Compare metrics of the same
version, and never relabel historical results as strict. None of these measures replaces the
validation of the schema or a human check of the categories.

Coverage is only a substitute for that property: it says which words were placed, never in which
order. An output that shuffled all the phrases would still get full coverage.

The behaviour of each metric is locked by golden cases in `tests/metrics/cases/`, with the expected
scores versioned next to them: an unexplained change is a defect, not a silent change of meaning.

### Constrained decoding

Setting `inference.structured_output: true` (needs the `structured` extra) asks for decoding
constrained by the schema. Backend compatibility, completion within the budget and the absence of
errors must be checked: **it is not an unconditional guarantee of valid JSON**. Measure validity,
truncations and failures anyway, and keep them apart from the quality of the content.

In that mode the observed validity is driven by the external constraint and does not measure by
itself how well the model learnt the format. The value lies in comparing the two runs: free, to
measure how much the model has internalised the format; constrained, to assess the content at equal
format.

## Expanding the corpus

The delivered spreadsheet comes from real conversations between developers and an assistant, where
direct requests dominate: the branches of the taxonomy that describe reasoning appeared three times
in 975 records, so three leaves out of nine could be neither learnt nor measured. The `synth` command
generates additional prompts for those paradigms, starting from thirty hand-written seeds in
`data/seeds/paradigms.yaml`.

```bash
uv run pstparser synth --config configs/experiments/baseline.yaml
```

The provider is any service with a chat-completions endpoint compatible with the OpenAI format; the
credential is read from the environment, never from a configuration file.

**The command produces prompts, not annotations.** The output is a file with the prompt and the
paradigm filled in and the nine leaves empty, in the same format `prepare-data` reads: it must be
labelled before it can come back in. That step stays human work.

The format is the same on purpose. Transcribing 150 prompts from one format to another means 150
chances of breaking the exact extraction, and the downstream check would report a transcription
error as if it were an annotation error.

The annotated file is not merged into the starting corpus: it is added to `data.sources`, which
concatenates the sources while keeping them as separate files. The delivered material stays apart
from what was annotated later, and `prepare-data` reports how many records each source brought.

## Corpus audit and downstream evaluation

Two tools stay outside the `pstparser` command and are invoked as modules, without dedicated entry
points.

```bash
# Read-only diagnostics: uncovered text, tolerant matches, possible families of duplicates
uv run python -m pstparser.data.audit --config configs/experiments/baseline.yaml --output outputs/new-run/corpus-audit.json
```

The audit corrects neither annotations nor partitions: the candidates (Jaccard similarity above
`--similarity-threshold`, 0.9 by default) are to be reviewed by a person. The output file must be new
and its directory must already exist.

```bash
uv run python -m pstparser.empirical render CANONICAL.jsonl VARIANTS.jsonl
uv run python -m pstparser.empirical generate VARIANTS.jsonl CONFIG.json GENERATION_DIR
uv run python -m pstparser.empirical check GENERATIONS.jsonl ORACLES.jsonl CHECK_DIR IMAGE@sha256:... ANALYSIS_ID
uv run python -m pstparser.empirical analyze DESIGN.json OUTCOMES.jsonl REPORT.json
```

`render` produces the paired T/H-gold variants from reviewed canonical annotations. `generate` needs
an explicit configuration with an immutable model revision and a review record. It checks the token
budget of every variant before generating, and writes a resumable journal (`generations.jsonl`).
`check` calibrates the given functional oracles and runs them in a local Docker container without
network, with the image pinned by digest. `analyze` runs the paired analysis on the declared design.
No command chooses a benchmark on its own. A run with purpose `final` starts only from a verified
freeze that lists its configuration and variants.

The other downstream commands, in the order they are used (`--help` describes their arguments):

| Stage | Commands |
|---|---|
| Data and samples | `import-benchmark`, `calibrate-bigcodebench`, `merge-calibrations`, `sample` (permutation and prefixes), `sample-pilot`, `sample-families`, `sample-hard`, `sample-reannotation` |
| Gold annotation | `propose-spans` (`draft` only), `gold-lint`, `canonicalize`, `gold-diff` (draft against gold), `agreement` (re-annotation) |
| Generation | `render`, `generate` (also on Colab with `notebooks/downstream.ipynb`), `model-digests`, `merge-journals` (two journals of one session) |
| H-auto | `parse-prompts` of the `pstparser` interface (on Colab with `notebooks/hauto-parser.ipynb`), `retarget-predictions` (target replaced by the frozen gold), `auto-annotate`, `parser-gate`, `merge-ledgers`, `hauto-ledgers` |
| Evaluation | `check-bigcodebench` (official harness, 2 + 1 runs), `calibrate-security`, `check-security` (CodeQL, one database per program), `blind-sample`, `blind-score`, `determinism-sample`, `determinism-compare` |
| Pilot and freeze | `pilot-report` (non-directional; `--merged` for two journals), `power`, `freeze`, `freeze-verify` |
| Analysis | `design`, `analyze`, `check`, `covariates`, `contamination`, `near-duplicates` |
| Thesis | `tables`, `figures`, `thesis-tables` |

`thesis-tables` regenerates every table (CSV and LaTeX) and figure (PDF and PNG) that the files of a
run allow, in a new directory. Every file records the `run_id`, the SHA-256 of every input and the
digest of the sources. Figures need the `analysis` extra (matplotlib):

```bash
make thesis-tables DS_RUN=<run directory> TABLES=<new directory>
```

## Colab notebooks

`notebooks/colab.ipynb` clones the repository and calls the same command-line interface used locally
and in a container. It holds no logic: duplicated logic would be the first thing to drift from the
rest of the project. `notebooks/downstream.ipynb` and `notebooks/hauto-parser.ipynb` are the
launchers of the generations and of the parser runs of the study, from a source archive of a clean
committed tree. The launchers of the exploratory runs are in `results/r4/notebooks/` and
`results/s/notebooks/`.

## Configuration

Configurations are YAML files composed on two levels:

- `configs/_base/` holds the reusable fragments (corpus, model, training, evaluation);
- `configs/experiments/` holds the experiments, which declare `extends` on the fragments and
  override only what changes.

`configs/experiments/baseline.yaml` is the reference experiment: every other configuration should
be expressed as a delta from it, so that comparisons stay meaningful.
`configs/experiments/hauto-parser.yaml` pins the parser used for H-auto. The generation
configurations of the study are in `protocol/configs/`.

The resolved configuration is serialised with the artefacts of every run, so the exact parameters a
result was produced with can always be traced.

## Layout

```
configs/         experiment configurations
docker/          multi-target Dockerfile and compose file
data/raw/        delivered corpus, immutable
data/annotated/  annotations written here, as editable text; gold of the empirical study
data/splits/     identifiers of the three partitions, frozen
notebooks/       Colab launchers of the CLI
prompts/         versioned system prompt
protocol/        frozen protocol of the empirical study, analysis plan and deviations
scripts/         archive verification
src/pstparser/
  config/        typed schema and YAML composition
  pst/           taxonomy, serialisation and derived schema
  data/          reading, normalisation, integrity, partitions, exchange formats
  models/        loading of the base model and of the adapters
  training/      fine-tuning recipe
  inference/     generation of the predictions
  evaluation/    metrics and reports
  synth/         generation of synthetic prompts
  empirical/     downstream contracts, renderers, oracles and paired analysis
  repro/         seeding and run manifests
results/         reference predictions and metrics, results of the empirical study
results/tools/   tools written after the freeze (declared in protocol/deviations.md)
tests/           test suite
```

Dependencies between packages are acyclic and point downwards: `pst/` depends on nothing, `data/`
depends on `pst/`, `evaluation/` depends on `data/`, and only `models/`, `training/` and `inference/`
touch PyTorch. This direction is what makes an evaluation environment without the model stack
possible.

## Corpus partition

Three sets, not two. Watching a set during training and then reporting results on the same set does
not measure generalisation: it measures how well training was stopped. The **validation** set is what
the training loop may look at; the **test** set is what results are quoted from, and it is read once.

Records that share a prompt end up on the same side. The corpus contains repeated prompts, and leaving
one in training and another in test would allow being evaluated on text already seen. Records that
are identical in both prompt and annotation are reduced to one.

`generate` writes predictions for the **test** split by default; `--split val` is for checks made
during development.

## Corpus formats

A source declares how it is written. Inferring it from the suffix would make the behaviour depend on
how the file was named, and would say the same thing twice (in the extension and in the presence of a
sheet) with nothing keeping the two in agreement.

| | `excel` | `yaml` |
|---|---|---|
| What it is | The delivered spreadsheet | Annotations written in this repository |
| `sheet` | required | absent |
| `fixes` | allowed | forbidden: the file itself is corrected |
| Leaf addressed by | column header | path in the taxonomy |
| Disjoint segments | `<sep>` token in the cell | one list item each |

```yaml
data:
  sources:
    - path: data/raw/prompt_dataset_v2.xlsx
      sheet: prompt_dataset_v2
      fixes: data/raw/annotation_fixes.yaml
    - path: data/annotated/reasoning.yaml
      format: yaml
```

A record annotated as text can be read in full in a diff, which is why the format exists: reviewing
a hundred annotations line by line is not the same as opening a binary. And by naming the leaves by
path, a file written here does not inherit the `CONSTRAINS` header that the delivered spreadsheet
carries for compatibility.

```yaml
records:
  - paradigm: tree_of_thoughts
    prompt: |-
      class A: pass

      class B: pass

      Rename these classes to something meaningful.
      Do not change their behaviour.
    leaves:
      main_instruction: Rename these classes to something meaningful.
      context.constraints: Do not change their behaviour.
      context.data:
        - "class A: pass"
        - "class B: pass"
```

An omitted leaf has no segments; a string is one segment; a list is one segment per item. The prompt
must be written as a **literal** scalar (`|-`): the folded one (`>-`) turns line breaks into spaces,
and the text would change without saying so.

## Corrections to the annotation

The spreadsheet in `data/raw/` is an input and stays exactly as it was delivered. The corrections
live in `data/raw/annotation_fixes.yaml`, each with the row it applies to and the reason it was made:
quotes added during copy and paste, one character typed for another, a passage rewritten in the first
person, a cell the spreadsheet read as a formula and destroyed. A correction that no longer describes
its cell is an error, not a no-op, so the corpus and its corrections cannot drift apart unnoticed.

The mechanism applies to spreadsheet sources only, a restriction stated by the schema. It exists
because that file cannot be edited; an annotation written here is corrected where it is, and the
history of what changed is already the diff. Two places to look for the truth of a record are worse
than one.

## Known limits

- **The delivered spreadsheet is not the one used in the original work.** The starting notebook reads
  a file of about 1,208 rows that was not delivered; this repository holds the 975-row version. The
  numbers are therefore not directly comparable with the published ones. If the original file were
  recovered, it would be enough to add it to `data.sources` in `configs/_base/data/`.
- **The support of the `reasoning` branch is modest and curated.** Its three leaves count 113, 86 and
  33 records out of 1,123, almost all from the prompts annotated here: the delivered spreadsheet had
  three in total. The scores on those nodes must be read with their support, and the prompts that
  exercise them were generated and chosen to exercise them; they are not a sample of how those
  paradigms occur in real conversations.

## Porting notes

Some elements of the starting notebook were not ported:

- a second inference loop, a duplicate of the first but without the chat template and the system
  prompt, and with a generation budget too small to complete the outputs;
- a cell without source;
- a set of validation classes never instantiated, which declared fields inconsistent with the targets
  actually produced. The validation schema is generated from the taxonomy, so it is consistent by
  construction;
- dependencies installed and never used;
- shell commands and interactive authentication, replaced by programmatic equivalents and environment
  variables.

## Licence

MIT. See [LICENSE](LICENSE).
