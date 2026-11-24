# Results

This directory holds two kinds of artefacts, versioned with the code:

- at its root, the **historical reference run of the parser** (2 August 2026), which reproduced the
  starting prototype;
- in its subdirectories, the **tracked results and tools of the empirical study**, written after the
  protocol freeze (`394c893`). The freeze admits new tracked files only here, and every one of them is
  declared in `protocol/deviations.md`.

| Path | Content |
|---|---|
| `predictions.jsonl`, `alignments.jsonl`, `results.json`, `eval_details.jsonl`, `conformant/`, `run_manifest.json`, `train_manifest.json`, `training_metrics.jsonl` | Historical reference run of the parser (below) |
| `attempts/` | Reports of runs that did not succeed, with their full logs |
| `reannotation/` | Agreement (κ on the judgement units) between the gold and the unassisted re-annotation of 30 BigCodeBench and 30 LLMSecEval prompts |
| `gold-diff/` | Changes from the rule drafts to the gold, per rule |
| `near-duplicates/` | Groups of near-duplicate BigCodeBench prompts, used by the S-NEARDUP sensitivity |
| `r4/` | Exploratory analysis R: labels, training data, identity and Colab notebooks of the retrained parser r4 |
| `s/notebooks/` | Exploratory analysis S: Colab notebooks of the LLMSecEval parse and generations with r3 and r4 |
| `tools/` | Tools written after the freeze, before the data they read (re-annotation sheets, exploratory analyses of the plan, decomposition of the H-auto effect, count on the unanalysable security cases, r4 data), with their tests. The `results_*.py` tools read the evaluated outcomes for the results analysis declared in `protocol/deviations.md` (24/11/2026): reproduction of the frozen reports, sample flow, correctness, heterogeneity (with the BigCodeBench library-to-domain table, `bigcodebench-lib2domain.json`, Apache-2.0), errors, H-auto, security, multiplicity and the parser versions |
| `downstream/` | Intended layout for downstream runs; see its README |

The generation journals, oracle outputs and analysis reports of the study are not versioned here:
they are large, they contain dataset texts under their own licences, and they carry paths of the
machine they ran on.

## The historical reference run

Its predictions are versioned because they make the metrics **recomputable without a GPU and
without a model**:

```bash
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl --run-dir results/conformant
```

Anyone can regenerate every number in a few seconds, check it, or apply different metrics to the
same predictions.

> **None of the scores in this directory is to be regenerated with the current code.**
>
> `results.json` and `eval_details.jsonl` are the output of the code at the tag `baseline-as-is`;
> reproducing them needs that tag (`git worktree add ../pst-asis baseline-as-is`).
>
> `conformant/` was produced after the metrics were corrected but **before** the schema was. The
> restrictive node was then called `constrains`, and that is what this model was trained to emit.
> With the schema corrected to `constraints`, all 82 parsable predictions are non-conformant, and the
> scores that would come out are meaningless. **This is the concrete proof that the rename requires a
> new training**, and the reason these numbers belong to the revision that produced them.
>
> Every new score goes into a subdirectory, with `--run-dir`.

The adapter weights are not versioned: they weigh 167 MB and are not needed to reproduce any metric.
They are needed only to generate new predictions, and for that it is enough to rerun the training
from the manifest.

### The run

| | |
|---|---|
| Date | 2 August 2026 |
| Hardware | NVIDIA L4, 24 GB (Google Colab) |
| Configuration | `configs/experiments/baseline.yaml`, without overrides |
| Training | 300 steps in 58 minutes, effective batch 8 |
| Generation | 88 predictions in 33 minutes, greedy decoding |
| Corpus | 975 records, split 887 / 88 |

#### Loss curve

| Step | Training | Validation |
|---:|---:|---:|
| 50 | 0.2906 | 0.2922 |
| 100 | 0.3171 | 0.2736 |
| 150 | 0.2477 | 0.2683 |
| 200 | 0.2252 | **0.2622** |
| 250 | 0.1686 | 0.2644 |
| 300 | 0.2076 | 0.2660 |

The validation loss reaches its minimum at step 200 and rises over the next hundred steps. Since
`load_best_model_at_end` is off, the saved adapter is the one of step 300, not the best of the six
evaluated. This is the behaviour of the reference work, faithfully reproduced, and the curve above is
what makes it visible.

### Metrics

**No prediction changed between the two columns.** Only the definitions they are measured with
change: the left column reproduces the computation of the starting work, the right one applies the
formulas as published.

| Metric | As originally computed | By the definitions |
|---|---:|---:|
| JSON validity | 93.18% | 93.18% |
| Coverage score | 99.71% | 99.60% |
| Hallucination rate | 0.39% | 0.41% |
| Tree edit distance | 1.55 | 1.55 |

| Leaf | Originally | By the definitions | Support |
|---|---:|---:|---:|
| `context.role` | 1.0000 | 1.0000 | 24 |
| `main_instruction` | 0.9127 | 0.8806 | 1294 |
| `context.data` | 0.7974 | **0.9070** | 3268 |
| `examples` | 0.7775 | 0.7021 | 198 |
| `context.format` | 0.5238 | **0.8105** | 668 |
| `context.constrains` | 0.4830 | **0.7097** | 585 |
| `reasoning.influence` | — | n/a | 0 |
| `reasoning.reasoning_examples` | — | n/a | 0 |
| `reasoning.paths` | — | n/a | 0 |

#### What changed, and why

**Coverage and hallucination: almost nothing.** The two formulas compare sets of tokens on both sides
of the fraction; the starting computation treated one of them as a list, in opposite ways in the two
metrics. Correcting it moves the values by a tenth of a point. It is a conformance fix, not a reversal
of results, and it is worth saying so explicitly because the opposite would have been easy to suggest.

**Per-field F1: a lot.** `context.format` rises by 0.29 and `context.constrains` by 0.23. The cause is
that segments were paired **by their position in the list**: if the model produced two segments where
the annotator had recorded one, the first was compared with the first and the second with the empty
string, penalising correct content twice. Pooling the tokens of the field before comparing them, as
the definition prescribes, removes that penalty. **The low agreement on the two "hard" fields was
largely an artefact of the pairing, not a weakness of the model**, which changes the reading of what
the open problem on those nodes is.

**The support now goes with every score.** A 1.0000 on 24 tokens and a 0.9070 on 3,268 are not the
same claim, and the three leaves of the `reasoning` branch are explicitly not assessable instead of
missing from the table.

**Validity is stricter than before and gives the same number.** A prediction now counts as valid if it
parses **and** follows the schema derived from the taxonomy: a `{}` no longer passes. All 82 parsable
predictions are also conformant, so the value does not move. The coincidence is a result, not an
oversight.

### Reconstruction of the prompt

This is the first measurement of the property the framework is argued on: collect the leaves, sort
them by position and recover the starting prompt. The coverage score was only a substitute, because
it reduces the prediction to a bag of tokens and says **which** words were placed, never **in which
order**.

| | Prediction | Reference |
|---|---:|---:|
| Phrases | 174 | 194 |
| Placed in the prompt | 88.51% | 88.14% |
| With more than one candidate occurrence | 0.00% | 0.00% |
| **Rebuild the prompt** | **72.73%** | 79.55% |
| ... among the parsable ones | 78.05% | 79.55% |

Reconstruction is compared up to every whitespace character, because the space between two segments
belongs to neither.

**The gap between 99.60% coverage and 72.73% reconstruction is what this measure adds.** It says that
almost every word of the prompt ends up somewhere, but that in one case out of four the set of leaves
is not enough to put the text back together. Of 88 predictions, 6 do not parse, and 18 leave part of
the prompt uncovered, from an opening sentence never segmented (index 869, 45 characters) to an
omitted final clause (index 136, 77 characters).

**The reference does not rebuild much better than the model.** The 12% of annotated phrases that
cannot be placed is not a defect of the model: it is ground truth that cannot be found in its own
prompt. Measuring the two sides separately is what keeps a defect of the annotation format from being
charged to the model, and the reason `alignments.jsonl` reports both.

**Ambiguity is zero, and must still be checked on every run.** No phrase, predicted or annotated, has
more than one candidate occurrence: the disambiguation rule was never asked to choose. This is a fact
about this corpus, not a theorem; shorter segments would raise it.

### Where the prompt tokens end up

The confusion matrix is built on the prompt tokens, not on whole field values: both trees are placed
in the prompt, and for every token the label of the reference is compared with that of the
prediction. `none` is a token one side assigned to no node.

| Expected | Predicted | Tokens |
|---|---|---:|
| `context.data` | `none` | 622 |
| `none` | `examples` | 473 |
| `none` | `context.data` | 200 |
| `none` | `main_instruction` | 174 |
| `context.constrains` | `context.format` | 117 |

The first four rows are **boundary** errors, not category errors: text one side places and the other
does not. The previous computation compared whole field values and silently dropped whatever did not
match literally, so it could not see any of them. The fifth row is the semantic confusion between
presentational and restrictive instructions, now quantified in tokens.

### Ablation: punctuation

The content metrics normalise punctuation away, so they never measure whether a segment was cut one
comma too early. Recomputing the same 88 predictions with a tokenisation that keeps every sign as a
token of its own:

| Metric | Punctuation dropped | Punctuation kept |
|---|---:|---:|
| Coverage score | 99.60% | 99.54% |
| Hallucination rate | 0.41% | 0.61% |
| F1 `context.constrains` | 0.7097 | 0.7101 |
| F1 `context.data` | 0.9070 | 0.9148 |
| F1 `context.format` | 0.8105 | 0.8216 |
| F1 `examples` | 0.7021 | 0.7583 |
| F1 `main_instruction` | 0.8806 | 0.8837 |

The difference is small everywhere except for hallucination, which grows by two tenths of a point.
The default is unchanged: changing it together with the definitions would have made it impossible to
attribute the deltas to one or the other, and the table says there is little to gain. If it is ever
changed, it will be a decision driven by these numbers and not by an impression.

### How to read them

**The two evaluation sets differ** from those of the published work: 88 examples here against 109,
drawn from corpora of different sizes. The comparisons are indicative, not paired.

**Three leaves out of nine have no support.** `reasoning.influence` has three annotated examples in
all, `reasoning.reasoning_examples` and `reasoning.paths` none: there is nothing to measure.

**JSON validity does not reach 100%.** Six predictions out of 88 do not parse. The generation budget
and the context window coincide, so prompts with a long `data` field can run out of room before the
object is closed. Schema-constrained generation (`inference.structured_output`) removes the problem by
construction, at the price of making the metric uninformative: for this reason it stays off in this
run, and comparing the two runs is more interesting than either on its own.

### Attempts

`attempts/` collects the reports of runs that did not succeed, with their full logs. They avoid
repeating diagnoses already made and record the hardware limits met.

| Date | Hardware | Outcome |
|---|---|---|
| 2026-08-01 | RTX 4070 Laptop, 8 GB | stopped at step 25 of 300 by thermal throttling; no model produced |
