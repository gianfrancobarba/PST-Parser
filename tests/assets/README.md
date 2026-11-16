# Test assets

Fixtures shared by the test suite. Everything here is small enough to run on CPU
in seconds and is committed so that tests never depend on the network.

## `tiny_corpus.xlsx`

Seven records on a worksheet named `corpus`, using the same eleven columns as the
production corpus. Each record targets a specific behaviour of the preparation
pipeline:

| Row | Purpose |
|----:|---------|
| 1 | Every populated leaf annotated at once |
| 2 | Instruction only, all other leaves empty |
| 3 | Discontinuous span joined by the multi-segment separator |
| 4 | Record without a main instruction |
| 5 | Annotation covering only a fraction of the prompt, below the integrity threshold |
| 6 | Reasoning trigger populated |
| 7 | Cell value shaped like a Python list literal |

The three reasoning columns that carry no annotation in the production corpus are
also empty here, so the inferred column types match.

## `tiny_annotations.yaml`

Three records annotated as text rather than on a worksheet, standing in for a
corpus written in this repository. All satisfy the extraction contract, so the
integrity check passes on them. Three is also the fewest a three-way partition
can be drawn from:

| Record | Purpose |
|-------:|---------|
| 1 | Every leaf a single segment, a paradigm recorded, most leaves left out |
| 2 | No paradigm, and a leaf holding two disjoint parts as two entries |
| 3 | A segment spanning several lines, at a leaf of the reasoning branch |

## `system_prompt.md`

Short stand-in for the production system message. Its content is irrelevant to the
assertions; it exists so that configurations referencing a prompt file validate.

## `configs/`

| File | Purpose |
|---|---|
| `_base/data.yaml`, `_base/model.yaml` | Fragments used to exercise composition |
| `valid.yaml` | Composes both fragments and validates successfully |
| `invalid.yaml` | Violates a field constraint and must be rejected |
| `circular_a.yaml`, `circular_b.yaml` | Mutually extending files, must be rejected |

## `ratesci_pairbinci.json`

Reference intervals for a paired difference of proportions, used by
`tests/unit/test_intervals.py`. Sixteen 2×2 tables (edge cases with no or
one-sided discordance, and tables of 800 tasks), each at 95% and 90%, computed
once with R 4.5.1 and the CRAN package `ratesci` 1.1.1:

- Tango's score interval: `pairbinci(x, level, contrast = "RD", method = "Score",
  skew = FALSE, bcf = FALSE, cc = FALSE, precis = 10)`;
- Newcombe's method 10: `pairbinci(x, level, contrast = "RD", method = "MOVER_newc",
  moverbase = "wilson", precis = 10)`.

`x` is `c(both_pass, treatment_only, baseline_only, both_fail)`, so the difference
is treatment minus baseline, as in the analysis.
