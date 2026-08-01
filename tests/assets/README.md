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
