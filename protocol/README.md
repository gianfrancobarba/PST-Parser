# Protocol of the downstream study

Everything the study is frozen on before any final generation, versioned with LF line endings.

- `samples/`: the frozen samples (BigCodeBench final sample and permutation, pilot pools, LLMSecEval identifiers), with the manifests of the commands that drew them.
- `oracles/`: settings of the BigCodeBench harness and of CodeQL, and the frozen exclusions of the calibration.
- `security/`: scenario cards and calibration controls of the security oracle (see its README).
- `analysis/`: frozen parameters of the prespecified analyses (for example the contamination threshold).
- `freeze-spec.yaml`: the list of files the freeze covers; `protocol-freeze.json` is written from it once, by `python -m pstparser.empirical freeze`, and checked by `freeze-verify`.
- After the freeze no listed file changes in place: a correction is a new file and an entry in `deviations.md`.
- The freeze proves what was fixed, not when: the date comes from the OSF registration and the e-mail to the supervisor, kept in `acks/`.
