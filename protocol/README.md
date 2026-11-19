# Protocol of the downstream study

Everything the study is frozen on before any final generation, versioned with LF line endings.

- `samples/`: the frozen samples: the stratified permutation of the eligible BigCodeBench tasks, the 600-task prefix (S_gold) and the nested 400-task prefix (time fallback) in `prefix-600/` and `prefix-400/`, the 800-task prefix the drafts were proposed on, the pilot pools and the LLMSecEval identifiers, with the manifests of the commands that drew them.
- `reannotation/`: the seeded samples of the from-scratch re-annotation (BigCodeBench and LLMSecEval).
- `configs/`: the generation configurations of the pilot and of the final runs.
- `hauto/`: the parser identity and the gate thresholds, committed before any benchmark parse.
- `oracles/`: settings of the BigCodeBench harness and of CodeQL, and the frozen exclusions of the calibration.
- `security/`: scenario cards and calibration controls of the security oracle (see its README).
- `analysis/`: the prespecified analysis plan with the time rules (`plan.md`) and frozen parameters of the analyses (for example the contamination threshold).
- `guidelines/`: the gold guidelines `gold-v2.0` the reviewed and frozen gold follows, with the declared procedure (rule drafts, LLM proposals on the judgement units, decision of the student).
- `rules/`: rules stated as text next to the code that applies them (`prompt-as-data.md`).
- `freeze-spec.yaml`: the list of files the single freeze covers; `protocol-freeze.json` is written from it once, by `python -m pstparser.empirical freeze`, and checked by `freeze-verify`.
- After the freeze no listed file changes in place: a correction is a new file and an entry in `deviations.md`.
- The freeze proves what was fixed, not when: the date comes from the single OSF deposit under embargo and from the exported messages to the supervisors, kept in `acks/`.
