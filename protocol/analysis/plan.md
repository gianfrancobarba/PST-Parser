# Prespecified analysis plan and time rules

Frozen in F4, the single protocol freeze, before any final generation. Every number below comes from a frozen file or from the code at the frozen source. An analysis added later is reported as exploratory and dated in `protocol/deviations.md`.

## 1. Design

- **Generator**: Qwen2.5-Coder-7B-Instruct at revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, bf16, `hf` backend, one NVIDIA L4 (`protocol/model.json`, `protocol/configs/final/`). Greedy decoding with 1280 new tokens; sampling at temperature 0.7, top_p 0.8, top_k 20; `repetition_penalty` 1.0; generation seed 7319.
- **Correctness sample (S_gold)**: the 600-task prefix (`protocol/samples/prefix-600/bcb-prefix-600.json`) of the frozen stratified permutation (seed 7319) of the 1,129 eligible tasks of BigCodeBench-Instruct v0.1.4. Nested 400-task fallback only under the time rules of §7. The 73 Hard tasks of the prefix form a descriptive subgroup.
- **Security sample**: LLMSecEval S2, 100 prompts in 35 families (`protocol/samples/llmseceval-ids.json`, selection S2).
  - The targeted measure covers the 16 `calibrated` families (47 prompts). Its sensitivity adds the 3 `calibrated-with-false-positives` families (56 prompts, 19 families).
  - States come from `protocol/security/cards.yaml`, backed by `protocol/oracles/codeql-calibration.json`.
- **Conditions**:
  - BigCodeBench core: T, H-gold, E on every task, plus F if CP3-F admits it (§7);
  - H-auto on S_parse only, in the same session (second journal);
  - LLMSecEval: T, H-gold, with 5 sampled replicates (3 if CP3-R applies) and 1 greedy.
- **Treatments**: the H-gold renderer is `pst-gold-json-v1`, E is `envelope-json-v1`, F is `pst-flat-json-v1`. The user wrapper and the system prompt are identical in all conditions. The gold follows `protocol/guidelines/gold-v2.0.md`.

## 2. Confirmatory family

| Hypothesis | Contrast | Set | Test | Weight |
|---|---|---|---|---|
| H1 (RQ1) | H-gold − T, functional pass | S_gold | McNemar mid-p, two-sided | α = 0.04 |
| H3 (RQ3b) | H-auto − T, functional pass | S_parse | McNemar mid-p, two-sided | α = 0.01 |

- **Weighted Holm with recycling** (FWER 0.05):
  - H1 is rejected if p₁ ≤ 0.04, or if p₁ ≤ 0.05 and H3 is rejected at 0.01;
  - H3 is rejected if p₃ ≤ 0.01, or if p₃ ≤ 0.05 and H1 is rejected at 0.04.
- **Access of H3.** H3 belongs to the family only if `parser-gate --purpose final --freeze --c22` reports `confirmatory`. The thresholds are in `protocol/hauto/gate-thresholds.json`:
  - share of `auto-strict-v1` failures ≤ 0.25;
  - macro F1 on the leaves ≥ 0.80 on S_gold.
  - F1 is measured against the frozen gold (`retarget-predictions --require-status frozen`). The primary conversion is `auto-strict-v1`, by the C-22 rule read on the D-01 conversion (`protocol/hauto/d01/`).
  - Otherwise RQ3b is descriptive and H1 is read at α = 0.05. The same applies if the prefix parse had not finished by CP2.
- **S_parse**: the tasks of S_gold whose `auto-strict-v1` conversion succeeds. Its list and hash are fixed before any generation.

## 3. Estimands and reading (correctness)

- **Estimand.** Y = 1 when the program passes every test of the official harness. Δ is the per-task mean of the paired differences; the inferential target is the superpopulation Δ_S.
- **Intervals.** Tango score interval at 95% and 90%. For a confirmatory row the "different from zero" part is read at the level Holm assigned (`holm_alpha`), with the Tango interval at 1 − `holm_alpha`. Newcombe and exact McNemar are reported as sensitivity.
- **Equivalence.** δ = 5 pp. TOST on the 90% Tango interval inside (−5, 5) pp. Readings at 3 and 10 pp are always reported.
- **Four-way reading.**

| Reading | Different from zero | Equivalent within ±δ |
|---|---|---|
| E1 | yes | no |
| E2 | yes | yes |
| E3 | no | yes |
| E4 | no | no (report the MDE) |

  If mid-p and Tango disagree on zero, the mid-p decides and the disagreement is stated.
- **No observed power** is computed. The MDE comes from `power` at the realised n and S_parse size.

## 4. Oracles, missing data and sensitivity

- **Correctness.**
  - Harness `bigcodebench/bigcodebench-evaluate@sha256:a3cd34ec…`, offline, official sanitize. Exclusions: `protocol/oracles/bcb-exclusions.json`.
  - Two runs in two line orders; a third where they disagree. The majority decides, with a `flaky` flag. A timeout, a cut output, an empty extraction, `json_response` and `prompt_as_data` (`protocol/rules/prompt-as-data.md`) are fails.
- **Missing.** Only infrastructure failures after the retries: a CUDA error or OOM that does not reproduce, a Docker image that does not start, a CodeQL crash of the environment, or a suspect harness run repeated once. Every missing value has a reason.
  - A task with a missing cell in any declared arm is excluded from every contrast of that ledger (`--ledger-conditions`).
  - Joint bounds for missing and U are reported. If they change the E1–E4 reading, the conclusion is declared not robust.
- **Correctness sensitivity.**
  - S-FLAKY: exclude the flaky tasks.
  - Fail if the first two runs disagree.
  - Exact McNemar, Newcombe.
  - S-NEARDUP: one task per near-duplicate group.
  - Per protocol: both answers usable; labelled as conditioning on a post-treatment variable.

## 5. Security (exploratory)

- **Measures.** Y = 1 only for state N: usable program and no alert. General measure on the 100/35, with the 83 Python-origin prompts as sensitivity. Targeted measure on 47/16, with 56/19 as sensitivity.
  - Usability: `usability-rules-v2`.
  - The analyser: CodeQL 2.22.1 with the settings of `protocol/oracles/codeql.json`.
  - U counts as 0 in the primary analysis; U as V and U as N are sensitivities.
- **Primary analysis.**
  - Per-task mean over the 5 sampled replicates; the greedy answer is descriptive.
  - Test: studentized sign-flip by family, by exact enumeration for G ≤ 16, otherwise 9,999 draws, seed 42.
  - Interval: CR1 with t(G − 1).
- **Security sensitivity.**
  - Δ_fam, equal weights per family; stated if it differs by more than 2 pp.
  - Leave-one-family-out.
  - Δ_I, the share of unusable programs.
  - Family bootstrap, 10,000 draws, seed 42.
  - Alerts per 100 LOC.
  - Per CWE: descriptive only.
- **Blind review** of the oracle (U and V_targeted first, then the full form if finished by 28/11 at 12:00).

## 6. Other exploratory analyses (point estimates with 95% intervals, no p-values in the main tables)

- **E − T and H-gold − E** (same method as RQ1, on the same complete tasks):
  - if H-gold − T is E1, H-gold − E is E3 or E4, and E − T has the same sign, the effect is read as mostly JSON;
  - if E − T is E3 and H-gold − E is E1, the effect is read as categorisation.
  - These readings are descriptive.
- **F**, only if collected: H-gold − F and F − T.
- **H-auto**:
  - RQ3b end-to-end on S_gold, with a parser failure counted as fail;
  - RQ3c, H-auto − H-gold;
  - Δ_auto = Δ_gold + Δ_par, by parser quality on the task: valid, strict reconstruction, F1 terciles.
- **The PST package**, with covariates frozen before generation (`covariates`, `prompt-covariates-v1`):
  - length in tokens: log ratio H/T, terciles, logistic model on the discordant pairs;
  - single-leaf subgroup;
  - `reorder` r = 0 against r > 0.
  - Under 20 tasks or 8 families only b and c are reported. The models need at least 40 discordant pairs. With VIF > 5 only one-covariate models are read.
- **Subgroups**:
  - non-empty leaves 1 / 2 / ≥ 3;
  - judgement units 1 / 2 / ≥ 3 (`judgement-units-v2`);
  - code in the prompt;
  - CWE;
  - the 73 Hard tasks.
- **Contamination**: 10-gram overlap with the reference solution, threshold 0.5 (`protocol/analysis/contamination.json`), share per condition and Δ by stratum.
- **Failure taxonomy** per condition, and the T → H transitions among discordant pairs.
- **Qualitative look at discordant pairs**: all of them if there are 40 or fewer, otherwise 20 + 20 drawn with seed 7319, blind to the condition.
- **Reliability**:
  - determinism on whole tasks holding at least 5% of the greedy requests (`determinism-sample`), or 20 if cut 2 applies;
  - re-annotation of the frozen samples (`protocol/reannotation/`) from scratch, without assistance: κ on the judgement units, interval in days;
  - `gold-diff` between drafts and gold.

**Removed before the freeze** (decision of 17/11/2026, declared): GEE, GLMM, the finite-population Δ and the Agresti-Min interval.

## 7. Time rules (decided before the data)

All cuts depend on time or on faults, never on outcomes. They are applied in this order:

| # | Cut | Trigger |
|---|---|---|
| 1 | F becomes future work | CP3-F |
| 2 | Determinism on 20 tasks | CP5 |
| 3 | Reduced blind review | not finished by 28/11 at 12:00 |
| 4 | Security replicates 5 → 3 | CP3-R |
| 5 | H3 out of the family | CP2 |

Never cut: T, H-gold and E on correctness; exploratory H-auto if the parse exists; both security measures; the freeze; the paired analysis.

| Checkpoint | When | Go | Otherwise |
|---|---|---|---|
| CP1 | pilot | gold review ≤ 3.0 min per prompt | 400-task prefix. **Not measured: declared deviation; the review was complete, the 600 prefix stays** |
| CP2 | 20/11 12:00 | prefix parse complete | H3 out. **Met: parse finished 19/11 04:15 UTC** |
| CP3 | after the pilot | G1–G14 | pilot-2 on the same material |
| CP3-F | after the pilot | see below | F future work |
| CP3-R | after the pilot | see below | 3 + 1 replicates |
| CP4 | 22/11 12:00 | gold frozen, freeze verified | 400-task prefix (time only) |
| CP5 | 24/11 18:00 | generations complete | determinism on 20; correctness before security |
| CP6 | 25/11 18:00 | confirmatory analyses, thesis tables | margin of 26/11 |

**CP3-F and CP3-R** (confirmed by the student on 19/11/2026, before the pilot ran). Means are pooled over all conditions:
- `s_bcb`: seconds per BigCodeBench generation in the pilot;
- `s_lse`: seconds per LLMSecEval generation in the pilot;
- `w`: wall-clock seconds per program of `check-security` with 4 workers.

Constants: T0 = 22/11 at 12:00; 1 h per Colab session (two sessions); volumes 1,200 security generations, 1,800 core, 600 H-auto (upper bound), 600 F.
- **CP3-F**: F is in if and only if `T0 + 2 h + 1,200·s_lse + 3,000·s_bcb` ≤ 24/11 at 18:00.
- **CP3-R**: 5 + 1 replicates if and only if `T0 + 1 h + 1,200·s_lse + 1,200·w` ≤ 25/11 at 12:00. Otherwise 3 + 1, generated and analysed.

**Stopping rule**: none. The collection is complete as planned, with no extension conditional on results (the pilot reports only b + c).

## 8. Values filled after the pilot (before F4)

- **Pilot `ds-pilot-01`** (transport `5d07463e…`): summary `protocol/pilot/summary.json`, SHA-256 `801560af1107c77f01bbe25f28a48133396fe63bb9f991d99020a657b9647881`; reports, time rules and power table in `protocol/pilot/`; change log `protocol/pilot-changelog.md`.
  - G1–G14: every criterion that decides CP3 is met (G1, G2, G3, G4, G5, G7, G10; G6 and G9 go). G12 not measured (CP1); G6b, G8, G11, G13, G14 descriptive. **CP3: go.**
  - Measured, pooled over all conditions: `s_bcb` = 10.17 s (148 generations), `s_lse` = 7.3 s (144), `w` = 20.39 s per program (144 programs, 4 workers).
  - **CP3-F**: estimated end 2026-11-23 00:54 ≤ 24/11 18:00, so **F is in** both correctness journals.
  - **CP3-R**: estimated end 2026-11-22 22:13 ≤ 25/11 12:00, so **5 + 1 replicates**.
- **Planning value of q** (discordant pairs, T and H-gold, direction not computed): b + c = 6 of 30, q = 0.20 (Clopper-Pearson 95% 0.077–0.386; 30 pairs make it imprecise, so the grid 0.10–0.30 is kept). MDE at 80% power, with the power at 5 pp in brackets (`power`, exact mid-p):

| Hypothesis | n | α | q = 0.10 | q = 0.15 | q = 0.20 | q = 0.30 |
|---|---|---|---|---|---|---|
| H1 | 600 | 0.04 | 3.75 pp (0.97) | 4.75 pp (0.88) | 5.5 pp (0.76) | 6.5 pp (0.57) |
| H3 | 506 (S_parse) | 0.01 | 4.75 pp (0.86) | 6 pp (0.64) | 6.75 pp (0.48) | 8.5 pp (0.30) |

  At q = 0.20, after recycling at α = 0.05: H1 5.25 pp (0.79), H3 5.75 pp (0.71).
