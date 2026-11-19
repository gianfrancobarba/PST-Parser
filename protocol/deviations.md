# Deviations

Every departure from the plan, dated. This file is outside the hashes of the freeze on purpose: entries added after F4 keep the frozen files unchanged. Entries before F4 are known when the freeze is made and are sent with the preregistration email (M3).

## Before the freeze F4

| Date | Deviation | Reason | Effect on validity |
|---|---|---|---|
| 17/11/2026 | GEE, GLMM, finite-population Δ and the Agresti-Min interval removed from the analysis plan | Unstable with 16–35 families; new dependency; no decision depends on them | Fewer robustness checks; declared |
| 18/11/2026 | Gold review time not measured, so CP1 is not evaluated; the 600-task prefix stays | No timesheet was kept | The 400 fallback existed only to finish the review by CP4; the review is complete, so the sample does not depend on it |
| 18/11/2026 | The gold leaves of the judgement units were proposed by an LLM (GPT, OpenAI; version not recorded) and decided by the student. The final decision equals the proposal in 1,205 of 1,279 units; no change is recorded as originating with the student. | Time | Possible anchoring on the LLM's preferences, which may also share tendencies with the parser (RQ3a, H3). The only independent measure is the re-annotation 30 + 30 from scratch, without assistance (`protocol/reannotation/`) |
| 18/11/2026 | 46 units decided under the draft A56 conventions were set back to their `gold-v2.0` leaf (22 prefix, 13 LLMSecEval, 7 and 4 in the pilots) | The A56 conventions are not part of `gold-v2.0` | The frozen gold follows `protocol/guidelines/gold-v2.0.md` only |
| 18/11/2026 | The message sent to the supervisors described the remaining gold work as a "manual check"; the planned integration stating the LLM assistance was not sent | Decision of the student | The procedure is declared here, in the guidelines and in the thesis |
| 18/11/2026 | A first parse of the 30 pilot tasks (`hauto-d01-01`, ended 12:22) predates the commit of the gate thresholds (17:10) and of the C-22 code (17:11). The official run `hauto-d01-02` gave byte-identical predictions. Whether the first predictions were opened before 17:11: not verified (answer of the student, 19/11/2026) | Run started before the commit | The thresholds still predate the parse of the prefix, on which the gate is measured; the pilot F1 (0.430) is descriptive |
| 19/11/2026 | The wrapper of Cicalese et al. was not checked against ours | Decision of the student | T/PST contrasts use the same wrapper W in all conditions; the ICSME T baseline is context only |
| 19/11/2026 | Pilot only: the 4 C-origin memory families of the pilot get `no-query` cards (`protocol/security/pilot-cards.yaml`) and only the generic usability rule | They had no card and no family rule | None on the final families, whose cards and rules are unchanged; pilot G9 covers the general measure only |
| 19/11/2026 | The gold was frozen in two commits: `6d0b223` (11:36) added the freeze records but, by mistake, not the annotations with status `frozen`; `79c2b9b` (11:39) applied that status. The annotations differ from the reviewed version only in `status` (600 + 100 rows) and their SHA-256 equal the records. The gold was committed before the protocol documents of A1 (`67e6ad6`), the reverse of the planned order | Error in staging | None: both commits precede every pilot outcome (the pilot run was still generating) and every opening of the prefix predictions |
| 19/11/2026 | No public registry (OSF was planned: one deposit under embargo at F4). The preregistration is a dated email to the supervisor carrying the SHA-256 of `protocol-freeze.json` (message M3), exported to `protocol/acks/` | Decision of the student | Weaker proof of timing than a public time-stamped registry: the date rests on the email and on the git history. The freeze hash still binds every frozen file; declared as a threat to validity |
| 19/11/2026 | The seed of the qualitative sample of discordant pairs is fixed at 7319 | The plan said "recorded seed" without a value | None: fixed before any outcome |

## After the freeze F4

(none yet)
