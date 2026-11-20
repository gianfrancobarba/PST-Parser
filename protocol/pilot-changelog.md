# Pilot change log

Changes made because of the pilot run `ds-pilot-01` (archive tree `704c970a2b07602793ba291c4b649d9bb825b209`, inputs `2deca8d5…`). The pilot measured technical criteria only, pooled over the conditions; no change below was chosen by looking at outcomes by condition.

| # | Found in | Change | Effect on the final runs |
|---|---|---|---|
| P-1 | `merge-journals` of the two BigCodeBench journals | The union refused the pilot journals: in a pilot `review_record` is the render record of each journal's own variant file, so `review_record_sha256` differs between journals by construction. `review_record_sha256` is now free when every journal is a pilot (`MERGE_FREE_KEYS_PILOT`); a final run, where every journal points at the same `protocol-freeze.json`, still needs it equal, and a pilot journal is never merged with a final one | None on the generations or on the measures; the check on final runs is unchanged |

## Other changes before F4, found by a dress rehearsal on synthetic outcomes

The analysis chain was run end to end on random outcomes drawn on the real task identifiers (no generation, no real outcome). None of these changes touches the generations, the samples, the gold or a threshold.

| # | Found in | Change |
|---|---|---|
| R-2 | `design` on S_parse | `design` reads the S_parse document (`load_s_parse`, hash checked): H3 is analysed on S_parse |
| R-3 | `thesis-tables` | A report is its `analysis_id` and its two conditions: E − T and H-gold − E share the core ledger of H1. H1 and H3 resolve to their prescribed pairs (T/H-gold, T/H-auto) |
| R-4 | `design` of the targeted measure | `design --cards [--sensitivity]` keeps the families judged by `check-security --measure targeted`: 47 prompts / 16 families, 56 / 19 with the sensitivity |
| R-5 | RQ3c and the decomposition of Δ_auto | `hauto-ledgers` also writes `outcomes-s-parse-all.jsonl`, every condition of the S_parse tasks; `analyze` keeps refusing rows outside a design |
| R-6 | final run in Colab | The final configurations pointed at a freeze next to `generate.json` inside `inputs.zip`, verified against the wrong root: every final run would have stopped before generating. They now point at `../../protocol-freeze.json`, and the notebook runs the repository copy of the configuration with the same SHA-256 |
