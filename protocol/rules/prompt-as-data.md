# Answers that treat the prompt as data (`json-response-v2`, `prompt-as-data-v2`)

Frozen before the pilot outcomes (decision T-8 of 15/11/2026). Implemented in `src/pstparser/empirical/bigcodebench.py` (`is_json_response`, `is_prompt_as_data`, `categorize`); the source of the freeze fixes the code, this file states the rule and its use.

## Why

A JSON prompt (H-gold, H-auto, E, F) can lead the model to answer in JSON, or to write a program that parses the prompt instead of solving the task. Such answers fail by construction. They are counted, never repaired, and they never change the wrapper, the system prompt or the conditions.

## Rules

**`json-response-v2`** (on the raw answer). True when:
- the whole answer is a JSON object or array; or
- a JSON object or array that decodes, inside a fenced block or in the prose, holds a key of the hierarchical prompt or of the envelope: `prompt`, `main_instruction`, `context`, `phrase`, `position`.

Keys merely quoted in prose do not count, so the rule does not depend on the prompt the model was shown.

**`prompt-as-data-v2`** (on the extracted program). True when the code:
- uses `main_instruction` as a dictionary key, a subscript or a `get` key;
- builds a dictionary with both `phrase` and `position` (a PST span);
- reads `x["prompt"][leaf]` with a top-level leaf of the hierarchical prompt (`main_instruction`, `context`, `examples`, `reasoning`);
- reads a prompt key (`prompt`, `context`, `phrase`, `position`) from the result of a `loads` call, directly or through a name bound to it;
- passes text holding `main_instruction` or `"prompt":` to `loads`.

Ordinary uses of `prompt` or `position` (an API body, a dataframe column) do not count. Code that does not parse is searched as text for `main_instruction` in key position.

## Use in the outcomes (BigCodeBench)

Failure categories are exclusive and tried in a fixed order:
1. `truncated` (output cut at the token budget);
2. `no_code` (empty extraction);
3. `pass` if the harness passes;
4. `json_response`;
5. `prompt_as_data`;
6. `no_code` (entry point not defined);
7. what the harness observed (`timeout`, `syntax_or_compile_error`, `import_error`, `module_level_exception`, `assertion_failure`, `test_exception`, `fail_no_detail`).

The category is the `reason` of the outcome; the outcome itself is `pass` only when the harness passes, so a program in categories 4 and 5 is a **fail** in every analysis (intention to treat). The same rule applies to every condition, T included.

## Reporting

- **Pilot.** Only the **presence** of each category per condition, and the counts summed over all conditions (`pilot-report-v2`). **No threshold and no action**: with 30 tasks any threshold would fire by chance too often.
- **Collection.** The share per condition is reported as a secondary, exploratory endpoint, next to the failure taxonomy.
- **Not done.** The H-explained arm (H plus a sentence explaining the schema in the system prompt) is a future extension. It is not part of this study and is never added in reaction to these counts.
