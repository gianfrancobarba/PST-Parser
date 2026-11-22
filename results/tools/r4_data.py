"""Training data of the parser r4 (exploratory analysis R, protocol/deviations.md, 22/11/2026).

Three steps, each writing into a fresh directory:

- ``apply``: the student's decisions on the judgement units (one CSV per batch, as
  written by ``outputs/ripresa/r/r_pool.py merge``) are applied to the rule drafts. Only
  the leaf (``path``) of a judgement span changes; ``status`` and ``reviewer`` become
  ``reviewed`` and ``studente`` only with ``--reviewed``, which the student passes.
  Writes ``annotations.jsonl``, ``review-log.csv`` (proposal and decision of every
  unit) and ``summary.json``.
- ``records``: the annotations become a corpus source in the YAML layout read by
  ``prepare-data`` (one record per task, each leaf listing its phrases in source
  order), so that the prepared target equals the gold target of the same spans.
- ``split``: the r3 preparation (``configs/experiments/baseline.yaml``) is run again,
  then the r4 one (the same sources plus this YAML, appended last); the r3 records
  must come out unchanged at the head of the r4 records. The r4 split is the r3 split
  with the new records added to train and validation (test unchanged), bound to the
  records by its manifest.

usage (repository root):
  python -I results/tools/r4_data.py apply --drafts D --decisions DIR --units CSV --out DIR
      [--reviewed]
  python -I results/tools/r4_data.py records --annotations A --sources S --split SPLIT --out FILE
  python -I results/tools/r4_data.py split --bcb FILE --split SPLIT --work DIR --out DIR
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

LEAVES = (
    "main_instruction",
    "context.data",
    "context.role",
    "context.format",
    "context.constraints",
    "examples",
    "reasoning.influence",
    "reasoning.reasoning_examples",
    "reasoning.paths",
)
LOG_HEADER = ("task_id", "start", "end", "proposta_llm", "decisione")
Unit = tuple[str, int, int]


def read_decisions(path: Path) -> list[dict[str, str]]:
    """Rows of a decision sheet, saved as UTF-8 or by Excel (cp1252, ``;``)."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    first = text.splitlines()[0] if text else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    return [{k.strip(): (v or "").strip() for k, v in row.items() if k is not None} for row in rows]


def apply_decisions(
    drafts: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, str]],
    units: set[Unit],
    *,
    reviewed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply one decision per judgement unit to the drafts; refuse anything else."""
    spans = {(d["task_id"], s["start"], s["end"]) for d in drafts for s in d["spans"]}
    foreign = sorted(units - spans)
    if foreign:
        raise ValueError(f"judgement unit not a draft span: {foreign[:3]}")
    decided: dict[Unit, Mapping[str, str]] = {}
    for row in decisions:
        key = (row["task_id"], int(row["start"]), int(row["end"]))
        if key not in units:
            raise ValueError(f"decision on a span that is not a judgement unit: {key}")
        if key in decided:
            raise ValueError(f"unit decided twice: {key}")
        leaf = row["decisione"].strip()
        if not leaf:
            raise ValueError(f"no decision for {key}")
        if leaf not in LEAVES:
            raise ValueError(f"unknown leaf {leaf!r} for {key}")
        decided[key] = row
    missing = sorted(units - set(decided))
    if missing:
        raise ValueError(f"{len(missing)} judgement units missing: {missing[:3]}")
    annotations: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for draft in drafts:
        out: dict[str, Any] = copy.deepcopy(dict(draft))
        for span in out["spans"]:
            unit: Unit = (out["task_id"], span["start"], span["end"])
            if unit in decided:
                span["path"] = decided[unit]["decisione"].strip()
                log.append(
                    {
                        "task_id": unit[0],
                        "start": unit[1],
                        "end": unit[2],
                        "proposta_llm": decided[unit]["proposta_ia"].strip(),
                        "decisione": span["path"],
                    }
                )
        if reviewed:
            out["status"], out["reviewer"] = "reviewed", "studente"
        annotations.append(out)
    return annotations, log


def training_records(
    annotations: Sequence[Mapping[str, Any]], sources: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One corpus record per task: the prompt and, per leaf, its phrases in source order."""
    records = []
    for annotation in annotations:
        text = sources[annotation["task_id"]]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != annotation["source_sha256"]:
            raise ValueError(f"source text differs from the annotation: {annotation['task_id']}")
        leaves: dict[str, list[str]] = {}
        for span in sorted(annotation["spans"], key=lambda s: (s["start"], s["end"])):
            if text[span["start"] : span["end"]] != span["phrase"]:
                raise ValueError(f"phrase is not its source span: {annotation['task_id']}")
            leaves.setdefault(span["path"], []).append(span["phrase"])
        records.append({"prompt": text, "leaves": leaves})
    return records


class _Dumper(yaml.SafeDumper):
    pass


def write_records_yaml(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Write the records in the layout of ``read_annotations`` (LF, new file only)."""
    body = yaml.dump(
        {"records": list(records)}, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100
    )
    header = (
        "# r4 training labels: BigCodeBench tasks 601-800 (exploratory analysis R).\n"
        "# Rule drafts (bcb-markers-v1); judgement units proposed by an LLM and decided by the\n"
        "# student. Generated by results/tools/r4_data.py records.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(header + body)
    return path


def r4_split(r3: Any, *, base: int, tasks: Sequence[str], validation: set[str]) -> Any:
    """The r3 split with record ``base + i`` (task ``tasks[i]``) added to train or validation."""
    from pstparser.data.splits import Split

    if max([*r3.train, *r3.val, *r3.test]) >= base:
        raise ValueError(f"an r3 index reaches the base of the new records ({base})")
    new_train = [base + i for i, t in enumerate(tasks) if t not in validation]
    new_val = [base + i for i, t in enumerate(tasks) if t in validation]
    return Split(
        train=sorted(r3.train + new_train), val=sorted(r3.val + new_val), test=list(r3.test)
    )


# ---- command line -----------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _fresh(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"choose a fresh directory: {directory}")


def _units(csv_path: Path, tasks: Iterable[str]) -> set[Unit]:
    wanted = set(tasks)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return {
            (r["task_id"], int(r["start"]), int(r["end"]))
            for r in csv.DictReader(handle)
            if r["kind"] == "judgement" and r["task_id"] in wanted
        }


def command_apply(args: argparse.Namespace) -> None:
    """Step ``apply``: decisions to annotations, review log and summary."""
    _fresh(args.out)
    drafts = _jsonl(args.drafts)
    decisions = [
        row
        for sheet in sorted(args.decisions.glob("*-decisioni.csv"))
        for row in read_decisions(sheet)
    ]
    units = _units(args.units, (d["task_id"] for d in drafts))
    annotations, log = apply_decisions(drafts, decisions, units, reviewed=args.reviewed)
    _write_new(
        args.out / "annotations.jsonl",
        "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in annotations),
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LOG_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(log)
    _write_new(args.out / "review-log.csv", buffer.getvalue())
    changed = [r for r in log if r["proposta_llm"] != r["decisione"]]
    summary = {
        "format": "r4-labels-summary-v1",
        "tasks": len(annotations),
        "judgement_units": len(log),
        "decision_equal_to_llm_proposal": len(log) - len(changed),
        "decision_changed_by_student": len(changed),
        "changes": dict(Counter(f"{r['proposta_llm']} -> {r['decisione']}" for r in changed)),
        "status": sorted({a["status"] for a in annotations}),
        "inputs_sha256": {
            "drafts": _sha(args.drafts),
            "units_csv": _sha(args.units),
            **{p.name: _sha(p) for p in sorted(args.decisions.glob("*-decisioni.csv"))},
        },
        "outputs_sha256": {
            "annotations": _sha(args.out / "annotations.jsonl"),
            "review_log": _sha(args.out / "review-log.csv"),
        },
    }
    _write_new(args.out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def command_records(args: argparse.Namespace) -> None:
    """Step ``records``: annotations of the train and validation tasks to the YAML source."""
    annotations = {a["task_id"]: a for a in _jsonl(args.annotations)}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    order = [t for t in annotations if t in set(split["train"]) | set(split["validation"])]
    if set(order) != set(split["train"]) | set(split["validation"]):
        raise ValueError("annotations do not cover the train and validation tasks")
    sources = {r["task_id"]: r["text"] for r in _jsonl(args.sources)}
    records = training_records([annotations[t] for t in order], sources)
    write_records_yaml(records, args.out)
    _write_new(
        args.out.with_suffix(".tasks.json"),
        json.dumps({"format": "r4-bcb-tasks-v1", "tasks": order}, indent=2) + "\n",
    )
    print(f"{len(records)} records -> {args.out} {_sha(args.out)}")


def _prepare(overrides: list[str]) -> Any:
    from pstparser.config.loader import load_experiment
    from pstparser.data.prepare import prepare_corpus

    experiment = load_experiment("configs/experiments/baseline.yaml", overrides)
    return prepare_corpus(experiment.data, experiment.evaluation.alignment_levels)


R3_SOURCES = (
    "[{path: data/raw/prompt_dataset_v2.xlsx, sheet: prompt_dataset_v2, "
    "fixes: data/raw/annotation_fixes.yaml}, {path: data/annotated/reasoning.yaml, format: yaml}"
)


def r4_sources_override(bcb: Path) -> str:
    """The ``data.sources`` override of r4: the r3 sources, then the BigCodeBench labels."""
    return f"data.sources={R3_SOURCES}, {{path: {bcb.as_posix()}, format: yaml}}]"


def command_split(args: argparse.Namespace) -> None:
    """Step ``split``: r3 and r4 preparations, the r4 records and the r4 split."""
    from pstparser.data.splits import save_split

    _fresh(args.work)
    _fresh(args.out)
    r3 = _prepare(
        [
            f"data.processed_dir={(args.work / 'r3/prepared').as_posix()}",
            f"data.split.output_dir={(args.work / 'r3/splits').as_posix()}",
        ]
    )
    r4 = _prepare(
        [
            r4_sources_override(args.bcb),
            f"data.processed_dir={(args.work / 'r4/prepared').as_posix()}",
            f"data.split.output_dir={(args.work / 'r4/unused-splits').as_posix()}",
        ]
    )
    r3_lines = r3.records_path.read_bytes().split(b"\n")
    r4_lines = r4.records_path.read_bytes().split(b"\n")
    base = len(r3.records)
    if r4_lines[:base] != r3_lines[:base]:
        raise ValueError("the r3 records changed at the head of the r4 records")
    tasks = json.loads(args.bcb.with_suffix(".tasks.json").read_text(encoding="utf-8"))["tasks"]
    if len(r4.records) != base + len(tasks):
        raise ValueError("r4 records are not the r3 records plus one per task")
    validation = set(json.loads(args.split.read_text(encoding="utf-8"))["validation"])
    split = r4_split(r3.split, base=base, tasks=tasks, validation=validation)
    prepared = args.out / "prepared"
    prepared.mkdir(parents=True)
    shutil.copyfile(r4.records_path, prepared / "records.jsonl")
    shutil.copyfile(r4.report_path, prepared / "quality_report.json")
    save_split(split, args.out / "splits", records_path=prepared / "records.jsonl")
    manifest = {
        "format": "r4-data-v1",
        "rule": (
            "r3 split (baseline.yaml, seed 42) kept; "
            "BigCodeBench records appended to train and validation"
        ),
        "r3": {
            "records": base,
            "train": len(r3.split.train),
            "val": len(r3.split.val),
            "test": len(r3.split.test),
            "records_sha256": _sha(r3.records_path),
            "split_sha256": {
                n: _sha(r3.split_dir / n) for n in ("train.json", "val.json", "test.json")
            },
        },
        "r4": {
            "records": len(r4.records),
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
            "records_sha256": _sha(prepared / "records.jsonl"),
            "split_sha256": {
                n: _sha(args.out / "splits" / n)
                for n in ("train.json", "val.json", "test.json", "split_manifest.json")
            },
        },
        "bcb": {
            "yaml": args.bcb.as_posix(),
            "yaml_sha256": _sha(args.bcb),
            "first_index": base,
            "train": len(tasks) - len(validation & set(tasks)),
            "val": len(validation & set(tasks)),
        },
        "quality_r4": r4.quality.as_dict(),
        "sources_override": r4_sources_override(args.bcb),
    }
    _write_new(args.out / "r4-data-manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """Run one step."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--drafts", type=Path, required=True)
    a.add_argument("--decisions", type=Path, required=True)
    a.add_argument("--units", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--reviewed", action="store_true")
    r = sub.add_parser("records")
    r.add_argument("--annotations", type=Path, required=True)
    r.add_argument("--sources", type=Path, required=True)
    r.add_argument("--split", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("split")
    s.add_argument("--bcb", type=Path, required=True)
    s.add_argument("--split", type=Path, required=True)
    s.add_argument("--work", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    {"apply": command_apply, "records": command_records, "split": command_split}[args.step](args)


if __name__ == "__main__":
    sys.path.insert(0, "src")
    main()
