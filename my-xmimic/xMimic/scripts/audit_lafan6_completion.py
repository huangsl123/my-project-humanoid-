#!/usr/bin/env python3
"""Audit every evidence gate for the six-action LAFAN1 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


ACTIONS = (
    "01_fall_getup",
    "02_fight_sports",
    "03_jumps",
    "04_ground",
    "05_push_stumble",
    "06_fight",
)

SUMMARY_GATES = (
    "all_actions_present",
    "all_preprocessing_validations_passed",
    "all_smoke_validations_passed",
    "deployment_matrix_preflight_passed",
    "smoke_onnx_matrix_preflight_passed",
    "all_four_evaluations_complete",
    "all_four_evaluations_full_integrity",
    "all_evaluations_finite",
    "all_evaluations_platform_safe",
    "all_checkpoint_validations_passed",
    "all_onnx_validations_passed",
    "all_video_validations_passed",
    "all_offline_deployments_passed",
    "all_sim2sim_passed",
    "all_selected_files_exist",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_analysis_passed(path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, ["missing"]
    checks = load_json(path).get("quality_checks", {})
    required = (
        "all_scalars_finite",
        "iteration_sequence_contiguous",
        "expected_global_step_end_reached",
        "iteration_records_complete",
        "iteration_count_complete",
        "checkpoint_matches_last_iteration",
        "checkpoint_reaches_expected_final_step",
    )
    failures = [name for name in required if checks.get(name) is not True]
    return not failures, failures


def checkpoint_selection_passed(path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, ["missing"]
    document = load_json(path)
    selected = document.get("selected_checkpoint")
    candidates = document.get("candidates") or []
    candidate_names = {item.get("checkpoint") for item in candidates}
    failures = []
    if not selected:
        failures.append("selected_checkpoint_missing")
    if selected not in candidate_names:
        failures.append("selected_checkpoint_not_in_candidates")
    if len(candidates) < 2:
        failures.append("fewer_than_two_candidates")
    for item in candidates:
        if int(item.get("case_count", 0)) != 4:
            failures.append(f"{item.get('checkpoint')}:not_four_cases")
    return not failures, failures


def manifest_passed(root: Path, path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, ["missing"]
    document = load_json(path)
    entries = document.get("files") or []
    failures = []
    names = {entry.get("path") for entry in entries}
    for required in ("REPORT.md", "SUMMARY.json", "preprocessing_validation.json"):
        if required not in names:
            failures.append(f"missing_entry:{required}")
    for entry in entries:
        relative = entry.get("path")
        file_path = root / str(relative)
        if not file_path.is_file():
            failures.append(f"missing_file:{relative}")
            continue
        if file_path.stat().st_size != int(entry.get("size_bytes", -1)):
            failures.append(f"size_mismatch:{relative}")
            continue
        if sha256(file_path) != entry.get("sha256"):
            failures.append(f"hash_mismatch:{relative}")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--success-marker", type=Path, required=True)
    parser.add_argument("--review-marker", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    summary_path = root / "SUMMARY.json"
    manifest_path = root / "MANIFEST.json"
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    if summary_path.is_file():
        summary = load_json(summary_path)
        for name in SUMMARY_GATES:
            checks[f"summary:{name}"] = summary.get(name) is True
        strict_count = int(summary.get("strict_pass_count", -1))
        evaluation_count = int(summary.get("strict_evaluation_count", -1))
        checks["summary:all_24_strict_evaluations_passed"] = (
            strict_count == evaluation_count == 24
        )
        details["strict_pass_count"] = strict_count
        details["strict_evaluation_count"] = evaluation_count
    else:
        checks["summary:exists"] = False

    checks["report:exists_nonempty"] = (root / "REPORT.md").is_file() and (
        root / "REPORT.md"
    ).stat().st_size > 0
    preprocessing = root / "preprocessing_validation.json"
    checks["preprocessing:evidence_passed"] = bool(
        preprocessing.is_file() and load_json(preprocessing).get("all_actions_passed")
    )

    action_details = {}
    for action in ACTIONS:
        action_dir = root / action
        action_checks: dict[str, bool] = {
            "smoke_marker": (action_dir / "SMOKE_COMPLETE").is_file(),
            "action_complete_marker": (action_dir / "TRAINING_EVALUATION_COMPLETE").is_file(),
            "selected_checkpoint_pointer": (action_dir / "selected_checkpoint.txt").is_file(),
            "selected_onnx_pointer": (action_dir / "selected_onnx.txt").is_file(),
        }
        analyses = (
            [action_dir / "training_analysis/summary.json"]
            if action == "01_fall_getup"
            else [
                action_dir / "stage1_training_analysis/summary.json",
                action_dir / "stage2_training_analysis/summary.json",
            ]
        )
        analysis_results = [training_analysis_passed(path) for path in analyses]
        action_checks["training_analyses_passed"] = all(item[0] for item in analysis_results)

        selections = (
            [action_dir / "single_checkpoint_selection.json"]
            if action == "01_fall_getup"
            else [
                action_dir / "stage1_checkpoint_selection.json",
                action_dir / "stage2_checkpoint_selection.json",
            ]
        )
        selection_results = [checkpoint_selection_passed(path) for path in selections]
        action_checks["checkpoint_selections_passed"] = all(item[0] for item in selection_results)

        fail_markers = sorted(
            path.relative_to(action_dir).as_posix()
            for path in action_dir.rglob("*FAILED*")
            if path.is_file()
        )
        action_checks["no_failure_markers"] = not fail_markers
        for pointer_name in ("selected_checkpoint.txt", "selected_onnx.txt"):
            pointer = action_dir / pointer_name
            if pointer.is_file():
                target = Path(pointer.read_text(encoding="utf-8").strip())
                action_checks[f"{pointer_name}:target_exists"] = target.is_file()

        checks[f"action:{action}"] = all(action_checks.values())
        action_details[action] = {
            "checks": action_checks,
            "analysis_failures": [item[1] for item in analysis_results],
            "selection_failures": [item[1] for item in selection_results],
            "failure_markers": fail_markers,
        }

    manifest_ok, manifest_failures = manifest_passed(root, manifest_path)
    checks["manifest:listed_files_exist_and_hash_match"] = manifest_ok
    details["manifest_failures"] = manifest_failures
    details["actions"] = action_details
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "audit_policy": "All named evidence gates and all 24 strict full-trajectory evaluations must pass.",
        "checks": checks,
        "failed_checks": failed_checks,
        "details": details,
        "passed": not failed_checks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# LAFAN1 六动作最终完成审计",
        "",
        f"生成时间：{result['generated_at']}",
        "",
        f"结论：{'PASS' if result['passed'] else 'REVIEW REQUIRED'}",
        "",
        f"严格完整轨迹通过：{details.get('strict_pass_count', 'N/A')}/"
        f"{details.get('strict_evaluation_count', 'N/A')}",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- [{'x' if value else ' '}] `{name}`" for name, value in checks.items())
    if failed_checks:
        lines += ["", "## 未通过项", ""]
        lines.extend(f"- `{name}`" for name in failed_checks)
    lines.append("")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")

    success_marker = args.success_marker.expanduser().resolve()
    review_marker = args.review_marker.expanduser().resolve()
    if result["passed"]:
        review_marker.unlink(missing_ok=True)
        success_marker.write_text(result["generated_at"] + "\n", encoding="utf-8")
    else:
        success_marker.unlink(missing_ok=True)
        review_marker.write_text("\n".join(failed_checks) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failed_checks": failed_checks}, indent=2))


if __name__ == "__main__":
    main()
