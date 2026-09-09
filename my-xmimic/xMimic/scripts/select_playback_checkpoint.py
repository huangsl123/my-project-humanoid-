#!/usr/bin/env python3
"""Rank checkpoint candidates from one or more playback analysis summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="CHECKPOINT=SUMMARY_JSON",
        help="Checkpoint label and one evaluation summary. Repeat for multiple seeds/candidates.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def metric(summary: dict, name: str, statistic: str, fallback: float = 1.0e6) -> float:
    value = summary.get("metrics", {}).get(name, {}).get(statistic)
    return fallback if value is None else float(value)


def case_rank(summary: dict) -> dict[str, object]:
    checks = summary.get("quality_checks", {})
    terminal_terms = summary.get("terminal_terms") or []
    failures = []
    required = {
        "motion_completed": True,
        "exact_motion_frame_sequence": True,
        "all_metrics_finite": True,
        "no_motion_wrap": True,
        "started_near_reference": True,
        "within_platform_bounds": True,
    }
    for name, expected in required.items():
        if checks.get(name) is not expected:
            failures.append(name)
    if terminal_terms:
        failures.append("terminal_terms")

    position = summary.get("platform_position", {})
    xy_p95 = float(position.get("xy_tracking_drift_p95_m", 1.0e6))
    xy_max = float(position.get("xy_tracking_drift_max_m", 1.0e6))
    xy_last = float(position.get("xy_tracking_drift_last_m", 1.0e6))
    # Dimensionless score. Hard validity failures dominate all tracking errors.
    body_pos_p95 = metric(summary, "error_body_pos", "p95")
    joint_pos_p95 = metric(summary, "error_joint_pos", "p95")
    anchor_rot_p95 = metric(summary, "error_anchor_rot", "p95")
    strict_pose_failure_count = sum(
        (
            body_pos_p95 > 0.08,
            joint_pos_p95 > 0.90,
            anchor_rot_p95 > 0.15,
        )
    )
    tracking_score = (
        xy_p95 / 0.25
        + xy_max / 0.50
        + xy_last / 0.35
        + body_pos_p95 / 0.08
        + joint_pos_p95 / 0.90
        + anchor_rot_p95 / 0.15
    )
    return {
        "failures": failures,
        "hard_failure_count": len(failures),
        "strict_pose_failure_count": strict_pose_failure_count,
        "tracking_score": tracking_score,
        "xy_p95_m": xy_p95,
        "xy_max_m": xy_max,
        "xy_last_m": xy_last,
        "body_pos_p95_m": body_pos_p95,
        "joint_pos_p95": joint_pos_p95,
        "anchor_rot_p95_rad": anchor_rot_p95,
    }


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[Path]] = defaultdict(list)
    for spec in args.candidate:
        if "=" not in spec:
            raise ValueError(f"Expected CHECKPOINT=SUMMARY_JSON, received: {spec}")
        checkpoint, raw_path = spec.split("=", 1)
        grouped[checkpoint.strip()].append(Path(raw_path).expanduser().resolve())

    candidates = []
    for checkpoint, paths in grouped.items():
        cases = []
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                summary = json.load(stream)
            cases.append({"summary": str(path), **case_rank(summary)})
        candidates.append(
            {
                "checkpoint": checkpoint,
                "case_count": len(cases),
                "hard_failure_count": sum(int(case["hard_failure_count"]) for case in cases),
                "worst_case_failure_count": max(int(case["hard_failure_count"]) for case in cases),
                "strict_pose_failure_count": sum(int(case["strict_pose_failure_count"]) for case in cases),
                "worst_case_strict_pose_failure_count": max(
                    int(case["strict_pose_failure_count"]) for case in cases
                ),
                "mean_tracking_score": sum(float(case["tracking_score"]) for case in cases) / len(cases),
                "worst_tracking_score": max(float(case["tracking_score"]) for case in cases),
                "cases": cases,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["hard_failure_count"],
            item["worst_case_failure_count"],
            item["strict_pose_failure_count"],
            item["worst_case_strict_pose_failure_count"],
            item["mean_tracking_score"],
            item["worst_tracking_score"],
            item["checkpoint"],
        )
    )
    result = {
        "selection_policy": (
            "Minimize hard validity failures first; then minimize mean and worst normalized "
            "full-trajectory tracking error across supplied evaluation cases."
        ),
        "selected_checkpoint": candidates[0]["checkpoint"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(result["selected_checkpoint"])


if __name__ == "__main__":
    main()
