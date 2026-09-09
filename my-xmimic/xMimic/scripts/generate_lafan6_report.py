#!/usr/bin/env python3
"""Generate the final evidence-backed report for the six-action LAFAN pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ACTIONS = (
    ("01_fall_getup", "FallAndGetUp", "fall_getup1_subject2_named.npz"),
    ("02_fight_sports", "FightAndSports", "fight_sports1_named.npz"),
    ("03_jumps", "Jumps", "jumps1_subject5_named.npz"),
    ("04_ground", "Ground", "ground2_named.npz"),
    ("05_push_stumble", "PushAndStumble", "push_stumble1_named.npz"),
    ("06_fight", "Fight", "fight1_named.npz"),
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def mean(records: list[dict], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def validation_passed(document: dict) -> bool:
    if "passed" in document:
        return bool(document["passed"])
    checks = document.get("checks", {})
    return bool(checks) and all(bool(value) for value in checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_root.expanduser().resolve()
    motion_root = args.motion_root.expanduser().resolve()
    preprocessing_path = root / "preprocessing_validation.json"
    if not preprocessing_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing evidence: {preprocessing_path}")
    preprocessing = load_json(preprocessing_path)
    deployment_matrix_path = root / "deployment_matrix_smoke_validation.json"
    if not deployment_matrix_path.is_file():
        raise FileNotFoundError(f"Missing deployment matrix evidence: {deployment_matrix_path}")
    deployment_matrix = load_json(deployment_matrix_path)
    smoke_onnx_matrix_path = root / "smoke_onnx_matrix_validation.json"
    if not smoke_onnx_matrix_path.is_file():
        raise FileNotFoundError(f"Missing smoke ONNX matrix evidence: {smoke_onnx_matrix_path}")
    smoke_onnx_matrix = load_json(smoke_onnx_matrix_path)
    preprocessing_actions = {
        item.get("slug"): item for item in preprocessing.get("actions", [])
    }
    actions = []
    missing = []
    for slug, category, motion_name in ACTIONS:
        action_dir = root / slug
        paths = {
            "smoke_training": action_dir / "smoke_training_analysis/summary.json",
            "smoke_checkpoint": action_dir / "smoke_checkpoint_validation.json",
            "evaluation": action_dir / "final_evaluation_comparison/evaluation_comparison.json",
            "checkpoint": action_dir / "selected_checkpoint.txt",
            "onnx": action_dir / "selected_onnx.txt",
            "checkpoint_validation": action_dir / "final_checkpoint_validation/summary.json",
            "onnx_validation": action_dir / "onnx_validation/summary.json",
            "video_validation": action_dir / "video_validation/summary.json",
            "deployment": action_dir / "deployment/install.json",
            "offline": action_dir / "deployment/offline_validation.json",
            "sim2sim": action_dir / "sim2sim_validation/summary.json",
        }
        for label, path in paths.items():
            if not path.is_file():
                missing.append(f"{slug}:{label}:{path}")
        if any(not path.is_file() for path in paths.values()):
            continue

        evaluation = load_json(paths["evaluation"])
        cases = evaluation["evaluations"]
        smoke_training = load_json(paths["smoke_training"])
        smoke_checkpoint = load_json(paths["smoke_checkpoint"])
        smoke_checks = smoke_training.get("quality_checks", {})
        deployment = load_json(paths["deployment"])
        checkpoint_validation = load_json(paths["checkpoint_validation"])
        onnx_validation = load_json(paths["onnx_validation"])
        video_validation = load_json(paths["video_validation"])
        offline = load_json(paths["offline"])
        sim2sim = load_json(paths["sim2sim"])
        motion_path = motion_root / motion_name
        with np.load(motion_path, allow_pickle=True) as motion:
            frames = int(len(motion["joint_pos"]))
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        selected_checkpoint = paths["checkpoint"].read_text(encoding="utf-8").strip()
        selected_onnx = paths["onnx"].read_text(encoding="utf-8").strip()
        expected_labels = {"nominal_seed42", "robust_seed0", "robust_seed1", "robust_seed2"}
        full_integrity_count = sum(
            bool(case.get("motion_completed"))
            and bool(case.get("exact_motion_frame_sequence"))
            and bool(case.get("all_metrics_finite"))
            and bool(case.get("no_motion_wrap"))
            and bool(case.get("started_near_reference"))
            and bool(case.get("within_platform_bounds"))
            and not case.get("terminal_terms")
            and int(case.get("logged_steps", -1)) == frames
            and int(case.get("motion_total_frames", -1)) == frames
            for case in cases
        )

        actions.append(
            {
                "slug": slug,
                "category": category,
                "motion": str(motion_path),
                "frames": frames,
                "fps": fps,
                "duration_s": (frames - 1) / fps,
                "selected_checkpoint": selected_checkpoint,
                "selected_onnx": selected_onnx,
                "selected_checkpoint_exists": Path(selected_checkpoint).is_file(),
                "selected_onnx_exists": Path(selected_onnx).is_file(),
                "smoke_training_passed": bool(
                    smoke_checks.get("all_scalars_finite")
                    and smoke_checks.get("iteration_count_complete")
                    and smoke_checks.get("checkpoint_reaches_expected_final_step")
                ),
                "smoke_checkpoint_passed": validation_passed(smoke_checkpoint),
                "evaluation_count": int(evaluation["evaluation_count"]),
                "evaluation_labels_exact": {case["label"] for case in cases} == expected_labels,
                "full_integrity_count": full_integrity_count,
                "strict_pass_count": int(evaluation["pass_count"]),
                "complete_count": sum(bool(case["motion_completed"]) for case in cases),
                "finite_count": sum(bool(case["all_metrics_finite"]) for case in cases),
                "platform_safe_count": sum(bool(case["within_platform_bounds"]) for case in cases),
                "xy_p95_mean_m": mean(cases, "xy_tracking_drift_p95_m"),
                "body_p95_mean_m": mean(cases, "body_pos_p95"),
                "joint_p95_mean": mean(cases, "joint_pos_p95"),
                "anchor_rot_p95_mean_rad": mean(cases, "anchor_rot_p95"),
                "failed_checks": sorted(
                    {item for case in cases for item in case.get("failed_checks", [])}
                ),
                "checkpoint_validation_passed": validation_passed(checkpoint_validation),
                "onnx_validation_passed": validation_passed(onnx_validation),
                "video_validation_passed": validation_passed(video_validation),
                "offline_deployment_passed": validation_passed(offline),
                "sim2sim_passed": validation_passed(sim2sim),
                "sim2sim_checks": sim2sim.get("checks", {}),
                "deployment": deployment,
            }
        )

    if missing:
        raise FileNotFoundError("Missing final evidence:\n" + "\n".join(missing))

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "action_count": len(actions),
        "all_actions_present": len(actions) == len(ACTIONS),
        "preprocessing_action_count": int(preprocessing.get("action_count", 0)),
        "all_preprocessing_validations_passed": bool(
            preprocessing.get("all_actions_passed")
            and preprocessing.get("body_name_orders_equal")
            and len(preprocessing_actions) == len(ACTIONS)
            and all(
                preprocessing_actions.get(slug, {}).get("passed")
                for slug, _, _ in ACTIONS
            )
        ),
        "all_smoke_validations_passed": all(
            action["smoke_training_passed"] and action["smoke_checkpoint_passed"]
            for action in actions
        ),
        "deployment_matrix_preflight_passed": bool(
            deployment_matrix.get("passed")
            and deployment_matrix.get("checks")
            and all(bool(value) for value in deployment_matrix["checks"].values())
        ),
        "smoke_onnx_matrix_preflight_passed": bool(
            smoke_onnx_matrix.get("passed")
            and smoke_onnx_matrix.get("checks")
            and all(bool(value) for value in smoke_onnx_matrix["checks"].values())
        ),
        "all_four_evaluations_complete": all(a["complete_count"] == 4 for a in actions),
        "all_four_evaluations_full_integrity": all(
            a["evaluation_count"] == 4
            and a["evaluation_labels_exact"]
            and a["full_integrity_count"] == 4
            for a in actions
        ),
        "all_evaluations_finite": all(a["finite_count"] == 4 for a in actions),
        "all_evaluations_platform_safe": all(a["platform_safe_count"] == 4 for a in actions),
        "all_checkpoint_validations_passed": all(a["checkpoint_validation_passed"] for a in actions),
        "all_onnx_validations_passed": all(a["onnx_validation_passed"] for a in actions),
        "all_video_validations_passed": all(a["video_validation_passed"] for a in actions),
        "all_offline_deployments_passed": all(a["offline_deployment_passed"] for a in actions),
        "all_sim2sim_passed": all(a["sim2sim_passed"] for a in actions),
        "all_selected_files_exist": all(
            a["selected_checkpoint_exists"] and a["selected_onnx_exists"] for a in actions
        ),
        "strict_pass_count": sum(a["strict_pass_count"] for a in actions),
        "strict_evaluation_count": sum(a["evaluation_count"] for a in actions),
        "actions": actions,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# LAFAN1 六类动作完整训练与部署汇总报告",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 总结",
        "",
        f"共处理 {len(actions)} 类动作、{summary['strict_evaluation_count']} 个完整回放工况。",
        f"严格精度通过 {summary['strict_pass_count']}/{summary['strict_evaluation_count']}；"
        "完整性、安全性和精度分别统计，不用完成动作代替精度达标。",
        "",
        f"- BVH→xGMR PKL→训练NPZ六动作预处理验证全部通过："
        f"{'是' if summary['all_preprocessing_validations_passed'] else '否'}",
        f"- 六动作64环境冒烟训练与checkpoint结构验证全部通过："
        f"{'是' if summary['all_smoke_validations_passed'] else '否'}",
        f"- xMIGCS六策略并存安装预检通过："
        f"{'是' if summary['deployment_matrix_preflight_passed'] else '否'}",
        f"- 六动作ONNX参考输出与各自NPZ首/中/末帧精确一致："
        f"{'是' if summary['smoke_onnx_matrix_preflight_passed'] else '否'}",
        f"- 四工况全部完成：{'是' if summary['all_four_evaluations_complete'] else '否'}",
        f"- 四工况帧序列、无回绕、无终止和参考起点全部严格一致：{'是' if summary['all_four_evaluations_full_integrity'] else '否'}",
        f"- 所有回放数值有限：{'是' if summary['all_evaluations_finite'] else '否'}",
        f"- 所有回放位于平台内：{'是' if summary['all_evaluations_platform_safe'] else '否'}",
        f"- Checkpoint 验证全部通过：{'是' if summary['all_checkpoint_validations_passed'] else '否'}",
        f"- ONNX 验证全部通过：{'是' if summary['all_onnx_validations_passed'] else '否'}",
        f"- 正常速度完整轨迹视频解码验证全部通过："
        f"{'是' if summary['all_video_validations_passed'] else '否'}",
        f"- xMIGCS 离线部署验证全部通过：{'是' if summary['all_offline_deployments_passed'] else '否'}",
        f"- xSIM MuJoCo 闭环全部通过：{'是' if summary['all_sim2sim_passed'] else '否'}",
        f"- 最终 checkpoint 与 ONNX 文件全部存在：{'是' if summary['all_selected_files_exist'] else '否'}",
        "",
        "## 六动作指标",
        "",
        "| 动作 | 帧数/时长 | 完整/安全 | 严格通过 | XY P95均值 | Body P95均值 | 关节P95均值 | MuJoCo |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for action in actions:
        lines.append(
            f"| {action['category']} | {action['frames']} / {action['duration_s']:.2f}s | "
            f"{action['complete_count']}/4 / {action['platform_safe_count']}/4 | "
            f"{action['strict_pass_count']}/4 | {action['xy_p95_mean_m']:.3f}m | "
            f"{action['body_p95_mean_m']:.3f}m | {action['joint_p95_mean']:.3f} | "
            f"{'PASS' if action['sim2sim_passed'] else 'FAIL'} |"
        )
    lines += ["", "## 模型与未达标项", ""]
    for action in actions:
        failures = ", ".join(action["failed_checks"]) or "无"
        lines += [
            f"### {action['category']}",
            "",
            f"- 最终 checkpoint：`{action['selected_checkpoint']}`",
            f"- 最终 ONNX：`{action['selected_onnx']}`",
            f"- 严格精度未达标项：{failures}",
            f"- xMIGCS 离线 / MuJoCo："
            f"{'PASS' if action['offline_deployment_passed'] else 'FAIL'} / "
            f"{'PASS' if action['sim2sim_passed'] else 'FAIL'}",
            "",
        ]
    lines += [
        "## 判定原则",
        "",
        "预处理证据先验证源BVH片段、xGMR PKL、预览视频和训练NPZ的哈希、维度与数值完整性。"
        "最终 checkpoint 再按完整轨迹、无终止、有限值和平台安全等硬条件排序，并比较名义与随机动力学工况的综合跟踪误差。"
        "报告保留所有真实失败项，不通过放宽阈值把结果改写为通过。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "actions"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
