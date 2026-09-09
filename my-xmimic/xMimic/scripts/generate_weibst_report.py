#!/usr/bin/env python3
"""Generate the final human-readable Weibst training and deployment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def yes(value: object) -> str:
    return "是" if bool(value) else "否"


def number(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def metric_tail(summary: dict, name: str) -> tuple[str, str]:
    metric = summary.get("metrics", {}).get(name, {})
    return number(metric.get("first")), number(metric.get("last_100_mean"))


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def selection_table(selection: dict) -> list[str]:
    lines = [
        "| Checkpoint | 工况数 | 硬失败数 | 平均归一化误差 | 最坏归一化误差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in selection["candidates"]:
        lines.append(
            f"| {item['checkpoint']} | {item['case_count']} | "
            f"{item['hard_failure_count']} | {number(item['mean_tracking_score'])} | "
            f"{number(item['worst_tracking_score'])} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.expanduser().resolve()

    stage1 = load(root / "stage1_training_analysis/summary.json")
    stage2 = load(root / "stage2_training_analysis/summary.json")
    stage1_selection = load(root / "stage1_checkpoint_selection.json")
    stage2_selection = load(root / "stage2_checkpoint_selection.json")
    comparison = load(root / "final_evaluation_comparison/evaluation_comparison.json")
    checkpoint = load(root / "final_checkpoint_validation/summary.json")
    onnx = load(root / "onnx_validation/summary.json")
    deployed_onnx = load(root / "deployment_onnx_validation/summary.json")
    adapter = load(root / "deployment_adapter_validation/summary.json")
    sim2sim = load(root / "sim2sim_validation/summary.json")

    evaluations = comparison["evaluations"]
    all_complete = all(item["motion_completed"] for item in evaluations)
    all_exact = all(item["exact_motion_frame_sequence"] for item in evaluations)
    all_finite = all(item["all_metrics_finite"] for item in evaluations)
    all_in_bounds = all(item["within_platform_bounds"] for item in evaluations)
    strict_pass_count = int(comparison["pass_count"])

    s1_reward = metric_tail(stage1, "Train/mean_reward")
    s2_reward = metric_tail(stage2, "Train/mean_reward")
    s1_length = metric_tail(stage1, "Train/mean_episode_length")
    s2_length = metric_tail(stage2, "Train/mean_episode_length")

    lines = [
        "# Weibst Hard 50,000轮训练、评估与sim2sim部署报告",
        "",
        "## 结论",
        "",
        f"最终选择 `{stage2_selection['selected_checkpoint']}`。四个完整动作工况全部完成：{yes(all_complete)}；"
        f"帧序列严格连续：{yes(all_exact)}；全部数值有限：{yes(all_finite)}；平台界内：{yes(all_in_bounds)}。",
        "",
        f"严格精度验收通过 {strict_pass_count}/{len(evaluations)} 个工况。表格中的 `failed_checks` 保留真实未达标项；"
        "完整执行、无倒地和平台安全与精度门槛分别报告，二者不会混为一谈。",
        "",
        "## 动作与训练设置",
        "",
        "- 动作：`motion_data/weibst_hard_named.npz`",
        f"- 动作帧数/频率/时长：{onnx['motion_frames']} / {number(onnx['motion_fps'], 1)} Hz / "
        f"{number(onnx['motion_frames'] / onnx['motion_fps'], 2)} s",
        "- 并行环境：4096",
        "- Stage 1：40,000 iterations，完整13.27秒状态反馈长时程任务",
        "- Stage 2：从Stage 1实测最佳候选继续10,000 iterations，完整13.27秒精度任务",
        "- actor观测/critic观测/action：130 / 346 / 23",
        "- 策略继续跟踪原动作的时变骨盆轨迹，没有增加平台中心吸引项。",
        "",
        "## 训练完整性",
        "",
        "| 阶段 | 首步 | 末步 | 记录数 | 连续 | 数值有限 | checkpoint到达末步 |",
        "|---|---:|---:|---:|---|---|---|",
        f"| Stage 1 | {stage1['iteration_progress']['first_iteration_step']} | "
        f"{stage1['iteration_progress']['last_iteration_step']} | "
        f"{stage1['iteration_progress']['observed_iteration_records']} | "
        f"{yes(stage1['iteration_progress']['iteration_sequence_contiguous'])} | "
        f"{yes(stage1['quality_checks']['all_scalars_finite'])} | "
        f"{yes(stage1['quality_checks']['checkpoint_reaches_expected_final_step'])} |",
        f"| Stage 2 | {stage2['iteration_progress']['first_iteration_step']} | "
        f"{stage2['iteration_progress']['last_iteration_step']} | "
        f"{stage2['iteration_progress']['observed_iteration_records']} | "
        f"{yes(stage2['iteration_progress']['iteration_sequence_contiguous'])} | "
        f"{yes(stage2['quality_checks']['all_scalars_finite'])} | "
        f"{yes(stage2['quality_checks']['checkpoint_reaches_expected_final_step'])} |",
        "",
        "| 阶段 | Reward首值 | Reward末100均值 | Episode length首值 | Episode length末100均值 |",
        "|---|---:|---:|---:|---:|",
        f"| Stage 1 | {s1_reward[0]} | {s1_reward[1]} | {s1_length[0]} | {s1_length[1]} |",
        f"| Stage 2 | {s2_reward[0]} | {s2_reward[1]} | {s2_length[0]} | {s2_length[1]} |",
        "",
        "## Stage 1候选选择",
        "",
        f"选择结果：`{stage1_selection['selected_checkpoint']}`。排序首先比较完整性、安全性和有限值等硬失败，"
        "随后比较名义与随机动力学工况的综合跟踪误差。",
        "",
        *selection_table(stage1_selection),
        "",
        "## Stage 2候选选择",
        "",
        f"选择结果：`{stage2_selection['selected_checkpoint']}`。最终模型不是按编号大小盲选。",
        "",
        *selection_table(stage2_selection),
        "",
        "## 最终四工况完整回放",
        "",
        "| 工况 | 完成帧 | XY P95(m) | XY Max(m) | Body P95(m) | 关节P95 | 朝向P95(rad) | 最小边缘余量(m) | 严格通过 | failed_checks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in evaluations:
        failures = ", ".join(item["failed_checks"]) if item["failed_checks"] else "无"
        lines.append(
            f"| {item['label']} | {item['logged_steps']}/{item['motion_total_frames']} | "
            f"{number(item['xy_tracking_drift_p95_m'])} | {number(item['xy_tracking_drift_max_m'])} | "
            f"{number(item['body_pos_p95'])} | {number(item['joint_pos_p95'])} | "
            f"{number(item['anchor_rot_p95'])} | {number(item['minimum_edge_margin_m'])} | "
            f"{yes(item['acceptance_pass'])} | {failures} |"
        )

    stage1_score = float(stage1_selection["candidates"][0]["mean_tracking_score"])
    stage2_score = float(stage2_selection["candidates"][0]["mean_tracking_score"])
    score_improvement = 100.0 * (stage1_score - stage2_score) / stage1_score
    mean_body_p95 = sum(float(item["body_pos_p95"]) for item in evaluations) / len(evaluations)
    mean_joint_p95 = sum(float(item["joint_pos_p95"]) for item in evaluations) / len(evaluations)
    mean_rot_p95 = sum(float(item["anchor_rot_p95"]) for item in evaluations) / len(evaluations)
    lines.extend(
        [
            "",
            "## 结果分析与后续建议",
            "",
            f"Stage 2最佳候选的综合跟踪分数由Stage 1的 {number(stage1_score)} 降至 "
            f"{number(stage2_score)}，相对改善 {number(score_improvement, 2)}%。训练记录完整、"
            "四组回放均跑满动作且sim2sim闭环最终保持站立，说明策略已经学会完整动作流程，并具备可部署的接口完整性。",
            "",
            f"但四工况严格精度门槛均未通过；Body P95、关节P95和骨盆朝向P95的四工况均值分别为 "
            f"{number(mean_body_p95)} m、{number(mean_joint_p95)} 和 {number(mean_rot_p95)} rad。"
            "因此当前模型适合继续仿真验证，不建议直接用于真实机器人。失败性质是动作保真度不足，"
            "不是轨迹未完成、越界、NaN或部署维度不兼容。",
            "",
            "下一轮优化应优先针对误差最大的动作片段做分段课程训练，并提高关节姿态、身体位置和骨盆朝向奖励权重；"
            "随后扩大动力学随机化范围并重复多种子严格回放。仅机械增加总轮数的收益可能低于针对困难片段的课程优化。"
            "另外，sim2sim中骨盆上轴Z分量的瞬时最小值为 "
            f"{number(sim2sim['base_up_axis_world_z']['minimum'])}，虽然最终恢复到 "
            f"{number(sim2sim['base_up_axis_world_z']['final'])}，仍应结合遥测图和视频确认该翻转是否属于原动作，"
            "再决定是否进入实机测试。",
        ]
    )

    video_files = sorted((root / "final_evaluation/nominal_seed42/video").glob("*.mp4"))
    video_link = relative(video_files[0], root) if video_files else "未找到"
    lines.extend(
        [
            "",
            "## Checkpoint、ONNX与部署验证",
            "",
            f"- 最终checkpoint：`{checkpoint['checkpoint']}`",
            f"- Checkpoint SHA-256：`{checkpoint['sha256']}`",
            f"- ONNX SHA-256：`{onnx['model_sha256']}`",
            f"- Checkpoint结构和有限值检查：{yes(checkpoint['checks']['all_passed'])}",
            f"- ONNX完整一致性检查：{yes(onnx['checks']['all_passed'])}",
            f"- 部署目录ONNX一致性检查：{yes(deployed_onnx['checks']['all_passed'])}",
            f"- xMIGCS真实适配器离线推理检查：{yes(adapter['passed'])}",
            f"- Actor/Critic观测归一化布局：{checkpoint['actor_normalizer_mode']} / "
            f"{checkpoint['critic_normalizer_mode']}",
            f"- xSIM+xMIGCS真实ROS闭环sim2sim验收：{yes(sim2sim['passed'])}",
            f"- sim2sim策略阶段时长/里程计样本：{number(sim2sim['duration_s'], 2)} s / "
            f"{sim2sim['odometry_samples_during_policy']}",
            f"- sim2sim基座高度（初始/最低/最终）："
            f"{number(sim2sim['base_height_m']['initial'])} / "
            f"{number(sim2sim['base_height_m']['minimum'])} / "
            f"{number(sim2sim['base_height_m']['final'])} m",
            f"- sim2sim末帧骨盆局部上轴的世界Z分量："
            f"{number(sim2sim['base_up_axis_world_z']['final'])}",
            "- 部署观测顺序：command 46 + anchor position 3 + anchor orientation 6 + base linear velocity 3 + "
            "base angular velocity 3 + joint position 23 + joint velocity 23 + previous action 23 = 130。",
            "- xSIM通过`/base/odometry`提供真实世界坐标骨盆位置和线速度。",
            "",
            "## 关键结果文件",
            "",
            f"- [Stage 1训练总图]({relative(root / 'stage1_training_analysis/training_overview.png', root)})",
            f"- [Stage 2训练总图]({relative(root / 'stage2_training_analysis/training_overview.png', root)})",
            f"- [四工况对比图]({relative(root / 'final_evaluation_comparison/evaluation_comparison.png', root)})",
            f"- [正常速度视频]({video_link})",
            f"- [最终checkpoint验证]({relative(root / 'final_checkpoint_validation/summary.json', root)})",
            f"- [ONNX验证]({relative(root / 'onnx_validation/summary.json', root)})",
            f"- [xMIGCS适配器验证]({relative(root / 'deployment_adapter_validation/summary.json', root)})",
            f"- [真实sim2sim闭环验证]({relative(root / 'sim2sim_validation/summary.json', root)})",
            f"- [sim2sim基座遥测]({relative(root / 'sim2sim_validation/telemetry.csv', root)})",
            f"- [sim2sim基座轨迹图]({relative(root / 'sim2sim_validation/telemetry.png', root)})",
            f"- [完整产物清单]({relative(root / 'MANIFEST.json', root)})",
            f"- [产物SHA-256清单]({relative(root / 'MANIFEST.sha256', root)})",
            "",
            "## sim2sim启动",
            "",
            "```bash",
            "cd \"/home/eai/my project/xmigcs_test\"",
            "./launch_weibst_hard_sim2sim.sh",
            "```",
            "",
            "控制节点启动后按 `u` 选择 `weibst_hard` 策略。MuJoCo中的ESC仅负责释放机器人约束，"
            "不是启动策略的按键。",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
