#!/usr/bin/env python3
"""Offline integration check for a BeyondMimic deployment configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from common.robot_data import RobotData
from policy.beyond_mimic.fsm_beyond_mimic import FSMStateBeyondMimic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="weibst_hard_validation")
    parser.add_argument("--expected-motion-length", type=int, default=1327)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    robot_data = RobotData(motor_num=int(config["motor_nums"]))
    state = FSMStateBeyondMimic(
        robot_data,
        config_path=str(config_path),
        variant_name=args.variant,
    )
    state.on_enter()

    # Supply a finite measured floating-base state so the exact 130-D measured
    # state-feedback path used in sim2sim is exercised, without requiring ROS.
    robot_data.base_state_received_ = True
    robot_data.base_position_w_[:] = (0.0, 0.0, 0.9)
    known_world_velocity = np.asarray((0.12, -0.08, 0.04), dtype=np.float64)
    robot_data.base_linear_velocity_w_[:] = known_world_velocity
    for _ in range(3):
        state.inner_run()

    observation = np.asarray(state.last_policy_observation)
    # 46 command values are followed by position error 3, relative anchor
    # orientation 6, and measured base velocity 3 in the 130-D profile.
    anchor_position_slice = observation[46:49]
    anchor_orientation_slice = observation[49:55]
    base_velocity_slice = observation[55:58]
    identity_orientation_6d = np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))

    model_path = Path(state.onnx_path).resolve()
    checks = {
        "model_exists": model_path.is_file() and model_path.stat().st_size > 0,
        "model_obs_matches_config": state.num_obs == int(config["num_obs"]) == 130,
        "actions_match_config": state.num_actions == int(config["num_actions"]) == 23,
        "motors_match_config": state.motor_nums == int(config["motor_nums"]) == 29,
        "body_count": state.num_bodies == len(config["body_names"]) == 24,
        "locked_joint_map_count": len(state.locked_joint_map) == state.num_actions,
        "locked_joint_map_unique_and_in_range": (
            len(np.unique(state.locked_joint_map)) == state.num_actions
            and int(np.min(state.locked_joint_map)) >= 0
            and int(np.max(state.locked_joint_map)) < state.motor_nums
        ),
        "joint_metadata_count": len(state.joint_seq) == state.num_actions,
        "joint_metadata_matches_deployment_xml": (
            set(state.joint_seq) == set(state.joint_xml)
            and np.array_equal(np.sort(state.mj2lab), np.arange(state.num_actions))
        ),
        "action_scale_count": len(state.action_scale_lab) == state.num_actions,
        "default_joint_pos_count": len(state.default_angles_lab) == state.num_actions,
        "gain_counts_and_values": (
            len(state.kps) == len(state.kds) == state.motor_nums
            and bool(np.isfinite(state.kps).all())
            and bool(np.isfinite(state.kds).all())
            and bool((np.asarray(state.kps) > 0.0).all())
            and bool((np.asarray(state.kds) > 0.0).all())
        ),
        "control_timing": (
            state.physical_dt == 0.01
            and state.decimation_ == 1
            and state.warm_start_time == 0.5
        ),
        "motion_length": (
            state.motion_length
            == int(config["motion_length"])
            == args.expected_motion_length
        ),
        "measured_state_feedback": state.state_feedback_source == "measured",
        "base_state_consumed": robot_data.base_state_received_,
        "diagnostic_observation_shape": list(observation.shape) == [130],
        "diagnostic_observation_finite": bool(np.isfinite(observation).all()),
        "initial_reference_translation_aligned": bool(
            np.allclose(anchor_position_slice, 0.0, atol=1.0e-5, rtol=0.0)
        ),
        "initial_reference_orientation_aligned": bool(
            np.allclose(anchor_orientation_slice, identity_orientation_6d, atol=1.0e-5, rtol=0.0)
        ),
        "measured_world_velocity_transformed_to_body": bool(
            np.allclose(base_velocity_slice, known_world_velocity, atol=1.0e-5, rtol=0.0)
        ),
        "action_shape": list(np.asarray(state.action).shape) == [1, 23],
        "action_finite": bool(np.isfinite(state.action).all()),
        "command_finite": bool(np.isfinite(robot_data.q_d_).all()),
    }
    result = {
        "config": str(config_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "onnx_inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in state.ort_session.get_inputs()
        ],
        "onnx_outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in state.ort_session.get_outputs()
        ],
        "checks": checks,
        "diagnostic_observation": {
            "anchor_position_b": anchor_position_slice.tolist(),
            "anchor_orientation_6d_b": anchor_orientation_slice.tolist(),
            "base_linear_velocity_b": base_velocity_slice.tolist(),
        },
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
