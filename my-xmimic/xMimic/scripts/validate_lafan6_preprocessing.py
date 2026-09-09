#!/usr/bin/env python3
"""Validate and document the six LAFAN1 BVH -> xGMR PKL -> xMimic NPZ chains."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import types
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np


ACTIONS = (
    {
        "slug": "01_fall_getup",
        "category": "FallAndGetUp",
        "bvh": "fallAndGetUp2_subject2.bvh",
        "start": 109,
        "end": 528,
        "pkl": "fall_getup1_subject2_clip109_528.pkl",
        "xgmr_pkl": "fallAndGetUp2_subject2_clip109_528dex_evt2.pkl",
        "preview": "fallAndGetUp2_subject2_clip109_528.mp4",
        "npz": "fall_getup1_subject2_named.npz",
        "frames": 1837,
    },
    {
        "slug": "02_fight_sports",
        "category": "FightAndSports",
        "bvh": "fightAndSports1_subject1.bvh",
        "start": 3615,
        "end": 4065,
        "pkl": "fight_sports1_clip3615_4065.pkl",
        "xgmr_pkl": "fight_sports1_clip3615_4065dex_evt2.pkl",
        "preview": "fight_sports1_clip3615_4065.mp4",
        "npz": "fight_sports1_named.npz",
        "frames": 1940,
    },
    {
        "slug": "03_jumps",
        "category": "Jumps",
        "bvh": "jumps1_subject5.bvh",
        "start": 6495,
        "end": 7020,
        "pkl": "jumps1_subject5_clip6495_7020.pkl",
        "xgmr_pkl": "jumps1_subject5_clip6495_7020dex_evt2.pkl",
        "preview": "jumps1_subject5_clip6495_7020.mp4",
        "npz": "jumps1_subject5_named.npz",
        "frames": 2190,
    },
    {
        "slug": "04_ground",
        "category": "Ground",
        "bvh": "ground2_subject2.bvh",
        "start": 135,
        "end": 735,
        "pkl": "ground2_clip135_735.pkl",
        "xgmr_pkl": "ground2_clip135_735dex_evt2.pkl",
        "preview": "ground2_clip135_735.mp4",
        "npz": "ground2_named.npz",
        "frames": 2440,
    },
    {
        "slug": "05_push_stumble",
        "category": "PushAndStumble",
        "bvh": "pushAndStumble1_subject2.bvh",
        "start": 1500,
        "end": 1950,
        "pkl": "push_stumble1_clip1500_1950.pkl",
        "xgmr_pkl": "push_stumble1_clip1500_1950dex_evt2.pkl",
        "preview": "push_stumble1_clip1500_1950.mp4",
        "npz": "push_stumble1_named.npz",
        "frames": 1940,
    },
    {
        "slug": "06_fight",
        "category": "Fight",
        "bvh": "fight1_subject2.bvh",
        "start": 2925,
        "end": 3375,
        "pkl": "fight1_clip2925_3375.pkl",
        "xgmr_pkl": "fight1_clip2925_3375dex_evt2.pkl",
        "preview": "fight1_clip2925_3375.mp4",
        "npz": "fight1_named.npz",
        "frames": 1940,
    },
)

NPZ_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
    "body_layout",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enable_numpy_pickle_compatibility() -> None:
    """Allow NumPy 1.x to read trusted xGMR pickles written by NumPy 2.x."""
    if "numpy._core" in sys.modules:
        return
    core = getattr(np, "core", None)
    if core is None:
        return
    shim = types.ModuleType("numpy._core")
    shim.__dict__.update(core.__dict__)
    sys.modules["numpy._core"] = shim
    for name in ("multiarray", "umath", "numerictypes", "_multiarray_umath", "numeric"):
        submodule = getattr(core, name, None)
        if submodule is not None:
            sys.modules[f"numpy._core.{name}"] = submodule


def bvh_metadata(path: Path) -> tuple[int, float]:
    header = path.read_text(encoding="utf-8", errors="replace")[:65536]
    frames_match = re.search(r"^Frames:\s*(\d+)\s*$", header, re.MULTILINE)
    dt_match = re.search(r"^Frame Time:\s*([0-9.eE+-]+)\s*$", header, re.MULTILINE)
    if not frames_match or not dt_match:
        raise ValueError(f"Could not parse BVH metadata: {path}")
    return int(frames_match.group(1)), float(dt_match.group(1))


def finite(value: object) -> bool:
    array = np.asarray(value)
    return bool(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all())


def validate_action(
    spec: dict,
    *,
    bvh_root: Path,
    xgmr_root: Path,
    motion_example: Path,
    motion_data: Path,
) -> dict:
    bvh = bvh_root / spec["bvh"]
    source_pkl = xgmr_root / "output" / spec["xgmr_pkl"]
    copied_pkl = motion_example / spec["pkl"]
    preview = xgmr_root / "videos" / spec["preview"]
    npz_path = motion_data / spec["npz"]
    required = (bvh, source_pkl, copied_pkl, preview, npz_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing preprocessing artifacts:\n" + "\n".join(missing))

    source_frames, frame_time = bvh_metadata(bvh)
    range_valid = 0 <= spec["start"] < spec["end"] <= source_frames

    enable_numpy_pickle_compatibility()
    with copied_pkl.open("rb") as stream:
        pkl = pickle.load(stream)  # trusted locally generated xGMR artifact
    pkl_shapes = {
        key: list(np.asarray(pkl[key]).shape) for key in ("root_pos", "root_rot", "dof_pos")
    }
    pkl_frames = pkl_shapes["root_pos"][0]
    pkl_checks = {
        "source_copy_hash_equal": sha256(source_pkl) == sha256(copied_pkl),
        "fps_30": int(pkl.get("fps", -1)) == 30,
        "root_pos_shape": pkl_shapes["root_pos"] == [pkl_frames, 3],
        "root_rot_shape": pkl_shapes["root_rot"] == [pkl_frames, 4],
        "dof_pos_shape_31": pkl_shapes["dof_pos"] == [pkl_frames, 31],
        "all_numeric_finite": all(
            finite(pkl[key]) for key in ("root_pos", "root_rot", "dof_pos")
        ),
        "quaternions_unit": bool(
            np.max(np.abs(np.linalg.norm(np.asarray(pkl["root_rot"]), axis=-1) - 1.0))
            < 1.0e-4
        ),
    }

    zip_valid = zipfile.is_zipfile(npz_path)
    zip_crc_valid = False
    if zip_valid:
        with zipfile.ZipFile(npz_path) as archive:
            zip_crc_valid = archive.testzip() is None
    with np.load(npz_path, allow_pickle=False) as motion:
        keys = set(motion.files)
        frames = int(motion["joint_pos"].shape[0])
        fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
        body_names = [str(name) for name in np.asarray(motion["body_names"]).tolist()]
        body_layout = str(np.asarray(motion["body_layout"]).item())
        numeric_keys = sorted(keys - {"body_names", "body_layout"})
        quat_norm_error = float(
            np.max(np.abs(np.linalg.norm(motion["body_quat_w"], axis=-1) - 1.0))
        )
        tail_hold_frames = 0
        joint_pos = np.asarray(motion["joint_pos"])
        for index in range(len(joint_pos) - 2, -1, -1):
            if np.array_equal(joint_pos[index], joint_pos[-1]):
                tail_hold_frames += 1
            else:
                break
        npz_checks = {
            "zip_container_valid": zip_valid,
            "zip_crc_valid": zip_crc_valid,
            "keys_exact": keys == NPZ_KEYS,
            "fps_100": fps == 100,
            "frame_count_expected": frames == spec["frames"],
            "joint_pos_shape_23": list(motion["joint_pos"].shape) == [frames, 23],
            "joint_vel_shape_23": list(motion["joint_vel"].shape) == [frames, 23],
            "body_pos_shape_39x3": list(motion["body_pos_w"].shape) == [frames, 39, 3],
            "body_quat_shape_39x4": list(motion["body_quat_w"].shape) == [frames, 39, 4],
            "body_lin_vel_shape_39x3": list(motion["body_lin_vel_w"].shape)
            == [frames, 39, 3],
            "body_ang_vel_shape_39x3": list(motion["body_ang_vel_w"].shape)
            == [frames, 39, 3],
            "body_names_39_unique": len(body_names) == 39 and len(set(body_names)) == 39,
            "pelvis_is_anchor_body_zero": bool(body_names and body_names[0] == "pelvis"),
            "body_layout_expected": body_layout == "dex_evt_legacy_39_named",
            "all_numeric_finite": all(finite(motion[key]) for key in numeric_keys),
            "quaternions_unit": quat_norm_error < 1.0e-4,
            "tail_hold_300_frames": tail_hold_frames == 300,
        }

    checks = {
        "selected_range_within_source": range_valid,
        "preview_nonempty": preview.stat().st_size > 0,
        **{f"pkl_{key}": value for key, value in pkl_checks.items()},
        **{f"npz_{key}": value for key, value in npz_checks.items()},
    }
    return {
        "slug": spec["slug"],
        "category": spec["category"],
        "selected_bvh": str(bvh.resolve()),
        "selected_range_start_inclusive": spec["start"],
        "selected_range_end_exclusive": spec["end"],
        "selected_source_frame_count": spec["end"] - spec["start"],
        "bvh_total_frames": source_frames,
        "bvh_frame_time_s": frame_time,
        "bvh_fps_approx": 1.0 / frame_time,
        "xgmr_pkl": str(source_pkl.resolve()),
        "copied_pkl": str(copied_pkl.resolve()),
        "preview_video": str(preview.resolve()),
        "npz": str(npz_path.resolve()),
        "pkl_frames": pkl_frames,
        "pkl_shapes": pkl_shapes,
        "npz_frames": frames,
        "npz_fps": fps,
        "npz_duration_s": (frames - 1) / fps,
        "body_layout": body_layout,
        "body_names": body_names,
        "tail_hold_frames": tail_hold_frames,
        "body_quaternion_max_norm_error": quat_norm_error,
        "hashes": {
            "bvh_sha256": sha256(bvh),
            "xgmr_pkl_sha256": sha256(source_pkl),
            "copied_pkl_sha256": sha256(copied_pkl),
            "preview_sha256": sha256(preview),
            "npz_sha256": sha256(npz_path),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-root", type=Path, required=True)
    parser.add_argument("--xgmr-root", type=Path, required=True)
    parser.add_argument("--motion-example", type=Path, required=True)
    parser.add_argument("--motion-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    actions = [
        validate_action(
            spec,
            bvh_root=args.bvh_root.expanduser().resolve(),
            xgmr_root=args.xgmr_root.expanduser().resolve(),
            motion_example=args.motion_example.expanduser().resolve(),
            motion_data=args.motion_data.expanduser().resolve(),
        )
        for spec in ACTIONS
    ]
    body_name_orders_equal = all(
        action["body_names"] == actions[0]["body_names"] for action in actions[1:]
    )
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "action_count": len(actions),
        "body_name_orders_equal": body_name_orders_equal,
        "all_actions_passed": body_name_orders_equal and all(action["passed"] for action in actions),
        "actions": actions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# LAFAN1 六动作预处理验证",
        "",
        f"生成时间：{result['generated_at']}",
        "",
        f"总体结果：{'PASS' if result['all_actions_passed'] else 'FAIL'}",
        "",
        "| 动作 | BVH片段 | xGMR PKL | 训练NPZ | 帧数/时长 | 结果 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for action in actions:
        lines.append(
            f"| {action['category']} | `{Path(action['selected_bvh']).name}` "
            f"[{action['selected_range_start_inclusive']}, {action['selected_range_end_exclusive']}) | "
            f"{action['pkl_frames']}帧/31 DOF | 23关节/39命名刚体 | "
            f"{action['npz_frames']}帧/{action['npz_duration_s']:.2f}s | "
            f"{'PASS' if action['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "验证覆盖源BVH范围、xGMR输出及复制文件哈希、预览视频、PKL帧率/31维重映射、"
        "NPZ ZIP完整性、100 Hz、23关节、39个有序刚体、有限值、单位四元数和300帧末姿态保持。",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in result if key != "actions"}, indent=2, ensure_ascii=False))
    if not result["all_actions_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
