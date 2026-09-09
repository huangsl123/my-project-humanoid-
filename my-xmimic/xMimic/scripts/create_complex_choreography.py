"""Build a reproducible long choreography from validated DexEVT motions.

The source motions already contain safe default-pose launch and landing
transitions.  This script trims redundant tail holds, aligns every following
clip to the preceding root position/yaw, inserts a short quintic bridge, and
writes MuJoCo-style qpos for a fresh Isaac Lab forward-kinematics export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_SOURCES = (
    "fight_sports1_named.npz",
    "jumps1_subject5_named.npz",
    "push_stumble1_named.npz",
    "fight1_named.npz",
)

# Named motion files store ``joint_pos`` in Isaac Articulation order.  The
# qpos converter, however, interprets a 23-DOF input in its explicit retarget
# order.  These orders are intentionally kept separate and the composed joint
# array is remapped before it is written as qpos.
SOURCE_JOINT_ORDER = (
    "hip_pitch_l_joint",
    "hip_pitch_r_joint",
    "waist_yaw_joint",
    "hip_roll_l_joint",
    "hip_roll_r_joint",
    "waist_roll_joint",
    "hip_yaw_l_joint",
    "hip_yaw_r_joint",
    "waist_pitch_joint",
    "knee_pitch_l_joint",
    "knee_pitch_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "ankle_pitch_l_joint",
    "ankle_pitch_r_joint",
    "shoulder_roll_l_joint",
    "shoulder_roll_r_joint",
    "ankle_roll_l_joint",
    "ankle_roll_r_joint",
    "shoulder_yaw_l_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_l_joint",
    "elbow_pitch_r_joint",
)

QPOS_JOINT_ORDER = (
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothstep5(t: np.ndarray) -> np.ndarray:
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def yaw_from_wxyz(quat_wxyz: np.ndarray) -> float:
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return float(Rotation.from_quat(quat_xyzw).as_euler("xyz")[2])


def rotate_wxyz(quat_wxyz: np.ndarray, yaw_delta: float) -> np.ndarray:
    source = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    rotated = Rotation.from_euler("z", yaw_delta) * source
    return rotated.as_quat()[:, [3, 0, 1, 2]]


def bridge(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    start_joints: np.ndarray,
    end_pos: np.ndarray,
    end_quat: np.ndarray,
    end_joints: np.ndarray,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Endpoints are excluded: they are supplied by the adjacent source clips.
    u = np.arange(1, frames + 1, dtype=np.float64) / (frames + 1)
    blend = smoothstep5(u)[:, None]
    positions = start_pos + blend * (end_pos - start_pos)
    joints = start_joints + blend * (end_joints - start_joints)
    key_rots = Rotation.from_quat(np.stack([start_quat, end_quat])[:, [1, 2, 3, 0]])
    quats_xyzw = Slerp([0.0, 1.0], key_rots)(smoothstep5(u)).as_quat()
    return positions, quats_xyzw[:, [3, 0, 1, 2]], joints


def load_source(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as motion:
        required = {"fps", "joint_pos", "body_pos_w", "body_quat_w", "body_names"}
        missing = required - set(motion.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        names = [str(name) for name in motion["body_names"].tolist()]
        pelvis = names.index("pelvis")
        joints = np.asarray(motion["joint_pos"], dtype=np.float64)
        root_pos = np.asarray(motion["body_pos_w"][:, pelvis], dtype=np.float64)
        root_quat = np.asarray(motion["body_quat_w"][:, pelvis], dtype=np.float64)
    if not np.isclose(fps, 100.0) or joints.shape[1] != 23:
        raise ValueError(f"Expected 100 Hz and 23 joints in {path}, got fps={fps}, shape={joints.shape}")
    return fps, root_pos, root_quat, joints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_dir", type=Path, default=Path("motion_data"))
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bridge_frames", type=int, default=20)
    parser.add_argument("--intermediate_tail_hold", type=int, default=50)
    parser.add_argument("--source_tail_hold", type=int, default=300)
    args = parser.parse_args()

    if args.bridge_frames < 1:
        raise ValueError("bridge_frames must be positive")
    if not 1 <= args.intermediate_tail_hold <= args.source_tail_hold:
        raise ValueError("intermediate_tail_hold must be within the source tail hold")

    position_chunks: list[np.ndarray] = []
    quaternion_chunks: list[np.ndarray] = []
    joint_chunks: list[np.ndarray] = []
    source_records: list[dict[str, object]] = []
    boundaries: list[dict[str, object]] = []

    current_xy = np.zeros(2, dtype=np.float64)
    current_yaw: float | None = None
    output_frame = 0

    for source_index, source_name in enumerate(DEFAULT_SOURCES):
        source_path = (args.motion_dir / source_name).resolve()
        fps, root_pos, root_quat, joints = load_source(source_path)
        original_frames = len(joints)
        if source_index < len(DEFAULT_SOURCES) - 1:
            trim = args.source_tail_hold - args.intermediate_tail_hold
            root_pos = root_pos[:-trim]
            root_quat = root_quat[:-trim]
            joints = joints[:-trim]

        source_start_yaw = yaw_from_wxyz(root_quat[0])
        if current_yaw is None:
            yaw_delta = -source_start_yaw
        else:
            yaw_delta = current_yaw - source_start_yaw

        rotation = Rotation.from_euler("z", yaw_delta).as_matrix()
        relative_xy = root_pos[:, :2] - root_pos[0, :2]
        aligned_pos = root_pos.copy()
        aligned_pos[:, :2] = np.column_stack([relative_xy, np.zeros(len(relative_xy))]) @ rotation.T[:, :2]
        aligned_pos[:, :2] += current_xy
        aligned_quat = rotate_wxyz(root_quat, yaw_delta)

        if position_chunks:
            bridge_pos, bridge_quat, bridge_joints = bridge(
                position_chunks[-1][-1],
                quaternion_chunks[-1][-1],
                joint_chunks[-1][-1],
                aligned_pos[0],
                aligned_quat[0],
                joints[0],
                args.bridge_frames,
            )
            position_chunks.append(bridge_pos)
            quaternion_chunks.append(bridge_quat)
            joint_chunks.append(bridge_joints)
            boundaries.append(
                {
                    "before_source": source_name,
                    "bridge_start_frame": output_frame,
                    "bridge_frames": args.bridge_frames,
                    "pre_bridge_position_gap_m": float(np.linalg.norm(aligned_pos[0] - position_chunks[-2][-1])),
                    "pre_bridge_joint_gap_l2_rad": float(np.linalg.norm(joints[0] - joint_chunks[-2][-1])),
                    "pre_bridge_quaternion_abs_dot": float(abs(np.dot(aligned_quat[0], quaternion_chunks[-2][-1]))),
                }
            )
            output_frame += args.bridge_frames

        source_start_frame = output_frame
        position_chunks.append(aligned_pos)
        quaternion_chunks.append(aligned_quat)
        joint_chunks.append(joints)
        output_frame += len(joints)
        source_records.append(
            {
                "file": str(source_path),
                "sha256": sha256(source_path),
                "original_frames": original_frames,
                "used_frames": len(joints),
                "output_frame_range": [source_start_frame, output_frame],
                "yaw_alignment_deg": float(np.degrees(yaw_delta)),
            }
        )
        current_xy = aligned_pos[-1, :2].copy()
        current_yaw = yaw_from_wxyz(aligned_quat[-1])

    root_pos = np.concatenate(position_chunks, axis=0)
    root_quat = np.concatenate(quaternion_chunks, axis=0)
    joints = np.concatenate(joint_chunks, axis=0)
    quat_norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
    root_quat /= quat_norm
    source_index = {name: index for index, name in enumerate(SOURCE_JOINT_ORDER)}
    qpos_joint_indexes = [source_index[name] for name in QPOS_JOINT_ORDER]
    qpos_joints = joints[:, qpos_joint_indexes]
    qpos = np.concatenate([root_pos, root_quat, qpos_joints], axis=1)

    if not np.isfinite(qpos).all():
        raise ValueError("Composed qpos contains non-finite values")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_file, qpos=qpos.astype(np.float32), fps=np.asarray([100], dtype=np.int64))

    joint_speed = np.abs(np.diff(joints, axis=0)) * 100.0
    root_speed = np.linalg.norm(np.diff(root_pos, axis=0), axis=1) * 100.0
    summary = {
        "schema_version": 1,
        "output_file": str(args.output_file.resolve()),
        "output_sha256": sha256(args.output_file),
        "fps": 100,
        "frames": len(qpos),
        "duration_s": (len(qpos) - 1) / 100.0,
        "qpos_shape": list(qpos.shape),
        "source_joint_order": list(SOURCE_JOINT_ORDER),
        "qpos_joint_order": list(QPOS_JOINT_ORDER),
        "qpos_joint_indexes_from_source": qpos_joint_indexes,
        "sources": source_records,
        "boundaries": boundaries,
        "metrics": {
            "root_xy_span_m": np.ptp(root_pos[:, :2], axis=0).tolist(),
            "root_max_radius_from_start_m": float(np.linalg.norm(root_pos[:, :2] - root_pos[0, :2], axis=1).max()),
            "root_height_min_m": float(root_pos[:, 2].min()),
            "root_height_max_m": float(root_pos[:, 2].max()),
            "root_speed_p99_mps": float(np.percentile(root_speed, 99)),
            "root_speed_max_mps": float(root_speed.max()),
            "joint_speed_p99_radps": float(np.percentile(joint_speed, 99)),
            "joint_speed_max_radps": float(joint_speed.max()),
            "root_quaternion_max_norm_error": float(np.abs(np.linalg.norm(root_quat, axis=1) - 1.0).max()),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
