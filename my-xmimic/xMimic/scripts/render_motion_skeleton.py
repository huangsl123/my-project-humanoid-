"""Render a named DexEVT NPZ motion as a lightweight skeleton video.

This renderer visualizes the reference trajectory itself, without a learned
policy or physics.  It is therefore useful as a preprocessing gate before an
expensive training run, but it is not evidence that a policy can track the
motion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


LINKS = (
    ("pelvis", "waist_yaw_link"),
    ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "waist_pitch_link"),
    ("waist_pitch_link", "head_yaw_link"),
    ("head_yaw_link", "head_pitch_link"),
    ("pelvis", "hip_pitch_l_link"),
    ("hip_pitch_l_link", "hip_roll_l_link"),
    ("hip_roll_l_link", "hip_yaw_l_link"),
    ("hip_yaw_l_link", "knee_pitch_l_link"),
    ("knee_pitch_l_link", "ankle_pitch_l_link"),
    ("ankle_pitch_l_link", "ankle_roll_l_link"),
    ("pelvis", "hip_pitch_r_link"),
    ("hip_pitch_r_link", "hip_roll_r_link"),
    ("hip_roll_r_link", "hip_yaw_r_link"),
    ("hip_yaw_r_link", "knee_pitch_r_link"),
    ("knee_pitch_r_link", "ankle_pitch_r_link"),
    ("ankle_pitch_r_link", "ankle_roll_r_link"),
    ("waist_pitch_link", "shoulder_pitch_l_link"),
    ("shoulder_pitch_l_link", "shoulder_roll_l_link"),
    ("shoulder_roll_l_link", "shoulder_yaw_l_link"),
    ("shoulder_yaw_l_link", "elbow_pitch_l_link"),
    ("elbow_pitch_l_link", "elbow_yaw_l_link"),
    ("elbow_yaw_l_link", "wrist_pitch_l_link"),
    ("wrist_pitch_l_link", "wrist_roll_l_link"),
    ("wrist_roll_l_link", "left_tcp_link"),
    ("waist_pitch_link", "shoulder_pitch_r_link"),
    ("shoulder_pitch_r_link", "shoulder_roll_r_link"),
    ("shoulder_roll_r_link", "shoulder_yaw_r_link"),
    ("shoulder_yaw_r_link", "elbow_pitch_r_link"),
    ("elbow_pitch_r_link", "elbow_yaw_r_link"),
    ("elbow_yaw_r_link", "wrist_pitch_r_link"),
    ("wrist_pitch_r_link", "wrist_roll_r_link"),
    ("wrist_roll_r_link", "right_tcp_link"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--output_fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.motion_file.expanduser().resolve(), allow_pickle=False) as motion:
        body_pos = np.asarray(motion["body_pos_w"], dtype=np.float32)
        names = [str(name) for name in motion["body_names"].tolist()]
        input_fps = float(np.asarray(motion["fps"]).reshape(-1)[0])

    name_to_index = {name: index for index, name in enumerate(names)}
    missing = sorted({name for link in LINKS for name in link} - set(name_to_index))
    if missing:
        raise ValueError(f"Motion is missing required bodies: {missing}")
    if args.output_fps <= 0.0 or args.output_fps > input_fps:
        raise ValueError("output_fps must be positive and no greater than input fps")

    frame_ids = np.arange(0, len(body_pos), input_fps / args.output_fps).astype(int)
    frame_ids = np.unique(np.clip(frame_ids, 0, len(body_pos) - 1))
    pelvis = name_to_index["pelvis"]
    root_path = body_pos[:, pelvis, :2]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.output_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the requested MP4 writer")

    # A root-following oblique projection keeps full-body motion readable while
    # the inset map preserves the choreography's true horizontal displacement.
    scale = min(args.width, args.height) * 0.38
    origin = np.array([args.width * 0.50, args.height * 0.82], dtype=np.float32)
    path_min = root_path.min(axis=0)
    path_span = np.maximum(np.ptp(root_path, axis=0), 0.1)

    for frame_id in frame_ids:
        image = np.full((args.height, args.width, 3), (30, 33, 38), dtype=np.uint8)
        current = body_pos[frame_id]
        root_xy = current[pelvis, :2]
        relative = current.copy()
        relative[:, :2] -= root_xy
        projected = np.empty((len(names), 2), dtype=np.int32)
        projected[:, 0] = np.rint(origin[0] + scale * (relative[:, 0] - 0.35 * relative[:, 1])).astype(int)
        projected[:, 1] = np.rint(origin[1] - scale * (relative[:, 2] + 0.18 * relative[:, 1])).astype(int)

        cv2.line(image, (80, int(origin[1])), (args.width - 80, int(origin[1])), (75, 80, 88), 2)
        for left, right in LINKS:
            color = (230, 185, 70) if "_l_" in left or left.startswith("left") else (80, 180, 245)
            if "_r_" not in left and not left.startswith("right") and "_l_" not in left and not left.startswith("left"):
                color = (205, 210, 215)
            cv2.line(image, tuple(projected[name_to_index[left]]), tuple(projected[name_to_index[right]]), color, 5, cv2.LINE_AA)
        for point in projected:
            cv2.circle(image, tuple(point), 4, (245, 245, 245), -1, cv2.LINE_AA)

        map_left, map_top, map_width, map_height = args.width - 330, 40, 280, 210
        cv2.rectangle(image, (map_left, map_top), (map_left + map_width, map_top + map_height), (60, 65, 72), -1)
        cv2.rectangle(image, (map_left, map_top), (map_left + map_width, map_top + map_height), (120, 125, 132), 1)
        path_pixels = np.empty((len(root_path), 2), dtype=np.int32)
        path_pixels[:, 0] = np.rint(map_left + 20 + (root_path[:, 0] - path_min[0]) / path_span[0] * (map_width - 40)).astype(int)
        path_pixels[:, 1] = np.rint(map_top + map_height - 20 - (root_path[:, 1] - path_min[1]) / path_span[1] * (map_height - 40)).astype(int)
        cv2.polylines(image, [path_pixels], False, (105, 110, 118), 2, cv2.LINE_AA)
        cv2.polylines(image, [path_pixels[: frame_id + 1]], False, (60, 210, 145), 3, cv2.LINE_AA)
        cv2.circle(image, tuple(path_pixels[frame_id]), 6, (60, 210, 145), -1, cv2.LINE_AA)

        elapsed = frame_id / input_fps
        duration = (len(body_pos) - 1) / input_fps
        cv2.putText(image, "complex_choreography_v1 - kinematic reference", (40, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(image, f"{elapsed:6.2f} / {duration:.2f} s", (40, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 220, 190), 2, cv2.LINE_AA)
        cv2.putText(image, "Reference only - not policy playback", (40, args.height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (145, 155, 165), 2, cv2.LINE_AA)
        cv2.putText(image, "root XY path", (map_left + 12, map_top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (215, 215, 215), 1, cv2.LINE_AA)
        writer.write(image)

    writer.release()
    print(f"[INFO] Saved {len(frame_ids)} frames at {args.output_fps:g} fps to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
