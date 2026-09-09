#!/usr/bin/env python3
"""Remove unstable xGMR prefix and smoothly slow the launch of a motion."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-prefix-frames", type=int, default=9)
    parser.add_argument("--source-blend-seconds", type=float, default=4.0)
    parser.add_argument("--output-blend-seconds", type=float, default=6.0)
    return parser.parse_args()


def interpolate_vectors(values: np.ndarray, source_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    result = np.empty((len(query), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(query, source_times, flat[:, column])
    return result.reshape((len(query),) + values.shape[1:])


def main() -> None:
    args = parse_args()
    with args.input.open("rb") as stream:
        data = pickle.load(stream)
    fps = float(np.asarray(data["fps"]).item())
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")

    arrays = {
        key: np.asarray(data[key], dtype=np.float64)[args.drop_prefix_frames :]
        for key in ("root_pos", "root_rot", "dof_pos")
    }
    frame_count = len(arrays["root_pos"])
    if any(len(value) != frame_count for value in arrays.values()):
        raise ValueError("Motion arrays have inconsistent frame counts")

    source_end_index = int(round(args.source_blend_seconds * fps))
    output_end_index = int(round(args.output_blend_seconds * fps))
    if not 1 < source_end_index < frame_count or output_end_index <= source_end_index:
        raise ValueError("Invalid launch smoothing interval")

    # Source-time speed ramps linearly from 1/3 speed to normal speed.  For
    # 4 source seconds over 6 output seconds, its integral is exactly 4 s,
    # so position and velocity join the untouched motion continuously.
    output_t = np.arange(output_end_index + 1, dtype=np.float64) / fps
    duration_out = args.output_blend_seconds
    initial_speed = 2.0 * args.source_blend_seconds / duration_out - 1.0
    if not 0.0 < initial_speed < 1.0:
        raise ValueError(f"Computed initial time scale is invalid: {initial_speed}")
    acceleration = (1.0 - initial_speed) / duration_out
    query_t = initial_speed * output_t + 0.5 * acceleration * output_t**2
    query_t[-1] = args.source_blend_seconds

    source_times = np.arange(source_end_index + 1, dtype=np.float64) / fps
    root_pos_head = interpolate_vectors(arrays["root_pos"][: source_end_index + 1], source_times, query_t)
    dof_pos_head = interpolate_vectors(arrays["dof_pos"][: source_end_index + 1], source_times, query_t)
    rotation_head = Slerp(source_times, Rotation.from_quat(arrays["root_rot"][: source_end_index + 1]))(query_t).as_quat()

    output = dict(data)
    output["fps"] = int(round(fps))
    output["root_pos"] = np.concatenate((root_pos_head, arrays["root_pos"][source_end_index + 1 :]), axis=0)
    output["root_rot"] = np.concatenate((rotation_head, arrays["root_rot"][source_end_index + 1 :]), axis=0)
    output["dof_pos"] = np.concatenate((dof_pos_head, arrays["dof_pos"][source_end_index + 1 :]), axis=0)

    for key in ("root_pos", "root_rot", "dof_pos"):
        if not np.isfinite(output[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    quat_error = float(np.max(np.abs(np.linalg.norm(output["root_rot"], axis=1) - 1.0)))
    if quat_error > 1.0e-8:
        raise ValueError(f"Quaternion normalization error: {quat_error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(output, stream)
    print(
        f"Saved {args.output}: {frame_count} cropped frames -> {len(output['root_pos'])} smoothed frames, "
        f"duration={(len(output['root_pos']) - 1) / fps:.3f}s"
    )


if __name__ == "__main__":
    main()
