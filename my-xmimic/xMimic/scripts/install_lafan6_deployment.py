#!/usr/bin/env python3
"""Install one selected LAFAN6 ONNX policy into xMIGCS atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument(
        "--control-root",
        type=Path,
        default=Path("/home/eai/my project/xmigcs_test"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    onnx = args.onnx.expanduser().resolve()
    motion = args.motion_file.expanduser().resolve()
    control_root = args.control_root.expanduser().resolve()
    if not onnx.is_file() or onnx.stat().st_size == 0:
        raise FileNotFoundError(f"Selected ONNX is missing: {onnx}")
    if not motion.is_file() or motion.stat().st_size == 0:
        raise FileNotFoundError(f"Motion file is missing: {motion}")

    with np.load(motion, allow_pickle=True) as data:
        motion_length = int(len(data["joint_pos"]))
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if data["joint_pos"].shape[1] != 23:
            raise ValueError(f"Expected 23 policy joints, got {data['joint_pos'].shape}")
        if not np.isfinite(data["joint_pos"]).all():
            raise ValueError("Motion joint positions contain non-finite values")

    config_dir = control_root / "policy/beyond_mimic/config"
    model_dir = control_root / "policy/beyond_mimic/model"
    template_path = config_dir / "BeyondMimic_weibst_hard.yaml"
    with template_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model_name = f"{args.variant}_best.onnx"
    config_name = f"BeyondMimic_{args.variant}.yaml"
    model_path = model_dir / model_name
    config_path = config_dir / config_name
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = model_dir / f".{model_name}.tmp"
    shutil.copy2(onnx, temporary_model)
    if sha256(temporary_model) != sha256(onnx):
        temporary_model.unlink(missing_ok=True)
        raise RuntimeError("ONNX copy hash mismatch")
    os.replace(temporary_model, model_path)

    config["onnx_path"] = model_name
    config["num_obs"] = 130
    config["num_actions"] = 23
    config["motor_nums"] = 29
    config["state_feedback_source"] = "measured"
    config["warm_start_time"] = 0.5
    config["hold_final_reference"] = True
    config["motion_length"] = motion_length
    config["physical_dt"] = 0.01
    config["decimation"] = 1
    atomic_yaml(config_path, config)

    main_config_path = control_root / "config/dex_config.yaml"
    with main_config_path.open(encoding="utf-8") as stream:
        main_config = yaml.safe_load(stream)
    variants = main_config.setdefault("beyond_mimic_variants", [])
    variants[:] = [item for item in variants if item.get("name") != args.variant]
    variants.append(
        {
            "name": args.variant,
            "config_path": f"policy/beyond_mimic/config/{config_name}",
        }
    )
    atomic_yaml(main_config_path, main_config)

    result = {
        "variant": args.variant,
        "source_onnx": str(onnx),
        "installed_onnx": str(model_path),
        "onnx_sha256": sha256(model_path),
        "config": str(config_path),
        "main_config": str(main_config_path),
        "motion_file": str(motion),
        "motion_length": motion_length,
        "fps": fps,
        "duration_s": (motion_length - 1) / fps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
