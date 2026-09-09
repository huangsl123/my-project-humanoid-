#!/usr/bin/env python3
"""Validate an RSL-RL checkpoint and save a machine-readable summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected_iteration", type=int, required=True)
    parser.add_argument("--expected_actor_obs", type=int, default=124)
    parser.add_argument("--expected_critic_obs", type=int, default=346)
    parser.add_argument("--expected_actions", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_audit(value: object, prefix: str = "") -> tuple[int, int, list[str]]:
    tensor_count = 0
    element_count = 0
    non_finite: list[str] = []
    if isinstance(value, torch.Tensor):
        tensor_count = 1
        element_count = value.numel()
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            non_finite.append(prefix or "<root>")
    elif isinstance(value, dict):
        for key, child in value.items():
            child_count, child_elements, child_non_finite = tensor_audit(
                child, f"{prefix}.{key}" if prefix else str(key)
            )
            tensor_count += child_count
            element_count += child_elements
            non_finite.extend(child_non_finite)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_count, child_elements, child_non_finite = tensor_audit(
                child, f"{prefix}[{index}]"
            )
            tensor_count += child_count
            element_count += child_elements
            non_finite.extend(child_non_finite)
    return tensor_count, element_count, non_finite


def normalizer_state(
    checkpoint: dict, model_state: dict, legacy_key: str, embedded_prefix: str
) -> tuple[str, dict[str, torch.Tensor] | None]:
    """Resolve legacy top-level, current embedded, or identity normalization."""
    if legacy_key in checkpoint:
        return "legacy_top_level", checkpoint[legacy_key]
    prefix = f"{embedded_prefix}."
    embedded = {
        name.removeprefix(prefix): value
        for name, value in model_state.items()
        if name.startswith(prefix)
    }
    if embedded:
        return "embedded_actor_critic", embedded
    return "identity", None


def normalizer_shape_matches(state: dict[str, torch.Tensor] | None, expected: int) -> bool:
    if state is None:
        return True
    required = {"_mean", "_var", "_std", "count"}
    return (
        required <= state.keys()
        and list(state["_mean"].shape) == [1, expected]
        and list(state["_var"].shape) == [1, expected]
        and list(state["_std"].shape) == [1, expected]
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required_keys = {"model_state_dict", "optimizer_state_dict", "iter"}
    tensor_count, tensor_element_count, non_finite = tensor_audit(checkpoint)
    model_state = checkpoint["model_state_dict"]
    actor_first = model_state["actor.0.weight"]
    actor_last = model_state["actor.6.weight"]
    critic_first = model_state["critic.0.weight"]
    optimizer_groups = checkpoint["optimizer_state_dict"]["param_groups"]
    actor_normalizer_mode, actor_normalizer = normalizer_state(
        checkpoint, model_state, "obs_norm_state_dict", "actor_obs_normalizer"
    )
    critic_normalizer_mode, critic_normalizer = normalizer_state(
        checkpoint,
        model_state,
        "privileged_obs_norm_state_dict",
        "critic_obs_normalizer",
    )
    checks = {
        "required_keys_present": required_keys <= checkpoint.keys(),
        "iteration_matches": int(checkpoint["iter"]) == args.expected_iteration,
        "all_floating_tensors_finite": not non_finite,
        "actor_observation_size_matches_expected": actor_first.shape[1] == args.expected_actor_obs,
        "critic_observation_size_matches_expected": critic_first.shape[1] == args.expected_critic_obs,
        "action_size_matches_expected": actor_last.shape[0] == args.expected_actions,
        "actor_normalizer_size_matches_expected": normalizer_shape_matches(
            actor_normalizer, args.expected_actor_obs
        ),
        "critic_normalizer_size_matches_expected": normalizer_shape_matches(
            critic_normalizer, args.expected_critic_obs
        ),
        "optimizer_has_one_parameter_group": len(optimizer_groups) == 1,
    }
    checks["all_passed"] = all(checks.values())
    document = {
        "checkpoint": str(checkpoint_path),
        "sha256": sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "iteration": int(checkpoint["iter"]),
        "top_level_keys": sorted(checkpoint.keys()),
        "tensor_count": tensor_count,
        "tensor_element_count": tensor_element_count,
        "non_finite_tensor_paths": non_finite,
        "model_tensor_count": len(model_state),
        "actor_observation_size": int(actor_first.shape[1]),
        "critic_observation_size": int(critic_first.shape[1]),
        "action_size": int(actor_last.shape[0]),
        "actor_normalizer_mode": actor_normalizer_mode,
        "critic_normalizer_mode": critic_normalizer_mode,
        "actor_normalizer_count": (
            float(actor_normalizer["count"]) if actor_normalizer is not None else None
        ),
        "critic_normalizer_count": (
            float(critic_normalizer["count"]) if critic_normalizer is not None else None
        ),
        "optimizer_learning_rates": [float(group["lr"]) for group in optimizer_groups],
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
