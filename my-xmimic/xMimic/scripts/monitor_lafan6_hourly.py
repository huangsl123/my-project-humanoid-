#!/usr/bin/env python3
"""Persist an hourly LAFAN6 pipeline status snapshot while the goal is active."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPO = Path("/home/eai/my project/my-xmimic/xMimic")
ROOT = REPO / "artifacts/2026-08-07_lafan6_full_pipeline"
OUTPUT = ROOT / "hourly_status"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout.strip()


def active_processes() -> list[dict[str, object]]:
    raw = command("ps", "-eo", "pid=,etimes=,stat=,%cpu=,%mem=,args=")
    wanted = (
        "scripts/rsl_rl/train.py",
        "scripts/rsl_rl/play.py",
        "run_strict_remediation_queue.sh",
        "run_strict_remediation_guard.sh",
        "validate_weibst_sim2sim.py",
    )
    records = []
    for line in raw.splitlines():
        if not any(token in line for token in wanted) or "monitor_lafan6_hourly.py" in line:
            continue
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)", line)
        if not match:
            continue
        pid, elapsed, state, cpu, memory, args = match.groups()
        records.append(
            {
                "pid": int(pid),
                "elapsed_s": int(elapsed),
                "state": state,
                "cpu_percent": float(cpu),
                "memory_percent": float(memory),
                "command": args,
            }
        )
    return records


def training_state(processes: list[dict[str, object]]) -> dict[str, object] | None:
    training = next((p for p in processes if "scripts/rsl_rl/train.py" in str(p["command"])), None)
    if training is None:
        return None
    args = str(training["command"])

    def option(name: str) -> str | None:
        match = re.search(rf"(?:^|\s)--{re.escape(name)}\s+(\S+)", args)
        return match.group(1) if match else None

    run_label = option("run_name")
    run_dir = None
    if run_label:
        matches = sorted(
            (REPO / "logs/rsl_rl/dex_evt_flat").glob(f"*_{run_label}"),
            key=lambda path: path.stat().st_mtime,
        )
        if matches:
            run_dir = matches[-1]
    state: dict[str, object] = {
        **training,
        "task": option("task"),
        "run_name": run_label,
        "max_iterations_argument": option("max_iterations"),
        "motion_file": option("motion_file"),
    }
    if run_dir is None:
        return state
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if checkpoints:
        step, path = max(checkpoints)
        state.update(latest_checkpoint_step=step, latest_checkpoint=str(path))
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags()["scalars"])
        metrics = {}
        for tag in (
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Episode_Termination/time_out",
            "Metrics/motion/error_anchor_pos",
            "Metrics/motion/error_anchor_rot",
            "Metrics/motion/error_body_pos",
            "Metrics/motion/error_joint_pos",
            "Perf/total_fps",
        ):
            if tag in scalar_tags:
                event = accumulator.Scalars(tag)[-1]
                metrics[tag] = {"step": event.step, "value": event.value}
        state["metrics"] = metrics
    except Exception as error:  # status collection must never stop the pipeline
        state["metrics_error"] = f"{type(error).__name__}: {error}"
    return state


def action_status() -> list[dict[str, object]]:
    records = []
    for action_dir in sorted(ROOT.glob("[0-9][0-9]_*")):
        comparison = action_dir / "final_evaluation_comparison/evaluation_comparison.json"
        record: dict[str, object] = {
            "action": action_dir.name,
            "pipeline_complete": (action_dir / "TRAINING_EVALUATION_COMPLETE").exists(),
            "strict_remediation_complete": (action_dir / "STRICT_REMEDIATION_COMPLETE").exists(),
            "strict_remediation_exhausted": (action_dir / "STRICT_REMEDIATION_EXHAUSTED").exists(),
        }
        if comparison.exists():
            data = json.loads(comparison.read_text(encoding="utf-8"))
            record.update(
                strict_pass_count=data.get("pass_count"),
                strict_evaluation_count=data.get("evaluation_count"),
                failed_cases=data.get("failed", []),
            )
        records.append(record)
    return records


def snapshot() -> dict[str, object]:
    now = datetime.now(TIMEZONE)
    processes = active_processes()
    failures = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and ("FAILED" in path.name or "RESTART_LIMIT" in path.name or "EXHAUSTED" in path.name)
    )
    gpu_raw = command(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    gpu = []
    for line in gpu_raw.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 5:
            gpu.append(
                dict(
                    index=int(values[0]),
                    utilization_percent=int(values[1]),
                    memory_used_mib=int(values[2]),
                    memory_total_mib=int(values[3]),
                    temperature_c=int(values[4]),
                )
            )
    actions = action_status()
    return {
        "generated_at": now.isoformat(),
        "strict_pass_count": sum(int(x.get("strict_pass_count") or 0) for x in actions),
        "strict_evaluation_count": sum(int(x.get("strict_evaluation_count") or 0) for x in actions),
        "strict_success_marker": (ROOT / "LAFAN6_STRICT_SUCCESS").exists(),
        "review_required_marker": (ROOT / "LAFAN6_REVIEW_REQUIRED").exists(),
        "remediation_queue_complete": (ROOT / "STRICT_REMEDIATION_QUEUE_COMPLETE").exists(),
        "failure_markers": failures,
        "actions": actions,
        "training": training_state(processes),
        "active_processes": processes,
        "gpu": gpu,
    }


def write_snapshot(data: dict[str, object]) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(TIMEZONE).strftime("%Y-%m-%d_%H-%M-%S")
    destination = OUTPUT / f"status_{stamp}.json"
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    latest_tmp = OUTPUT / "latest.tmp"
    latest_tmp.write_text(payload, encoding="utf-8")
    os.replace(latest_tmp, OUTPUT / "latest.json")
    with (OUTPUT / "monitor.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"{data['generated_at']} strict={data['strict_pass_count']}/{data['strict_evaluation_count']} "
            f"training={bool(data['training'])} failures={len(data['failure_markers'])}\n"
        )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_stream = (OUTPUT / "monitor.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    (OUTPUT / "monitor.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    elapsed = args.interval_seconds
    while True:
        if elapsed >= args.interval_seconds:
            write_snapshot(snapshot())
            elapsed = 0
        if not args.watch:
            return
        if (ROOT / "LAFAN6_STRICT_SUCCESS").exists():
            if elapsed:
                write_snapshot(snapshot())
            return
        time.sleep(60)
        elapsed += 60


if __name__ == "__main__":
    main()
