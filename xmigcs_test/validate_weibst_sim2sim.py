#!/usr/bin/env python3
"""Run an automated xSIM/xMIGCS closed-loop BeyondMimic acceptance test."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import json
import math
import os
import pty
import re
import select
import signal
import subprocess
import threading
import time
from pathlib import Path

import matplotlib
import numpy as np
import rclpy
from bodyctrl_msgs.msg import CmdMotorCtrl, MotorStatusMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SIM_ROOT = Path("/home/eai/my project/xSIM_MUJOCO")
CONTROL_ROOT = Path("/home/eai/my project/xmigcs_test")
DEPLOY_PYTHON = Path("/home/eai/Tools/miniforge3/envs/xmimic_deploy/bin/python")
WINDOW_TITLE = "MuJoCo : evt2"


class TelemetryMonitor(Node):
    """Collect independent ROS evidence from both sides of the sim2sim bridge."""

    def __init__(self) -> None:
        super().__init__("weibst_sim2sim_acceptance_monitor")
        self.lock = threading.Lock()
        self.started_at = time.monotonic()
        self.policy_started_at: float | None = None
        self.odom: list[dict[str, float]] = []
        self.command_counts = {"leg": 0, "arm": 0, "waist": 0}
        self.status_counts = {"leg": 0, "arm": 0, "waist": 0}
        self.create_subscription(Odometry, "/base/odometry", self._on_odom, 20)
        for group in self.command_counts:
            self.create_subscription(
                CmdMotorCtrl,
                f"/{group}/cmd_ctrl",
                lambda _msg, name=group: self._count(self.command_counts, name),
                20,
            )
            self.create_subscription(
                MotorStatusMsg,
                f"/{group}/status",
                lambda _msg, name=group: self._count(self.status_counts, name),
                20,
            )

    def _count(self, target: dict[str, int], name: str) -> None:
        with self.lock:
            target[name] += 1

    def _on_odom(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        velocity = msg.twist.twist.linear
        sample = {
            "wall_time_s": time.monotonic() - self.started_at,
            "policy_time_s": (
                time.monotonic() - self.policy_started_at
                if self.policy_started_at is not None
                else math.nan
            ),
            "x_m": float(position.x),
            "y_m": float(position.y),
            "z_m": float(position.z),
            "quat_x": float(orientation.x),
            "quat_y": float(orientation.y),
            "quat_z": float(orientation.z),
            "quat_w": float(orientation.w),
            "vx_mps": float(velocity.x),
            "vy_mps": float(velocity.y),
            "vz_mps": float(velocity.z),
        }
        with self.lock:
            self.odom.append(sample)

    def mark_policy_start(self) -> dict[str, int]:
        with self.lock:
            self.policy_started_at = time.monotonic()
            return dict(self.command_counts)

    def snapshot_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        with self.lock:
            return dict(self.command_counts), dict(self.status_counts)


class PtyProcess:
    """Spawn the keyboard controller on a real PTY and retain diagnostic lines."""

    IMPORTANT = re.compile(
        rb"Waiting for start|Start signal|Keyboard listener|Command:|FSM transition|"
        rb"weibst|BEYONDMIMIC|Traceback|Error|Exception|mismatch|failed",
        re.IGNORECASE,
    )

    def __init__(self, command: list[str], cwd: Path, env: dict[str, str], log_path: Path):
        self.master, slave = pty.openpty()
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            close_fds=True,
        )
        os.close(slave)
        self.stop_event = threading.Event()
        self.log_path = log_path
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        pending = b""
        with self.log_path.open("wb") as stream:
            while not self.stop_event.is_set():
                try:
                    ready, _, _ = select.select([self.master], [], [], 0.2)
                    if not ready:
                        if self.process.poll() is not None:
                            break
                        continue
                    chunk = os.read(self.master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                pending += chunk
                lines = pending.split(b"\n")
                pending = lines.pop()
                for line in lines:
                    if self.IMPORTANT.search(line):
                        stream.write(line + b"\n")
                        stream.flush()
            if pending and self.IMPORTANT.search(pending):
                stream.write(pending + b"\n")

    def send(self, value: bytes) -> None:
        os.write(self.master, value)

    def close(self) -> None:
        self.stop_event.set()
        if self.process.poll() is None:
            try:
                self.send(b"\x03")
            except OSError:
                pass
            try:
                self.process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
        self.thread.join(timeout=2)
        try:
            os.close(self.master)
        except OSError:
            pass


def spin_for(node: Node, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=min(0.05, deadline - time.monotonic()))


def wait_for(node: TelemetryMonitor, predicate, timeout_s: float, message: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    raise RuntimeError(message)


def find_mujoco_window(display: str, simulator_pid: int) -> str:
    env = {**os.environ, "DISPLAY": display}
    root = subprocess.check_output(
        ["xprop", "-root", "_NET_CLIENT_LIST"], env=env, text=True
    )
    window_ids = re.findall(r"0x[0-9a-fA-F]+", root)
    title_matches = []
    for window_id in window_ids:
        props = subprocess.run(
            ["xprop", "-id", window_id, "WM_NAME", "_NET_WM_NAME", "_NET_WM_PID"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
        if WINDOW_TITLE not in props:
            continue
        title_matches.append(window_id)
        pid_match = re.search(r"_NET_WM_PID\(CARDINAL\) = (\d+)", props)
        if pid_match and int(pid_match.group(1)) == simulator_pid:
            return window_id
    if len(title_matches) == 1:
        return title_matches[0]
    raise RuntimeError(
        f"Unable to uniquely find X11 window {WINDOW_TITLE!r} for PID {simulator_pid}; "
        f"title matches: {title_matches}"
    )


def wait_for_log(node: Node, path: Path, text: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if path.exists() and text in path.read_text(encoding="utf-8", errors="replace"):
            return
    raise RuntimeError(f"Controller log did not contain {text!r} within {timeout_s:.1f}s")


def send_escape_to_window(display_name: str, window_id: str) -> None:
    """Focus the MuJoCo GLFW window briefly and inject a real Escape key event."""
    x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
    xtst = ctypes.CDLL(ctypes.util.find_library("Xtst"))
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
    x11.XStringToKeysym.restype = ctypes.c_ulong
    x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XKeysymToKeycode.restype = ctypes.c_uint
    x11.XGetInputFocus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int)]
    x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    display = x11.XOpenDisplay(display_name.encode())
    if not display:
        raise RuntimeError(f"Unable to open X display {display_name}")
    previous = ctypes.c_ulong()
    revert_to = ctypes.c_int()
    try:
        x11.XGetInputFocus(display, ctypes.byref(previous), ctypes.byref(revert_to))
        target = int(window_id, 16)
        x11.XRaiseWindow(display, target)
        x11.XSetInputFocus(display, target, 1, 0)
        x11.XFlush(display)
        time.sleep(0.15)
        keycode = x11.XKeysymToKeycode(display, x11.XStringToKeysym(b"Escape"))
        xtst.XTestFakeKeyEvent(display, keycode, 1, 0)
        xtst.XTestFakeKeyEvent(display, keycode, 0, 0)
        x11.XFlush(display)
        time.sleep(0.15)
        if previous.value:
            x11.XSetInputFocus(display, previous.value, revert_to.value, 0)
            x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def save_telemetry_plot(rows: list[dict[str, float]], path: Path) -> None:
    policy_rows = [row for row in rows if math.isfinite(row["policy_time_s"])]
    if not policy_rows:
        return
    time_s = np.asarray([row["policy_time_s"] for row in policy_rows])
    x = np.asarray([row["x_m"] for row in policy_rows])
    y = np.asarray([row["y_m"] for row in policy_rows])
    z = np.asarray([row["z_m"] for row in policy_rows])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(time_s, z, linewidth=1.5, color="#2563eb")
    axes[0].axhline(0.45, color="#dc2626", linestyle="--", linewidth=1, label="final safety gate")
    axes[0].set(title="MuJoCo pelvis height", xlabel="Policy time (s)", ylabel="Height (m)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, y, linewidth=1.5, color="#059669")
    axes[1].scatter([x[0]], [y[0]], marker="o", color="#2563eb", label="start")
    axes[1].scatter([x[-1]], [y[-1]], marker="x", color="#dc2626", label="end")
    axes[1].set(title="MuJoCo pelvis XY trajectory", xlabel="X (m)", ylabel="Y (m)")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--domain_id", type=int, default=142)
    parser.add_argument("--duration_s", type=float, default=15.0)
    parser.add_argument("--prepare_s", type=float, default=3.0)
    parser.add_argument("--post-release-delay-s", type=float, default=0.15)
    parser.add_argument("--minimum-height-m", type=float, default=0.25)
    parser.add_argument(
        "--maximum-abs-axis-position-m",
        type=float,
        default=3.75,
        help="Maximum allowed absolute pelvis X or Y position during policy execution.",
    )
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--variant", default="weibst_hard")
    parser.add_argument(
        "--model",
        type=Path,
        default=CONTROL_ROOT / "policy/beyond_mimic/model/weibst_hard_best.onnx",
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        raise ValueError("--duration_s must be positive")
    if args.maximum_abs_axis_position_m <= 0.0:
        raise ValueError("--maximum-abs-axis-position-m must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model.expanduser().resolve()
    if not model.is_file() or model.stat().st_size == 0:
        raise FileNotFoundError(f"Validated deployment model is missing: {model}")

    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(args.domain_id)
    env["DISPLAY"] = args.display
    env["XMIGCS_BEYONDMIMIC_VARIANT"] = args.variant
    start_signal = Path(f"/tmp/rl_start_signal_{args.domain_id}_{os.getpid()}")
    env["XMIGCS_START_SIGNAL"] = str(start_signal)
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ["DISPLAY"] = args.display
    if start_signal.exists():
        start_signal.unlink()

    rclpy.init()
    monitor = TelemetryMonitor()
    simulator_log = (output_dir / "simulator.log").open("wb")
    simulator = subprocess.Popen(
        [str(DEPLOY_PYTHON), "scripts/simulator_view_asyn.py", "-m", "evt2"],
        cwd=SIM_ROOT,
        env=env,
        stdout=simulator_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    controller: PtyProcess | None = None
    result: dict[str, object] = {}
    error: str | None = None
    try:
        wait_for(
            monitor,
            lambda: len(monitor.odom) >= 10 and all(v > 0 for v in monitor.status_counts.values()),
            20.0,
            "xSIM did not publish odometry and all motor status groups",
        )
        window_id = find_mujoco_window(args.display, simulator.pid)
        controller_log = output_dir / "controller_events.log"
        controller = PtyProcess(
            [str(DEPLOY_PYTHON), "rl_control_node.py"],
            CONTROL_ROOT,
            env,
            controller_log,
        )
        wait_for_log(monitor, controller_log, "Waiting for start signal", 30.0)
        wait_for_log(monitor, controller_log, "Keyboard listener ready", 10.0)
        start_signal.touch()
        wait_for_log(monitor, controller_log, "Start signal received", 10.0)
        controller.send(b"b")
        wait_for_log(monitor, controller_log, "BEYONDZERO", 10.0)
        spin_for(monitor, args.prepare_s)
        send_escape_to_window(args.display, window_id)
        spin_for(monitor, args.post_release_delay_s)
        counts_at_policy_start = monitor.mark_policy_start()
        controller.send(b";")
        spin_for(monitor, args.duration_s)

        command_counts, status_counts = monitor.snapshot_counts()
        policy_samples = [row for row in monitor.odom if math.isfinite(row["policy_time_s"])]
        values = np.asarray(
            [
                [
                    row["x_m"], row["y_m"], row["z_m"],
                    row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"],
                    row["vx_mps"], row["vy_mps"], row["vz_mps"],
                ]
                for row in policy_samples
            ],
            dtype=np.float64,
        )
        quaternion_norms = np.linalg.norm(values[:, 3:7], axis=1) if values.size else np.asarray([])
        # World-Z component of the pelvis local up axis.  1.0 is upright,
        # 0.0 is lying on a side, and -1.0 is upside down.
        up_z = (
            1.0 - 2.0 * (values[:, 3] ** 2 + values[:, 4] ** 2)
            if values.size
            else np.asarray([])
        )
        command_deltas = {
            name: command_counts[name] - counts_at_policy_start[name]
            for name in command_counts
        }
        controller_text = (output_dir / "controller_events.log").read_text(
            encoding="utf-8", errors="replace"
        )
        checks = {
            "simulator_alive_through_test": simulator.poll() is None,
            "controller_alive_through_test": controller.process.poll() is None,
            "all_motor_status_groups_received": all(value >= 100 for value in status_counts.values()),
            "all_motor_command_groups_received_during_policy": all(
                value >= int(args.duration_s * 50) for value in command_deltas.values()
            ),
            "complete_duration_observed": bool(
                policy_samples and policy_samples[-1]["policy_time_s"] >= args.duration_s - 0.25
            ),
            "sufficient_odometry_rate": len(policy_samples) >= int(args.duration_s * 50),
            "all_base_state_finite": bool(values.size and np.isfinite(values).all()),
            "base_not_collapsed_at_end": bool(values.size and values[-1, 2] > 0.45),
            "base_never_below_safety_floor": bool(
                values.size and np.min(values[:, 2]) > args.minimum_height_m
            ),
            "base_remained_within_platform_bounds": bool(
                values.size
                and np.max(np.abs(values[:, :2])) < args.maximum_abs_axis_position_m
            ),
            "base_orientation_finite_and_normalized": bool(
                quaternion_norms.size
                and np.isfinite(quaternion_norms).all()
                and np.max(np.abs(quaternion_norms - 1.0)) < 1.0e-3
            ),
            "base_upright_at_end": bool(up_z.size and up_z[-1] > 0.5),
            "requested_variant_entered": (
                f"Command: gotoBEYONDMIMIC ({args.variant})" in controller_text
                and bool(re.search(r"FSM transition: .* -> BEYONDMIMIC", controller_text))
            ),
            "no_controller_exception": not re.search(
                r"Traceback|Observation length mismatch|\bException\b", controller_text, re.IGNORECASE
            ),
        }
        result = {
            "domain_id": args.domain_id,
            "variant": args.variant,
            "model": str(model),
            "display": args.display,
            "duration_s": args.duration_s,
            "minimum_height_m": args.minimum_height_m,
            "maximum_abs_axis_position_limit_m": args.maximum_abs_axis_position_m,
            "window_id": window_id,
            "command_counts_total": command_counts,
            "command_counts_during_policy": command_deltas,
            "status_counts_total": status_counts,
            "odometry_samples_during_policy": len(policy_samples),
            "base_height_m": {
                "initial": float(values[0, 2]) if values.size else None,
                "minimum": float(np.min(values[:, 2])) if values.size else None,
                "final": float(values[-1, 2]) if values.size else None,
            },
            "base_up_axis_world_z": {
                "minimum": float(np.min(up_z)) if up_z.size else None,
                "final": float(up_z[-1]) if up_z.size else None,
            },
            "quaternion_norm_max_error": (
                float(np.max(np.abs(quaternion_norms - 1.0)))
                if quaternion_norms.size else None
            ),
            "maximum_xy_radius_m": (
                float(np.max(np.linalg.norm(values[:, :2], axis=1))) if values.size else None
            ),
            "maximum_abs_axis_position_m": (
                float(np.max(np.abs(values[:, :2]))) if values.size else None
            ),
            "checks": checks,
            "passed": all(checks.values()),
        }
    except Exception as exc:  # preserve telemetry and logs for post-mortem
        error = f"{type(exc).__name__}: {exc}"
        result = {"domain_id": args.domain_id, "error": error, "passed": False}
    finally:
        with (output_dir / "telemetry.csv").open("w", encoding="utf-8", newline="") as stream:
            fieldnames = list(monitor.odom[0]) if monitor.odom else ["wall_time_s"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(monitor.odom)
        save_telemetry_plot(monitor.odom, output_dir / "telemetry.png")
        if controller is not None:
            controller.close()
        terminate_process(simulator)
        simulator_log.close()
        monitor.destroy_node()
        rclpy.shutdown()
        if start_signal.exists():
            start_signal.unlink()

    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if error or not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
