from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort
import yaml
from joblib import load as joblib_load
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class MotionReference:
    anchor_pos: np.ndarray
    anchor_ori: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray


class PBHCPolicy29:
    """
    ONNX policy wrapper aligned with the new 29‑DOF PBHC observation layout:

        obs = [actions(29), base_ang_vel(3), dof_pos(29), dof_vel(29),
               generated_commands(67), history_actor(372), projected_gravity(3)]

    ``generated_commands`` = anchor_pos(3) + anchor_ori(6) + ref_joint_pos(29) + ref_joint_vel(29).
    """

    def __init__(
        self,
        config_path: str,
        model_path: Optional[str] = None,
        motion_data_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ) -> None:
        cfg_path = Path(config_path)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # --- basic config ---
        self.num_actions: int = int(cfg.get("num_actions", 29))
        self.motor_num: int = int(cfg.get("motor_num", self.num_actions))
        self.generated_command_dim: int = int(cfg.get("generated_command_dim", 2 * self.num_actions + 9))
        self.obs_history_len: int = int(cfg.get("obs_history_len", 4))
        self.num_obs: int = int(
            cfg.get(
                "num_obs",
                self.num_actions  # previous actions
                + 3               # base ang vel
                + self.num_actions  # dof pos
                + self.num_actions  # dof vel
                + self.generated_command_dim
                + (self.obs_history_len * (self.num_actions * 3 + 3 * 2))  # history_actor
                + 3               # projected gravity
            )
        )

        self.dt: float = float(cfg.get("dt", 0.01))
        self.action_scale: float = float(cfg.get("action_scale", 0.25))
        self.ang_vel_scale: float = float(cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale: float = float(cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale: float = float(cfg.get("dof_vel_scale", 0.05))

        default_angles_cfg = cfg.get("default_angles", [0.0] * self.num_actions)
        if len(default_angles_cfg) < self.num_actions:
            raise ValueError("default_angles 列表长度不足以覆盖全部关节")
        self.default_angles = np.asarray(default_angles_cfg, dtype=np.float32)[: self.num_actions]

        # Motion meta
        self.motion_cycle_time: float = float(cfg.get("motion_cycle_time", 12.54))
        motion_file_cfg = cfg.get("motion_file")
        if motion_data_path is not None:
            motion_file_cfg = motion_data_path
        if motion_file_cfg is None:
            raise ValueError("必须在配置或参数中提供 motion_file")

        self.motion_path = (cfg_path.parent / motion_file_cfg).resolve()
        if not self.motion_path.exists():
            raise FileNotFoundError(f"motion 数据文件不存在: {self.motion_path}")

        # Load motion
        self._load_motion_data(cfg)

        # Histories (newest first)
        H = self.obs_history_len
        A = self.num_actions
        self.history_action = np.zeros((H, A), dtype=np.float32)
        self.history_omega = np.zeros((H, 3), dtype=np.float32)
        self.history_qj = np.zeros((H, A), dtype=np.float32)
        self.history_dqj = np.zeros((H, A), dtype=np.float32)
        self.history_gravity = np.zeros((H, 3), dtype=np.float32)
        self.last_action = np.zeros(A, dtype=np.float32)

        # ONNX session
        onnx_path = model_path or cfg.get("model_path")
        if onnx_path is None:
            raise ValueError("缺少 model_path")
        onnx_path = str((cfg_path.parent / onnx_path).resolve())
        if use_gpu is None:
            use_gpu = ort.get_device() == "GPU"
        providers: Sequence[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[PBHCPolicy29] ONNX loaded: {onnx_path} | providers={providers}")

    # ------------------------------------------------------------------
    # Motion handling
    # ------------------------------------------------------------------

    def _load_motion_data(self, cfg: dict) -> None:
        raw = joblib_load(self.motion_path)
        if isinstance(raw, dict) and len(raw) == 1:
            motion = next(iter(raw.values()))
        elif isinstance(raw, dict):
            motion = raw
        else:
            raise TypeError("motion 数据格式不支持")

        self.motion_root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float32)
        self.motion_root_rot = np.asarray(motion["root_rot"], dtype=np.float32)  # [T,4] xyzw
        self.motion_joint_pos = np.asarray(motion["dof"], dtype=np.float32)      # [T, 29]
        self.motion_len = self.motion_root_pos.shape[0]
        if "fps" in motion:
            self.motion_dt = 1.0 / float(motion["fps"])
        else:
            self.motion_dt = float(cfg.get("motion_dt", 0.033))

        # Velocities via finite-difference
        self.motion_joint_vel = np.gradient(self.motion_joint_pos, self.motion_dt, axis=0, edge_order=2).astype(np.float32)

        times = np.linspace(0.0, 1.0, self.motion_len)
        rotations = Rotation.from_quat(self.motion_root_rot)
        self.slerp = Slerp(times, rotations)

    def _sample_reference(self, phase: float) -> MotionReference:
        phase = float(np.clip(phase, 0.0, 1.0))
        idx_float = phase * (self.motion_len - 1)
        idx0 = int(np.floor(idx_float))
        idx1 = min(idx0 + 1, self.motion_len - 1)
        w = idx_float - idx0

        def lerp(arr: np.ndarray) -> np.ndarray:
            return (1.0 - w) * arr[idx0] + w * arr[idx1]

        anchor_pos = lerp(self.motion_root_pos)
        anchor_rot_mat = self.slerp([phase]).as_matrix()[0]  # 3x3
        anchor_ori = anchor_rot_mat[:, :2].reshape(-1)  # first two columns (6 dims)

        joint_pos = lerp(self.motion_joint_pos)
        joint_vel = lerp(self.motion_joint_vel)

        return MotionReference(
            anchor_pos.astype(np.float32),
            anchor_ori.astype(np.float32),
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Policy API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.history_action.fill(0.0)
        self.history_omega.fill(0.0)
        self.history_qj.fill(0.0)
        self.history_dqj.fill(0.0)
        self.history_gravity.fill(0.0)
        self.last_action[:] = 0.0

    def build_observation(
        self,
        omega_3: np.ndarray,
        qj_29: np.ndarray,
        dqj_29: np.ndarray,
        gravity_3: np.ndarray,
        motion_phase: float,
        anchor_pos_3: Optional[np.ndarray] = None,
        anchor_ori_6: Optional[np.ndarray] = None,
        ref_joint_pos_29: Optional[np.ndarray] = None,
        ref_joint_vel_29: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if (
            anchor_pos_3 is None
            or anchor_ori_6 is None
            or ref_joint_pos_29 is None
            or ref_joint_vel_29 is None
        ):
            ref = self._sample_reference(motion_phase)
            anchor_pos_3 = ref.anchor_pos
            anchor_ori_6 = ref.anchor_ori
            ref_joint_pos_29 = ref.joint_pos
            ref_joint_vel_29 = ref.joint_vel

        qj_29 = np.asarray(qj_29, dtype=np.float32)
        dqj_29 = np.asarray(dqj_29, dtype=np.float32)

        qj_n = (qj_29 - self.default_angles) * self.dof_pos_scale
        dqj_n = dqj_29 * self.dof_vel_scale
        omega_n = np.asarray(omega_3, dtype=np.float32) * self.ang_vel_scale
        gravity = np.asarray(gravity_3, dtype=np.float32)

        generated_commands = np.concatenate(
            [
                np.asarray(anchor_pos_3, dtype=np.float32).reshape(-1),
                np.asarray(anchor_ori_6, dtype=np.float32).reshape(-1),
                np.asarray(ref_joint_pos_29, dtype=np.float32).reshape(-1),
                np.asarray(ref_joint_vel_29, dtype=np.float32).reshape(-1),
            ]
        )
        if generated_commands.shape[0] != self.generated_command_dim:
            raise ValueError(
                f"generated_commands 维度不匹配：期望 {self.generated_command_dim}，得到 {generated_commands.shape[0]}"
            )

        # Update histories (newest first)
        action_t = self.last_action.astype(np.float32, copy=True)
        omega_t = omega_n.astype(np.float32, copy=True)
        qj_t = qj_n.astype(np.float32, copy=True)
        dqj_t = dqj_n.astype(np.float32, copy=True)
        grav_t = gravity.astype(np.float32, copy=True)

        def _push_history(buffer: np.ndarray, value: np.ndarray) -> None:
            buffer[1:] = buffer[:-1]
            buffer[0] = value

        _push_history(self.history_action, action_t)
        _push_history(self.history_omega, omega_t)
        _push_history(self.history_qj, qj_t)
        _push_history(self.history_dqj, dqj_t)
        _push_history(self.history_gravity, grav_t)

        history_flat = np.concatenate(
            [
                self.history_action.reshape(-1),
                self.history_omega.reshape(-1),
                self.history_qj.reshape(-1),
                self.history_dqj.reshape(-1),
                self.history_gravity.reshape(-1),
            ]
        ).astype(np.float32)

        obs = np.concatenate(
            [
                action_t,
                omega_t,
                qj_t,
                dqj_t,
                generated_commands,
                history_flat,
                grav_t,
            ]
        ).astype(np.float32)

        if obs.shape[0] != self.num_obs:
            raise RuntimeError(f"观察维度装填错误：得到 {obs.shape[0]}，期望 {self.num_obs}")
        return obs

    def infer(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: obs.reshape(1, -1).astype(np.float32)})
        action = out[0].squeeze().astype(np.float32)
        self.last_action = action
        return action

    def action_to_targets(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_scale + self.default_angles

    def step(
        self,
        omega_3: np.ndarray,
        qj_29: np.ndarray,
        dqj_29: np.ndarray,
        gravity_3: np.ndarray,
        motion_phase: float,
        return_raw: bool = False,
    ):
        obs = self.build_observation(omega_3, qj_29, dqj_29, gravity_3, motion_phase)
        raw = self.infer(obs)
        targets = self.action_to_targets(raw)
        return (targets, raw) if return_raw else targets


__all__ = ["PBHCPolicy29"]
