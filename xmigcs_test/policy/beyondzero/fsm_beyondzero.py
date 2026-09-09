"""
BeyondZero FSM state
Moves the robot to the BeyondMimic default pose using smooth interpolation.
"""
import os
from typing import List

import numpy as np
import yaml
from types import SimpleNamespace
try:
    import onnx
except ImportError:  # pragma: no cover
    onnx = None
import onnxruntime

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData


class FSMStateBeyondZero(FSMState):
    """Zero pose specifically aligned with the BeyondMimic policy metadata."""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.BEYONDZERO
        self.q_factor = 0.0
        self.motor_nums = 29
        self.start_pose = np.zeros(self.motor_nums, dtype=np.float32)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "beyondzero.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.motor_nums: int = int(config["motor_nums"])
        self.locked_joint_map: List[int] = config["locked_joint_map"]
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.interp_step = float(config.get("interp_step", 0.001))
        self.interp_max = float(config.get("interp_max", 1.0))

        model_dir = os.path.join(os.path.dirname(current_dir), "beyond_mimic", "model")
        candidate_path = os.path.join(model_dir, config["onnx_path"])
        if not os.path.exists(candidate_path) and not candidate_path.endswith(".onnx"):
            alt_path = candidate_path + ".onnx"
            if os.path.exists(alt_path):
                candidate_path = alt_path
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"[FSMStateBeyondZero] ONNX file not found: {candidate_path}")
        self.onnx_path = candidate_path

        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.default_angles_lab = None
        metadata_props = self._collect_metadata_props()
        self._load_metadata(metadata_props)
        self._build_mappings()

        self.zero_target = self._build_zero_pose()

    def _collect_metadata_props(self):
        metadata_props = []
        if onnx is not None and hasattr(onnx, "load"):
            try:
                model = onnx.load(self.onnx_path)
                metadata_props = getattr(model, "metadata_props", [])
            except Exception as exc:
                print(f"[FSMStateBeyondZero] Failed to load metadata via onnx.load: {exc}")
        else:
            print("[FSMStateBeyondZero] Python onnx package unavailable, using onnxruntime metadata.")

        if not metadata_props:
            session = onnxruntime.InferenceSession(self.onnx_path)
            model_meta = session.get_modelmeta()
            custom_map = getattr(model_meta, "custom_metadata_map", {})
            metadata_props = [SimpleNamespace(key=k, value=v) for k, v in custom_map.items()]
        return metadata_props

    def _load_metadata(self, metadata_props):
        for prop in metadata_props:
            if prop.key == "joint_names":
                self.joint_seq = [name.strip() for name in prop.value.split(",")]
            elif prop.key == "default_joint_pos":
                self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")], dtype=np.float32)

        if self.joint_seq is None or self.joint_pos_array_seq is None:
            raise RuntimeError("[FSMStateBeyondZero] Missing joint metadata inside ONNX model.")

    def _build_mappings(self):
        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", 
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", 
        ]
        try:
            self.mj2lab = np.array([self.joint_xml.index(joint) for joint in self.joint_seq], dtype=np.int64)
        except ValueError as exc:
            raise RuntimeError(f"[FSMStateBeyondZero] Joint mapping error: {exc}") from exc

    def _build_zero_pose(self) -> np.ndarray:
        """Convert default pose from policy metadata to the mj joint ordering."""
        default_lab = np.zeros(len(self.locked_joint_map), dtype=np.float32)
        default_lab[self.mj2lab] = self.joint_pos_array_seq
        self.default_angles_lab = default_lab

        zero_pose = np.zeros(self.motor_nums, dtype=np.float32)
        zero_pose[self.locked_joint_map] = default_lab
        return zero_pose

    def on_enter(self):
        print("[FSMStateBeyondZero] Enter zero pose for BeyondMimic policy")
        self.q_factor = 0.0
        if self.robot_data_ is not None:
            self.start_pose = self.robot_data_.get_joint_pos().copy()
        else:
            self.start_pose = np.zeros(self.motor_nums, dtype=np.float32)

    def run(self, flag: ControlFlag):
        if self.robot_data_ is None:
            return

        target = self.zero_target
        if self.q_factor < self.interp_max:
            pos_cmd = (1.0 - self.q_factor) * self.start_pose + self.q_factor * target
            self.q_factor = min(self.q_factor + self.interp_step, self.interp_max)
        else:
            pos_cmd = target

        joint_start_idx = 35 - self.motor_nums
        self.robot_data_.q_d_[joint_start_idx:] = pos_cmd
        self.robot_data_.q_dot_d_[joint_start_idx:] = 0.0
        self.robot_data_.tau_d_[joint_start_idx:] = 0.0

        self.robot_data_.joint_kp_p_[:self.motor_nums] = self.kps
        self.robot_data_.joint_kd_p_[:self.motor_nums] = self.kds

    def on_exit(self):
        print("[FSMStateBeyondZero] Exit BeyondZero state")

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """Allow transitions to other FSM states."""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDMIMIC":
            return FSMStateName.BEYONDMIMIC
        elif flag.fsm_state_command == "gotoBEYONDZERO":
            return FSMStateName.BEYONDZERO
        else:
            return None
