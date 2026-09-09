"""
FSM State Implementations
Concrete implementations of different FSM states
"""
from typing import Dict

import numpy as np
import onnxruntime as ort

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
from scipy.spatial.transform import Rotation
import os
import yaml
import time
from policy.dh.tool_functions import quat_rotate_inverse_numpy, euler_to_quaternion_scipy
from policy.dh.dex_dh_agent import dex_dh_Agent

class FSMStateDH(FSMState):
    """DH策略状态实现"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "dh.yaml")
        with open(config_path, "r") as f:
            policy_config = yaml.safe_load(f)

        # Load configuration exactly like C++
        self.num_actions = policy_config.get("num_actions", 29)

        self.dt = 0.01
        self.physics_dt = policy_config.get("physics_dt", 0.002)
        self.last_run_time = time.perf_counter()
        # Read kp/kd gains
        self.kp = np.array(policy_config.get("kp_pos"))
        self.kd = np.array(policy_config.get("kd_pos"))
        self.decimation_ = round(self.dt / self.physics_dt)

        self.init_flag = False
        self.slow_start_max = 100.
        # self.file_logger = FileLogger(self.dt, "/home/ubuntu/MIG/xmigcs_10_16_1/xmigcs/policy/dh", "prop_obs")
        # self.file_logger2 = FileLogger(self.dt, "/home/ubuntu/MIG/xmigcs_10_16_1/xmigcs/policy/dh", "dof_pos_target")
        # self.file_logger3 = FileLogger(self.dt, "/home/ubuntu/MIG/xmigcs_10_16_1/xmigcs/policy/dh", "slow_start")

        self.agent = dex_dh_Agent()

    def on_enter(self):
        """进入DH状态"""
        print("[FSMStateDH] enter")
        # self.bootstrap()
        self.init_flag = True
        self.slow_start = self.slow_start_max

    def on_exit(self):
        """退出DH状态"""
        print("[FSMStateDH] exit")

    def run(self, flag: ControlFlag):
        """运行完成后给self.robot_data_赋值
        """
        # Only run policy inference every decimation_ steps
        # if int(self.robot_data_.time_now_ / self.dt) % self.decimation_ == 0:
        if int(self.robot_data_.time_now_ / self.physics_dt) % self.decimation_ == 0:
            current_time = time.perf_counter()
            print(f"Inference hz: {1/(current_time - self.last_run_time)}")
            self.last_run_time = current_time

            obs_group = self.get_observation_group(flag)
            if self.init_flag:
                self.init_flag = False
                self.agent.reset(obs_group)
            output_joint_pos = self.agent.inference(obs_group)
            self.write_to_robot_data(output_joint_pos)

        self.set_kp_kd()

    def set_kp_kd(self):
        # Set kp/kd gains
        self.robot_data_.joint_kp_p_[:self.num_actions] = self.kp
        self.robot_data_.joint_kd_p_[:self.num_actions] = self.kd

    def write_to_robot_data(self, output_joint_pos):
        # waist_yaw = output_joint_pos[12]
        # waist_roll = output_joint_pos[13]
        # waist_pitch = output_joint_pos[14]
        # output_joint_pos[12:15] = [waist_pitch,waist_roll,waist_yaw]

        # Set joint commands exactly like C++
        for i in range(self.num_actions):
            # C++: robot_data_->q_d_(35 - motor_num_ + i)
            joint_idx = 35 - self.num_actions + i
            self.robot_data_.q_d_[joint_idx] = output_joint_pos[i]
            self.robot_data_.q_dot_d_[joint_idx] = 0.0
            self.robot_data_.tau_d_[joint_idx] = 0.0

    def get_observation_group(self, flag: ControlFlag) -> Dict:
        joint_start_idx = 35 - self.num_actions
        dof_pos = self.robot_data_.q_a_[joint_start_idx:].astype(np.float32)
        dof_vel = self.robot_data_.q_dot_a_[joint_start_idx:].astype(np.float32)
        # waist_pitch = joint_pos[12]
        # waist_roll = joint_pos[13]
        # waist_yaw = joint_pos[14]
        # joint_pos[12] = waist_yaw
        # joint_pos[13] = waist_roll
        # joint_pos[14] = waist_pitch

        commands = np.array([
            flag.x_speed_command,
            flag.y_speed_command,
            flag.yaw_speed_command])

        ang_vel = np.array([
            self.robot_data_.imu_data_[3],
            self.robot_data_.imu_data_[4],
            self.robot_data_.imu_data_[5]])

        rpy = np.array([
            self.robot_data_.imu_data_[2],  # roll
            self.robot_data_.imu_data_[1],  # pitch
            self.robot_data_.imu_data_[0]   # yaw
        ], dtype=np.float32) * 1.0

        q_wxyz = euler_to_quaternion_scipy(rpy[0], rpy[1], rpy[2])
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        g = np.array([0., 0., -1.])
        projected_gravity = quat_rotate_inverse_numpy(q_xyzw, g)

        obs_group = {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "angular_velocity": ang_vel,
            "commands": commands,
            "projected_gravity": projected_gravity,
        }

        return obs_group

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoMLP":
            return FSMStateName.MLP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoDH":
            return FSMStateName.DH
        elif flag.fsm_state_command == "gotoDHZERO":
            return FSMStateName.DHZERO
        else:
            return None  # 无状态转换
