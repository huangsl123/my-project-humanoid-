"""
FSM State Implementations
Concrete implementations of different FSM states
"""
from typing import Dict
from enum import Enum, auto

import numpy as np
import onnxruntime as ort

from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
import math
import onnxruntime
import os
import yaml
from scipy.spatial.transform import Rotation


class JointOrderMode(Enum):
    """关节顺序模式"""
    GYM = auto()  # 原始gym顺序（配置文件中的顺序）
    LAB = auto()  # lab顺序（推理使用的顺序）


class FSMSGetup(FSMState):
    """MLP策略状态实现 - 支持GYM/LAB顺序切换"""
    
    # ========== 顺序切换开关 ==========
    USE_ORDER_MODE: JointOrderMode = JointOrderMode.LAB  # 设置为LAB使用lab顺序，GYM使用gym顺序
    
    # ========== GYM顺序（23自由度）==========
    GYM_JOINT_ORDER = [
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
    
    # ========== LAB顺序（23自由度）==========
    LAB_JOINT_ORDER = [
        "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
        "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
        "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
        "knee_pitch_l_joint", "knee_pitch_r_joint", "shoulder_pitch_l_joint",
        "shoulder_pitch_r_joint", "ankle_pitch_l_joint", "ankle_pitch_r_joint",
        "shoulder_roll_l_joint", "shoulder_roll_r_joint", "ankle_roll_l_joint",
        "ankle_roll_r_joint", "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
        "elbow_pitch_l_joint", "elbow_pitch_r_joint",
    ]

    # 【修改1】新增：29自由度GYM顺序（在23基础上增加6个手腕关节）
    GYM_JOINT_ORDER_29 = [
        "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
        "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
        "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
        "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
        "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
        "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
    ]
    
    # 【修改2】新增：29自由度LAB顺序（在23基础上增加6个手腕关节）
    LAB_JOINT_ORDER_29 = [
        "hip_pitch_l_joint", "hip_pitch_r_joint", "waist_yaw_joint",
        "hip_roll_l_joint", "hip_roll_r_joint", "waist_roll_joint",
        "hip_yaw_l_joint", "hip_yaw_r_joint", "waist_pitch_joint",
        "knee_pitch_l_joint", "knee_pitch_r_joint", "shoulder_pitch_l_joint",
        "shoulder_pitch_r_joint", "ankle_pitch_l_joint", "ankle_pitch_r_joint",
        "shoulder_roll_l_joint", "shoulder_roll_r_joint", "ankle_roll_l_joint",
        "ankle_roll_r_joint", "shoulder_yaw_l_joint", "shoulder_yaw_r_joint",
        "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
        "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
    ]

    def _reset_internal_state(self):
        """把所有随时间变化的内部状态重置成初始值"""
        # 1) 清空 obs / hist / actions
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)

        # 2) 标志位重置
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True

        # 3) 期望关节 / 期望速度 / 力矩重置为"初始姿态"
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        # 期望角 = 初始角（使用当前模式的顺序）
        self.robot_data_.q_d_[base:base + len(self.joint_xml)] = self.joint_pos_array
        # 期望速度 = 0
        self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
        # 期望力矩 = 0（位置控制）
        self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "getup_config.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
            
        # Load configuration exactly like C++
        self.action_num_ = policy_config.get('actions_size')
        self.motor_num_ = policy_config.get('motor_num')
        self.dt_ = policy_config.get('dt')

        # Size configuration
        size_config = policy_config.get('size', {})
        self.num_hist_ = size_config.get('num_hist')
        self.obs_size_ = size_config.get('observations_size')

        # Control configuration
        control_config = policy_config.get('control', {})
        self.action_scale_ = control_config.get('action_scale')
        self.decimation_ = control_config.get('decimation')
        self.warm_start_time_ = control_config.get('warm_start_time', 0.3)

        # Normalization configuration
        norm_config = policy_config.get('normalization', {})
        clip_config = norm_config.get('clip_scales', {})
        obs_config = norm_config.get('obs_scales', {})

        self.clip_obs_ = clip_config.get('clip_observations', 100.0)
        self.clip_act_ = clip_config.get('clip_actions', 100.0)
        self.lin_vel_scale_ = obs_config.get('lin_vel')
        self.ang_vel_scale_ = obs_config.get('ang_vel')
        self.dof_pos_scale_ = obs_config.get('dof_pos')
        self.dof_vel_scale_ = obs_config.get('dof_vel')
        
        # ========== 顺序切换核心逻辑 ==========
        
        # 【修改3】根据action_num_选择对应的关节顺序（23或29）
        if self.action_num_ == 23:
            self.GYM_JOINT_ORDER = FSMSGetup.GYM_JOINT_ORDER
            self.LAB_JOINT_ORDER = FSMSGetup.LAB_JOINT_ORDER
            print(f"[FSMStateMLP] Configured for 23-DOF")
        elif self.action_num_ == 29:
            self.GYM_JOINT_ORDER = FSMSGetup.GYM_JOINT_ORDER_29
            self.LAB_JOINT_ORDER = FSMSGetup.LAB_JOINT_ORDER_29
            print(f"[FSMStateMLP] Configured for 29-DOF")
        else:
            raise ValueError(f"[FSMStateMLP] Unsupported action_num: {self.action_num_}, must be 23 or 29")
        
        # 从配置文件读取（始终是GYM顺序的值，长度需与action_num_匹配或为29）
        gym_stiffness = list(policy_config.get('gains')["kp"])
        gym_damping = list(policy_config.get('gains')["kd"])
        gym_init_state = policy_config.get('init_state')  # dict: joint_name -> value
        gym_action_scales = list(policy_config.get('action_scales'))
        
        # 构建映射表：lab索引 -> gym索引
        self.lab2gym_idx = np.array([
            self.GYM_JOINT_ORDER.index(joint) for joint in self.LAB_JOINT_ORDER
        ], dtype=int)
        # 反向映射：gym索引 -> lab索引
        self.gym2lab_idx = np.array([
            self.LAB_JOINT_ORDER.index(joint) for joint in self.GYM_JOINT_ORDER
        ], dtype=int)
        
        # 根据模式选择当前使用的顺序和数组
        if self.USE_ORDER_MODE == JointOrderMode.LAB:
            # LAB模式：使用LAB顺序，值通过映射从GYM配置转换
            self.joint_seq = np.array(self.LAB_JOINT_ORDER)
            self.joint_pos_array_seq = np.array([
                gym_init_state[joint] for joint in self.LAB_JOINT_ORDER
            ])
            print(f"default joint pos is: {self.joint_pos_array_seq}")
            self.stiffness_array_seq = np.array([gym_stiffness[i] for i in self.lab2gym_idx])
            self.damping_array_seq = np.array([gym_damping[i] for i in self.lab2gym_idx])
            self.action_scale = np.array([gym_action_scales[i] for i in self.lab2gym_idx])
            print(f"[FSMStateMLP] Using LAB order ({self.action_num_} joints)")
        else:
            # GYM模式：直接使用配置文件中的GYM顺序
            self.joint_seq = np.array(self.GYM_JOINT_ORDER)
            self.joint_pos_array_seq = np.array([
                gym_init_state[joint] for joint in self.GYM_JOINT_ORDER
            ])
            # 【修改4】GYM模式下也只取前action_num_个，避免配置是29维但模式是23时越界
            self.stiffness_array_seq = np.array(gym_stiffness[:self.action_num_])
            self.damping_array_seq = np.array(gym_damping[:self.action_num_])
            self.action_scale = np.array(gym_action_scales[:self.action_num_])
            print(f"[FSMStateMLP] Using GYM order ({self.action_num_} joints)")

        # Initialize buffers and actions
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)

        # Flags matching C++
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self.is_first_step_ = True
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0

        # Initialize ONNX session
        self.model_path = os.path.join(current_dir, "model", policy_config["model_path"]) 
        self._init_onnx_session()

        # ========== MuJoCo XML顺序（29个关节，固定不变）==========
        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ]

        # 构建lab/gym顺序到XML29维的映射
        self.lab2mj = []
        for name in self.joint_seq:  # joint_seq已经是当前模式的顺序
            if name not in self.joint_xml:
                raise ValueError(f"[FSMStateMLP] joint '{name}' not found in joint_xml!")
            self.lab2mj.append(self.joint_xml.index(name))
        self.lab2mj = np.array(self.lab2mj, dtype=int)

        # ========== 把23/29个关节scatter到29个XML维度 ==========
        n_mj = len(self.joint_xml)
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.ones(n_mj, dtype=np.float32) * 50
        self.damping_array = np.ones(n_mj, dtype=np.float32) * 2.0

        # 将当前模式顺序映射到29维XML顺序
        for lab_idx, mj_idx in enumerate(self.lab2mj):
            self.joint_pos_array[mj_idx] = self.joint_pos_array_seq[lab_idx]
            self.stiffness_array[mj_idx] = self.stiffness_array_seq[lab_idx]
            self.damping_array[mj_idx] = self.damping_array_seq[lab_idx]

        # 设置其他参数（lab顺序的别名，保持代码兼容性）
        self.kps_lab = self.stiffness_array_seq
        self.kds_lab = self.damping_array_seq
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

    def _init_onnx_session(self):
        """初始化ONNX推理会话"""
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.enable_mem_reuse = True

            self.ort_session_ = ort.InferenceSession(
                self.model_path, options, providers=['CPUExecutionProvider']
            )
            print(f"[FSMStateMLP-ONNX] ONNX model loaded: {self.model_path}")
        except Exception as e:
            print(f"[FSMStateMLP] Failed to load ONNX model: {e}")
            self.ort_session_ = None

    def on_enter(self):
        """进入MLP状态"""
        self._reset_internal_state()
        print(f"[FSMStateMLP] enter (mode: {self.USE_ORDER_MODE.name}, DOF: {self.action_num_})")
        self.is_first_obs_ = True
        self.is_first_action_ = True
        self._warmup_inference_counter = 0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    def run(self, flag: ControlFlag):
        """运行MLP状态"""
        print(f"[FSMStateMLP] run (DOF: {self.action_num_})")
        
        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            self.compute_observation(flag)
            self.compute_actions()

            # 当前模式顺序的目标位置（23或29维）
            target_dof_pos_current = self.actions_ * self.action_scale_lab + self.default_angles_lab
            
            # 构造29维XML顺序数组
            target_dof_pos_xml = np.zeros(29, dtype=np.float32)
            for current_idx, xml_idx in enumerate(self.lab2mj):
                target_dof_pos_xml[xml_idx] = target_dof_pos_current[current_idx]

            # Warm start处理
            commanded_pos = target_dof_pos_xml
            if self._warm_start_steps > 0 and self._warmup_inference_counter < self._warm_start_steps:
                self._warmup_inference_counter += 1
                blend = self._warmup_inference_counter / float(self._warm_start_steps)
                commanded_pos = (1.0 - blend) * self._warm_start_pose + blend * target_dof_pos_xml

            # 逐个写入机器人数据
            for i in range(self.motor_num_):
                joint_idx = 35 - self.motor_num_ + i
                self.robot_data_.q_d_[joint_idx] = commanded_pos[i]
                self.robot_data_.q_dot_d_[joint_idx] = 0.0
                self.robot_data_.tau_d_[joint_idx] = 0.0

            self.last_actions_[:] = self.actions_

        # PD增益设置
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def compute_observation(self, flag: ControlFlag):
        """计算观测量"""
        roll, pitch, yaw = (
            float(self.robot_data_.imu_data_[2]),
            float(self.robot_data_.imu_data_[1]),
            float(self.robot_data_.imu_data_[0]),
        )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0., 0., -1.], dtype=np.float32))

        # Command vector
        command = np.array([
            flag.x_speed_command,
            flag.y_speed_command,
            flag.yaw_speed_command
        ], dtype=np.float32)
        print(f'Input command: {command}')
        command = np.zeros(3)

        # IMU数据
        gyro = np.array([
            self.robot_data_.imu_data_[3],
            self.robot_data_.imu_data_[4],
            self.robot_data_.imu_data_[5]
        ], dtype=np.float32)

        # 获取MJ数据（29维XML顺序）
        ang_vel = self.robot_data_.get_angular_velocity()
        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel()

        # 通过lab2mj映射，提取当前自由度（23或29）的数据
        if self.USE_ORDER_MODE == JointOrderMode.LAB:
            q_current = q_mj[self.lab2mj] - self.default_angles_lab
        else:
            q_current = q_mj[self.lab2mj]
        dq_current = dq_mj[self.lab2mj]

        lin_vel = np.zeros(3)
        others = np.zeros(6)
        
        # 构建观测向量
        if self.USE_ORDER_MODE == JointOrderMode.LAB:
            proprio = np.concatenate([
                # lin_vel * self.lin_vel_scale_,
                ang_vel * self.ang_vel_scale_,
                gravity_init,
                # command,
                q_current * self.dof_pos_scale_,      # 已按当前模式顺序
                dq_current * self.dof_vel_scale_,     # 已按当前模式顺序
                self.last_actions_,
                # others
            ])
        else:
            proprio = np.concatenate([
                lin_vel * self.lin_vel_scale_* 0,
                ang_vel * self.ang_vel_scale_,
                gravity_init,
                command,
                q_current * self.dof_pos_scale_,      # 已按当前模式顺序
                dq_current * self.dof_vel_scale_,     # 已按当前模式顺序
                self.last_actions_,
                others
            ])

        # History buffer管理
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        # Clip observations
        self.observations_ = np.clip(proprio, -self.clip_obs_, self.clip_obs_)

    @staticmethod
    def euler_to_quaternion_scipy(roll, pitch, yaw, degrees=False):
        r = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
        q_xyzw = r.as_quat()
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    @staticmethod
    def quat_rotate_inverse_numpy(q_xyzw, v):
        q_w = q_xyzw[3]
        q_v = q_xyzw[:3]
        a = v * (2.0 * q_w * q_w - 1.0)
        b = np.cross(q_v, v) * (2.0 * q_w)
        c = q_v * (2.0 * np.dot(q_v, v))
        return a - b + c

    def compute_actions(self):
        """使用ONNX模型计算动作"""
        if self.ort_session_ is None:
            return

        try:
            input_data = self.observations_.reshape(1, -1).astype(np.float32)
            input_name = self.ort_session_.get_inputs()[0].name
            outputs = self.ort_session_.run(None, {input_name: input_data})

            # 提取并裁剪动作（输出是当前模式顺序）
            output_data = outputs[0][0]
            for i in range(self.action_num_):
                self.actions_[i] = np.clip(output_data[i], -self.clip_act_, self.clip_act_)

            if self.is_first_action_:
                print(f"[FSMStateMLP-ONNX] First Observation ({self.action_num_} DOF):")
                for i in range(self.obs_size_):
                    print(f"{self.observations_[i]:.6f} ", end="")
                print()
                self.is_first_action_ = False

        except Exception as e:
            print(f"[FSMStateMLP] ONNX Runtime inference error: {e}")

    def on_exit(self):
        """退出MLP状态"""
        print("[FSMStateMLP] exit")
        if getattr(self, "obs_log_file", None) is not None:
            try:
                self.obs_log_file.flush()
                self.obs_log_file.close()
                print(f"[FSMStateMLP] obs log saved")
            except Exception as e:
                print(f"[FSMStateMLP] failed to close obs log: {e}")
            self.obs_log_file = None

    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        """检查状态转换"""
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoMLP":
            return FSMStateName.MLP
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        elif flag.fsm_state_command == "gotoBEYONDMIMIC":
            return FSMStateName.BEYONDMIMIC
        elif flag.fsm_state_command == "gotoPBHC":
            return FSMStateName.PBHC
        elif flag.fsm_state_command == "gotoHhdGetup":
            return FSMStateName.HhdGetUp
        else:
            return None