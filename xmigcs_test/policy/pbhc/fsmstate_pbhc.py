# policy/pbhc/fsm_state_pbhc.py
import numpy as np
from FSM.fsm_base import FSMState, FSMStateName
from common.joystick import ControlFlag
from common.robot_data import RobotData
from scipy.spatial.transform import Rotation
from pathlib import Path
import os, time

from policy.pbhc.pbhc_infer import PBHCPolicy29 as PBHCPolicy

class FSMStatePBHC(FSMState):
    """FSM 中 PBHC 策略状态：只在输入/输出处做 PRY↔YRP，对内全部交给 PBHCPolicy。"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)

        # --- 基本参数（保持你此前设置） ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "pbhc.yaml")
        model_dir = Path(current_dir) / "model"
        default_model = model_dir / "model_8000_1119.onnx"
        # if not default_model.exists():
        #     all_models = sorted(model_dir.glob("*.onnx"))
        #     if not all_models:
        #         raise FileNotFoundError(f"ONNX 模型未找到: {model_dir}")
        #     default_model = all_models[-1]

        self.motor_num_  = 29
        self.dt          = 0.01    # policy 节拍
        self.decimation_ = 1        # 每 1*0.01 = 0.01s 推理一次
        self.warm_start_time = 0.3
        self._is_freeze = False
      
        self.kp = np.array([
            300, 300, 150, 350, 30, 16.8,
            300, 300, 150, 350, 30, 16.8,
            400, 400, 400,
            150, 50, 50, 150,80,40,30,
            150, 50, 50, 150,80,40,30,
        ], dtype=np.float32)

        self.kd = np.array([
            10, 10, 5, 10, 2.5, 1.4,
            10, 10, 5, 10, 2.5, 1.4,
            5, 10, 10,
            5, 2.5, 2.5, 5,3,1.5,1,
            5, 2.5, 2.5, 5,3,1.5,1,
        ], dtype=np.float32)
        # --- Policy ---
        self.policy = PBHCPolicy(config_path=config_path, model_path=str(default_model))
        self.motion_len = self.policy.motion_cycle_time

        # 运行时缓存
        self._last_step_idx = -1
        self.step_index = 0
        self._infer_count   = 0      # 只要触发一次推理就 +1（用于相位）
        self._sys_t0        = 0.0    # 本机单调时钟起点
        self._last_infer_ts = 0.0    # 上一次打印推理频率的时间戳（perf_counter）
        self.start_time = 0.0
        self.run_count = 0.0
        self._warmup_inference_counter = 0
        self._freeze_final_pose = False
        self._frozen_target_pos = None
          # allow switching final-pose freeze on/off
        self._last_target_pos = self.policy.default_angles.copy()

    # -----------------------------------------------------------
    def on_enter(self):
        print("[FSMStatePBHC] enter")
        self._last_step_idx = -1
        self._infer_count   = 0
        self._warmup_inference_counter = 0
        self._sys_t0        = time.perf_counter()
        self._last_infer_ts = self._sys_t0
        self._freeze_final_pose = False
        self._frozen_target_pos = None
        self._last_target_pos = self.policy.default_angles.copy()
        self.policy.reset()
        target_dof_pos_29 = self.policy.default_angles.copy()
        zero_vel = np.zeros_like(target_dof_pos_29)
        roll, pitch, yaw = (
                        float(self.robot_data_.imu_data_[2]),
                        float(self.robot_data_.imu_data_[1]),
                        float(self.robot_data_.imu_data_[0]),
                    )
        quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
        gravity_init   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
        for _ in range(self.policy.obs_history_len):
            self.policy.build_observation(
                omega_3=np.zeros(3, dtype=np.float32),
                qj_29=target_dof_pos_29,
                dqj_29=zero_vel,
                gravity_3=gravity_init,
                motion_phase=0.0
            )
        self.warm_start_step = max(1, int(self.warm_start_time / (self.decimation_ * self.dt)))

    # -----------------------------------------------------------
    def run(self, flag: ControlFlag):
        # —— 用系统时钟计算当前 step_idx（完全与 robot_time 脱钩）——
        if self.run_count == 0:
            self._sys_t0        = time.perf_counter()
            self.start_time = self.robot_data_.time_now_
        self.run_count += 1
        sys_now  = time.perf_counter()
        elapsed  = sys_now - self._sys_t0
        # step_idx = int(self.robot_data_.time_now_ / self.dt)
        step_idx = int((self.robot_data_.time_now_- self.start_time) / self.dt)
        start_inter = time.perf_counter()
        # print(f"step_idx:{step_idx}  _last_step_idx{self._last_step_idx}")
        if step_idx != self._last_step_idx:
            # for s in range(self._last_step_idx + 1, step_idx + 1):
                # 每 decimation_ 个 step 触发一次推理
                start_inter_time = time.perf_counter()
                if (step_idx > 0) and (step_idx % self.decimation_ == 0):
                    # ====== 读取 29DOF 状态 ======
                    start  = 35 - self.motor_num_
                    qj_29  = self.robot_data_.q_a_[start:start+self.motor_num_].astype(np.float32)
                    dqj_29 = self.robot_data_.q_dot_a_[start:start+self.motor_num_].astype(np.float32)

                    if self._freeze_final_pose:
                        if self._frozen_target_pos is None:
                            self._frozen_target_pos = self._last_target_pos.copy()
                        target_dof_pos_29 = self._frozen_target_pos
                        motion_phase = 1.0
                    else:
                        # —— 相位：基于推理调用次数（稳）——
                        PHASE_PERIOD = self.motion_len
                        phase_progress = self.decimation_ * self.dt * (self._infer_count + 1) / PHASE_PERIOD
                        motion_phase = min(1.0, phase_progress)

                        if motion_phase >= 1.0 and self._is_freeze:
                            self._freeze_final_pose = True
                            if self._frozen_target_pos is None:
                                self._frozen_target_pos = self._last_target_pos.copy()
                            target_dof_pos_29 = self._frozen_target_pos
                        else:
                            self._infer_count  += 1

                            # IMU 数据（重力投影与角速度）
                            roll, pitch, yaw = (
                                float(self.robot_data_.imu_data_[2]),
                                float(self.robot_data_.imu_data_[1]),
                                float(self.robot_data_.imu_data_[0]),
                            )
                            quat_wxyz = self.euler_to_quaternion_scipy(roll, pitch, yaw)
                            q_xyzw    = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
                            gravity   = self.quat_rotate_inverse_numpy(q_xyzw, np.array([0.,0.,-1.], dtype=np.float32))
                            omega     = np.array(self.robot_data_.imu_data_[3:6], dtype=np.float32)

                            # ====== Policy 推理 ======
                            robot_q_cmd = self.policy.step(
                                omega_3=omega,
                                qj_29=qj_29,
                                dqj_29=dqj_29,
                                gravity_3=gravity,
                                motion_phase=motion_phase,
                            )

                            self._warmup_inference_counter += 1
                            if self._warmup_inference_counter <= self.warm_start_step:
                                blend = self._warmup_inference_counter / self.warm_start_step
                                target_dof_pos_29 = (1.0 - blend) * self.policy.default_angles + blend * robot_q_cmd
                            else:
                                target_dof_pos_29 = robot_q_cmd

                            self._last_target_pos = target_dof_pos_29.copy()

                            if self._is_freeze and motion_phase >= 1.0:
                                self._freeze_final_pose = True
                                self._frozen_target_pos = self._last_target_pos.copy()

                    # --- 写回 robot_data ---
                    for i in range(self.motor_num_):
                        j = 35 - self.motor_num_ + i
                        self.robot_data_.q_d_[j]     = target_dof_pos_29[i]
                        self.robot_data_.q_dot_d_[j] = 0.0
                        self.robot_data_.tau_d_[j]   = 0.0

                    self._last_step_idx = step_idx
                end_infer_time = time.perf_counter()
                print(f"inter time is :{end_infer_time - start_inter_time}")
        # 增益  
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd


    # -----------------------------------------------------------
    def on_exit(self):
        print("[FSMStatePBHC] exit")
        self._last_step_idx = -1

    # ---------------- 工具 ----------------
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

    # ---------------- 状态跳转 ----------------
    def check_transition(self, flag: ControlFlag) -> FSMStateName:
        if flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        elif flag.fsm_state_command == "gotoPBHC":
            return FSMStateName.PBHC
        elif flag.fsm_state_command == "gotoPBHCSTAND":
            return FSMStateName.PBHCSTAND
        elif flag.fsm_state_command == "gotoPBHCZERO":
            return FSMStateName.PBHCZERO
        elif flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        else:
            return None
