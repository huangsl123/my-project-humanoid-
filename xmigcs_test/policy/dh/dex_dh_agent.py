# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
from policy.dh.base_agent import baseAgent
from policy.dh.model import MODEL_DIR
import os.path as osp
import numpy as np
from policy.dh.exp_filter import expFilter
import onnxruntime as ort


class dex_dh_Agent(baseAgent):
    def __init__(self):
        self.num_prop_obs_input = 98
        self.num_estimator_input = 93
        self.include_history_steps = 5
        self.long_history = 64
        self.num_actions = 29
        # policy_path = osp.join(MODEL_DIR, "dex_1030_big_cmd.onnx")
        policy_path = osp.join(MODEL_DIR, "dex_1030_small_cmd.onnx")

        providers = [
            "CUDAExecutionProvider",  # 优先使用GPU
            "CPUExecutionProvider"    # 回退到CPU
        ] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]

        # 启用线程优化配置
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 设置计算线程数
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # 创建推理会话
        self.onnx_session = ort.InferenceSession(
            policy_path,
            providers=providers,
            sess_options=options
        )

        print([ipt.name for ipt in self.onnx_session.get_inputs()])

        # dof_pos归一化范围
        dof_pos_limits = [[-3.1416, 3.1416],
                          [-0.5236, 2.6180],
                          [-1.5708, 4.5379],
                          [-0.0873, 2.4435],
                          [-1.2217, 0.5236],
                          [-0.5236, 0.5236],

                          [-3.1416, 3.1416],
                          [-2.6180, 0.5236],
                          [-4.5379, 1.5708],
                          [-0.0873, 2.4435],
                          [-1.2217, 0.5236],
                          [-0.5236, 0.5236],

                          [-2.9671, 3.1416],
                          [-0.5236, 0.5236],
                          [-0.5236, 1.0472],

                          [-2.9671, 2.9671],
                          [-0.2618, 3.4034],
                          [-2.9671, 2.9671],
                          [-2.6180, 0.2618],
                          [-2.9671, 2.9671],
                          [-1.3090, 1.6581],
                          [-1.0472, 0.7854],

                          [-2.9671, 2.9671],
                          [-3.4034, 0.2618],
                          [-2.9671, 2.9671],
                          [-2.6180, 0.2618],
                          [-2.9671, 2.9671],
                          [-1.3090, 1.6581],
                          [-0.7854, 1.0472]]
        self.dof_pos_limits = np.array(dof_pos_limits)

        self.default_dof_pos = [
            0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.,
            0., 0., 0.,
            0., 0.15, 0., 0., 0., 0., 0.,
            0., -0.15, 0., 0., 0., 0., 0.,
        ]
        self.default_dof_pos = np.array(self.default_dof_pos)

        l_low = self.default_dof_pos - self.dof_pos_limits[:, 0]
        l_up = self.dof_pos_limits[:, 1] - self.default_dof_pos
        dof_scale = 1.0 / np.max(np.array([l_low, l_up]), axis=0)

        self.obs_scale = {
            "dof_pos": dof_scale,
            "dof_vel": 0.1,
            "ang_vel": 0.1,
        }
        # self.action_scale = 1.0
        self.action_scale = [  # v4
            1., 1., 1., 2.4435, 2., 1.,
            1., 1., 1., 2.4435, 2., 1.,
            1., 2.0, 2.0,
            1., 0.3, 0.3, 1.57, 0.3, 0., 0.,
            1., 0.3, 0.3, 1.57, 0.3, 0., 0.,]
        self.action_scale = np.array(self.action_scale)

        self.clip_observation = 10.
        self.clip_action = 10.

        self.prop_obs_history = np.zeros((self.num_prop_obs_input, self.long_history))
        self.estimator_obs_history = np.zeros((self.num_estimator_input, self.include_history_steps))

        self.last_actions_buf = np.zeros(self.num_actions)
        self.inference_count = 0
        self.dt = 0.01
        self.gait_period = 1.3
        self.exp_filter = expFilter(0.6)
        self.reset()

    def bootstrap(self):
        "预热用"
        obs_group = {
            "dof_pos": np.zeros(29, dtype=float),
            "dof_vel": np.zeros(29, dtype=float),
            "angular_velocity": np.zeros(3, dtype=float),
            "commands": np.zeros(3, dtype=float),
            "projected_gravity": np.array([0., 0., -1.], dtype=float),
        }
        self.inference(obs_group)

    def get_phase(self):
        phase = self.inference_count * self.dt / self.gait_period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        obs = np.array([sin_phase, cos_phase])
        return obs

    def build_observations(self, obs_group):
        prop_obs, estimator_obs = self.build_observations_one_step(obs_group)

        self.prop_obs_history = np.roll(self.prop_obs_history, shift=-1, axis=1)
        self.prop_obs_history[:, -1] = prop_obs

        self.estimator_obs_history = np.roll(self.estimator_obs_history, shift=-1, axis=1)
        self.estimator_obs_history[:, -1] = estimator_obs

        return self.prop_obs_history, self.estimator_obs_history

    def build_observations_one_step(self, obs_group):
        for obs in obs_group.values():
            obs = obs.clip(-self.clip_observation, self.clip_observation)

        obs_dof_pos = (obs_group["dof_pos"] - self.default_dof_pos) * self.obs_scale["dof_pos"]
        obs_dof_vel = obs_group["dof_vel"] * self.obs_scale["dof_vel"]
        obs_last_actions = self.last_actions_buf
        obs_projected_gravity = obs_group["projected_gravity"]
        obs_base_ang_vel = obs_group["angular_velocity"] * self.obs_scale["ang_vel"]
        obs_commands = obs_group["commands"]
        obs_commands[..., 2] *= self.obs_scale["ang_vel"]
        obs_phase = self.get_phase()

        # 本体感知proprioception 98
        prop_obs = np.concatenate((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
            obs_commands,
            obs_phase
        ), axis=-1)

        # estimator 93
        estimator_obs = np.concatenate((
            obs_dof_pos,
            obs_dof_vel,
            obs_last_actions,
            obs_base_ang_vel,
            obs_projected_gravity,
        ), axis=-1)

        return prop_obs, estimator_obs

    def inference(self, obs_group):
        prop_obs_history, estimator_obs_history = self.build_observations(obs_group)

        input_feed = {
            "prop_obs": prop_obs_history.flatten()[None, :].astype(np.float32),
            "estimator_obs": estimator_obs_history.flatten()[None, :].astype(np.float32),
        }
        actions = np.squeeze(self.onnx_session.run(["output"], input_feed))  # test
        actions = np.clip(actions, -self.clip_action, self.clip_action)

        self.last_actions_buf = actions

        dof_pos_target_urdf = actions * self.action_scale + self.default_dof_pos

        # dof_pos_target_urdf = self.exp_filter.filter(dof_pos_target_urdf)
        self.inference_count += 1
        # dof_pos_target_urdf = np.clip(dof_pos_target_urdf, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        return dof_pos_target_urdf

    def reset(self, first_obs_group=None):
        self.last_actions_buf = np.zeros(self.num_actions)
        if first_obs_group is None:
            first_obs_group = {
                "dof_pos": np.zeros(29),
                "dof_vel": np.zeros(29),
                "angular_velocity": np.zeros(3),
                "projected_gravity": np.array([0., 0., -1.]),
                "commands": np.zeros(3),
            }
        self.inference_count = 0
        norminal_obs, estimator_obs = self.build_observations_one_step(first_obs_group)
        self.prop_obs_history[:, :] = norminal_obs[:, None]
        self.estimator_obs_history[:, :] = estimator_obs[:, None]
        self.exp_filter.reset()

if __name__ == "__main__":
    a = dex_dh_Agent()
    obs_group = {
        "dof_pos": np.zeros(29),
        "dof_vel": np.zeros(29),
        "angular_velocity": np.zeros(3),
        "projected_gravity": np.array([0., 0., -1.]),
        "commands": np.zeros(3),
    }
    np.set_printoptions(formatter={"float": "{:.2f}".format})
    for i in range(100):
        print(a.inference(obs_group))
