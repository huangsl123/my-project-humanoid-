# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
class baseAgent:
    def __init__(self, device):
        self.device = device
        self.last_actions_buf = None

    def build_observations(self, obs_group):
        raise NotImplementedError

    def inference(self, obs_group):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def last_actions(self):
        return self.last_actions_buf
