# Copyright (c) 2025 Xuxin 747302550@qq.com. 保留所有权利. 未经许可，禁止复制、修改或分发
import numpy as np

class expFilter:
    def __init__(self, alpha):
        """
        指数滤波器\\
        alpha=0,完全不滤波,等于current\\
        alpha=1,全部等于last
        """
        self.alpha = alpha
        self.last = None

    def filter(self, current):
        if self.last is None:
            if type(current) is np.ndarray:
                self.last = np.zeros_like(current)
            else:
                raise NotImplementedError
        filtered = current * (1 - self.alpha) + self.last * self.alpha
        self.last = filtered
        return filtered

    def reset(self):
        self.last = None
