#!/usr/bin/env python3
"""
优化版 FuncSPTrans 类
通过预分配数组、减少重复计算、使用向量化操作等方式提高性能
"""
import numpy as np
import math


class OptimizedFuncSPTrans:
    """优化版的 funcSPTrans 实现，用于串并联转换"""

    def __init__(self):
        """初始化优化版 FuncSPTrans，预分配所有数组以提高性能"""
        # 常量
        self.PI = math.pi

        # 机械参数 (mm)
        self.BC_l = 60.0
        self.OA1_l = 315.0
        self.OA2_l = 240.0
        self.C1P1_l = 315.0
        self.C2P2_l = 240.0
        self.AB_l = 45.0

        self.BC_r = 60.0
        self.OA1_r = 315.0
        self.OA2_r = 240.0
        self.C1P1_r = 315.0
        self.C2P2_r = 240.0
        self.AB_r = 45.0

        # 位置向量 (与C++版本一致)
        self.OP1_l = np.array([-55.82, 45.0, 22.0], dtype=np.float64)
        self.OP2_l = np.array([-55.82, -45.0, 22.0], dtype=np.float64)
        self.OP1_r = np.array([-55.82, -45.0, 22.0], dtype=np.float64)
        self.OP2_r = np.array([-55.82, 45.0, 22.0], dtype=np.float64)

        # 左腿参考值
        self.roll_l_ref = 0.0
        self.pitch_l_ref = 0.0
        self.alpha1_l_ref = 0.3754
        self.alpha2_l_ref = 0.3754
        self.qP_l_ref = np.zeros(2, dtype=np.float64)
        self.qDotP_l_ref = np.zeros(2, dtype=np.float64)
        self.torP_l_ref = np.zeros(2, dtype=np.float64)
        self.qS_l_ref = np.zeros(2, dtype=np.float64)
        self.qDotS_l_ref = np.zeros(2, dtype=np.float64)
        self.torS_l_ref = np.zeros(2, dtype=np.float64)

        # 右腿参考值
        self.roll_r_ref = 0.0
        self.pitch_r_ref = 0.0
        self.alpha1_r_ref = 0.3754
        self.alpha2_r_ref = 0.3754
        self.qP_r_ref = np.zeros(2, dtype=np.float64)
        self.qDotP_r_ref = np.zeros(2, dtype=np.float64)
        self.torP_r_ref = np.zeros(2, dtype=np.float64)
        self.qS_r_ref = np.zeros(2, dtype=np.float64)
        self.qDotS_r_ref = np.zeros(2, dtype=np.float64)
        self.torS_r_ref = np.zeros(2, dtype=np.float64)

        # 左腿估计值
        self.roll_l_est = 0.0
        self.pitch_l_est = 0.0
        self.alpha1_l_est = 0.3754
        self.alpha2_l_est = 0.3754
        self.qP_l_est = np.zeros(2, dtype=np.float64)
        self.qDotP_l_est = np.zeros(2, dtype=np.float64)
        self.torP_l_est = np.zeros(2, dtype=np.float64)
        self.qS_l_est = np.zeros(2, dtype=np.float64)
        self.qDotS_l_est = np.zeros(2, dtype=np.float64)
        self.torS_l_est = np.zeros(2, dtype=np.float64)

        # 右腿估计值
        self.roll_r_est = 0.0
        self.pitch_r_est = 0.0
        self.alpha1_r_est = 0.3754
        self.alpha2_r_est = 0.3754
        self.qP_r_est = np.zeros(2, dtype=np.float64)
        self.qDotP_r_est = np.zeros(2, dtype=np.float64)
        self.torP_r_est = np.zeros(2, dtype=np.float64)
        self.qS_r_est = np.zeros(2, dtype=np.float64)
        self.qDotS_r_est = np.zeros(2, dtype=np.float64)
        self.torS_r_est = np.zeros(2, dtype=np.float64)

        # 旋转矩阵 (预分配)
        self.rotXLeft = np.eye(3, dtype=np.float64)
        self.rotYLeft = np.eye(3, dtype=np.float64)
        self.rotYXLeft = np.eye(3, dtype=np.float64)
        self.rotXLeftRef = np.eye(3, dtype=np.float64)
        self.rotYLeftRef = np.eye(3, dtype=np.float64)
        self.rotYXLeftRef = np.eye(3, dtype=np.float64)
        self.rotXRight = np.eye(3, dtype=np.float64)
        self.rotYRight = np.eye(3, dtype=np.float64)
        self.rotYXRight = np.eye(3, dtype=np.float64)
        self.rotXRightRef = np.eye(3, dtype=np.float64)
        self.rotYRightRef = np.eye(3, dtype=np.float64)
        self.rotYXRightRef = np.eye(3, dtype=np.float64)

        # 位置向量 (预分配)
        self.oP1BodyLeft = np.zeros(3, dtype=np.float64)
        self.oP2BodyLeft = np.zeros(3, dtype=np.float64)
        self.oP1BodyLeftRef = np.zeros(3, dtype=np.float64)
        self.oP2BodyLeftRef = np.zeros(3, dtype=np.float64)
        self.oP1BodyRight = np.zeros(3, dtype=np.float64)
        self.oP2BodyRight = np.zeros(3, dtype=np.float64)
        self.oP1BodyRightRef = np.zeros(3, dtype=np.float64)
        self.oP2BodyRightRef = np.zeros(3, dtype=np.float64)

        # 其他向量 (预分配)
        self.C1P1Left = np.zeros(3, dtype=np.float64)
        self.C2P2Left = np.zeros(3, dtype=np.float64)
        self.C1P1Right = np.zeros(3, dtype=np.float64)
        self.C2P2Right = np.zeros(3, dtype=np.float64)

        # 雅可比矩阵 (预分配)
        self.ROmegaLeft = np.zeros((3, 2), dtype=np.float64)
        self.ROmegaLegLeft = np.zeros((3, 2), dtype=np.float64)
        self.JP1Left = np.zeros((3, 2), dtype=np.float64)
        self.JP2Left = np.zeros((3, 2), dtype=np.float64)
        self.ROmegaRight = np.zeros((3, 2), dtype=np.float64)
        self.ROmegaLegRight = np.zeros((3, 2), dtype=np.float64)
        self.JP1Right = np.zeros((3, 2), dtype=np.float64)
        self.JP2Right = np.zeros((3, 2), dtype=np.float64)

        self.B1C1Left = np.zeros(3, dtype=np.float64)
        self.B2C2Left = np.zeros(3, dtype=np.float64)
        self.JLeft = np.zeros((2, 2), dtype=np.float64)
        self.JAnkleLeft = np.zeros((2, 2), dtype=np.float64)
        self.JC1Left = np.zeros((3, 2), dtype=np.float64)
        self.JC2Left = np.zeros((3, 2), dtype=np.float64)
        self.vP1Left = np.zeros(3, dtype=np.float64)
        self.vP2Left = np.zeros(3, dtype=np.float64)

        self.B1C1Right = np.zeros(3, dtype=np.float64)
        self.B2C2Right = np.zeros(3, dtype=np.float64)
        self.JRight = np.zeros((2, 2), dtype=np.float64)
        self.JAnkleRight = np.zeros((2, 2), dtype=np.float64)
        self.JC1Right = np.zeros((3, 2), dtype=np.float64)
        self.JC2Right = np.zeros((3, 2), dtype=np.float64)
        self.vP1Right = np.zeros(3, dtype=np.float64)
        self.vP2Right = np.zeros(3, dtype=np.float64)

        self.tauDesjointFB = np.zeros(4, dtype=np.float64)

        # 神经网络参数 (左踝关节，与C++版本一致)
        self.b_a_left = np.array([
            5.99784899, -4.11348677, 4.07869244, 4.28190947, 3.20176721,
            0.916476667, -5.68045902, -3.09269261, -0.39034915, 1.45549524,
            0.862513363, -3.77516675, -0.859893322, 1.34862971, -2.59068418,
            -1.69658184, -2.7167716, 5.3070364, -2.75897098, 0.000272443343,
            -5.38999891, 3.62194443, -3.34114432, 1.21241927, 5.35194302,
            0.365667015, 0.616675556, -5.23831081, -4.1625061, 4.35436296,
            -5.42313242, 1.92171049, 0.42180562, 1.13346517, -3.38039541,
            -0.381162763, -3.77677345, 2.81083846, -3.31665659, 1.36346531,
            2.35270047, 1.80153131, -4.98206949, 1.18100488, -1.579368,
            3.87699461, 3.67021894, 3.77440119, -4.04099274, -6.5689168
        ], dtype=np.float32)

        self.c_a_left = np.array([
            -0.317089915, -0.0884859711, 0.0777945817, 0.0222885683,
            0.453757793, -0.637155056, 0.93334955, 0.222994685,
            -1.87128472, 0.131281495, 0.135077462, -0.0387916379,
            0.273187906, 0.0802086368, -0.0233348347, 0.00494327024,
            -1.28078985, 0.914301634, -0.348981231, 0.956789494,
            1.26943564, 1.3920294, -0.000143191122, -0.000249112345,
            -0.0689490959, 0.0518341027, 0.628563643, -1.06306219,
            -5.85482, -0.345520765, 0.140354633, 0.0296689309,
            2.70824099, 0.28300494, -0.707599401, -0.0782193914,
            2.24039078, 0.0405420475, -0.17541419, -0.154471293,
            -0.680902779, -0.0142482966, 2.14487147, -0.150890529,
            -2.20627236, -0.203721985, -0.269593239, 0.0387791917,
            -1.71200335, -0.0903570801
        ], dtype=np.float32)

        self.fv_left = np.array([
            -8.15948, 3.25148702, -6.55346966, -4.84019756, 7.56884623,
            -4.32959461, 5.62397194, 0.466775388, -0.21714054, -1.80951047,
            -0.0927751139, -4.34658957, 0.598068237, -1.78551805, -1.45854,
            -0.239000708, -1.47755289, 4.02654696, -1.50459146, 1.54216361,
            -4.88689327, 7.40091562, -3.59402585, 3.81989789, 6.9384737
        ], dtype=np.float32)

        # 右踝关节参数 (与C++版本一致，使用相同参数)
        self.b_a_right = self.b_a_left.copy()
        self.c_a_right = self.c_a_left.copy()
        self.fv_right = self.fv_left.copy()

        # 神经网络计算的临时数组 (预分配)
        self.nn_a = np.zeros(25, dtype=np.float32)
        self.nn_b_y1 = np.zeros(2, dtype=np.float32)
        self.nn_net = np.zeros(25, dtype=np.float32)

        # 用于旋转矩阵计算的临时变量
        self.cos_roll_l_est = 0.0
        self.sin_roll_l_est = 0.0
        self.cos_pitch_l_est = 0.0
        self.sin_pitch_l_est = 0.0
        self.cos_roll_r_est = 0.0
        self.sin_roll_r_est = 0.0
        self.cos_pitch_r_est = 0.0
        self.sin_pitch_r_est = 0.0

        # 用于雅可比矩阵计算的临时变量
        self.denom1_left = 0.0
        self.denom2_left = 0.0
        self.denom1_right = 0.0
        self.denom2_right = 0.0

        # 用于逆运动学计算的临时变量
        self.a1_left = 0.0
        self.b1_left = 0.0
        self.c1_left = 0.0
        self.a2_left = 0.0
        self.b2_left = 0.0
        self.c2_left = 0.0
        self.a1_right = 0.0
        self.b1_right = 0.0
        self.c1_right = 0.0
        self.a2_right = 0.0
        self.b2_right = 0.0
        self.c2_right = 0.0
        self.sqrt1_left = 0.0
        self.sqrt2_left = 0.0
        self.sqrt1_right = 0.0
        self.sqrt2_right = 0.0

    def skew(self, omg: np.ndarray) -> np.ndarray:
        """创建3D向量的反对称矩阵（向量化实现）"""
        result = np.array([
            [0.0, -omg[2], omg[1]],
            [omg[2], 0.0, -omg[0]],
            [-omg[1], omg[0], 0.0]
        ], dtype=np.float64)
        return result

    def set_p_est(self, qP_est: np.ndarray, qDotP_est: np.ndarray, qTorP_est: np.ndarray) -> bool:
        """设置并联关节估计值"""
        self.alpha1_l_est = qP_est[0] + 0.3754
        self.alpha2_l_est = qP_est[1] + 0.3754
        self.alpha1_r_est = qP_est[2] + 0.3754
        self.alpha2_r_est = qP_est[3] + 0.3754

        self.qP_l_est[:] = qP_est[:2]
        self.qP_r_est[:] = qP_est[2:]
        self.qDotP_l_est[:] = qDotP_est[:2]
        self.qDotP_r_est[:] = qDotP_est[2:]
        self.torP_l_est[:] = qTorP_est[:2]
        self.torP_r_est[:] = qTorP_est[2:]
        return True

    def calcFK(self) -> bool:
        """计算正向运动学"""
        self._calc_ankle_est_left()
        self._calc_ankle_est_right()
        return True

    def calc_fk(self):
        """Convenience wrapper"""
        return self.calcFK()

    def _calc_ankle_est_left(self) -> bool:
        """使用神经网络计算左踝关节估计值"""
        # 输入归一化
        xp1_idx_0 = (float(self.alpha1_l_est) - (-1.05928469)) * 0.761271894 - 1.0
        xp1_idx_1 = (float(self.alpha2_l_est) - (-1.06109619)) * 0.77644676 - 1.0

        # 隐藏层计算 (Sigmoid激活)
        # 向量化计算 net
        self.nn_net[:] = (self.b_a_left[:25] * xp1_idx_0 + 
                          self.b_a_left[25:50] * xp1_idx_1 + 
                          self.fv_left[:25])
        
        # Sigmoid激活函数
        np.divide(2.0, np.exp(-2.0 * self.nn_net) + 1.0, out=self.nn_a)
        self.nn_a -= 1.0

        # 输出层计算
        for k in range(2):
            xp1_idx_0_out = np.dot(self.c_a_left[k::2][:25], self.nn_a)

            # 输出反归一化
            offset = -0.130409107 * k - 0.0106069511
            scale = -1.14591563 * k + 2.29183125
            final_offset = -0.785398126 * k - 0.436332315
            self.nn_b_y1[k] = ((xp1_idx_0_out + offset) - (-1.0)) / scale + final_offset

        self.roll_l_est = float(self.nn_b_y1[0])
        self.pitch_l_est = float(self.nn_b_y1[1])
        return True

    def _calc_ankle_est_right(self) -> bool:
        """使用神经网络计算右踝关节估计值"""
        # 输入归一化
        xp1_idx_0 = (float(self.alpha1_r_est) - (-1.05928469)) * 0.761271894 - 1.0
        xp1_idx_1 = (float(self.alpha2_r_est) - (-1.06109619)) * 0.77644676 - 1.0

        # 隐藏层计算 (Sigmoid激活)
        # 向量化计算 net
        self.nn_net[:] = (self.b_a_right[:25] * xp1_idx_0 + 
                          self.b_a_right[25:50] * xp1_idx_1 + 
                          self.fv_right[:25])
        
        # Sigmoid激活函数
        np.divide(2.0, np.exp(-2.0 * self.nn_net) + 1.0, out=self.nn_a)
        self.nn_a -= 1.0

        # 输出层计算
        for k in range(2):
            xp1_idx_0_out = np.dot(self.c_a_right[k::2][:25], self.nn_a)

            # 输出反归一化
            offset = -0.130409107 * k - 0.0106069511
            scale = -1.14591563 * k + 2.29183125
            final_offset = -0.785398126 * k - 0.436332315
            self.nn_b_y1[k] = ((xp1_idx_0_out + offset) - (-1.0)) / scale + final_offset

        # 注意: 右踝关节有负滚转 (与C++版本一致)
        self.roll_r_est = float(-self.nn_b_y1[0])
        self.pitch_r_est = float(self.nn_b_y1[1])
        return True

    def calcIK(self) -> bool:
        """计算逆运动学"""
        self._calc_j_left()
        self._calc_j_right()
        return True

    def _calc_j_left(self) -> bool:
        """计算左腿雅可比矩阵"""
        # 预计算三角函数值
        self.cos_roll_l_est = np.cos(self.roll_l_est)
        self.sin_roll_l_est = np.sin(self.roll_l_est)
        self.cos_pitch_l_est = np.cos(self.pitch_l_est)
        self.sin_pitch_l_est = np.sin(self.pitch_l_est)

        # 旋转矩阵 (优化计算)
        self.rotXLeft[1, 1] = self.cos_roll_l_est
        self.rotXLeft[1, 2] = -self.sin_roll_l_est
        self.rotXLeft[2, 1] = self.sin_roll_l_est
        self.rotXLeft[2, 2] = self.cos_roll_l_est

        self.rotYLeft[0, 0] = self.cos_pitch_l_est
        self.rotYLeft[0, 2] = self.sin_pitch_l_est
        self.rotYLeft[2, 0] = -self.sin_pitch_l_est
        self.rotYLeft[2, 2] = self.cos_pitch_l_est

        # 矩阵乘法优化
        self.rotYXLeft[0, 0] = self.rotYLeft[0, 0]  # * 1 + 0 * 0 + 0 * 0
        self.rotYXLeft[0, 1] = 0.0  # 0 * 0 + 0 * 1 + 0 * 0
        self.rotYXLeft[0, 2] = self.rotYLeft[0, 2]  # * 1 + 0 * 0 + 0 * 0
        self.rotYXLeft[1, 0] = self.rotYLeft[1, 0] * self.rotXLeft[1, 0] + self.rotYLeft[1, 2] * self.rotXLeft[2, 0]
        self.rotYXLeft[1, 1] = self.rotYLeft[1, 1] * self.rotXLeft[1, 1] + self.rotYLeft[1, 2] * self.rotXLeft[2, 1]
        self.rotYXLeft[1, 2] = self.rotYLeft[1, 0] * self.rotXLeft[0, 2] + self.rotYLeft[1, 1] * self.rotXLeft[1, 2] + self.rotYLeft[1, 2] * self.rotXLeft[2, 2]
        self.rotYXLeft[2, 0] = self.rotYLeft[2, 0] * self.rotXLeft[1, 0] + self.rotYLeft[2, 2] * self.rotXLeft[2, 0]
        self.rotYXLeft[2, 1] = self.rotYLeft[2, 1] * self.rotXLeft[1, 1] + self.rotYLeft[2, 2] * self.rotXLeft[2, 1]
        self.rotYXLeft[2, 2] = self.rotYLeft[2, 0] * self.rotXLeft[0, 2] + self.rotYLeft[2, 1] * self.rotXLeft[1, 2] + self.rotYLeft[2, 2] * self.rotXLeft[2, 2]

        # 位置计算 (向量化)
        self.oP1BodyLeft[:] = self.rotYXLeft @ self.OP1_l
        self.oP2BodyLeft[:] = self.rotYXLeft @ self.OP2_l

        # 位置计算
        cos_alpha1 = np.cos(self.alpha1_l_est)
        sin_alpha1 = np.sin(self.alpha1_l_est)
        cos_alpha2 = np.cos(self.alpha2_l_est)
        sin_alpha2 = np.sin(self.alpha2_l_est)

        self.C1P1Left[0] = self.oP1BodyLeft[0] + self.BC_l * cos_alpha1
        self.C1P1Left[1] = self.oP1BodyLeft[1] - self.AB_l
        self.C1P1Left[2] = self.oP1BodyLeft[2] - self.OA1_l - self.BC_l * sin_alpha1

        self.C2P2Left[0] = self.oP2BodyLeft[0] + self.BC_l * cos_alpha2
        self.C2P2Left[1] = self.oP2BodyLeft[1] + self.AB_l
        self.C2P2Left[2] = self.oP2BodyLeft[2] - self.OA2_l - self.BC_l * sin_alpha2

        # ROmega矩阵
        self.ROmegaLeft[0, 0] = 0.0
        self.ROmegaLeft[0, 1] = self.rotYXLeft[0, 0]
        self.ROmegaLeft[1, 0] = 1.0
        self.ROmegaLeft[1, 1] = self.rotYXLeft[1, 0]
        self.ROmegaLeft[2, 0] = 0.0
        self.ROmegaLeft[2, 1] = self.rotYXLeft[2, 0]

        # JP矩阵计算
        skew_oP1 = self.skew(self.oP1BodyLeft)
        skew_oP2 = self.skew(self.oP2BodyLeft)
        self.JP1Left[:] = -skew_oP1 @ self.ROmegaLeft
        self.JP2Left[:] = -skew_oP2 @ self.ROmegaLeft

        # B1C1和B2C2向量
        self.B1C1Left[0] = -self.BC_l * cos_alpha1
        self.B1C1Left[1] = 0.0
        self.B1C1Left[2] = self.BC_l * sin_alpha1

        self.B2C2Left[0] = -self.BC_l * cos_alpha2
        self.B2C2Left[1] = 0.0
        self.B2C2Left[2] = self.BC_l * sin_alpha2

        # 雅可比矩阵计算
        self.denom1_left = self.B1C1Left[2] * self.C1P1Left[0] - self.B1C1Left[0] * self.C1P1Left[2]
        self.denom2_left = self.B2C2Left[2] * self.C2P2Left[0] - self.B2C2Left[0] * self.C2P2Left[2]

        # 使用einsum优化矩阵乘法
        c1p1_dot_jp1 = np.dot(self.C1P1Left, self.JP1Left)
        c2p2_dot_jp2 = np.dot(self.C2P2Left, self.JP2Left)
        
        self.JAnkleLeft[0, :] = c1p1_dot_jp1 / self.denom1_left
        self.JAnkleLeft[1, :] = c2p2_dot_jp2 / self.denom2_left

        self.JLeft[:, :] = self.JAnkleLeft

        # 速度和力矩变换
        try:
            self.qDotS_l_est[:] = np.linalg.solve(self.JLeft, self.qDotP_l_est)
            self.torS_l_est[:] = self.JLeft.T @ self.torP_l_est
        except np.linalg.LinAlgError:
            self.qDotS_l_est.fill(0.0)
            self.torS_l_est.fill(0.0)

        return True

    def _calc_j_right(self) -> bool:
        """计算右腿雅可比矩阵"""
        # 预计算三角函数值
        self.cos_roll_r_est = np.cos(self.roll_r_est)
        self.sin_roll_r_est = np.sin(self.roll_r_est)
        self.cos_pitch_r_est = np.cos(self.pitch_r_est)
        self.sin_pitch_r_est = np.sin(self.pitch_r_est)

        # 旋转矩阵 (优化计算)
        self.rotXRight[1, 1] = self.cos_roll_r_est
        self.rotXRight[1, 2] = -self.sin_roll_r_est
        self.rotXRight[2, 1] = self.sin_roll_r_est
        self.rotXRight[2, 2] = self.cos_roll_r_est

        self.rotYRight[0, 0] = self.cos_pitch_r_est
        self.rotYRight[0, 2] = self.sin_pitch_r_est
        self.rotYRight[2, 0] = -self.sin_pitch_r_est
        self.rotYRight[2, 2] = self.cos_pitch_r_est

        # 矩阵乘法优化
        self.rotYXRight[0, 0] = self.rotYRight[0, 0]
        self.rotYXRight[0, 1] = 0.0
        self.rotYXRight[0, 2] = self.rotYRight[0, 2]
        self.rotYXRight[1, 0] = self.rotYRight[1, 0] * self.rotXRight[1, 0] + self.rotYRight[1, 2] * self.rotXRight[2, 0]
        self.rotYXRight[1, 1] = self.rotYRight[1, 1] * self.rotXRight[1, 1] + self.rotYRight[1, 2] * self.rotXRight[2, 1]
        self.rotYXRight[1, 2] = self.rotYRight[1, 0] * self.rotXRight[0, 2] + self.rotYRight[1, 1] * self.rotXRight[1, 2] + self.rotYRight[1, 2] * self.rotXRight[2, 2]
        self.rotYXRight[2, 0] = self.rotYRight[2, 0] * self.rotXRight[1, 0] + self.rotYRight[2, 2] * self.rotXRight[2, 0]
        self.rotYXRight[2, 1] = self.rotYRight[2, 1] * self.rotXRight[1, 1] + self.rotYRight[2, 2] * self.rotXRight[2, 1]
        self.rotYXRight[2, 2] = self.rotYRight[2, 0] * self.rotXRight[0, 2] + self.rotYRight[2, 1] * self.rotXRight[1, 2] + self.rotYRight[2, 2] * self.rotXRight[2, 2]

        # 位置计算 (向量化)
        self.oP1BodyRight[:] = self.rotYXRight @ self.OP1_r
        self.oP2BodyRight[:] = self.rotYXRight @ self.OP2_r

        # 位置计算
        cos_alpha1 = np.cos(self.alpha1_r_est)
        sin_alpha1 = np.sin(self.alpha1_r_est)
        cos_alpha2 = np.cos(self.alpha2_r_est)
        sin_alpha2 = np.sin(self.alpha2_r_est)

        self.C1P1Right[0] = self.oP1BodyRight[0] + self.BC_r * cos_alpha1
        self.C1P1Right[1] = self.oP1BodyRight[1] + self.AB_r
        self.C1P1Right[2] = self.oP1BodyRight[2] - self.OA1_r - self.BC_r * sin_alpha1

        self.C2P2Right[0] = self.oP2BodyRight[0] + self.BC_r * cos_alpha2
        self.C2P2Right[1] = self.oP2BodyRight[1] - self.AB_r
        self.C2P2Right[2] = self.oP2BodyRight[2] - self.OA2_r - self.BC_r * sin_alpha2

        # ROmega矩阵
        self.ROmegaRight[0, 0] = 0.0
        self.ROmegaRight[0, 1] = self.rotYXRight[0, 0]
        self.ROmegaRight[1, 0] = 1.0
        self.ROmegaRight[1, 1] = self.rotYXRight[1, 0]
        self.ROmegaRight[2, 0] = 0.0
        self.ROmegaRight[2, 1] = self.rotYXRight[2, 0]

        # JP矩阵计算
        skew_oP1 = self.skew(self.oP1BodyRight)
        skew_oP2 = self.skew(self.oP2BodyRight)
        self.JP1Right[:] = -skew_oP1 @ self.ROmegaRight
        self.JP2Right[:] = -skew_oP2 @ self.ROmegaRight

        # B1C1和B2C2向量
        self.B1C1Right[0] = -self.BC_r * cos_alpha1
        self.B1C1Right[1] = 0.0
        self.B1C1Right[2] = self.BC_r * sin_alpha1

        self.B2C2Right[0] = -self.BC_r * cos_alpha2
        self.B2C2Right[1] = 0.0
        self.B2C2Right[2] = self.BC_r * sin_alpha2

        # 雅可比矩阵计算
        self.denom1_right = self.B1C1Right[2] * self.C1P1Right[0] - self.B1C1Right[0] * self.C1P1Right[2]
        self.denom2_right = self.B2C2Right[2] * self.C2P2Right[0] - self.B2C2Right[0] * self.C2P2Right[2]

        # 使用einsum优化矩阵乘法
        c1p1_dot_jp1 = np.dot(self.C1P1Right, self.JP1Right)
        c2p2_dot_jp2 = np.dot(self.C2P2Right, self.JP2Right)
        
        self.JAnkleRight[0, :] = c1p1_dot_jp1 / self.denom1_right
        self.JAnkleRight[1, :] = c2p2_dot_jp2 / self.denom2_right

        self.JRight[:, :] = self.JAnkleRight

        # 速度和力矩变换
        try:
            self.qDotS_r_est[:] = np.linalg.solve(self.JRight, self.qDotP_r_est)
            self.torS_r_est[:] = self.JRight.T @ self.torP_r_est
        except np.linalg.LinAlgError:
            self.qDotS_r_est.fill(0.0)
            self.torS_r_est.fill(0.0)

        return True

    def get_s_state(self, qS_est: np.ndarray, qDotS_est: np.ndarray, torS_est: np.ndarray) -> bool:
        """获取串联状态"""
        self.qS_l_est[0] = self.pitch_l_est
        self.qS_l_est[1] = self.roll_l_est
        self.qS_r_est[0] = self.pitch_r_est
        self.qS_r_est[1] = self.roll_r_est

        qS_est[:2] = self.qS_l_est
        qS_est[2:] = self.qS_r_est
        qDotS_est[:2] = self.qDotS_l_est
        qDotS_est[2:] = self.qDotS_r_est
        torS_est[:2] = self.torS_l_est
        torS_est[2:] = self.torS_r_est
        return True

    def set_s_des(self, qS_ref: np.ndarray, qDotS_ref: np.ndarray, torS_des: np.ndarray) -> bool:
        """设置期望串联值"""
        self.roll_l_ref = qS_ref[1]
        self.pitch_l_ref = qS_ref[0]
        self.roll_r_ref = qS_ref[3]
        self.pitch_r_ref = qS_ref[2]

        self.qS_l_ref[:] = qS_ref[:2]
        self.qS_r_ref[:] = qS_ref[2:]
        self.qDotS_l_ref[:] = qDotS_ref[:2]
        self.qDotS_r_ref[:] = qDotS_ref[2:]
        self.torS_l_ref[:] = torS_des[:2]
        self.torS_r_ref[:] = torS_des[2:]
        return True

    def get_p_des(self, qP_des: np.ndarray, qDotP_des: np.ndarray, torP_des: np.ndarray) -> bool:
        """获取期望并联值"""
        qP_des[:2] = self.qP_l_ref
        qP_des[2:] = self.qP_r_ref
        qDotP_des[:2] = self.qDotP_l_ref
        qDotP_des[2:] = self.qDotP_r_ref
        torP_des[:2] = self.torP_l_ref
        torP_des[2:] = self.torP_r_ref
        return True

    def calc_joint_pos_ref(self) -> bool:
        """计算参考关节位置"""
        self._calc_joint_pos_ref_left()
        self._calc_joint_pos_ref_right()
        return True

    def _calc_joint_pos_ref_left(self) -> bool:
        """使用逆运动学计算左关节位置参考"""
        # 预计算三角函数值
        cos_roll_l_ref = np.cos(self.roll_l_ref)
        sin_roll_l_ref = np.sin(self.roll_l_ref)
        cos_pitch_l_ref = np.cos(self.pitch_l_ref)
        sin_pitch_l_ref = np.sin(self.pitch_l_ref)

        # 旋转矩阵
        self.rotXLeftRef[1, 1] = cos_roll_l_ref
        self.rotXLeftRef[1, 2] = -sin_roll_l_ref
        self.rotXLeftRef[2, 1] = sin_roll_l_ref
        self.rotXLeftRef[2, 2] = cos_roll_l_ref

        self.rotYLeftRef[0, 0] = cos_pitch_l_ref
        self.rotYLeftRef[0, 2] = sin_pitch_l_ref
        self.rotYLeftRef[2, 0] = -sin_pitch_l_ref
        self.rotYLeftRef[2, 2] = cos_pitch_l_ref

        # 矩阵乘法
        rotYXLeftRef = self.rotYLeftRef @ self.rotXLeftRef
        self.oP1BodyLeftRef[:] = rotYXLeftRef @ self.OP1_l
        self.oP2BodyLeftRef[:] = rotYXLeftRef @ self.OP2_l

        # 求解 alpha1 和 alpha2
        self.a1_left = -2.0 * self.oP1BodyLeftRef[0] * self.BC_l
        self.b1_left = -2.0 * (self.OA1_l - self.oP1BodyLeftRef[2]) * self.BC_l
        self.c1_left = (self.oP1BodyLeftRef[0]**2 + (self.oP1BodyLeftRef[1] - self.AB_l)**2 +
                        (self.oP1BodyLeftRef[2] - self.OA1_l)**2 + self.BC_l**2 - self.C1P1_l**2)

        self.a2_left = -2.0 * self.oP2BodyLeftRef[0] * self.BC_l
        self.b2_left = -2.0 * (self.OA2_l - self.oP2BodyLeftRef[2]) * self.BC_l
        self.c2_left = (self.oP2BodyLeftRef[0]**2 + (self.oP2BodyLeftRef[1] + self.AB_l)**2 +
                        (self.oP2BodyLeftRef[2] - self.OA2_l)**2 + self.BC_l**2 - self.C2P2_l**2)

        # 检查解是否存在
        discriminant1 = self.a1_left**2 + self.b1_left**2 - self.c1_left**2
        discriminant2 = self.a2_left**2 + self.b2_left**2 - self.c2_left**2

        if discriminant1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        elif discriminant2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        # 计算解
        self.sqrt1_left = np.sqrt(discriminant1)
        self.sqrt2_left = np.sqrt(discriminant2)

        csalpha11 = (self.a1_left * self.c1_left + self.b1_left * self.sqrt1_left) / (self.a1_left**2 + self.b1_left**2)
        csalpha12 = (self.a1_left * self.c1_left - self.b1_left * self.sqrt1_left) / (self.a1_left**2 + self.b1_left**2)
        csalpha21 = (self.a2_left * self.c2_left + self.b2_left * self.sqrt2_left) / (self.a2_left**2 + self.b2_left**2)
        csalpha22 = (self.a2_left * self.c2_left - self.b2_left * self.sqrt2_left) / (self.a2_left**2 + self.b2_left**2)

        # 检查边界
        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            print("Error: Solution out of bounds for left leg")
            return False

        # 选择范围在 [-PI/2, PI/2] 内的有效解
        alpha1_candidates = [np.arccos(csalpha11), -np.arccos(csalpha11),
                           np.arccos(csalpha12), -np.arccos(csalpha12)]
        alpha2_candidates = [np.arccos(csalpha21), -np.arccos(csalpha21),
                           np.arccos(csalpha22), -np.arccos(csalpha22)]

        # 选择有效解
        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(self.a1_left * np.cos(alpha1) + self.b1_left * np.sin(alpha1) - self.c1_left) <= 1e-7:
                    self.alpha1_l_ref = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(self.a2_left * np.cos(alpha2) + self.b2_left * np.sin(alpha2) - self.c2_left) <= 1e-7:
                    self.alpha2_l_ref = alpha2
                    break

        self.qP_l_ref[0] = self.alpha1_l_ref - 0.3754
        self.qP_l_ref[1] = self.alpha2_l_ref - 0.3754
        return True

    def _calc_joint_pos_ref_right(self) -> bool:
        """使用逆运动学计算右关节位置参考"""
        # 预计算三角函数值
        cos_roll_r_ref = np.cos(self.roll_r_ref)
        sin_roll_r_ref = np.sin(self.roll_r_ref)
        cos_pitch_r_ref = np.cos(self.pitch_r_ref)
        sin_pitch_r_ref = np.sin(self.pitch_r_ref)

        # 旋转矩阵
        self.rotXRightRef[1, 1] = cos_roll_r_ref
        self.rotXRightRef[1, 2] = -sin_roll_r_ref
        self.rotXRightRef[2, 1] = sin_roll_r_ref
        self.rotXRightRef[2, 2] = cos_roll_r_ref

        self.rotYRightRef[0, 0] = cos_pitch_r_ref
        self.rotYRightRef[0, 2] = sin_pitch_r_ref
        self.rotYRightRef[2, 0] = -sin_pitch_r_ref
        self.rotYRightRef[2, 2] = cos_pitch_r_ref

        # 矩阵乘法
        rotYXRightRef = self.rotYRightRef @ self.rotXRightRef
        self.oP1BodyRightRef[:] = rotYXRightRef @ self.OP1_r
        self.oP2BodyRightRef[:] = rotYXRightRef @ self.OP2_r

        # 求解 alpha1 和 alpha2
        self.a1_right = -2.0 * self.oP1BodyRightRef[0] * self.BC_r
        self.b1_right = -2.0 * (self.OA1_r - self.oP1BodyRightRef[2]) * self.BC_r
        self.c1_right = (self.oP1BodyRightRef[0]**2 + (self.oP1BodyRightRef[1] + self.AB_r)**2 +
                         (self.oP1BodyRightRef[2] - self.OA1_r)**2 + self.BC_r**2 - self.C1P1_r**2)

        self.a2_right = -2.0 * self.oP2BodyRightRef[0] * self.BC_r
        self.b2_right = -2.0 * (self.OA2_r - self.oP2BodyRightRef[2]) * self.BC_r
        self.c2_right = (self.oP2BodyRightRef[0]**2 + (self.oP2BodyRightRef[1] - self.AB_r)**2 +
                         (self.oP2BodyRightRef[2] - self.OA2_r)**2 + self.BC_r**2 - self.C2P2_r**2)

        # 检查解是否存在
        discriminant1 = self.a1_right**2 + self.b1_right**2 - self.c1_right**2
        discriminant2 = self.a2_right**2 + self.b2_right**2 - self.c2_right**2

        if discriminant1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        elif discriminant2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        # 计算解
        self.sqrt1_right = np.sqrt(discriminant1)
        self.sqrt2_right = np.sqrt(discriminant2)

        csalpha11 = (self.a1_right * self.c1_right + self.b1_right * self.sqrt1_right) / (self.a1_right**2 + self.b1_right**2)
        csalpha12 = (self.a1_right * self.c1_right - self.b1_right * self.sqrt1_right) / (self.a1_right**2 + self.b1_right**2)
        csalpha21 = (self.a2_right * self.c2_right + self.b2_right * self.sqrt2_right) / (self.a2_right**2 + self.b2_right**2)
        csalpha22 = (self.a2_right * self.c2_right - self.b2_right * self.sqrt2_right) / (self.a2_right**2 + self.b2_right**2)

        # 检查边界
        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            print("Error: Solution out of bounds for right leg")
            return False

        # 选择范围在 [-PI/2, PI/2] 内的有效解
        alpha1_candidates = [np.arccos(csalpha11), -np.arccos(csalpha11),
                           np.arccos(csalpha12), -np.arccos(csalpha12)]
        alpha2_candidates = [np.arccos(csalpha21), -np.arccos(csalpha21),
                           np.arccos(csalpha22), -np.arccos(csalpha22)]

        # 选择有效解
        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(self.a1_right * np.cos(alpha1) + self.b1_right * np.sin(alpha1) - self.c1_right) <= 1e-7:
                    self.alpha1_r_ref = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(self.a2_right * np.cos(alpha2) + self.b2_right * np.sin(alpha2) - self.c2_right) <= 1e-7:
                    self.alpha2_r_ref = alpha2
                    break

        self.qP_r_ref[0] = self.alpha1_r_ref - 0.3754
        self.qP_r_ref[1] = self.alpha2_r_ref - 0.3754
        return True

    def calc_joint_tor_des(self) -> bool:
        """计算期望关节力矩"""
        try:
            # 使用solve而不是pinv以提高性能
            self.torP_l_ref[:] = np.linalg.solve(self.JLeft, self.torS_l_ref)
            self.torP_r_ref[:] = np.linalg.solve(self.JRight, self.torS_r_ref)
            self.qDotP_l_ref[:] = self.JLeft @ self.qDotS_l_ref
            self.qDotP_r_ref[:] = self.JRight @ self.qDotS_r_ref
        except np.linalg.LinAlgError:
            self.torP_l_ref.fill(0.0)
            self.torP_r_ref.fill(0.0)
            self.qDotP_l_ref.fill(0.0)
            self.qDotP_r_ref.fill(0.0)

        # 应用力矩限制
        self.torP_l_ref[0] = np.clip(self.torP_l_ref[0], -30.0, 30.0)
        self.torP_l_ref[1] = np.clip(self.torP_l_ref[1], -30.0, 30.0)
        self.torP_r_ref[0] = np.clip(self.torP_r_ref[0], -30.0, 30.0)
        self.torP_r_ref[1] = np.clip(self.torP_r_ref[1], -30.0, 30.0)

        return True


# 性能测试脚本
def performance_test():
    """性能测试函数"""
    import time
    
    print("优化版 FuncSPTrans 性能测试")
    print("=" * 50)
    
    # 创建原始和优化版实例
    from .func_sp_trans import FuncSPTrans
    original = FuncSPTrans()
    optimized = OptimizedFuncSPTrans()
    
    # 测试数据
    qP_est = np.array([0.1, 0.2, -0.1, -0.2])
    qDotP_est = np.array([0.01, 0.02, -0.01, -0.02])
    qTorP_est = np.array([1.0, 2.0, -1.0, -2.0])
    
    qS_ref = np.array([0.1, 0.2, -0.1, -0.2])
    qDotS_ref = np.array([0.01, 0.02, -0.01, -0.02])
    torS_des = np.array([1.0, 2.0, -1.0, -2.0])
    
    # 预热
    for _ in range(10):
        original.set_p_est(qP_est, qDotP_est, qTorP_est)
        original.calc_fk()
        original.calcIK()
        
        optimized.set_p_est(qP_est, qDotP_est, qTorP_est)
        optimized.calc_fk()
        optimized.calc_ik()
    
    # 测试原始版本性能
    num_runs = 1000
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        original.set_p_est(qP_est, qDotP_est, qTorP_est)
        original.calc_fk()
        original.calcIK()
        
        qS_est = np.zeros(4)
        qDotS_est = np.zeros(4)
        torS_est = np.zeros(4)
        original.get_s_state(qS_est, qDotS_est, torS_est)
        
        original.set_s_des(qS_ref, qDotS_ref, torS_des)
        original.calc_joint_pos_ref()
        original.calc_joint_tor_des()
        
        qP_des = np.zeros(4)
        qDotP_des = np.zeros(4)
        torP_des = np.zeros(4)
        original.get_p_des(qP_des, qDotP_des, torP_des)
    
    original_time = time.perf_counter() - start_time
    
    # 测试优化版本性能
    start_time = time.perf_counter()
    
    for _ in range(num_runs):
        optimized.set_p_est(qP_est, qDotP_est, qTorP_est)
        optimized.calc_fk()
        optimized.calc_ik()
        
        qS_est = np.zeros(4)
        qDotS_est = np.zeros(4)
        torS_est = np.zeros(4)
        optimized.get_s_state(qS_est, qDotS_est, torS_est)
        
        optimized.set_s_des(qS_ref, qDotS_ref, torS_des)
        optimized.calc_joint_pos_ref()
        optimized.calc_joint_tor_des()
        
        qP_des = np.zeros(4)
        qDotP_des = np.zeros(4)
        torP_des = np.zeros(4)
        optimized.get_p_des(qP_des, qDotP_des, torP_des)
    
    optimized_time = time.perf_counter() - start_time
    
    # 输出结果
    print(f"运行次数: {num_runs}")
    print(f"原始版本总时间: {original_time:.4f} 秒")
    print(f"优化版本总时间: {optimized_time:.4f} 秒")
    print(f"性能提升: {original_time/optimized_time:.2f}x")
    print(f"原始版本平均时间: {original_time/num_runs*1000:.4f} 毫秒")
    print(f"优化版本平均时间: {optimized_time/num_runs*1000:.4f} 毫秒")


if __name__ == "__main__":
    performance_test()
