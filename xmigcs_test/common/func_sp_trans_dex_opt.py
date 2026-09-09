#!/usr/bin/env python3
"""
优化后的 FuncSPTrans（_new.py 风格），使用 `func_sp_trans.py` 的常量与神经网络参数，
向量化并预分配临时数组以加速计算，并采用与NEW版本类似的优化策略。
"""
import numpy as np
import math

class OptimizedFuncSPTrans:
    """优化版（保留 func_sp_trans.py 数值语义，但采用 _new.py 的向量化/预分配策略）。"""
    def __init__(self):
        self.PI = math.pi
        # 机械参数（与 func_sp_trans.py 保持一致）
        self.BC_l = 46.0
        self.OA1_l = 290.0
        self.OA2_l = 220.0
        self.C1P1_l = 290.41
        self.C2P2_l = 220.54
        self.AB_l = 21.0

        self.BC_r = 46.0
        self.OA1_r = 290.0
        self.OA2_r = 220.0
        self.C1P1_r = 290.41
        self.C2P2_r = 220.54
        self.AB_r = 21.0

        # OP 向量
        self.OP1_l = np.array([-41.48, 36.4, 19.88], dtype=np.float64)
        self.OP2_l = np.array([-41.48, -36.4, 19.88], dtype=np.float64)
        self.OP1_r = np.array([-41.48, -36.4, 19.88], dtype=np.float64)
        self.OP2_r = np.array([-41.48, 36.4, 19.88], dtype=np.float64)

        # 参考/估计值（预分配）
        self.roll_l_ref = 0.0
        self.pitch_l_ref = 0.0
        self.alpha1_l_ref = 0.3492
        self.alpha2_l_ref = 0.3492
        self.qP_l_ref = np.zeros(2, dtype=np.float64)
        self.qDotP_l_ref = np.zeros(2, dtype=np.float64)
        self.torP_l_ref = np.zeros(2, dtype=np.float64)
        self.qS_l_ref = np.zeros(2, dtype=np.float64)
        self.qDotS_l_ref = np.zeros(2, dtype=np.float64)
        self.torS_l_ref = np.zeros(2, dtype=np.float64)

        self.roll_r_ref = 0.0
        self.pitch_r_ref = 0.0
        self.alpha1_r_ref = 0.3492
        self.alpha2_r_ref = 0.3492
        self.qP_r_ref = np.zeros(2, dtype=np.float64)
        self.qDotP_r_ref = np.zeros(2, dtype=np.float64)
        self.torP_r_ref = np.zeros(2, dtype=np.float64)
        self.qS_r_ref = np.zeros(2, dtype=np.float64)
        self.qDotS_r_ref = np.zeros(2, dtype=np.float64)
        self.torS_r_ref = np.zeros(2, dtype=np.float64)

        # 估计
        self.roll_l_est = 0.0
        self.pitch_l_est = 0.0
        self.alpha1_l_est = 0.3492
        self.alpha2_l_est = 0.3492
        self.qP_l_est = np.zeros(2, dtype=np.float64)
        self.qDotP_l_est = np.zeros(2, dtype=np.float64)
        self.torP_l_est = np.zeros(2, dtype=np.float64)
        self.qS_l_est = np.zeros(2, dtype=np.float64)
        self.qDotS_l_est = np.zeros(2, dtype=np.float64)
        self.torS_l_est = np.zeros(2, dtype=np.float64)

        self.roll_r_est = 0.0
        self.pitch_r_est = 0.0
        self.alpha1_r_est = 0.3492
        self.alpha2_r_est = 0.3492
        self.qP_r_est = np.zeros(2, dtype=np.float64)
        self.qDotP_r_est = np.zeros(2, dtype=np.float64)
        self.torP_r_est = np.zeros(2, dtype=np.float64)
        self.qS_r_est = np.zeros(2, dtype=np.float64)
        self.qDotS_r_est = np.zeros(2, dtype=np.float64)
        self.torS_r_est = np.zeros(2, dtype=np.float64)

        # 旋转矩阵与中间量（预分配以减少分配开销）
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

        self.oP1BodyLeft = np.zeros(3, dtype=np.float64)
        self.oP2BodyLeft = np.zeros(3, dtype=np.float64)
        self.oP1BodyLeftRef = np.zeros(3, dtype=np.float64)
        self.oP2BodyLeftRef = np.zeros(3, dtype=np.float64)
        self.oP1BodyRight = np.zeros(3, dtype=np.float64)
        self.oP2BodyRight = np.zeros(3, dtype=np.float64)
        self.oP1BodyRightRef = np.zeros(3, dtype=np.float64)
        self.oP2BodyRightRef = np.zeros(3, dtype=np.float64)

        self.C1P1Left = np.zeros(3, dtype=np.float64)
        self.C2P2Left = np.zeros(3, dtype=np.float64)
        self.C1P1Right = np.zeros(3, dtype=np.float64)
        self.C2P2Right = np.zeros(3, dtype=np.float64)

        self.ROmegaLeft = np.zeros((3,2), dtype=np.float64)
        self.JP1Left = np.zeros((3,2), dtype=np.float64)
        self.JP2Left = np.zeros((3,2), dtype=np.float64)
        self.ROmegaRight = np.zeros((3,2), dtype=np.float64)
        self.JP1Right = np.zeros((3,2), dtype=np.float64)
        self.JP2Right = np.zeros((3,2), dtype=np.float64)

        self.B1C1Left = np.zeros(3, dtype=np.float64)
        self.B2C2Left = np.zeros(3, dtype=np.float64)
        self.JLeft = np.zeros((2,2), dtype=np.float64)
        self.JAnkleLeft = np.zeros((2,2), dtype=np.float64)
        self.JC1Left = np.zeros((3,2), dtype=np.float64)
        self.JC2Left = np.zeros((3,2), dtype=np.float64)
        self.vP1Left = np.zeros(3, dtype=np.float64)
        self.vP2Left = np.zeros(3, dtype=np.float64)

        self.B1C1Right = np.zeros(3, dtype=np.float64)
        self.B2C2Right = np.zeros(3, dtype=np.float64)
        self.JRight = np.zeros((2,2), dtype=np.float64)
        self.JAnkleRight = np.zeros((2,2), dtype=np.float64)
        self.JC1Right = np.zeros((3,2), dtype=np.float64)
        self.JC2Right = np.zeros((3,2), dtype=np.float64)
        self.vP1Right = np.zeros(3, dtype=np.float64)
        self.vP2Right = np.zeros(3, dtype=np.float64)

        # 神经网络参数（从 func_sp_trans.py 提取）
        self.b_a = np.array([
            5.94356918, 0.698980629, -2.87409472, -1.01680517, -4.25445032,
            2.43503571, -1.23554814, 1.23667514, 0.786474645, -2.66088629,
            -2.22718477, -2.23365259, 1.81063032, 0.0637441, -1.25995994,
            2.73385882, 2.79696178, -2.45870185, 0.643692911, 5.62071323,
            -0.147124752, 0.378021479, -4.25320578, 3.90843987, -2.9711256,
            -4.35533476, 5.77593756, -5.64612532, 5.67641592, -0.773670137,
            -1.84441483, -2.87270236, -2.22745919, 0.10766869, 2.75126028,
            -4.04213095, -4.04839754, 2.68976188, 1.00575793, -1.78429079,
            -1.47443664, -2.17987609, 3.26092958, -1.83372235, -1.28457642,
            -3.66284776, 3.99858475, 5.09313583, 3.16851783, -4.53481102
        ], dtype=np.float64)
        self.c_a = np.array([
            -0.12816304, -0.0906492099, 0.0967225, -0.0920438766,
            0.0311508775, -0.0592247397, 0.957266, -0.0120375948,
            0.0395333804, 0.0282558668, -0.165704682, -0.169937313,
            0.0365005322, -0.0896773115, 1.39905655, 1.22731364,
            0.810329378, 0.567874134, -0.15331161, -0.049582541,
            0.698119521, -0.400004566, -0.693767846, 0.397573084,
            0.00196632301, -0.00473483559, -0.655982494, 0.821581066,
            0.0312418789, -0.0368167497, 0.548184454, 0.0165984798,
            0.19014442, 0.0510320961, -0.130089715, -0.0527587309,
            -0.513291419, -1.32376087, 0.0928636342, 0.00108589,
            -0.0479600728, 0.0617735051, -0.22801739, 0.0593482256,
            0.0210175235, 0.0085986387, 2.64173675, -0.115078852,
            2.19358563, -0.0805778876
        ], dtype=np.float64)
        self.fv = np.array([
            -5.81407595, -5.75171757, 5.80118036, 6.18498325, 4.05050945,
            -1.92026055, 2.33524013, -1.62735295, -0.229573011, 0.0187892355,
            1.23916209, 1.24042356, 0.108361147, -0.0811663568, -1.8306812,
            1.65682483, 0.759762943, 0.885608077, -1.40363491, 4.284688,
            2.85224295, 4.18213511, -5.30001831, 7.50161648, -7.814466
        ], dtype=np.float64)
        self.fv1 = np.array([-1.01818216, 0.110170461], dtype=np.float64)

        # NN temporary arrays (preallocated)
        self.nn_net = np.zeros(25, dtype=np.float64)
        self.nn_a = np.zeros(25, dtype=np.float64)
        self.nn_out = np.zeros(2, dtype=np.float64)

        # 预计算/向量化常量，避免每次调用时 reshape/slice 分配
        # 将 c_a 重排为 (2,25) 的权重矩阵，分割 b_a 为两个长度为25的向量
        try:
            self.W_ca = self.c_a.reshape((25, 2)).T.copy()  # shape (2,25)
        except Exception:
            # 保护性回退，保持原有行为
            self.W_ca = np.reshape(self.c_a, (25, 2)).T
        self.b_a0 = self.b_a[:25].copy()
        self.b_a1 = self.b_a[25:50].copy()
        # 归一化常量向量化
        self._denom = np.array([2.29183125, 2.29183125 - 1.11047494], dtype=np.float64)
        self._final_off = np.array([-0.436332315, (-0.872664571) - 0.436332315], dtype=np.float64)

        # temp trig/cache vars
        self.cos_roll_l_est = 0.0
        self.sin_roll_l_est = 0.0
        self.cos_pitch_l_est = 0.0
        self.sin_pitch_l_est = 0.0
        self.cos_roll_r_est = 0.0
        self.sin_roll_r_est = 0.0
        self.cos_pitch_r_est = 0.0
        self.sin_pitch_r_est = 0.0

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
        # 使用更高效的数组填充方式
        result = np.zeros((3, 3), dtype=np.float64)
        result[0, 1] = -omg[2]
        result[0, 2] = omg[1]
        result[1, 0] = omg[2]
        result[1, 2] = -omg[0]
        result[2, 0] = -omg[1]
        result[2, 1] = omg[0]
        return result

    def set_p_est(self, qP_est: np.ndarray, qDotP_est: np.ndarray, qTorP_est: np.ndarray) -> bool:
        self.alpha1_l_est = qP_est[0] + 0.3492
        self.alpha2_l_est = qP_est[1] + 0.3492
        self.alpha1_r_est = qP_est[2] + 0.3492
        self.alpha2_r_est = qP_est[3] + 0.3492
        self.qP_l_est[:] = qP_est[:2]
        self.qP_r_est[:] = qP_est[2:]
        self.qDotP_l_est[:] = qDotP_est[:2]
        self.qDotP_r_est[:] = qDotP_est[2:]
        self.torP_l_est[:] = qTorP_est[:2]
        self.torP_r_est[:] = qTorP_est[2:]
        return True

    def calc_fk(self) -> bool:
        return self.calcFK()

    def calcFK(self) -> bool:
        self._calc_ankle_est_left()
        self._calc_ankle_est_right()
        return True

    def _calc_ankle_est_left(self) -> bool:
        # normalization (use same constants as func_sp_trans.py)
        xp0 = (float(self.alpha1_l_est) - -1.04791963) * 0.77919364 - 1.0
        xp1 = (float(self.alpha2_l_est) - -1.05400908) * 0.784508467 - 1.0
        # 使用预计算切片和矩阵，尽量原地计算以减少临时数组
        # nn_net = b_a0 * xp0 + b_a1 * xp1 + fv
        np.multiply(self.b_a0, xp0, out=self.nn_net)
        np.add(self.nn_net, self.b_a1 * xp1, out=self.nn_net)
        np.add(self.nn_net, self.fv, out=self.nn_net)
        # sigmoid: nn_a = 2/(1+exp(-2*nn_net)) - 1
        np.divide(2.0, np.exp(-2.0 * self.nn_net) + 1.0, out=self.nn_a)
        self.nn_a -= 1.0
        # output using precomputed weight matrix
        self.nn_out[:] = self.W_ca @ self.nn_a
        # vectorized denormalize
        self.nn_out = (self.nn_out + self.fv1 + 1.0) / self._denom + self._final_off
        self.roll_l_est = float(self.nn_out[0])
        self.pitch_l_est = float(self.nn_out[1])
        return True

    def _calc_ankle_est_right(self) -> bool:
        xp0 = (float(self.alpha1_r_est) - -1.04791963) * 0.77919364 - 1.0
        xp1 = (float(self.alpha2_r_est) - -1.05400908) * 0.784508467 - 1.0
        # vectorized net using precomputed slices
        np.multiply(self.b_a0, xp0, out=self.nn_net)
        np.add(self.nn_net, self.b_a1 * xp1, out=self.nn_net)
        np.add(self.nn_net, self.fv, out=self.nn_net)
        np.divide(2.0, np.exp(-2.0 * self.nn_net) + 1.0, out=self.nn_a)
        self.nn_a -= 1.0
        self.nn_out[:] = self.W_ca @ self.nn_a
        self.nn_out = (self.nn_out + self.fv1 + 1.0) / self._denom + self._final_off
        # right roll is negated in original implementation
        self.roll_r_est = float(-self.nn_out[0])
        self.pitch_r_est = float(self.nn_out[1])
        return True

    def calc_ik(self) -> bool:
        return self.calcIK()

    def calcIK(self) -> bool:
        self._calc_j_left()
        self._calc_j_right()
        return True

    def _calc_j_left(self) -> bool:
        # precompute trig
        self.cos_roll_l_est = math.cos(self.roll_l_est)
        self.sin_roll_l_est = math.sin(self.roll_l_est)
        self.cos_pitch_l_est = math.cos(self.pitch_l_est)
        self.sin_pitch_l_est = math.sin(self.pitch_l_est)

        # rotation matrices - 优化计算
        self.rotXLeft[0, 0] = 1.0
        self.rotXLeft[0, 1] = 0.0
        self.rotXLeft[0, 2] = 0.0
        self.rotXLeft[1, 0] = 0.0
        self.rotXLeft[1, 1] = self.cos_roll_l_est
        self.rotXLeft[1, 2] = -self.sin_roll_l_est
        self.rotXLeft[2, 0] = 0.0
        self.rotXLeft[2, 1] = self.sin_roll_l_est
        self.rotXLeft[2, 2] = self.cos_roll_l_est

        self.rotYLeft[0, 0] = self.cos_pitch_l_est
        self.rotYLeft[0, 1] = 0.0
        self.rotYLeft[0, 2] = self.sin_pitch_l_est
        self.rotYLeft[1, 0] = 0.0
        self.rotYLeft[1, 1] = 1.0
        self.rotYLeft[1, 2] = 0.0
        self.rotYLeft[2, 0] = -self.sin_pitch_l_est
        self.rotYLeft[2, 1] = 0.0
        self.rotYLeft[2, 2] = self.cos_pitch_l_est

        # 矩阵乘法优化
        self.rotYXLeft[0, 0] = self.rotYLeft[0, 0] * self.rotXLeft[0, 0] + self.rotYLeft[0, 1] * self.rotXLeft[1, 0] + self.rotYLeft[0, 2] * self.rotXLeft[2, 0]
        self.rotYXLeft[0, 1] = self.rotYLeft[0, 0] * self.rotXLeft[0, 1] + self.rotYLeft[0, 1] * self.rotXLeft[1, 1] + self.rotYLeft[0, 2] * self.rotXLeft[2, 1]
        self.rotYXLeft[0, 2] = self.rotYLeft[0, 0] * self.rotXLeft[0, 2] + self.rotYLeft[0, 1] * self.rotXLeft[1, 2] + self.rotYLeft[0, 2] * self.rotXLeft[2, 2]
        self.rotYXLeft[1, 0] = self.rotYLeft[1, 0] * self.rotXLeft[0, 0] + self.rotYLeft[1, 1] * self.rotXLeft[1, 0] + self.rotYLeft[1, 2] * self.rotXLeft[2, 0]
        self.rotYXLeft[1, 1] = self.rotYLeft[1, 0] * self.rotXLeft[0, 1] + self.rotYLeft[1, 1] * self.rotXLeft[1, 1] + self.rotYLeft[1, 2] * self.rotXLeft[2, 1]
        self.rotYXLeft[1, 2] = self.rotYLeft[1, 0] * self.rotXLeft[0, 2] + self.rotYLeft[1, 1] * self.rotXLeft[1, 2] + self.rotYLeft[1, 2] * self.rotXLeft[2, 2]
        self.rotYXLeft[2, 0] = self.rotYLeft[2, 0] * self.rotXLeft[0, 0] + self.rotYLeft[2, 1] * self.rotXLeft[1, 0] + self.rotYLeft[2, 2] * self.rotXLeft[2, 0]
        self.rotYXLeft[2, 1] = self.rotYLeft[2, 0] * self.rotXLeft[0, 1] + self.rotYLeft[2, 1] * self.rotXLeft[1, 1] + self.rotYLeft[2, 2] * self.rotXLeft[2, 1]
        self.rotYXLeft[2, 2] = self.rotYLeft[2, 0] * self.rotXLeft[0, 2] + self.rotYLeft[2, 1] * self.rotXLeft[1, 2] + self.rotYLeft[2, 2] * self.rotXLeft[2, 2]

        # positions
        self.oP1BodyLeft[0] = self.rotYXLeft[0, 0] * self.OP1_l[0] + self.rotYXLeft[0, 1] * self.OP1_l[1] + self.rotYXLeft[0, 2] * self.OP1_l[2]
        self.oP1BodyLeft[1] = self.rotYXLeft[1, 0] * self.OP1_l[0] + self.rotYXLeft[1, 1] * self.OP1_l[1] + self.rotYXLeft[1, 2] * self.OP1_l[2]
        self.oP1BodyLeft[2] = self.rotYXLeft[2, 0] * self.OP1_l[0] + self.rotYXLeft[2, 1] * self.OP1_l[1] + self.rotYXLeft[2, 2] * self.OP1_l[2]
        
        self.oP2BodyLeft[0] = self.rotYXLeft[0, 0] * self.OP2_l[0] + self.rotYXLeft[0, 1] * self.OP2_l[1] + self.rotYXLeft[0, 2] * self.OP2_l[2]
        self.oP2BodyLeft[1] = self.rotYXLeft[1, 0] * self.OP2_l[0] + self.rotYXLeft[1, 1] * self.OP2_l[1] + self.rotYXLeft[1, 2] * self.OP2_l[2]
        self.oP2BodyLeft[2] = self.rotYXLeft[2, 0] * self.OP2_l[0] + self.rotYXLeft[2, 1] * self.OP2_l[1] + self.rotYXLeft[2, 2] * self.OP2_l[2]

        # compute C1P1, C2P2
        cos_a1 = self.cos_alpha1_l_est = math.cos(self.alpha1_l_est)
        sin_a1 = self.sin_alpha1_l_est = math.sin(self.alpha1_l_est)
        cos_a2 = self.cos_alpha2_l_est = math.cos(self.alpha2_l_est)
        sin_a2 = self.sin_alpha2_l_est = math.sin(self.alpha2_l_est)

        self.C1P1Left[0] = self.oP1BodyLeft[0] + self.BC_l * cos_a1
        self.C1P1Left[1] = self.oP1BodyLeft[1] - self.AB_l
        self.C1P1Left[2] = self.oP1BodyLeft[2] - self.OA1_l - self.BC_l * sin_a1

        self.C2P2Left[0] = self.oP2BodyLeft[0] + self.BC_l * cos_a2
        self.C2P2Left[1] = self.oP2BodyLeft[1] + self.AB_l
        self.C2P2Left[2] = self.oP2BodyLeft[2] - self.OA2_l - self.BC_l * sin_a2

        # ROmega
        self.ROmegaLeft[0, 0] = 0.0
        self.ROmegaLeft[0, 1] = self.rotYXLeft[0, 0]
        self.ROmegaLeft[1, 0] = 1.0
        self.ROmegaLeft[1, 1] = self.rotYXLeft[1, 0]
        self.ROmegaLeft[2, 0] = 0.0
        self.ROmegaLeft[2, 1] = self.rotYXLeft[2, 0]

        # JP matrices
        skew_oP1 = self.skew(self.oP1BodyLeft)
        skew_oP2 = self.skew(self.oP2BodyLeft)
        self.JP1Left[0, 0] = -(skew_oP1[0, 0] * self.ROmegaLeft[0, 0] + skew_oP1[0, 1] * self.ROmegaLeft[1, 0] + skew_oP1[0, 2] * self.ROmegaLeft[2, 0])
        self.JP1Left[0, 1] = -(skew_oP1[0, 0] * self.ROmegaLeft[0, 1] + skew_oP1[0, 1] * self.ROmegaLeft[1, 1] + skew_oP1[0, 2] * self.ROmegaLeft[2, 1])
        self.JP1Left[1, 0] = -(skew_oP1[1, 0] * self.ROmegaLeft[0, 0] + skew_oP1[1, 1] * self.ROmegaLeft[1, 0] + skew_oP1[1, 2] * self.ROmegaLeft[2, 0])
        self.JP1Left[1, 1] = -(skew_oP1[1, 0] * self.ROmegaLeft[0, 1] + skew_oP1[1, 1] * self.ROmegaLeft[1, 1] + skew_oP1[1, 2] * self.ROmegaLeft[2, 1])
        self.JP1Left[2, 0] = -(skew_oP1[2, 0] * self.ROmegaLeft[0, 0] + skew_oP1[2, 1] * self.ROmegaLeft[1, 0] + skew_oP1[2, 2] * self.ROmegaLeft[2, 0])
        self.JP1Left[2, 1] = -(skew_oP1[2, 0] * self.ROmegaLeft[0, 1] + skew_oP1[2, 1] * self.ROmegaLeft[1, 1] + skew_oP1[2, 2] * self.ROmegaLeft[2, 1])
        
        self.JP2Left[0, 0] = -(skew_oP2[0, 0] * self.ROmegaLeft[0, 0] + skew_oP2[0, 1] * self.ROmegaLeft[1, 0] + skew_oP2[0, 2] * self.ROmegaLeft[2, 0])
        self.JP2Left[0, 1] = -(skew_oP2[0, 0] * self.ROmegaLeft[0, 1] + skew_oP2[0, 1] * self.ROmegaLeft[1, 1] + skew_oP2[0, 2] * self.ROmegaLeft[2, 1])
        self.JP2Left[1, 0] = -(skew_oP2[1, 0] * self.ROmegaLeft[0, 0] + skew_oP2[1, 1] * self.ROmegaLeft[1, 0] + skew_oP2[1, 2] * self.ROmegaLeft[2, 0])
        self.JP2Left[1, 1] = -(skew_oP2[1, 0] * self.ROmegaLeft[0, 1] + skew_oP2[1, 1] * self.ROmegaLeft[1, 1] + skew_oP2[1, 2] * self.ROmegaLeft[2, 1])
        self.JP2Left[2, 0] = -(skew_oP2[2, 0] * self.ROmegaLeft[0, 0] + skew_oP2[2, 1] * self.ROmegaLeft[1, 0] + skew_oP2[2, 2] * self.ROmegaLeft[2, 0])
        self.JP2Left[2, 1] = -(skew_oP2[2, 0] * self.ROmegaLeft[0, 1] + skew_oP2[2, 1] * self.ROmegaLeft[1, 1] + skew_oP2[2, 2] * self.ROmegaLeft[2, 1])

        # B vectors
        self.B1C1Left[0] = -self.BC_l * cos_a1
        self.B1C1Left[1] = 0.0
        self.B1C1Left[2] = self.BC_l * sin_a1

        self.B2C2Left[0] = -self.BC_l * cos_a2
        self.B2C2Left[1] = 0.0
        self.B2C2Left[2] = self.BC_l * sin_a2

        # denom and ankle jacobian
        self.denom1_left = self.B1C1Left[2] * self.C1P1Left[0] - self.B1C1Left[0] * self.C1P1Left[2]
        self.denom2_left = self.B2C2Left[2] * self.C2P2Left[0] - self.B2C2Left[0] * self.C2P2Left[2]

        c1 = np.dot(self.C1P1Left, self.JP1Left)
        c2 = np.dot(self.C2P2Left, self.JP2Left)
        self.JAnkleLeft[0, 0] = c1[0] / self.denom1_left
        self.JAnkleLeft[0, 1] = c1[1] / self.denom1_left
        self.JAnkleLeft[1, 0] = c2[0] / self.denom2_left
        self.JAnkleLeft[1, 1] = c2[1] / self.denom2_left
        self.JLeft[:, :] = self.JAnkleLeft

        # velocity & torque transform - 优先使用solve提升性能
        try:
            # 使用solve替代pinv以提高性能和数值稳定性
            self.qDotS_l_est[:] = np.linalg.solve(self.JLeft, self.qDotP_l_est)
            self.torS_l_est[:] = self.JLeft.T @ self.torP_l_est
        except np.linalg.LinAlgError:
            # fallback to pseudo-inverse semantics
            pinv = np.linalg.pinv(self.JLeft)
            self.qDotS_l_est[:] = pinv @ self.qDotP_l_est
            self.torS_l_est[:] = self.JLeft.T @ self.torP_l_est
        return True

    def _calc_j_right(self) -> bool:
        # precompute trig
        self.cos_roll_r_est = math.cos(self.roll_r_est)
        self.sin_roll_r_est = math.sin(self.roll_r_est)
        self.cos_pitch_r_est = math.cos(self.pitch_r_est)
        self.sin_pitch_r_est = math.sin(self.pitch_r_est)

        # rotation matrices - 优化计算
        self.rotXRight[0, 0] = 1.0
        self.rotXRight[0, 1] = 0.0
        self.rotXRight[0, 2] = 0.0
        self.rotXRight[1, 0] = 0.0
        self.rotXRight[1, 1] = self.cos_roll_r_est
        self.rotXRight[1, 2] = -self.sin_roll_r_est
        self.rotXRight[2, 0] = 0.0
        self.rotXRight[2, 1] = self.sin_roll_r_est
        self.rotXRight[2, 2] = self.cos_roll_r_est

        self.rotYRight[0, 0] = self.cos_pitch_r_est
        self.rotYRight[0, 1] = 0.0
        self.rotYRight[0, 2] = self.sin_pitch_r_est
        self.rotYRight[1, 0] = 0.0
        self.rotYRight[1, 1] = 1.0
        self.rotYRight[1, 2] = 0.0
        self.rotYRight[2, 0] = -self.sin_pitch_r_est
        self.rotYRight[2, 1] = 0.0
        self.rotYRight[2, 2] = self.cos_pitch_r_est

        # 矩阵乘法优化
        self.rotYXRight[0, 0] = self.rotYRight[0, 0] * self.rotXRight[0, 0] + self.rotYRight[0, 1] * self.rotXRight[1, 0] + self.rotYRight[0, 2] * self.rotXRight[2, 0]
        self.rotYXRight[0, 1] = self.rotYRight[0, 0] * self.rotXRight[0, 1] + self.rotYRight[0, 1] * self.rotXRight[1, 1] + self.rotYRight[0, 2] * self.rotXRight[2, 1]
        self.rotYXRight[0, 2] = self.rotYRight[0, 0] * self.rotXRight[0, 2] + self.rotYRight[0, 1] * self.rotXRight[1, 2] + self.rotYRight[0, 2] * self.rotXRight[2, 2]
        self.rotYXRight[1, 0] = self.rotYRight[1, 0] * self.rotXRight[0, 0] + self.rotYRight[1, 1] * self.rotXRight[1, 0] + self.rotYRight[1, 2] * self.rotXRight[2, 0]
        self.rotYXRight[1, 1] = self.rotYRight[1, 0] * self.rotXRight[0, 1] + self.rotYRight[1, 1] * self.rotXRight[1, 1] + self.rotYRight[1, 2] * self.rotXRight[2, 1]
        self.rotYXRight[1, 2] = self.rotYRight[1, 0] * self.rotXRight[0, 2] + self.rotYRight[1, 1] * self.rotXRight[1, 2] + self.rotYRight[1, 2] * self.rotXRight[2, 2]
        self.rotYXRight[2, 0] = self.rotYRight[2, 0] * self.rotXRight[0, 0] + self.rotYRight[2, 1] * self.rotXRight[1, 0] + self.rotYRight[2, 2] * self.rotXRight[2, 0]
        self.rotYXRight[2, 1] = self.rotYRight[2, 0] * self.rotXRight[0, 1] + self.rotYRight[2, 1] * self.rotXRight[1, 1] + self.rotYRight[2, 2] * self.rotXRight[2, 1]
        self.rotYXRight[2, 2] = self.rotYRight[2, 0] * self.rotXRight[0, 2] + self.rotYRight[2, 1] * self.rotXRight[1, 2] + self.rotYRight[2, 2] * self.rotXRight[2, 2]

        # positions
        self.oP1BodyRight[0] = self.rotYXRight[0, 0] * self.OP1_r[0] + self.rotYXRight[0, 1] * self.OP1_r[1] + self.rotYXRight[0, 2] * self.OP1_r[2]
        self.oP1BodyRight[1] = self.rotYXRight[1, 0] * self.OP1_r[0] + self.rotYXRight[1, 1] * self.OP1_r[1] + self.rotYXRight[1, 2] * self.OP1_r[2]
        self.oP1BodyRight[2] = self.rotYXRight[2, 0] * self.OP1_r[0] + self.rotYXRight[2, 1] * self.OP1_r[1] + self.rotYXRight[2, 2] * self.OP1_r[2]
        
        self.oP2BodyRight[0] = self.rotYXRight[0, 0] * self.OP2_r[0] + self.rotYXRight[0, 1] * self.OP2_r[1] + self.rotYXRight[0, 2] * self.OP2_r[2]
        self.oP2BodyRight[1] = self.rotYXRight[1, 0] * self.OP2_r[0] + self.rotYXRight[1, 1] * self.OP2_r[1] + self.rotYXRight[1, 2] * self.OP2_r[2]
        self.oP2BodyRight[2] = self.rotYXRight[2, 0] * self.OP2_r[0] + self.rotYXRight[2, 1] * self.OP2_r[1] + self.rotYXRight[2, 2] * self.OP2_r[2]

        # compute C1P1, C2P2
        cos_a1 = self.cos_alpha1_r_est = math.cos(self.alpha1_r_est)
        sin_a1 = self.sin_alpha1_r_est = math.sin(self.alpha1_r_est)
        cos_a2 = self.cos_alpha2_r_est = math.cos(self.alpha2_r_est)
        sin_a2 = self.sin_alpha2_r_est = math.sin(self.alpha2_r_est)

        self.C1P1Right[0] = self.oP1BodyRight[0] + self.BC_r * cos_a1
        self.C1P1Right[1] = self.oP1BodyRight[1] + self.AB_r
        self.C1P1Right[2] = self.oP1BodyRight[2] - self.OA1_r - self.BC_r * sin_a1

        self.C2P2Right[0] = self.oP2BodyRight[0] + self.BC_r * cos_a2
        self.C2P2Right[1] = self.oP2BodyRight[1] - self.AB_r
        self.C2P2Right[2] = self.oP2BodyRight[2] - self.OA2_r - self.BC_r * sin_a2

        # ROmega
        self.ROmegaRight[0, 0] = 0.0
        self.ROmegaRight[0, 1] = self.rotYXRight[0, 0]
        self.ROmegaRight[1, 0] = 1.0
        self.ROmegaRight[1, 1] = self.rotYXRight[1, 0]
        self.ROmegaRight[2, 0] = 0.0
        self.ROmegaRight[2, 1] = self.rotYXRight[2, 0]

        # JP matrices
        skew_oP1 = self.skew(self.oP1BodyRight)
        skew_oP2 = self.skew(self.oP2BodyRight)
        self.JP1Right[0, 0] = -(skew_oP1[0, 0] * self.ROmegaRight[0, 0] + skew_oP1[0, 1] * self.ROmegaRight[1, 0] + skew_oP1[0, 2] * self.ROmegaRight[2, 0])
        self.JP1Right[0, 1] = -(skew_oP1[0, 0] * self.ROmegaRight[0, 1] + skew_oP1[0, 1] * self.ROmegaRight[1, 1] + skew_oP1[0, 2] * self.ROmegaRight[2, 1])
        self.JP1Right[1, 0] = -(skew_oP1[1, 0] * self.ROmegaRight[0, 0] + skew_oP1[1, 1] * self.ROmegaRight[1, 0] + skew_oP1[1, 2] * self.ROmegaRight[2, 0])
        self.JP1Right[1, 1] = -(skew_oP1[1, 0] * self.ROmegaRight[0, 1] + skew_oP1[1, 1] * self.ROmegaRight[1, 1] + skew_oP1[1, 2] * self.ROmegaRight[2, 1])
        self.JP1Right[2, 0] = -(skew_oP1[2, 0] * self.ROmegaRight[0, 0] + skew_oP1[2, 1] * self.ROmegaRight[1, 0] + skew_oP1[2, 2] * self.ROmegaRight[2, 0])
        self.JP1Right[2, 1] = -(skew_oP1[2, 0] * self.ROmegaRight[0, 1] + skew_oP1[2, 1] * self.ROmegaRight[1, 1] + skew_oP1[2, 2] * self.ROmegaRight[2, 1])
        
        self.JP2Right[0, 0] = -(skew_oP2[0, 0] * self.ROmegaRight[0, 0] + skew_oP2[0, 1] * self.ROmegaRight[1, 0] + skew_oP2[0, 2] * self.ROmegaRight[2, 0])
        self.JP2Right[0, 1] = -(skew_oP2[0, 0] * self.ROmegaRight[0, 1] + skew_oP2[0, 1] * self.ROmegaRight[1, 1] + skew_oP2[0, 2] * self.ROmegaRight[2, 1])
        self.JP2Right[1, 0] = -(skew_oP2[1, 0] * self.ROmegaRight[0, 0] + skew_oP2[1, 1] * self.ROmegaRight[1, 0] + skew_oP2[1, 2] * self.ROmegaRight[2, 0])
        self.JP2Right[1, 1] = -(skew_oP2[1, 0] * self.ROmegaRight[0, 1] + skew_oP2[1, 1] * self.ROmegaRight[1, 1] + skew_oP2[1, 2] * self.ROmegaRight[2, 1])
        self.JP2Right[2, 0] = -(skew_oP2[2, 0] * self.ROmegaRight[0, 0] + skew_oP2[2, 1] * self.ROmegaRight[1, 0] + skew_oP2[2, 2] * self.ROmegaRight[2, 0])
        self.JP2Right[2, 1] = -(skew_oP2[2, 0] * self.ROmegaRight[0, 1] + skew_oP2[2, 1] * self.ROmegaRight[1, 1] + skew_oP2[2, 2] * self.ROmegaRight[2, 1])

        # B vectors
        self.B1C1Right[0] = -self.BC_r * cos_a1
        self.B1C1Right[1] = 0.0
        self.B1C1Right[2] = self.BC_r * sin_a1

        self.B2C2Right[0] = -self.BC_r * cos_a2
        self.B2C2Right[1] = 0.0
        self.B2C2Right[2] = self.BC_r * sin_a2

        # denom and ankle jacobian
        self.denom1_right = self.B1C1Right[2] * self.C1P1Right[0] - self.B1C1Right[0] * self.C1P1Right[2]
        self.denom2_right = self.B2C2Right[2] * self.C2P2Right[0] - self.B2C2Right[0] * self.C2P2Right[2]

        c1 = np.dot(self.C1P1Right, self.JP1Right)
        c2 = np.dot(self.C2P2Right, self.JP2Right)
        self.JAnkleRight[0, 0] = c1[0] / self.denom1_right
        self.JAnkleRight[0, 1] = c1[1] / self.denom1_right
        self.JAnkleRight[1, 0] = c2[0] / self.denom2_right
        self.JAnkleRight[1, 1] = c2[1] / self.denom2_right
        self.JRight[:, :] = self.JAnkleRight

        # velocity & torque transform - 优先使用solve提升性能
        try:
            # 使用solve替代pinv以提高性能和数值稳定性
            self.qDotS_r_est[:] = np.linalg.solve(self.JRight, self.qDotP_r_est)
            self.torS_r_est[:] = self.JRight.T @ self.torP_r_est
        except np.linalg.LinAlgError:
            # fallback to pseudo-inverse semantics
            pinv = np.linalg.pinv(self.JRight)
            self.qDotS_r_est[:] = pinv @ self.qDotP_r_est
            self.torS_r_est[:] = self.JRight.T @ self.torP_r_est
        return True

    def get_s_state(self, qS_est: np.ndarray, qDotS_est: np.ndarray, torS_est: np.ndarray) -> bool:
        # keep same mapping as func_sp_trans.py getSState
        qS_est[0] = self.pitch_l_est + 0.09765
        qS_est[1] = self.roll_l_est
        qS_est[2] = self.pitch_r_est + 0.09765
        qS_est[3] = self.roll_r_est
        qDotS_est[:2] = self.qDotS_l_est
        qDotS_est[2:] = self.qDotS_r_est
        torS_est[:2] = self.torS_l_est
        torS_est[2:] = self.torS_r_est
        return True

    def set_s_des(self, qS_ref: np.ndarray, qDotS_ref: np.ndarray, torS_des: np.ndarray) -> bool:
        self.roll_l_ref = qS_ref[1]
        self.pitch_l_ref = qS_ref[0] - 0.09765  # 修正偏移
        self.roll_r_ref = qS_ref[3]
        self.pitch_r_ref = qS_ref[2] - 0.09765  # 修正偏移
        
        self.qS_l_ref[:] = qS_ref[:2]
        self.qS_r_ref[:] = qS_ref[2:]
        self.qDotS_l_ref[:] = qDotS_ref[:2]
        self.qDotS_r_ref[:] = qDotS_ref[2:]
        self.torS_l_ref[:] = torS_des[:2]
        self.torS_r_ref[:] = torS_des[2:]
        return True

    def get_p_des(self, qP_des: np.ndarray, qDotP_des: np.ndarray, torP_des: np.ndarray) -> bool:
        qP_des[:2] = self.qP_l_ref
        qP_des[2:] = self.qP_r_ref
        qDotP_des[:2] = self.qDotP_l_ref
        qDotP_des[2:] = self.qDotP_r_ref
        torP_des[:2] = self.torP_l_ref
        torP_des[2:] = self.torP_r_ref
        return True

    def calc_joint_pos_ref(self) -> bool:
        # compute left reference joint positions (mirrors calcJointPosRefLeft)
        self._calc_joint_pos_ref_left()
        self._calc_joint_pos_ref_right()
        return True

    def _calc_joint_pos_ref_left(self) -> bool:
        # 预计算三角函数值
        cos_roll_l_ref = math.cos(self.roll_l_ref)
        sin_roll_l_ref = math.sin(self.roll_l_ref)
        cos_pitch_l_ref = math.cos(self.pitch_l_ref)
        sin_pitch_l_ref = math.sin(self.pitch_l_ref)

        # rotation matrices - 优化计算
        self.rotXLeftRef[0, 0] = 1.0
        self.rotXLeftRef[0, 1] = 0.0
        self.rotXLeftRef[0, 2] = 0.0
        self.rotXLeftRef[1, 0] = 0.0
        self.rotXLeftRef[1, 1] = cos_roll_l_ref
        self.rotXLeftRef[1, 2] = -sin_roll_l_ref
        self.rotXLeftRef[2, 0] = 0.0
        self.rotXLeftRef[2, 1] = sin_roll_l_ref
        self.rotXLeftRef[2, 2] = cos_roll_l_ref

        self.rotYLeftRef[0, 0] = cos_pitch_l_ref
        self.rotYLeftRef[0, 1] = 0.0
        self.rotYLeftRef[0, 2] = sin_pitch_l_ref
        self.rotYLeftRef[1, 0] = 0.0
        self.rotYLeftRef[1, 1] = 1.0
        self.rotYLeftRef[1, 2] = 0.0
        self.rotYLeftRef[2, 0] = -sin_pitch_l_ref
        self.rotYLeftRef[2, 1] = 0.0
        self.rotYLeftRef[2, 2] = cos_pitch_l_ref

        # 矩阵乘法优化
        self.rotYXLeftRef[0, 0] = self.rotYLeftRef[0, 0] * self.rotXLeftRef[0, 0] + self.rotYLeftRef[0, 1] * self.rotXLeftRef[1, 0] + self.rotYLeftRef[0, 2] * self.rotXLeftRef[2, 0]
        self.rotYXLeftRef[0, 1] = self.rotYLeftRef[0, 0] * self.rotXLeftRef[0, 1] + self.rotYLeftRef[0, 1] * self.rotXLeftRef[1, 1] + self.rotYLeftRef[0, 2] * self.rotXLeftRef[2, 1]
        self.rotYXLeftRef[0, 2] = self.rotYLeftRef[0, 0] * self.rotXLeftRef[0, 2] + self.rotYLeftRef[0, 1] * self.rotXLeftRef[1, 2] + self.rotYLeftRef[0, 2] * self.rotXLeftRef[2, 2]
        self.rotYXLeftRef[1, 0] = self.rotYLeftRef[1, 0] * self.rotXLeftRef[0, 0] + self.rotYLeftRef[1, 1] * self.rotXLeftRef[1, 0] + self.rotYLeftRef[1, 2] * self.rotXLeftRef[2, 0]
        self.rotYXLeftRef[1, 1] = self.rotYLeftRef[1, 0] * self.rotXLeftRef[0, 1] + self.rotYLeftRef[1, 1] * self.rotXLeftRef[1, 1] + self.rotYLeftRef[1, 2] * self.rotXLeftRef[2, 1]
        self.rotYXLeftRef[1, 2] = self.rotYLeftRef[1, 0] * self.rotXLeftRef[0, 2] + self.rotYLeftRef[1, 1] * self.rotXLeftRef[1, 2] + self.rotYLeftRef[1, 2] * self.rotXLeftRef[2, 2]

        self.rotYXLeftRef[2, 0] = self.rotYLeftRef[2, 0] * self.rotXLeftRef[0, 0] + self.rotYLeftRef[2, 1] * self.rotXLeftRef[1, 0] + self.rotYLeftRef[2, 2] * self.rotXLeftRef[2, 0]
        self.rotYXLeftRef[2, 1] = self.rotYLeftRef[2, 0] * self.rotXLeftRef[0, 1] + self.rotYLeftRef[2, 1] * self.rotXLeftRef[1, 1] + self.rotYLeftRef[2, 2] * self.rotXLeftRef[2, 1]
        self.rotYXLeftRef[2, 2] = self.rotYLeftRef[2, 0] * self.rotXLeftRef[0, 2] + self.rotYLeftRef[2, 1] * self.rotXLeftRef[1, 2] + self.rotYLeftRef[2, 2] * self.rotXLeftRef[2, 2]

        # positions
        self.oP1BodyLeftRef[0] = self.rotYXLeftRef[0, 0] * self.OP1_l[0] + self.rotYXLeftRef[0, 1] * self.OP1_l[1] + self.rotYXLeftRef[0, 2] * self.OP1_l[2]
        self.oP1BodyLeftRef[1] = self.rotYXLeftRef[1, 0] * self.OP1_l[0] + self.rotYXLeftRef[1, 1] * self.OP1_l[1] + self.rotYXLeftRef[1, 2] * self.OP1_l[2]
        self.oP1BodyLeftRef[2] = self.rotYXLeftRef[2, 0] * self.OP1_l[0] + self.rotYXLeftRef[2, 1] * self.OP1_l[1] + self.rotYXLeftRef[2, 2] * self.OP1_l[2]
        
        self.oP2BodyLeftRef[0] = self.rotYXLeftRef[0, 0] * self.OP2_l[0] + self.rotYXLeftRef[0, 1] * self.OP2_l[1] + self.rotYXLeftRef[0, 2] * self.OP2_l[2]
        self.oP2BodyLeftRef[1] = self.rotYXLeftRef[1, 0] * self.OP2_l[0] + self.rotYXLeftRef[1, 1] * self.OP2_l[1] + self.rotYXLeftRef[1, 2] * self.OP2_l[2]
        self.oP2BodyLeftRef[2] = self.rotYXLeftRef[2, 0] * self.OP2_l[0] + self.rotYXLeftRef[2, 1] * self.OP2_l[1] + self.rotYXLeftRef[2, 2] * self.OP2_l[2]

        # 求解 alpha1 和 alpha2
        a1 = -2.0 * self.oP1BodyLeftRef[0] * self.BC_l
        b1 = -2.0 * (self.OA1_l - self.oP1BodyLeftRef[2]) * self.BC_l
        c1 = (self.oP1BodyLeftRef[0]**2 + (self.oP1BodyLeftRef[1] - self.AB_l)**2 +
                        (self.oP1BodyLeftRef[2] - self.OA1_l)**2 + self.BC_l**2 - self.C1P1_l**2)

        a2 = -2.0 * self.oP2BodyLeftRef[0] * self.BC_l
        b2 = -2.0 * (self.OA2_l - self.oP2BodyLeftRef[2]) * self.BC_l
        c2 = (self.oP2BodyLeftRef[0]**2 + (self.oP2BodyLeftRef[1] + self.AB_l)**2 +
                        (self.oP2BodyLeftRef[2] - self.OA2_l)**2 + self.BC_l**2 - self.C2P2_l**2)

        # 检查解是否存在
        discriminant1 = a1*a1 + b1*b1 - c1*c1
        discriminant2 = a2*a2 + b2*b2 - c2*c2

        if discriminant1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        elif discriminant2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        # 计算解
        sqrt1 = math.sqrt(discriminant1)
        sqrt2 = math.sqrt(discriminant2)

        csalpha11 = (a1 * c1 + b1 * sqrt1) / (a1*a1 + b1*b1)
        csalpha12 = (a1 * c1 - b1 * sqrt1) / (a1*a1 + b1*b1)
        csalpha21 = (a2 * c2 + b2 * sqrt2) / (a2*a2 + b2*b2)
        csalpha22 = (a2 * c2 - b2 * sqrt2) / (a2*a2 + b2*b2)

        # 检查边界
        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            print("Error: Solution out of bounds for left leg")
            return False

        # 选择范围在 [-PI/2, PI/2] 内的有效解
        alpha1_candidates = [math.acos(csalpha11), -math.acos(csalpha11),
                           math.acos(csalpha12), -math.acos(csalpha12)]
        alpha2_candidates = [math.acos(csalpha21), -math.acos(csalpha21),
                           math.acos(csalpha22), -math.acos(csalpha22)]

        # 选择有效解
        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(a1 * math.cos(alpha1) + b1 * math.sin(alpha1) - c1) <= 1e-7:
                    self.alpha1_l_ref = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(a2 * math.cos(alpha2) + b2 * math.sin(alpha2) - c2) <= 1e-7:
                    self.alpha2_l_ref = alpha2
                    break

        self.qP_l_ref[0] = self.alpha1_l_ref - 0.3492
        self.qP_l_ref[1] = self.alpha2_l_ref - 0.3492
        return True

    def _calc_joint_pos_ref_right(self) -> bool:
        # 预计算三角函数值
        cos_roll_r_ref = math.cos(self.roll_r_ref)
        sin_roll_r_ref = math.sin(self.roll_r_ref)
        cos_pitch_r_ref = math.cos(self.pitch_r_ref)
        sin_pitch_r_ref = math.sin(self.pitch_r_ref)

        # rotation matrices - 优化计算
        self.rotXRightRef[0, 0] = 1.0
        self.rotXRightRef[0, 1] = 0.0
        self.rotXRightRef[0, 2] = 0.0
        self.rotXRightRef[1, 0] = 0.0
        self.rotXRightRef[1, 1] = cos_roll_r_ref
        self.rotXRightRef[1, 2] = -sin_roll_r_ref
        self.rotXRightRef[2, 0] = 0.0
        self.rotXRightRef[2, 1] = sin_roll_r_ref
        self.rotXRightRef[2, 2] = cos_roll_r_ref

        self.rotYRightRef[0, 0] = cos_pitch_r_ref
        self.rotYRightRef[0, 1] = 0.0
        self.rotYRightRef[0, 2] = sin_pitch_r_ref
        self.rotYRightRef[1, 0] = 0.0
        self.rotYRightRef[1, 1] = 1.0
        self.rotYRightRef[1, 2] = 0.0
        self.rotYRightRef[2, 0] = -sin_pitch_r_ref
        self.rotYRightRef[2, 1] = 0.0
        self.rotYRightRef[2, 2] = cos_pitch_r_ref

        # 矩阵乘法优化
        self.rotYXRightRef[0, 0] = self.rotYRightRef[0, 0] * self.rotXRightRef[0, 0] + self.rotYRightRef[0, 1] * self.rotXRightRef[1, 0] + self.rotYRightRef[0, 2] * self.rotXRightRef[2, 0]
        self.rotYXRightRef[0, 1] = self.rotYRightRef[0, 0] * self.rotXRightRef[0, 1] + self.rotYRightRef[0, 1] * self.rotXRightRef[1, 1] + self.rotYRightRef[0, 2] * self.rotXRightRef[2, 1]
        self.rotYXRightRef[0, 2] = self.rotYRightRef[0, 0] * self.rotXRightRef[0, 2] + self.rotYRightRef[0, 1] * self.rotXRightRef[1, 2] + self.rotYRightRef[0, 2] * self.rotXRightRef[2, 2]
        self.rotYXRightRef[1, 0] = self.rotYRightRef[1, 0] * self.rotXRightRef[0, 0] + self.rotYRightRef[1, 1] * self.rotXRightRef[1, 0] + self.rotYRightRef[1, 2] * self.rotXRightRef[2, 0]
        self.rotYXRightRef[1, 1] = self.rotYRightRef[1, 0] * self.rotXRightRef[0, 1] + self.rotYRightRef[1, 1] * self.rotXRightRef[1, 1] + self.rotYRightRef[1, 2] * self.rotXRightRef[2, 1]
        self.rotYXRightRef[1, 2] = self.rotYRightRef[1, 0] * self.rotXRightRef[0, 2] + self.rotYRightRef[1, 1] * self.rotXRightRef[1, 2] + self.rotYRightRef[1, 2] * self.rotXRightRef[2, 2]
        self.rotYXRightRef[2, 0] = self.rotYRightRef[2, 0] * self.rotXRightRef[0, 0] + self.rotYRightRef[2, 1] * self.rotXRightRef[1, 0] + self.rotYRightRef[2, 2] * self.rotXRightRef[2, 0]
        self.rotYXRightRef[2, 1] = self.rotYRightRef[2, 0] * self.rotXRightRef[0, 1] + self.rotYRightRef[2, 1] * self.rotXRightRef[1, 1] + self.rotYRightRef[2, 2] * self.rotXRightRef[2, 1]
        self.rotYXRightRef[2, 2] = self.rotYRightRef[2, 0] * self.rotXRightRef[0, 2] + self.rotYRightRef[2, 1] * self.rotXRightRef[1, 2] + self.rotYRightRef[2, 2] * self.rotXRightRef[2, 2]

        # positions
        self.oP1BodyRightRef[0] = self.rotYXRightRef[0, 0] * self.OP1_r[0] + self.rotYXRightRef[0, 1] * self.OP1_r[1] + self.rotYXRightRef[0, 2] * self.OP1_r[2]
        self.oP1BodyRightRef[1] = self.rotYXRightRef[1, 0] * self.OP1_r[0] + self.rotYXRightRef[1, 1] * self.OP1_r[1] + self.rotYXRightRef[1, 2] * self.OP1_r[2]
        self.oP1BodyRightRef[2] = self.rotYXRightRef[2, 0] * self.OP1_r[0] + self.rotYXRightRef[2, 1] * self.OP1_r[1] + self.rotYXRightRef[2, 2] * self.OP1_r[2]
        
        self.oP2BodyRightRef[0] = self.rotYXRightRef[0, 0] * self.OP2_r[0] + self.rotYXRightRef[0, 1] * self.OP2_r[1] + self.rotYXRightRef[0, 2] * self.OP2_r[2]
        self.oP2BodyRightRef[1] = self.rotYXRightRef[1, 0] * self.OP2_r[0] + self.rotYXRightRef[1, 1] * self.OP2_r[1] + self.rotYXRightRef[1, 2] * self.OP2_r[2]
        self.oP2BodyRightRef[2] = self.rotYXRightRef[2, 0] * self.OP2_r[0] + self.rotYXRightRef[2, 1] * self.OP2_r[1] + self.rotYXRightRef[2, 2] * self.OP2_r[2]

        # 求解 alpha1 和 alpha2
        a1 = -2.0 * self.oP1BodyRightRef[0] * self.BC_r
        b1 = -2.0 * (self.OA1_r - self.oP1BodyRightRef[2]) * self.BC_r
        c1 = (self.oP1BodyRightRef[0]**2 + (self.oP1BodyRightRef[1] + self.AB_r)**2 +
                         (self.oP1BodyRightRef[2] - self.OA1_r)**2 + self.BC_r**2 - self.C1P1_r**2)

        a2 = -2.0 * self.oP2BodyRightRef[0] * self.BC_r
        b2 = -2.0 * (self.OA2_r - self.oP2BodyRightRef[2]) * self.BC_r
        c2 = (self.oP2BodyRightRef[0]**2 + (self.oP2BodyRightRef[1] - self.AB_r)**2 +
                         (self.oP2BodyRightRef[2] - self.OA2_r)**2 + self.BC_r**2 - self.C2P2_r**2)

        # 检查解是否存在
        discriminant1 = a1*a1 + b1*b1 - c1*c1
        discriminant2 = a2*a2 + b2*b2 - c2*c2

        if discriminant1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        elif discriminant2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        # 计算解
        sqrt1 = math.sqrt(discriminant1)
        sqrt2 = math.sqrt(discriminant2)

        csalpha11 = (a1 * c1 + b1 * sqrt1) / (a1*a1 + b1*b1)
        csalpha12 = (a1 * c1 - b1 * sqrt1) / (a1*a1 + b1*b1)
        csalpha21 = (a2 * c2 + b2 * sqrt2) / (a2*a2 + b2*b2)
        csalpha22 = (a2 * c2 - b2 * sqrt2) / (a2*a2 + b2*b2)

        # 检查边界
        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            print("Error: Solution out of bounds for right leg")
            return False

        # 选择范围在 [-PI/2, PI/2] 内的有效解
        alpha1_candidates = [math.acos(csalpha11), -math.acos(csalpha11),
                           math.acos(csalpha12), -math.acos(csalpha12)]
        alpha2_candidates = [math.acos(csalpha21), -math.acos(csalpha21),
                           math.acos(csalpha22), -math.acos(csalpha22)]

        # 选择有效解
        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(a1 * math.cos(alpha1) + b1 * math.sin(alpha1) - c1) <= 1e-7:
                    self.alpha1_r_ref = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(a2 * math.cos(alpha2) + b2 * math.sin(alpha2) - c2) <= 1e-7:
                    self.alpha2_r_ref = alpha2
                    break

        self.qP_r_ref[0] = self.alpha1_r_ref - 0.3492
        self.qP_r_ref[1] = self.alpha2_r_ref - 0.3492
        return True

    def calc_joint_tor_des(self) -> bool:
        # try to use solve where possible for speed, fallback to pinv semantics
        try:
            # 使用solve替代pinv以提高性能和数值稳定性
            self.torP_l_ref[:] = np.linalg.solve(self.JLeft.T, self.torS_l_ref)
            self.torP_r_ref[:] = np.linalg.solve(self.JRight.T, self.torS_r_ref)
            
            # 速度变换使用矩阵乘法
            self.qDotP_l_ref[:] = self.JLeft @ self.qDotS_l_ref
            self.qDotP_r_ref[:] = self.JRight @ self.qDotS_r_ref
        except np.linalg.LinAlgError:
            # fallback to pseudo-inverse semantics
            pinvL = np.linalg.pinv(self.JLeft)
            pinvR = np.linalg.pinv(self.JRight)
            self.torP_l_ref[:] = pinvL.T @ self.torS_l_ref
            self.torP_r_ref[:] = pinvR.T @ self.torS_r_ref
            self.qDotP_l_ref[:] = self.JLeft @ self.qDotS_l_ref
            self.qDotP_r_ref[:] = self.JRight @ self.qDotS_r_ref
        # clamp to +/-50 to match func_sp_trans.py
        np.clip(self.torP_l_ref, -50.0, 50.0, out=self.torP_l_ref)
        np.clip(self.torP_r_ref, -50.0, 50.0, out=self.torP_r_ref)
        return True
