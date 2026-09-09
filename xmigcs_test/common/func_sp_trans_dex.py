#!/usr/bin/env python3
"""
直接从 funcSPTrans.cpp 翻译而来，保留命名和数值以便与 C++ 输出一致。
"""
import numpy as np
import math

class FuncSPTrans:
    def __init__(self):
        # constants
        self.PI = math.pi
        # mechanical params (from C++)
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

        # OP vectors
        self.OP1_l = np.array([-41.48, 36.4, 19.88], dtype=np.float64)
        self.OP2_l = np.array([-41.48, -36.4, 19.88], dtype=np.float64)
        self.OP1_r = np.array([-41.48, -36.4, 19.88], dtype=np.float64)
        self.OP2_r = np.array([-41.48, 36.4, 19.88], dtype=np.float64)

        # refs
        self.roll_l_ref = 0.0
        self.pitch_l_ref = 0.0
        self.alpha1_l_ref = 0.3492
        self.alpha2_l_ref = 0.3492
        self.qP_l_ref = np.zeros(2)
        self.qDotP_l_ref = np.zeros(2)
        self.torP_l_ref = np.zeros(2)
        self.qS_l_ref = np.zeros(2)
        self.qDotS_l_ref = np.zeros(2)
        self.torS_l_ref = np.zeros(2)

        self.roll_r_ref = 0.0
        self.pitch_r_ref = 0.0
        self.alpha1_r_ref = 0.3492
        self.alpha2_r_ref = 0.3492
        self.qP_r_ref = np.zeros(2)
        self.qDotP_r_ref = np.zeros(2)
        self.torP_r_ref = np.zeros(2)
        self.qS_r_ref = np.zeros(2)
        self.qDotS_r_ref = np.zeros(2)
        self.torS_r_ref = np.zeros(2)

        # est
        self.roll_l_est = 0.0
        self.pitch_l_est = 0.0
        self.alpha1_l_est = 0.3492
        self.alpha2_l_est = 0.3492
        self.qP_l_est = np.zeros(2)
        self.qDotP_l_est = np.zeros(2)
        self.torP_l_est = np.zeros(2)
        self.qS_l_est = np.zeros(2)
        self.qDotS_l_est = np.zeros(2)
        self.torS_l_est = np.zeros(2)

        self.roll_r_est = 0.0
        self.pitch_r_est = 0.0
        self.alpha1_r_est = 0.3492
        self.alpha2_r_est = 0.3492
        self.qP_r_est = np.zeros(2)
        self.qDotP_r_est = np.zeros(2)
        self.torP_r_est = np.zeros(2)
        self.qS_r_est = np.zeros(2)
        self.qDotS_r_est = np.zeros(2)
        self.torS_r_est = np.zeros(2)

        # rotation matrices
        self.rotXLeft = np.zeros((3,3))
        self.rotYLeft = np.zeros((3,3))
        self.rotYXLeft = np.zeros((3,3))
        self.rotXLeftRef = np.zeros((3,3))
        self.rotYLeftRef = np.zeros((3,3))
        self.rotYXLeftRef = np.zeros((3,3))
        self.rotXRight = np.zeros((3,3))
        self.rotYRight = np.zeros((3,3))
        self.rotYXRight = np.zeros((3,3))
        self.rotXRightRef = np.zeros((3,3))
        self.rotYRightRef = np.zeros((3,3))
        self.rotYXRightRef = np.zeros((3,3))

        # body points
        self.oP1BodyLeft = np.zeros(3)
        self.oP2BodyLeft = np.zeros(3)
        self.oP1BodyLeftRef = np.zeros(3)
        self.oP2BodyLeftRef = np.zeros(3)
        self.oP1BodyRight = np.zeros(3)
        self.oP2BodyRight = np.zeros(3)
        self.oP1BodyRightRef = np.zeros(3)
        self.oP2BodyRightRef = np.zeros(3)

        # other vecs
        self.C1P1Left = np.zeros(3)
        self.C2P2Left = np.zeros(3)
        self.C1P1Right = np.zeros(3)
        self.C2P2Right = np.zeros(3)

        # Jacobians
        self.ROmegaLeft = np.zeros((3,2))
        self.ROmegaLegLeft = np.zeros((3,2))
        self.JP1Left = np.zeros((3,2))
        self.JP2Left = np.zeros((3,2))
        self.ROmegaRight = np.zeros((3,2))
        self.ROmegaLegRight = np.zeros((3,2))
        self.JP1Right = np.zeros((3,2))
        self.JP2Right = np.zeros((3,2))

        self.B1C1Left = np.zeros(3)
        self.B2C2Left = np.zeros(3)
        self.JLeft = np.zeros((2,2))
        self.JAnkleLeft = np.zeros((2,2))
        self.JC1Left = np.zeros((3,2))
        self.JC2Left = np.zeros((3,2))
        self.vP1Left = np.zeros(3)
        self.vP2Left = np.zeros(3)

        self.B1C1Right = np.zeros(3)
        self.B2C2Right = np.zeros(3)
        self.JRight = np.zeros((2,2))
        self.JAnkleRight = np.zeros((2,2))
        self.JC1Right = np.zeros((3,2))
        self.JC2Right = np.zeros((3,2))
        self.vP1Right = np.zeros(3)
        self.vP2Right = np.zeros(3)

    def Skew(self, omg):
        m = np.zeros((3,3))
        m[0,1] = -omg[2]
        m[0,2] = omg[1]
        m[1,0] = omg[2]
        m[1,2] = -omg[0]
        m[2,0] = -omg[1]
        m[2,1] = omg[0]
        return m

    def setPEst(self, qPEst, qDotPEst, qTorPEst):
        self.alpha1_l_est = qPEst[0] + 0.3492
        self.alpha2_l_est = qPEst[1] + 0.3492
        self.alpha1_r_est = qPEst[2] + 0.3492
        self.alpha2_r_est = qPEst[3] + 0.3492
        self.qP_l_est[:] = qPEst[:2]
        self.qP_r_est[:] = qPEst[2:]
        self.qDotP_l_est[:] = qDotPEst[:2]
        self.qDotP_r_est[:] = qDotPEst[2:]
        self.torP_l_est[:] = qTorPEst[:2]
        self.torP_r_est[:] = qTorPEst[2:]
        return True

    def calcFK(self):
        self.calcAnkleEstLeft()
        self.calcAnkleEstRight()
        return True

    def calcAnkleEstLeft(self):
        # neural net constants taken from C++
        b_a = np.array([5.94356918,   0.698980629, -2.87409472, -1.01680517, -4.25445032,
      2.43503571,   -1.23554814, 1.23667514,  0.786474645, -2.66088629,
      -2.22718477,  -2.23365259, 1.81063032,  0.0637441,   -1.25995994,
      2.73385882,   2.79696178,  -2.45870185, 0.643692911, 5.62071323,
      -0.147124752, 0.378021479, -4.25320578, 3.90843987,  -2.9711256,
      -4.35533476,  5.77593756,  -5.64612532, 5.67641592,  -0.773670137,
      -1.84441483,  -2.87270236, -2.22745919, 0.10766869,  2.75126028,
      -4.04213095,  -4.04839754, 2.68976188,  1.00575793,  -1.78429079,
      -1.47443664,  -2.17987609, 3.26092958,  -1.83372235, -1.28457642,
      -3.66284776,  3.99858475,  5.09313583,  3.16851783,  -4.53481102])
        c_a = np.array([-0.12816304,   -0.0906492099,  0.0967225,    -0.0920438766,
      0.0311508775,  -0.0592247397,  0.957266,     -0.0120375948,
      0.0395333804,  0.0282558668,   -0.165704682, -0.169937313,
      0.0365005322,  -0.0896773115,  1.39905655,   1.22731364,
      0.810329378,   0.567874134,    -0.15331161,  -0.049582541,
      0.698119521,   -0.400004566,   -0.693767846, 0.397573084,
      0.00196632301, -0.00473483559, -0.655982494, 0.821581066,
      0.0312418789,  -0.0368167497,  0.548184454,  0.0165984798,
      0.19014442,    0.0510320961,   -0.130089715, -0.0527587309,
      -0.513291419,  -1.32376087,    0.0928636342, 0.00108589,
      -0.0479600728, 0.0617735051,   -0.22801739,  0.0593482256,
      0.0210175235,  0.0085986387,   2.64173675,   -0.115078852,
      2.19358563,    -0.0805778876])
        fv = np.array([-5.81407595, -5.75171757, 5.80118036,  6.18498325,    4.05050945,
      -1.92026055, 2.33524013,  -1.62735295, -0.229573011,  0.0187892355,
      1.23916209,  1.24042356,  0.108361147, -0.0811663568, -1.8306812,
      1.65682483,  0.759762943, 0.885608077, -1.40363491,   4.284688,
      2.85224295,  4.18213511,  -5.30001831, 7.50161648,    -7.814466])
        fv1 = np.array([-1.01818216, 0.110170461])

        xp1_idx_0 = (float(self.alpha1_l_est) - -1.04791963) * 0.77919364 - 1.0
        xp1_idx_1 = (float(self.alpha2_l_est) - -1.05400908) * 0.784508467 - 1.0

        a = np.zeros(25)
        for k in range(25):
            # b_a has first-25 and second-25 groups like in C++: use b_a[k] and b_a[k+25]
            a[k] = 2.0 / (math.exp(-2.0 * ((b_a[k] * xp1_idx_0 + b_a[k + 25] * xp1_idx_1) + fv[k])) + 1.0) - 1.0
        # output
        b_y1 = np.zeros(2)
        for k in range(2):
            s = 0.0
            for i in range(25):
                s += c_a[k + (i<<1)] * a[i]
            b_y1[k] = ((s + fv1[k]) - -1.0) / ((-1.11047494 * k) + 2.29183125) + ((-0.872664571 * k) - 0.436332315)

        self.roll_l_est = float(b_y1[0])
        self.pitch_l_est = float(b_y1[1])
        return True

    def calcAnkleEstRight(self):
        # use same constants as left
        b_a = np.array([5.94356918,   0.698980629, -2.87409472, -1.01680517, -4.25445032,
      2.43503571,   -1.23554814, 1.23667514,  0.786474645, -2.66088629,
      -2.22718477,  -2.23365259, 1.81063032,  0.0637441,   -1.25995994,
      2.73385882,   2.79696178,  -2.45870185, 0.643692911, 5.62071323,
      -0.147124752, 0.378021479, -4.25320578, 3.90843987,  -2.9711256,
      -4.35533476,  5.77593756,  -5.64612532, 5.67641592,  -0.773670137,
      -1.84441483,  -2.87270236, -2.22745919, 0.10766869,  2.75126028,
      -4.04213095,  -4.04839754, 2.68976188,  1.00575793,  -1.78429079,
      -1.47443664,  -2.17987609, 3.26092958,  -1.83372235, -1.28457642,
      -3.66284776,  3.99858475,  5.09313583,  3.16851783,  -4.53481102])
        c_a = np.array([-0.12816304,   -0.0906492099,  0.0967225,    -0.0920438766,
      0.0311508775,  -0.0592247397,  0.957266,     -0.0120375948,
      0.0395333804,  0.0282558668,   -0.165704682, -0.169937313,
      0.0365005322,  -0.0896773115,  1.39905655,   1.22731364,
      0.810329378,   0.567874134,    -0.15331161,  -0.049582541,
      0.698119521,   -0.400004566,   -0.693767846, 0.397573084,
      0.00196632301, -0.00473483559, -0.655982494, 0.821581066,
      0.0312418789,  -0.0368167497,  0.548184454,  0.0165984798,
      0.19014442,    0.0510320961,   -0.130089715, -0.0527587309,
      -0.513291419,  -1.32376087,    0.0928636342, 0.00108589,
      -0.0479600728, 0.0617735051,   -0.22801739,  0.0593482256,
      0.0210175235,  0.0085986387,   2.64173675,   -0.115078852,
      2.19358563,    -0.0805778876])
        fv = np.array([-5.81407595, -5.75171757, 5.80118036,  6.18498325,    4.05050945,
      -1.92026055, 2.33524013,  -1.62735295, -0.229573011,  0.0187892355,
      1.23916209,  1.24042356,  0.108361147, -0.0811663568, -1.8306812,
      1.65682483,  0.759762943, 0.885608077, -1.40363491,   4.284688,
      2.85224295,  4.18213511,  -5.30001831, 7.50161648,    -7.814466])
        fv1 = np.array([-1.01818216, 0.110170461])

        xp1_idx_0 = (float(self.alpha1_r_est) - -1.04791963) * 0.77919364 - 1.0
        xp1_idx_1 = (float(self.alpha2_r_est) - -1.05400908) * 0.784508467 - 1.0

        a = np.zeros(25)
        for k in range(25):
            a[k] = 2.0 / (math.exp(-2.0 * ((b_a[k] * xp1_idx_0 + b_a[k + 25] * xp1_idx_1) + fv[k])) + 1.0) - 1.0
        # output
        b_y1 = np.zeros(2)
        for k in range(2):
            s = 0.0
            for i in range(25):
                s += c_a[k + (i<<1)] * a[i]
            b_y1[k] = ((s + fv1[k]) - -1.0) / ((-1.11047494 * k) + 2.29183125) + ((-0.872664571 * k) - 0.436332315)

        self.roll_r_est = float(-b_y1[0])
        self.pitch_r_est = float(b_y1[1])
        return True

    def calcIK(self):
        self.calcJLeft()
        self.calcJRight()
        return True

    def calcJLeft(self):
        # rotX
        self.rotXLeft[:] = np.array([[1.0,0.0,0.0],[0.0,math.cos(self.roll_l_est),-math.sin(self.roll_l_est)],[0.0,math.sin(self.roll_l_est),math.cos(self.roll_l_est)]])
        self.rotYLeft[:] = np.array([[math.cos(self.pitch_l_est),0.0,math.sin(self.pitch_l_est)],[0.0,1.0,0.0],[-math.sin(self.pitch_l_est),0.0,math.cos(self.pitch_l_est)]])
        self.rotYXLeft = self.rotYLeft @ self.rotXLeft
        self.oP1BodyLeft = self.rotYXLeft @ self.OP1_l
        self.oP2BodyLeft = self.rotYXLeft @ self.OP2_l

        self.C1P1Left[0] = self.oP1BodyLeft[0] + self.BC_l * math.cos(self.alpha1_l_est)
        self.C1P1Left[1] = self.oP1BodyLeft[1] - self.AB_l
        self.C1P1Left[2] = self.oP1BodyLeft[2] - self.OA1_l - self.BC_l * math.sin(self.alpha1_l_est)

        self.C2P2Left[0] = self.oP2BodyLeft[0] + self.BC_l * math.cos(self.alpha2_l_est)
        self.C2P2Left[1] = self.oP2BodyLeft[1] + self.AB_l
        self.C2P2Left[2] = self.oP2BodyLeft[2] - self.OA2_l - self.BC_l * math.sin(self.alpha2_l_est)

        self.ROmegaLeft[0,0] = 0.0
        self.ROmegaLeft[0,1] = self.rotYXLeft[0,0]
        self.ROmegaLeft[1,0] = 1.0
        self.ROmegaLeft[1,1] = self.rotYXLeft[1,0]
        self.ROmegaLeft[2,0] = 0.0
        self.ROmegaLeft[2,1] = self.rotYXLeft[2,0]

        self.JP1Left = -self.Skew(self.oP1BodyLeft) @ self.ROmegaLeft
        self.JP2Left = -self.Skew(self.oP2BodyLeft) @ self.ROmegaLeft

        self.B1C1Left[0] = -self.BC_l * math.cos(self.alpha1_l_est)
        self.B1C1Left[1] = 0.0
        self.B1C1Left[2] = self.BC_l * math.sin(self.alpha1_l_est)

        self.B2C2Left[0] = -self.BC_l * math.cos(self.alpha2_l_est)
        self.B2C2Left[1] = 0.0
        self.B2C2Left[2] = self.BC_l * math.sin(self.alpha2_l_est)

        denom1 = self.B1C1Left[2] * self.C1P1Left[0] - self.B1C1Left[0] * self.C1P1Left[2]
        denom2 = self.B2C2Left[2] * self.C2P2Left[0] - self.B2C2Left[0] * self.C2P2Left[2]

        c1 = np.dot(self.C1P1Left, self.JP1Left)
        c2 = np.dot(self.C2P2Left, self.JP2Left)
        self.JAnkleLeft[0,:] = c1 / denom1
        self.JAnkleLeft[1,:] = c2 / denom2
        self.JLeft = self.JAnkleLeft.copy()

        try:
            # match Eigen pseudoInverse behavior
            pinv = np.linalg.pinv(self.JLeft)
            self.qDotS_l_est = pinv @ self.qDotP_l_est
            self.torS_l_est = self.JLeft.T @ self.torP_l_est
        except np.linalg.LinAlgError:
            self.qDotS_l_est.fill(0.0)
            self.torS_l_est.fill(0.0)
        return True

    def calcJRight(self):
        self.rotXRight[:] = np.array([[1.0,0.0,0.0],[0.0,math.cos(self.roll_r_est),-math.sin(self.roll_r_est)],[0.0,math.sin(self.roll_r_est),math.cos(self.roll_r_est)]])
        self.rotYRight[:] = np.array([[math.cos(self.pitch_r_est),0.0,math.sin(self.pitch_r_est)],[0.0,1.0,0.0],[-math.sin(self.pitch_r_est),0.0,math.cos(self.pitch_r_est)]])
        self.rotYXRight = self.rotYRight @ self.rotXRight
        self.oP1BodyRight = self.rotYXRight @ self.OP1_r
        self.oP2BodyRight = self.rotYXRight @ self.OP2_r

        self.C1P1Right[0] = self.oP1BodyRight[0] + self.BC_r * math.cos(self.alpha1_r_est)
        self.C1P1Right[1] = self.oP1BodyRight[1] + self.AB_r
        self.C1P1Right[2] = self.oP1BodyRight[2] - self.OA1_r - self.BC_r * math.sin(self.alpha1_r_est)

        self.C2P2Right[0] = self.oP2BodyRight[0] + self.BC_r * math.cos(self.alpha2_r_est)
        self.C2P2Right[1] = self.oP2BodyRight[1] - self.AB_r
        self.C2P2Right[2] = self.oP2BodyRight[2] - self.OA2_r - self.BC_r * math.sin(self.alpha2_r_est)

        self.ROmegaRight[0,0] = 0.0
        self.ROmegaRight[0,1] = self.rotYXRight[0,0]
        self.ROmegaRight[1,0] = 1.0
        self.ROmegaRight[1,1] = self.rotYXRight[1,0]
        self.ROmegaRight[2,0] = 0.0
        self.ROmegaRight[2,1] = self.rotYXRight[2,0]

        self.JP1Right = -self.Skew(self.oP1BodyRight) @ self.ROmegaRight
        self.JP2Right = -self.Skew(self.oP2BodyRight) @ self.ROmegaRight

        self.B1C1Right[0] = -self.BC_r * math.cos(self.alpha1_r_est)
        self.B1C1Right[1] = 0.0
        self.B1C1Right[2] = self.BC_r * math.sin(self.alpha1_r_est)

        self.B2C2Right[0] = -self.BC_r * math.cos(self.alpha2_r_est)
        self.B2C2Right[1] = 0.0
        self.B2C2Right[2] = self.BC_r * math.sin(self.alpha2_r_est)

        denom1 = self.B1C1Right[2] * self.C1P1Right[0] - self.B1C1Right[0] * self.C1P1Right[2]
        denom2 = self.B2C2Right[2] * self.C2P2Right[0] - self.B2C2Right[0] * self.C2P2Right[2]

        c1 = np.dot(self.C1P1Right, self.JP1Right)
        c2 = np.dot(self.C2P2Right, self.JP2Right)
        self.JAnkleRight[0,:] = c1 / denom1
        self.JAnkleRight[1,:] = c2 / denom2
        self.JRight = self.JAnkleRight.copy()

        try:
            pinv = np.linalg.pinv(self.JRight)
            self.qDotS_r_est = pinv @ self.qDotP_r_est
            self.torS_r_est = self.JRight.T @ self.torP_r_est
        except np.linalg.LinAlgError:
            self.qDotS_r_est.fill(0.0)
            self.torS_r_est.fill(0.0)
        return True

    def getSState(self):
        qS_est = np.zeros(4)
        qDotS_est = np.zeros(4)
        torS_est = np.zeros(4)
        qS_est[0] = self.pitch_l_est + 0.09765
        qS_est[1] = self.roll_l_est
        qS_est[2] = self.pitch_r_est + 0.09765
        qS_est[3] = self.roll_r_est
        qDotS_est[:2] = self.qDotS_l_est
        qDotS_est[2:] = self.qDotS_r_est
        torS_est[:2] = self.torS_l_est
        torS_est[2:] = self.torS_r_est
        return qS_est, qDotS_est, torS_est

    def setSState(self, qSEst):
        self.roll_l_est = qSEst[1]
        self.pitch_l_est = qSEst[0] - 0.09765
        self.roll_r_est = qSEst[3]
        self.pitch_r_est = qSEst[2] - 0.09765
        self.qS_l_est = qSEst[:2].copy()
        self.qS_r_est = qSEst[2:].copy()
        return True

    def calcJointPosEst(self):
        self.calcJointPosEstLeft()
        self.calcJointPosEstRight()
        return True

    def calcJointPosEstLeft(self):
        # rotation ref
        self.rotXLeftRef[:] = np.array([[1.0,0.0,0.0],[0.0,math.cos(self.roll_l_est),-math.sin(self.roll_l_est)],[0.0,math.sin(self.roll_l_est),math.cos(self.roll_l_est)]])
        self.rotYLeftRef[:] = np.array([[math.cos(self.pitch_l_est),0.0,math.sin(self.pitch_l_est)],[0.0,1.0,0.0],[-math.sin(self.pitch_l_est),0.0,math.cos(self.pitch_l_est)]])
        rot = self.rotYLeftRef @ self.rotXLeftRef
        self.oP1BodyLeftRef = rot @ self.OP1_l
        self.oP2BodyLeftRef = rot @ self.OP2_l

        a1 = -2.0 * self.oP1BodyLeftRef[0] * self.BC_l
        b1 = -2.0 * (self.OA1_l - self.oP1BodyLeftRef[2]) * self.BC_l
        c1 = (self.oP1BodyLeftRef[0]**2 + (self.oP1BodyLeftRef[1] - self.AB_l)**2 + (self.oP1BodyLeftRef[2] - self.OA1_l)**2 + self.BC_l**2 - self.C1P1_l**2)

        a2 = -2.0 * self.oP2BodyLeftRef[0] * self.BC_l
        b2 = -2.0 * (self.OA2_l - self.oP2BodyLeftRef[2]) * self.BC_l
        c2 = (self.oP2BodyLeftRef[0]**2 + (self.oP2BodyLeftRef[1] + self.AB_l)**2 + (self.oP2BodyLeftRef[2] - self.OA2_l)**2 + self.BC_l**2 - self.C2P2_l**2)

        if a1*a1 + b1*b1 - c1*c1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        if a2*a2 + b2*b2 - c2*c2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        csalpha11 = (a1*c1 + b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha12 = (a1*c1 - b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha21 = (a2*c2 + b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)
        csalpha22 = (a2*c2 - b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)

        if abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0:
            print("Error: Solution out of bounds for left leg")
            return False

        alpha1_p = [math.acos(csalpha11), -math.acos(csalpha11), math.acos(csalpha12), -math.acos(csalpha12)]
        alpha2_p = [math.acos(csalpha21), -math.acos(csalpha21), math.acos(csalpha22), -math.acos(csalpha22)]

        for a1c in alpha1_p:
            if -self.PI/2 <= a1c <= self.PI/2:
                if abs(a1*math.cos(a1c) + b1*math.sin(a1c) - c1) <= 1e-7:
                    self.alpha1_l_est = a1c
                    break
        for a2c in alpha2_p:
            if -self.PI/2 <= a2c <= self.PI/2:
                if abs(a2*math.cos(a2c) + b2*math.sin(a2c) - c2) <= 1e-7:
                    self.alpha2_l_est = a2c
                    break
        self.qP_l_est[0] = self.alpha1_l_est - 0.3492
        self.qP_l_est[1] = self.alpha2_l_est - 0.3492
        return True

    def calcJointPosEstRight(self):
        self.rotXRightRef[:] = np.array([[1.0,0.0,0.0],[0.0,math.cos(self.roll_r_est),-math.sin(self.roll_r_est)],[0.0,math.sin(self.roll_r_est),math.cos(self.roll_r_est)]])
        self.rotYRightRef[:] = np.array([[math.cos(self.pitch_r_est),0.0,math.sin(self.pitch_r_est)],[0.0,1.0,0.0],[-math.sin(self.pitch_r_est),0.0,math.cos(self.pitch_r_est)]])
        rot = self.rotYRightRef @ self.rotXRightRef
        self.oP1BodyRightRef = rot @ self.OP1_r
        self.oP2BodyRightRef = rot @ self.OP2_r

        a1 = -2.0 * self.oP1BodyRightRef[0] * self.BC_r
        b1 = -2.0 * (self.OA1_r - self.oP1BodyRightRef[2]) * self.BC_r
        c1 = (self.oP1BodyRightRef[0]**2 + (self.oP1BodyRightRef[1] + self.AB_r)**2 + (self.oP1BodyRightRef[2] - self.OA1_r)**2 + self.BC_r**2 - self.C1P1_r**2)

        a2 = -2.0 * self.oP2BodyRightRef[0] * self.BC_r
        b2 = -2.0 * (self.OA2_r - self.oP2BodyRightRef[2]) * self.BC_r
        c2 = (self.oP2BodyRightRef[0]**2 + (self.oP2BodyRightRef[1] - self.AB_r)**2 + (self.oP2BodyRightRef[2] - self.OA2_r)**2 + self.BC_r**2 - self.C2P2_r**2)

        if a1*a1 + b1*b1 - c1*c1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        if a2*a2 + b2*b2 - c2*c2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        csalpha11 = (a1*c1 + b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha12 = (a1*c1 - b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha21 = (a2*c2 + b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)
        csalpha22 = (a2*c2 - b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)

        if abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0:
            print("Error: Solution out of bounds for right leg")
            return False

        alpha1_p = [math.acos(csalpha11), -math.acos(csalpha11), math.acos(csalpha12), -math.acos(csalpha12)]
        alpha2_p = [math.acos(csalpha21), -math.acos(csalpha21), math.acos(csalpha22), -math.acos(csalpha22)]

        for a1c in alpha1_p:
            if -self.PI/2 <= a1c <= self.PI/2:
                if abs(a1*math.cos(a1c) + b1*math.sin(a1c) - c1) <= 1e-7:
                    self.alpha1_r_ref = a1c
                    break
        for a2c in alpha2_p:
            if -self.PI/2 <= a2c <= self.PI/2:
                if abs(a2*math.cos(a2c) + b2*math.sin(a2c) - c2) <= 1e-7:
                    self.alpha2_r_ref = a2c
                    break
        self.qP_r_ref[0] = self.alpha1_r_ref - 0.3492
        self.qP_r_ref[1] = self.alpha2_r_ref - 0.3492
        return True

    def setSDes(self, qSRef, qDotSRef, torSDes):
        self.roll_l_ref = qSRef[1]
        self.pitch_l_ref = qSRef[0] - 0.09765
        self.roll_r_ref = qSRef[3]
        self.pitch_r_ref = qSRef[2] - 0.09765
        self.qS_l_ref[:] = qSRef[:2]
        self.qS_r_ref[:] = qSRef[2:]
        self.qDotS_l_ref[:] = qDotSRef[:2]
        self.qDotS_r_ref[:] = qDotSRef[2:]
        self.torS_l_ref[:] = torSDes[:2]
        self.torS_r_ref[:] = torSDes[2:]
        return True

    def getPDes(self):
        qPDes = np.zeros(4)
        qDotPDes = np.zeros(4)
        torPDes = np.zeros(4)
        qPDes[:2] = self.qP_l_ref
        qPDes[2:] = self.qP_r_ref
        qDotPDes[:2] = self.qDotP_l_ref
        qDotPDes[2:] = self.qDotP_r_ref
        torPDes[:2] = self.torP_l_ref
        torPDes[2:] = self.torP_r_ref
        return qPDes, qDotPDes, torPDes

    def calcJointPosRef(self):
        self.calcJointPosRefLeft()
        self.calcJointPosRefRight()
        return True

    def calcJointPosRefLeft(self):
        self.rotXLeftRef[:] = np.array([[1.0,0.0,0.0],[0.0,math.cos(self.roll_l_ref),-math.sin(self.roll_l_ref)],[0.0,math.sin(self.roll_l_ref),math.cos(self.roll_l_ref)]])
        self.rotYLeftRef[:] = np.array([[math.cos(self.pitch_l_ref),0.0,math.sin(self.pitch_l_ref)],[0.0,1.0,0.0],[-math.sin(self.pitch_l_ref),0.0,math.cos(self.pitch_l_ref)]])
        rot = self.rotYLeftRef @ self.rotXLeftRef
        self.oP1BodyLeftRef = rot @ self.OP1_l
        self.oP2BodyLeftRef = rot @ self.OP2_l

        a1 = -2.0 * self.oP1BodyLeftRef[0] * self.BC_l
        b1 = -2.0 * (self.OA1_l - self.oP1BodyLeftRef[2]) * self.BC_l
        c1 = (self.oP1BodyLeftRef[0]**2 + (self.oP1BodyLeftRef[1] - self.AB_l)**2 + (self.oP1BodyLeftRef[2] - self.OA1_l)**2 + self.BC_l**2 - self.C1P1_l**2)

        a2 = -2.0 * self.oP2BodyLeftRef[0] * self.BC_l
        b2 = -2.0 * (self.OA2_l - self.oP2BodyLeftRef[2]) * self.BC_l
        c2 = (self.oP2BodyLeftRef[0]**2 + (self.oP2BodyLeftRef[1] + self.AB_l)**2 + (self.oP2BodyLeftRef[2] - self.OA2_l)**2 + self.BC_l**2 - self.C2P2_l**2)

        if a1*a1 + b1*b1 - c1*c1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        if a2*a2 + b2*b2 - c2*c2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        csalpha11 = (a1*c1 + b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha12 = (a1*c1 - b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha21 = (a2*c2 + b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)
        csalpha22 = (a2*c2 - b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)

        if abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0:
            print("Error: Solution out of bounds for left leg")
            return False

        alpha1_p = [math.acos(csalpha11), -math.acos(csalpha11), math.acos(csalpha12), -math.acos(csalpha12)]
        alpha2_p = [math.acos(csalpha21), -math.acos(csalpha21), math.acos(csalpha22), -math.acos(csalpha22)]

        for a1c in alpha1_p:
            if -self.PI/2 <= a1c <= self.PI/2:
                if abs(a1*math.cos(a1c) + b1*math.sin(a1c) - c1) <= 1e-7:
                    self.alpha1_l_ref = a1c
                    break
        for a2c in alpha2_p:
            if -self.PI/2 <= a2c <= self.PI/2:
                if abs(a2*math.cos(a2c) + b2*math.sin(a2c) - c2) <= 1e-7:
                    self.alpha2_l_ref = a2c
                    break
        self.qP_l_ref[0] = self.alpha1_l_ref - 0.3492
        self.qP_l_ref[1] = self.alpha2_l_ref - 0.3492
        return True

    def calcJointPosRefRight(self):
        self.rotXRightRef[:] = np.array([[1.0,0.0,0.0],[0.0,math.cos(self.roll_r_ref),-math.sin(self.roll_r_ref)],[0.0,math.sin(self.roll_r_ref),math.cos(self.roll_r_ref)]])
        self.rotYRightRef[:] = np.array([[math.cos(self.pitch_r_ref),0.0,math.sin(self.pitch_r_ref)],[0.0,1.0,0.0],[-math.sin(self.pitch_r_ref),0.0,math.cos(self.pitch_r_ref)]])
        rot = self.rotYRightRef @ self.rotXRightRef
        self.oP1BodyRightRef = rot @ self.OP1_r
        self.oP2BodyRightRef = rot @ self.OP2_r

        a1 = -2.0 * self.oP1BodyRightRef[0] * self.BC_r
        b1 = -2.0 * (self.OA1_r - self.oP1BodyRightRef[2]) * self.BC_r
        c1 = (self.oP1BodyRightRef[0]**2 + (self.oP1BodyRightRef[1] + self.AB_r)**2 + (self.oP1BodyRightRef[2] - self.OA1_r)**2 + self.BC_r**2 - self.C1P1_r**2)

        a2 = -2.0 * self.oP2BodyRightRef[0] * self.BC_r
        b2 = -2.0 * (self.OA2_r - self.oP2BodyRightRef[2]) * self.BC_r
        c2 = (self.oP2BodyRightRef[0]**2 + (self.oP2BodyRightRef[1] - self.AB_r)**2 + (self.oP2BodyRightRef[2] - self.OA2_r)**2 + self.BC_r**2 - self.C2P2_r**2)

        if a1*a1 + b1*b1 - c1*c1 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        if a2*a2 + b2*b2 - c2*c2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        csalpha11 = (a1*c1 + b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha12 = (a1*c1 - b1*math.sqrt(a1*a1 + b1*b1 - c1*c1)) / (a1*a1 + b1*b1)
        csalpha21 = (a2*c2 + b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)
        csalpha22 = (a2*c2 - b2*math.sqrt(a2*a2 + b2*b2 - c2*c2)) / (a2*a2 + b2*b2)

        if abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0:
            print("Error: Solution out of bounds for right leg")
            return False

        alpha1_p = [math.acos(csalpha11), -math.acos(csalpha11), math.acos(csalpha12), -math.acos(csalpha12)]
        alpha2_p = [math.acos(csalpha21), -math.acos(csalpha21), math.acos(csalpha22), -math.acos(csalpha22)]

        for a1c in alpha1_p:
            if -self.PI/2 <= a1c <= self.PI/2:
                if abs(a1*math.cos(a1c) + b1*math.sin(a1c) - c1) <= 1e-7:
                    self.alpha1_r_ref = a1c
                    break
        for a2c in alpha2_p:
            if -self.PI/2 <= a2c <= self.PI/2:
                if abs(a2*math.cos(a2c) + b2*math.sin(a2c) - c2) <= 1e-7:
                    self.alpha2_r_ref = a2c
                    break
        self.qP_r_ref[0] = self.alpha1_r_ref - 0.3492
        self.qP_r_ref[1] = self.alpha2_r_ref - 0.3492
        return True

    def calcJointTorDes(self):
        try:
            pinvL = np.linalg.pinv(self.JLeft)
            pinvR = np.linalg.pinv(self.JRight)
            # match C++: torP = pinv(J).T * torS
            self.torP_l_ref[:] = pinvL.T @ self.torS_l_ref
            self.torP_r_ref[:] = pinvR.T @ self.torS_r_ref
            self.qDotP_l_ref[:] = self.JLeft @ self.qDotS_l_ref
            self.qDotP_r_ref[:] = self.JRight @ self.qDotS_r_ref
        except np.linalg.LinAlgError:
            self.torP_l_ref.fill(0.0)
            self.torP_r_ref.fill(0.0)
            self.qDotP_l_ref.fill(0.0)
            self.qDotP_r_ref.fill(0.0)
        # match C++ max torque limit
        max_torque = 50.0
        if self.torP_l_ref[0] >= max_torque:
            self.torP_l_ref[0] = max_torque
        if self.torP_l_ref[0] <= -max_torque:
            self.torP_l_ref[0] = -max_torque
        if self.torP_l_ref[1] >= max_torque:
            self.torP_l_ref[1] = max_torque
        if self.torP_l_ref[1] <= -max_torque:
            self.torP_l_ref[1] = -max_torque
        if self.torP_r_ref[0] >= max_torque:
            self.torP_r_ref[0] = max_torque
        if self.torP_r_ref[0] <= -max_torque:
            self.torP_r_ref[0] = -max_torque
        if self.torP_r_ref[1] >= max_torque:
            self.torP_r_ref[1] = max_torque
        if self.torP_r_ref[1] <= -max_torque:
            self.torP_r_ref[1] = -max_torque
        return True

# end of file
