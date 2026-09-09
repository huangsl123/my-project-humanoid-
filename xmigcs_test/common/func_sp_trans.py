"""
FuncSPTrans Python Implementation
Complete Python equivalent of the C++ funcSPTrans class for serial-parallel transformation
"""
import numpy as np
import math
from typing import Tuple, Union


class FuncSPTrans:
    """Python implementation of funcSPTrans for serial-parallel transformation"""

    def __init__(self):
        """Initialize the FuncSPTrans with all parameters and variables"""
        # Constants
        self.PI = math.pi

        # Mechanical parameters (mm)
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

        # Position vectors (exactly as in C++)
        self.OP1_l = np.array([-55.82, 45.0, 22.0])
        self.OP2_l = np.array([-55.82, -45.0, 22.0])
        self.OP1_r = np.array([-55.82, -45.0, 22.0])
        self.OP2_r = np.array([-55.82, 45.0, 22.0])

        # Left leg reference values
        self.roll_l_ref = 0.0
        self.pitch_l_ref = 0.0
        self.alpha1_l_ref = 0.3754
        self.alpha2_l_ref = 0.3754
        self.qP_l_ref = np.zeros(2)
        self.qDotP_l_ref = np.zeros(2)
        self.torP_l_ref = np.zeros(2)
        self.qS_l_ref = np.zeros(2)
        self.qDotS_l_ref = np.zeros(2)
        self.torS_l_ref = np.zeros(2)

        # Right leg reference values
        self.roll_r_ref = 0.0
        self.pitch_r_ref = 0.0
        self.alpha1_r_ref = 0.3754
        self.alpha2_r_ref = 0.3754
        self.qP_r_ref = np.zeros(2)
        self.qDotP_r_ref = np.zeros(2)
        self.torP_r_ref = np.zeros(2)
        self.qS_r_ref = np.zeros(2)
        self.qDotS_r_ref = np.zeros(2)
        self.torS_r_ref = np.zeros(2)

        # Left leg estimated values
        self.roll_l_est = 0.0
        self.pitch_l_est = 0.0
        self.alpha1_l_est = 0.3754
        self.alpha2_l_est = 0.3754
        self.qP_l_est = np.zeros(2)
        self.qDotP_l_est = np.zeros(2)
        self.torP_l_est = np.zeros(2)
        self.qS_l_est = np.zeros(2)
        self.qDotS_l_est = np.zeros(2)
        self.torS_l_est = np.zeros(2)

        # Right leg estimated values
        self.roll_r_est = 0.0
        self.pitch_r_est = 0.0
        self.alpha1_r_est = 0.3754
        self.alpha2_r_est = 0.3754
        self.qP_r_est = np.zeros(2)
        self.qDotP_r_est = np.zeros(2)
        self.torP_r_est = np.zeros(2)
        self.qS_r_est = np.zeros(2)
        self.qDotS_r_est = np.zeros(2)
        self.torS_r_est = np.zeros(2)

        # Rotation matrices
        self.rotXLeft = np.zeros((3, 3))
        self.rotYLeft = np.zeros((3, 3))
        self.rotYXLeft = np.zeros((3, 3))
        self.rotXLeftRef = np.zeros((3, 3))
        self.rotYLeftRef = np.zeros((3, 3))
        self.rotYXLeftRef = np.zeros((3, 3))
        self.rotXRight = np.zeros((3, 3))
        self.rotYRight = np.zeros((3, 3))
        self.rotYXRight = np.zeros((3, 3))
        self.rotXRightRef = np.zeros((3, 3))
        self.rotYRightRef = np.zeros((3, 3))
        self.rotYXRightRef = np.zeros((3, 3))

        # Position vectors
        self.oP1BodyLeft = np.zeros(3)
        self.oP2BodyLeft = np.zeros(3)
        self.oP1BodyLeftRef = np.zeros(3)
        self.oP2BodyLeftRef = np.zeros(3)
        self.oP1BodyRight = np.zeros(3)
        self.oP2BodyRight = np.zeros(3)
        self.oP1BodyRightRef = np.zeros(3)
        self.oP2BodyRightRef = np.zeros(3)

        # Other vectors
        self.C1P1Left = np.zeros(3)
        self.C2P2Left = np.zeros(3)
        self.C1P1Right = np.zeros(3)
        self.C2P2Right = np.zeros(3)

        # Jacobian matrices
        self.ROmegaLeft = np.zeros((3, 2))
        self.ROmegaLegLeft = np.zeros((3, 2))
        self.JP1Left = np.zeros((3, 2))
        self.JP2Left = np.zeros((3, 2))
        self.ROmegaRight = np.zeros((3, 2))
        self.ROmegaLegRight = np.zeros((3, 2))
        self.JP1Right = np.zeros((3, 2))
        self.JP2Right = np.zeros((3, 2))

        self.B1C1Left = np.zeros(3)
        self.B2C2Left = np.zeros(3)
        self.JLeft = np.zeros((2, 2))
        self.JAnkleLeft = np.zeros((2, 2))
        self.JC1Left = np.zeros((3, 2))
        self.JC2Left = np.zeros((3, 2))
        self.vP1Left = np.zeros(3)
        self.vP2Left = np.zeros(3)

        self.B1C1Right = np.zeros(3)
        self.B2C2Right = np.zeros(3)
        self.JRight = np.zeros((2, 2))
        self.JAnkleRight = np.zeros((2, 2))
        self.JC1Right = np.zeros((3, 2))
        self.JC2Right = np.zeros((3, 2))
        self.vP1Right = np.zeros(3)
        self.vP2Right = np.zeros(3)

        self.tauDesjointFB = np.zeros(4)

        # Neural network parameters for left ankle (exactly as in C++)
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

        # Same parameters for right ankle (in the C++ code, they use identical parameters)
        self.b_a_right = self.b_a_left.copy()
        self.c_a_right = self.c_a_left.copy()
        self.fv_right = self.fv_left.copy()

    def Skew(self, omg: np.ndarray) -> np.ndarray:
        """Create skew symmetric matrix from 3D vector"""
        return np.array([
            [0.0, -omg[2], omg[1]],
            [omg[2], 0.0, -omg[0]],
            [-omg[1], omg[0], 0.0]
        ])

    def setPEst(self, qPEst: np.ndarray, qDotPEst: np.ndarray, qTorPEst: np.ndarray) -> bool:
        """Set parallel joint estimations (exactly as C++)"""
        self.alpha1_l_est = qPEst[0] + 0.3754
        self.alpha2_l_est = qPEst[1] + 0.3754
        self.alpha1_r_est = qPEst[2] + 0.3754
        self.alpha2_r_est = qPEst[3] + 0.3754

        self.qP_l_est = qPEst[:2].copy()
        self.qP_r_est = qPEst[2:].copy()
        self.qDotP_l_est = qDotPEst[:2].copy()
        self.qDotP_r_est = qDotPEst[2:].copy()
        self.torP_l_est = qTorPEst[:2].copy()
        self.torP_r_est = qTorPEst[2:].copy()
        return True

    def calcFK(self) -> bool:
        """Calculate forward kinematics (exactly as C++)"""
        self.calcAnkleEstLeft()
        self.calcAnkleEstRight()
        return True

    def calcAnkleEstLeft(self) -> bool:
        """Calculate left ankle estimation using neural network (exactly as C++)"""
        # Input normalization (exactly as C++)
        xp1_idx_0 = (float(self.alpha1_l_est) - (-1.05928469)) * 0.761271894 - 1.0
        xp1_idx_1 = (float(self.alpha2_l_est) - (-1.06109619)) * 0.77644676 - 1.0

        # Hidden layer computation (Sigmoid activation) - exactly as C++
        a = np.zeros(25, dtype=np.float32)
        for k in range(25):
            net = (self.b_a_left[k] * xp1_idx_0 + self.b_a_left[k + 25] * xp1_idx_1 + self.fv_left[k])
            a[k] = 2.0 / (np.exp(-2.0 * net) + 1.0) - 1.0

        # Output layer computation - exactly as C++
        b_y1 = np.zeros(2, dtype=np.float32)
        for k in range(2):
            xp1_idx_0_out = 0.0
            for i in range(25):
                xp1_idx_0_out += self.c_a_left[k + (i << 1)] * a[i]

            # Output denormalization - exactly as C++
            offset = -0.130409107 * k - 0.0106069511
            scale = -1.14591563 * k + 2.29183125
            final_offset = -0.785398126 * k - 0.436332315
            b_y1[k] = ((xp1_idx_0_out + offset) - (-1.0)) / scale + final_offset

        self.roll_l_est = float(b_y1[0])
        self.pitch_l_est = float(b_y1[1])
        return True

    def calcAnkleEstRight(self) -> bool:
        """Calculate right ankle estimation using neural network (exactly as C++)"""
        # Input normalization (exactly as C++)
        xp1_idx_0 = (float(self.alpha1_r_est) - (-1.05928469)) * 0.761271894 - 1.0
        xp1_idx_1 = (float(self.alpha2_r_est) - (-1.06109619)) * 0.77644676 - 1.0

        # Hidden layer computation (Sigmoid activation) - exactly as C++
        a = np.zeros(25, dtype=np.float32)
        for k in range(25):
            net = (self.b_a_right[k] * xp1_idx_0 + self.b_a_right[k + 25] * xp1_idx_1 + self.fv_right[k])
            a[k] = 2.0 / (np.exp(-2.0 * net) + 1.0) - 1.0

        # Output layer computation - exactly as C++
        b_y1 = np.zeros(2, dtype=np.float32)
        for k in range(2):
            xp1_idx_0_out = 0.0
            for i in range(25):
                xp1_idx_0_out += self.c_a_right[k + (i << 1)] * a[i]

            # Output denormalization - exactly as C++
            offset = -0.130409107 * k - 0.0106069511
            scale = -1.14591563 * k + 2.29183125
            final_offset = -0.785398126 * k - 0.436332315
            b_y1[k] = ((xp1_idx_0_out + offset) - (-1.0)) / scale + final_offset

        # Note: Right ankle has negative roll (exactly as in C++)
        self.roll_r_est = float(-b_y1[0])
        self.pitch_r_est = float(b_y1[1])
        return True

    def calcIK(self) -> bool:
        """Calculate inverse kinematics (exactly as C++)"""
        self.calcJLeft()
        self.calcJRight()
        return True

    def calcJLeft(self) -> bool:
        """Calculate Jacobian for left leg (exactly as C++)"""
        # Rotation matrices - exactly as C++
        self.rotXLeft = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll_l_est), -np.sin(self.roll_l_est)],
            [0.0, np.sin(self.roll_l_est), np.cos(self.roll_l_est)]
        ])

        self.rotYLeft = np.array([
            [np.cos(self.pitch_l_est), 0.0, np.sin(self.pitch_l_est)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch_l_est), 0.0, np.cos(self.pitch_l_est)]
        ])

        self.rotYXLeft = self.rotYLeft @ self.rotXLeft
        self.oP1BodyLeft = self.rotYXLeft @ self.OP1_l
        self.oP2BodyLeft = self.rotYXLeft @ self.OP2_l

        # Position calculations - exactly as C++
        self.C1P1Left = np.array([
            self.oP1BodyLeft[0] + self.BC_l * np.cos(self.alpha1_l_est),
            self.oP1BodyLeft[1] - self.AB_l,
            self.oP1BodyLeft[2] - self.OA1_l - self.BC_l * np.sin(self.alpha1_l_est)
        ])

        self.C2P2Left = np.array([
            self.oP2BodyLeft[0] + self.BC_l * np.cos(self.alpha2_l_est),
            self.oP2BodyLeft[1] + self.AB_l,
            self.oP2BodyLeft[2] - self.OA2_l - self.BC_l * np.sin(self.alpha2_l_est)
        ])

        self.ROmegaLeft = np.array([
            [0.0, self.rotYXLeft[0, 0]],
            [1.0, self.rotYXLeft[1, 0]],
            [0.0, self.rotYXLeft[2, 0]]
        ])

        self.JP1Left = -self.Skew(self.oP1BodyLeft) @ self.ROmegaLeft
        self.JP2Left = -self.Skew(self.oP2BodyLeft) @ self.ROmegaLeft

        self.B1C1Left = np.array([
            -self.BC_l * np.cos(self.alpha1_l_est),
            0.0,
            self.BC_l * np.sin(self.alpha1_l_est)
        ])

        self.B2C2Left = np.array([
            -self.BC_l * np.cos(self.alpha2_l_est),
            0.0,
            self.BC_l * np.sin(self.alpha2_l_est)
        ])

        # Jacobian calculation - exactly as C++
        denom1 = self.B1C1Left[2] * self.C1P1Left[0] - self.B1C1Left[0] * self.C1P1Left[2]
        denom2 = self.B2C2Left[2] * self.C2P2Left[0] - self.B2C2Left[0] * self.C2P2Left[2]

        self.JAnkleLeft = np.array([
            self.C1P1Left.T @ self.JP1Left / denom1,
            self.C2P2Left.T @ self.JP2Left / denom2
        ])

        self.JLeft = self.JAnkleLeft

        # Velocity and torque transformations - exactly as C++
        try:
            self.qDotS_l_est = np.linalg.pinv(self.JLeft) @ self.qDotP_l_est
            self.torS_l_est = self.JLeft.T @ self.torP_l_est
        except np.linalg.LinAlgError:
            self.qDotS_l_est = np.zeros(2)
            self.torS_l_est = np.zeros(2)

        return True

    def calcJRight(self) -> bool:
        """Calculate Jacobian for right leg (exactly as C++)"""
        # Rotation matrices - exactly as C++
        self.rotXRight = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll_r_est), -np.sin(self.roll_r_est)],
            [0.0, np.sin(self.roll_r_est), np.cos(self.roll_r_est)]
        ])

        self.rotYRight = np.array([
            [np.cos(self.pitch_r_est), 0.0, np.sin(self.pitch_r_est)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch_r_est), 0.0, np.cos(self.pitch_r_est)]
        ])

        self.rotYXRight = self.rotYRight @ self.rotXRight
        self.oP1BodyRight = self.rotYXRight @ self.OP1_r
        self.oP2BodyRight = self.rotYXRight @ self.OP2_r

        # Position calculations - exactly as C++
        self.C1P1Right = np.array([
            self.oP1BodyRight[0] + self.BC_r * np.cos(self.alpha1_r_est),
            self.oP1BodyRight[1] + self.AB_r,
            self.oP1BodyRight[2] - self.OA1_r - self.BC_r * np.sin(self.alpha1_r_est)
        ])

        self.C2P2Right = np.array([
            self.oP2BodyRight[0] + self.BC_r * np.cos(self.alpha2_r_est),
            self.oP2BodyRight[1] - self.AB_r,
            self.oP2BodyRight[2] - self.OA2_r - self.BC_r * np.sin(self.alpha2_r_est)
        ])

        self.ROmegaRight = np.array([
            [0.0, self.rotYXRight[0, 0]],
            [1.0, self.rotYXRight[1, 0]],
            [0.0, self.rotYXRight[2, 0]]
        ])

        self.JP1Right = -self.Skew(self.oP1BodyRight) @ self.ROmegaRight
        self.JP2Right = -self.Skew(self.oP2BodyRight) @ self.ROmegaRight

        self.B1C1Right = np.array([
            -self.BC_r * np.cos(self.alpha1_r_est),
            0.0,
            self.BC_r * np.sin(self.alpha1_r_est)
        ])

        self.B2C2Right = np.array([
            -self.BC_r * np.cos(self.alpha2_r_est),
            0.0,
            self.BC_r * np.sin(self.alpha2_r_est)
        ])

        # Jacobian calculation - exactly as C++
        denom1 = self.B1C1Right[2] * self.C1P1Right[0] - self.B1C1Right[0] * self.C1P1Right[2]
        denom2 = self.B2C2Right[2] * self.C2P2Right[0] - self.B2C2Right[0] * self.C2P2Right[2]

        self.JAnkleRight = np.array([
            self.C1P1Right.T @ self.JP1Right / denom1,
            self.C2P2Right.T @ self.JP2Right / denom2
        ])

        self.JRight = self.JAnkleRight

        # Velocity and torque transformations - exactly as C++
        try:
            self.qDotS_r_est = np.linalg.pinv(self.JRight) @ self.qDotP_r_est
            self.torS_r_est = self.JRight.T @ self.torP_r_est
        except np.linalg.LinAlgError:
            self.qDotS_r_est = np.zeros(2)
            self.torS_r_est = np.zeros(2)

        return True

    def getSState(self, qSEst: np.ndarray, qDotSEst: np.ndarray, torSEst: np.ndarray) -> bool:
        """Get serial state (exactly as C++)"""
        self.qS_l_est = np.array([self.pitch_l_est, self.roll_l_est])
        self.qS_r_est = np.array([self.pitch_r_est, self.roll_r_est])

        qSEst[:2] = self.qS_l_est
        qSEst[2:] = self.qS_r_est
        qDotSEst[:2] = self.qDotS_l_est
        qDotSEst[2:] = self.qDotS_r_est
        torSEst[:2] = self.torS_l_est
        torSEst[2:] = self.torS_r_est
        return True

    def setSState(self, qSEst: np.ndarray) -> bool:
        """Set serial state (exactly as C++)"""
        self.roll_l_est = qSEst[1]
        self.pitch_l_est = qSEst[0]
        self.roll_r_est = qSEst[3]
        self.pitch_r_est = qSEst[2]

        self.qS_l_est = qSEst[:2].copy()
        self.qS_r_est = qSEst[2:].copy()
        return True

    def setSDes(self, qSRef: np.ndarray, qDotSRef: np.ndarray, torSDes: np.ndarray) -> bool:
        """Set desired serial values (exactly as C++)"""
        self.roll_l_ref = qSRef[1]
        self.pitch_l_ref = qSRef[0]
        self.roll_r_ref = qSRef[3]
        self.pitch_r_ref = qSRef[2]

        self.qS_l_ref = qSRef[:2].copy()
        self.qS_r_ref = qSRef[2:].copy()
        self.qDotS_l_ref = qDotSRef[:2].copy()
        self.qDotS_r_ref = qDotSRef[2:].copy()
        self.torS_l_ref = torSDes[:2].copy()
        self.torS_r_ref = torSDes[2:].copy()
        return True

    def getPDes(self, qPDes: np.ndarray, qPDotDes: np.ndarray, tauPDes: np.ndarray) -> bool:
        """Get desired parallel values (exactly as C++)"""
        qPDes[:2] = self.qP_l_ref
        qPDes[2:] = self.qP_r_ref
        qPDotDes[:2] = self.qDotP_l_ref
        qPDotDes[2:] = self.qDotP_r_ref
        tauPDes[:2] = self.torP_l_ref
        tauPDes[2:] = self.torP_r_ref
        return True

    def calcJointPosRef(self) -> bool:
        """Calculate reference joint positions (exactly as C++)"""
        self.calcJointPosRefLeft()
        self.calcJointPosRefRight()
        return True

    def calcJointPosRefLeft(self) -> bool:
        """Calculate left joint position reference using inverse kinematics (exactly as C++)"""
        # Rotation matrices - exactly as C++
        self.rotXLeftRef = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll_l_ref), -np.sin(self.roll_l_ref)],
            [0.0, np.sin(self.roll_l_ref), np.cos(self.roll_l_ref)]
        ])

        self.rotYLeftRef = np.array([
            [np.cos(self.pitch_l_ref), 0.0, np.sin(self.pitch_l_ref)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch_l_ref), 0.0, np.cos(self.pitch_l_ref)]
        ])

        self.oP1BodyLeftRef = self.rotYLeftRef @ self.rotXLeftRef @ self.OP1_l
        self.oP2BodyLeftRef = self.rotYLeftRef @ self.rotXLeftRef @ self.OP2_l

        # Solve for alpha1 and alpha2 - exactly as C++
        a1 = -2.0 * self.oP1BodyLeftRef[0] * self.BC_l
        b1 = -2.0 * (self.OA1_l - self.oP1BodyLeftRef[2]) * self.BC_l
        c1 = (self.oP1BodyLeftRef[0]**2 + (self.oP1BodyLeftRef[1] - self.AB_l)**2 +
              (self.oP1BodyLeftRef[2] - self.OA1_l)**2 + self.BC_l**2 - self.C1P1_l**2)

        a2 = -2.0 * self.oP2BodyLeftRef[0] * self.BC_l
        b2 = -2.0 * (self.OA2_l - self.oP2BodyLeftRef[2]) * self.BC_l
        c2 = (self.oP2BodyLeftRef[0]**2 + (self.oP2BodyLeftRef[1] + self.AB_l)**2 +
              (self.oP2BodyLeftRef[2] - self.OA2_l)**2 + self.BC_l**2 - self.C2P2_l**2)

        # Check if solution exists - exactly as C++
        if a1**2 + b1**2 - c1**2 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        elif a2**2 + b2**2 - c2**2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        # Calculate solutions - exactly as C++
        sqrt1 = np.sqrt(a1**2 + b1**2 - c1**2)
        sqrt2 = np.sqrt(a2**2 + b2**2 - c2**2)

        csalpha11 = (a1 * c1 + b1 * sqrt1) / (a1**2 + b1**2)
        csalpha12 = (a1 * c1 - b1 * sqrt1) / (a1**2 + b1**2)
        csalpha21 = (a2 * c2 + b2 * sqrt2) / (a2**2 + b2**2)
        csalpha22 = (a2 * c2 - b2 * sqrt2) / (a2**2 + b2**2)

        # Check bounds - exactly as C++
        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            print("Error: Solution out of bounds for left leg")
            return False

        # Find valid solutions - exactly as C++
        alpha1_candidates = [np.arccos(csalpha11), -np.arccos(csalpha11),
                           np.arccos(csalpha12), -np.arccos(csalpha12)]
        alpha2_candidates = [np.arccos(csalpha21), -np.arccos(csalpha21),
                           np.arccos(csalpha22), -np.arccos(csalpha22)]

        # Select valid solutions within range [-PI/2, PI/2] - exactly as C++
        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(a1 * np.cos(alpha1) + b1 * np.sin(alpha1) - c1) <= 1e-7:
                    self.alpha1_l_ref = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(a2 * np.cos(alpha2) + b2 * np.sin(alpha2) - c2) <= 1e-7:
                    self.alpha2_l_ref = alpha2
                    break

        self.qP_l_ref = np.array([self.alpha1_l_ref - 0.3754, self.alpha2_l_ref - 0.3754])
        return True

    def calcJointPosRefRight(self) -> bool:
        """Calculate right joint position reference using inverse kinematics (exactly as C++)"""
        # Rotation matrices - exactly as C++
        self.rotXRightRef = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll_r_ref), -np.sin(self.roll_r_ref)],
            [0.0, np.sin(self.roll_r_ref), np.cos(self.roll_r_ref)]
        ])

        self.rotYRightRef = np.array([
            [np.cos(self.pitch_r_ref), 0.0, np.sin(self.pitch_r_ref)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch_r_ref), 0.0, np.cos(self.pitch_r_ref)]
        ])

        self.oP1BodyRightRef = self.rotYRightRef @ self.rotXRightRef @ self.OP1_r
        self.oP2BodyRightRef = self.rotYRightRef @ self.rotXRightRef @ self.OP2_r

        # Solve for alpha1 and alpha2 - exactly as C++
        a1 = -2.0 * self.oP1BodyRightRef[0] * self.BC_r
        b1 = -2.0 * (self.OA1_r - self.oP1BodyRightRef[2]) * self.BC_r
        c1 = (self.oP1BodyRightRef[0]**2 + (self.oP1BodyRightRef[1] + self.AB_r)**2 +
              (self.oP1BodyRightRef[2] - self.OA1_r)**2 + self.BC_r**2 - self.C1P1_r**2)

        a2 = -2.0 * self.oP2BodyRightRef[0] * self.BC_r
        b2 = -2.0 * (self.OA2_r - self.oP2BodyRightRef[2]) * self.BC_r
        c2 = (self.oP2BodyRightRef[0]**2 + (self.oP2BodyRightRef[1] - self.AB_r)**2 +
              (self.oP2BodyRightRef[2] - self.OA2_r)**2 + self.BC_r**2 - self.C2P2_r**2)

        # Check if solution exists - exactly as C++
        if a1**2 + b1**2 - c1**2 < 0.0:
            print("ErrorID00 : Cannot solved !")
            return False
        elif a2**2 + b2**2 - c2**2 < 0.0:
            print("ErrorID01 : Cannot solved !")
            return False

        # Calculate solutions - exactly as C++
        sqrt1 = np.sqrt(a1**2 + b1**2 - c1**2)
        sqrt2 = np.sqrt(a2**2 + b2**2 - c2**2)

        csalpha11 = (a1 * c1 + b1 * sqrt1) / (a1**2 + b1**2)
        csalpha12 = (a1 * c1 - b1 * sqrt1) / (a1**2 + b1**2)
        csalpha21 = (a2 * c2 + b2 * sqrt2) / (a2**2 + b2**2)
        csalpha22 = (a2 * c2 - b2 * sqrt2) / (a2**2 + b2**2)

        # Check bounds - exactly as C++
        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            print("Error: Solution out of bounds for right leg")
            return False

        # Find valid solutions - exactly as C++
        alpha1_candidates = [np.arccos(csalpha11), -np.arccos(csalpha11),
                           np.arccos(csalpha12), -np.arccos(csalpha12)]
        alpha2_candidates = [np.arccos(csalpha21), -np.arccos(csalpha21),
                           np.arccos(csalpha22), -np.arccos(csalpha22)]

        # Select valid solutions within range [-PI/2, PI/2] - exactly as C++
        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(a1 * np.cos(alpha1) + b1 * np.sin(alpha1) - c1) <= 1e-7:
                    self.alpha1_r_ref = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(a2 * np.cos(alpha2) + b2 * np.sin(alpha2) - c2) <= 1e-7:
                    self.alpha2_r_ref = alpha2
                    break

        self.qP_r_ref = np.array([self.alpha1_r_ref - 0.3754, self.alpha2_r_ref - 0.3754])
        return True

    def calcJointTorDes(self) -> bool:
        """Calculate desired joint torques (exactly as C++)"""
        try:
            self.torP_l_ref = np.linalg.pinv(self.JLeft).T @ self.torS_l_ref
            self.torP_r_ref = np.linalg.pinv(self.JRight).T @ self.torS_r_ref
            self.qDotP_l_ref = self.JLeft @ self.qDotS_l_ref
            self.qDotP_r_ref = self.JRight @ self.qDotS_r_ref
        except np.linalg.LinAlgError:
            self.torP_l_ref = np.zeros(2)
            self.torP_r_ref = np.zeros(2)
            self.qDotP_l_ref = np.zeros(2)
            self.qDotP_r_ref = np.zeros(2)

        # Apply torque limits - exactly as C++
        if self.torP_l_ref[0] >= 30.0:
            print(f"Torque1 beyond upper limit!!! {self.torP_l_ref[0]}")
            self.torP_l_ref[0] = 30.0
        elif self.torP_l_ref[0] <= -30.0:
            print(f"Torque1 beyond lower limit!!! {self.torP_l_ref[0]}")
            self.torP_l_ref[0] = -30.0

        if self.torP_l_ref[1] >= 30.0:
            print(f"Torque2 beyond upper limit!!! {self.torP_l_ref[1]}")
            self.torP_l_ref[1] = 30.0
        elif self.torP_l_ref[1] <= -30.0:
            print(f"Torque2 beyond lower limit!!! {self.torP_l_ref[1]}")
            self.torP_l_ref[1] = -30.0

        if self.torP_r_ref[0] >= 30.0:
            print(f"Torque5 beyond upper limit!!! {self.torP_r_ref[0]}")
            self.torP_r_ref[0] = 30.0
        elif self.torP_r_ref[0] <= -30.0:
            print(f"Torque5 beyond lower limit!!! {self.torP_r_ref[0]}")
            self.torP_r_ref[0] = -30.0

        if self.torP_r_ref[1] >= 30.0:
            print(f"Torque6 beyond upper limit!!! {self.torP_r_ref[1]}")
            self.torP_r_ref[1] = 30.0
        elif self.torP_r_ref[1] <= -30.0:
            print(f"Torque6 beyond lower limit!!! {self.torP_r_ref[1]}")
            self.torP_r_ref[1] = -30.0

        return True

    def calcJointPosEst(self) -> bool:
        """Calculate estimated joint positions (exactly as C++)"""
        self.calcJointPosEstLeft()
        self.calcJointPosEstRight()
        return True

    def calcJointPosEstLeft(self) -> bool:
        """Calculate left joint position estimation using inverse kinematics (exactly as C++)"""
        # Exactly same implementation as calcJointPosRefLeft but using estimated values
        self.rotXLeftRef = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll_l_est), -np.sin(self.roll_l_est)],
            [0.0, np.sin(self.roll_l_est), np.cos(self.roll_l_est)]
        ])

        self.rotYLeftRef = np.array([
            [np.cos(self.pitch_l_est), 0.0, np.sin(self.pitch_l_est)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch_l_est), 0.0, np.cos(self.pitch_l_est)]
        ])

        self.oP1BodyLeftRef = self.rotYLeftRef @ self.rotXLeftRef @ self.OP1_l
        self.oP2BodyLeftRef = self.rotYLeftRef @ self.rotXLeftRef @ self.OP2_l

        # Same solving logic as calcJointPosRefLeft
        a1 = -2.0 * self.oP1BodyLeftRef[0] * self.BC_l
        b1 = -2.0 * (self.OA1_l - self.oP1BodyLeftRef[2]) * self.BC_l
        c1 = (self.oP1BodyLeftRef[0]**2 + (self.oP1BodyLeftRef[1] - self.AB_l)**2 +
              (self.oP1BodyLeftRef[2] - self.OA1_l)**2 + self.BC_l**2 - self.C1P1_l**2)

        a2 = -2.0 * self.oP2BodyLeftRef[0] * self.BC_l
        b2 = -2.0 * (self.OA2_l - self.oP2BodyLeftRef[2]) * self.BC_l
        c2 = (self.oP2BodyLeftRef[0]**2 + (self.oP2BodyLeftRef[1] + self.AB_l)**2 +
              (self.oP2BodyLeftRef[2] - self.OA2_l)**2 + self.BC_l**2 - self.C2P2_l**2)

        if a1**2 + b1**2 - c1**2 < 0.0 or a2**2 + b2**2 - c2**2 < 0.0:
            return False

        # Solve and select valid solutions
        sqrt1 = np.sqrt(a1**2 + b1**2 - c1**2)
        sqrt2 = np.sqrt(a2**2 + b2**2 - c2**2)

        csalpha11 = (a1 * c1 + b1 * sqrt1) / (a1**2 + b1**2)
        csalpha12 = (a1 * c1 - b1 * sqrt1) / (a1**2 + b1**2)
        csalpha21 = (a2 * c2 + b2 * sqrt2) / (a2**2 + b2**2)
        csalpha22 = (a2 * c2 - b2 * sqrt2) / (a2**2 + b2**2)

        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            return False

        alpha1_candidates = [np.arccos(csalpha11), -np.arccos(csalpha11),
                           np.arccos(csalpha12), -np.arccos(csalpha12)]
        alpha2_candidates = [np.arccos(csalpha21), -np.arccos(csalpha21),
                           np.arccos(csalpha22), -np.arccos(csalpha22)]

        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(a1 * np.cos(alpha1) + b1 * np.sin(alpha1) - c1) <= 1e-7:
                    self.alpha1_l_est = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(a2 * np.cos(alpha2) + b2 * np.sin(alpha2) - c2) <= 1e-7:
                    self.alpha2_l_est = alpha2
                    break

        self.qP_l_est = np.array([self.alpha1_l_est - 0.3754, self.alpha2_l_est - 0.3754])
        return True

    def calcJointPosEstRight(self) -> bool:
        """Calculate right joint position estimation using inverse kinematics (exactly as C++)"""
        # Exactly same implementation as calcJointPosRefRight but using estimated values
        self.rotXRightRef = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(self.roll_r_est), -np.sin(self.roll_r_est)],
            [0.0, np.sin(self.roll_r_est), np.cos(self.roll_r_est)]
        ])

        self.rotYRightRef = np.array([
            [np.cos(self.pitch_r_est), 0.0, np.sin(self.pitch_r_est)],
            [0.0, 1.0, 0.0],
            [-np.sin(self.pitch_r_est), 0.0, np.cos(self.pitch_r_est)]
        ])

        self.oP1BodyRightRef = self.rotYRightRef @ self.rotXRightRef @ self.OP1_r
        self.oP2BodyRightRef = self.rotYRightRef @ self.rotXRightRef @ self.OP2_r

        # Same solving logic as calcJointPosRefRight
        a1 = -2.0 * self.oP1BodyRightRef[0] * self.BC_r
        b1 = -2.0 * (self.OA1_r - self.oP1BodyRightRef[2]) * self.BC_r
        c1 = (self.oP1BodyRightRef[0]**2 + (self.oP1BodyRightRef[1] + self.AB_r)**2 +
              (self.oP1BodyRightRef[2] - self.OA1_r)**2 + self.BC_r**2 - self.C1P1_r**2)

        a2 = -2.0 * self.oP2BodyRightRef[0] * self.BC_r
        b2 = -2.0 * (self.OA2_r - self.oP2BodyRightRef[2]) * self.BC_r
        c2 = (self.oP2BodyRightRef[0]**2 + (self.oP2BodyRightRef[1] - self.AB_r)**2 +
              (self.oP2BodyRightRef[2] - self.OA2_r)**2 + self.BC_r**2 - self.C2P2_r**2)

        if a1**2 + b1**2 - c1**2 < 0.0 or a2**2 + b2**2 - c2**2 < 0.0:
            return False

        # Solve and select valid solutions
        sqrt1 = np.sqrt(a1**2 + b1**2 - c1**2)
        sqrt2 = np.sqrt(a2**2 + b2**2 - c2**2)

        csalpha11 = (a1 * c1 + b1 * sqrt1) / (a1**2 + b1**2)
        csalpha12 = (a1 * c1 - b1 * sqrt1) / (a1**2 + b1**2)
        csalpha21 = (a2 * c2 + b2 * sqrt2) / (a2**2 + b2**2)
        csalpha22 = (a2 * c2 - b2 * sqrt2) / (a2**2 + b2**2)

        if (abs(csalpha11) > 1.0 or abs(csalpha12) > 1.0 or
            abs(csalpha21) > 1.0 or abs(csalpha22) > 1.0):
            return False

        alpha1_candidates = [np.arccos(csalpha11), -np.arccos(csalpha11),
                           np.arccos(csalpha12), -np.arccos(csalpha12)]
        alpha2_candidates = [np.arccos(csalpha21), -np.arccos(csalpha21),
                           np.arccos(csalpha22), -np.arccos(csalpha22)]

        for alpha1 in alpha1_candidates:
            if -self.PI/2 <= alpha1 <= self.PI/2:
                if abs(a1 * np.cos(alpha1) + b1 * np.sin(alpha1) - c1) <= 1e-7:
                    self.alpha1_r_est = alpha1
                    break

        for alpha2 in alpha2_candidates:
            if -self.PI/2 <= alpha2 <= self.PI/2:
                if abs(a2 * np.cos(alpha2) + b2 * np.sin(alpha2) - c2) <= 1e-7:
                    self.alpha2_r_est = alpha2
                    break

        self.qP_r_est = np.array([self.alpha1_r_est - 0.3754, self.alpha2_r_est - 0.3754])
        return True

    # Convenience methods using the same naming as in our main code
    def set_p_est(self, qP_est: np.ndarray, qDotP_est: np.ndarray, qTorP_est: np.ndarray):
        """Convenience wrapper"""
        return self.setPEst(qP_est, qDotP_est, qTorP_est)

    def get_s_state(self, qS_est: np.ndarray, qDotS_est: np.ndarray, torS_est: np.ndarray):
        """Convenience wrapper"""
        return self.getSState(qS_est, qDotS_est, torS_est)

    def set_s_des(self, qS_ref: np.ndarray, qDotS_ref: np.ndarray, torS_des: np.ndarray):
        """Convenience wrapper"""
        return self.setSDes(qS_ref, qDotS_ref, torS_des)

    def get_p_des(self, qP_des: np.ndarray, qDotP_des: np.ndarray, torP_des: np.ndarray):
        """Convenience wrapper"""
        return self.getPDes(qP_des, qDotP_des, torP_des)

    def calc_fk(self):
        """Convenience wrapper"""
        return self.calcFK()

    def calc_joint_pos_ref(self):
        """Convenience wrapper"""
        return self.calcJointPosRef()

    def calc_joint_tor_des(self):
        """Convenience wrapper"""
        return self.calcJointTorDes()
