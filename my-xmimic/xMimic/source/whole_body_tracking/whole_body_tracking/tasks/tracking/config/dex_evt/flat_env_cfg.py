import whole_body_tracking.tasks.tracking.mdp as mdp

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from whole_body_tracking.robots.dex_evt import D3_ACTION_SCALE, DEX_EVT_CFG
from whole_body_tracking.tasks.tracking.config.dex_evt.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


def _configure_long_horizon_tracking(cfg: TrackingEnvCfg) -> None:
    """Enable long contiguous segments and two-scale reference-path rewards."""
    # Keep a conservative default for historical staged runs.  train.py may
    # override this with the exact duration of any supplied motion.
    cfg.episode_length_s = 10.0
    cfg.commands.motion.sample_phase_with_episode_horizon = True

    # Keep the original 3-D fine reward, then add a wide XY basin so the
    # policy still receives a useful gradient after long-horizon drift.
    cfg.rewards.motion_global_anchor_xy_coarse = RewTerm(
        func=mdp.motion_global_anchor_xy_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 1.0},
    )
    cfg.rewards.motion_global_anchor_xy_vel = RewTerm(
        func=mdp.motion_global_anchor_xy_velocity_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.5},
    )
    cfg.rewards.motion_joint_pos = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0},
    )


class DexEVTFlatEnvConfig(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Update the robot configuration
        self.scene.robot = DEX_EVT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = D3_ACTION_SCALE

        # Set the anchor body for motion commands
        self.commands.motion.anchor_body = "pelvis"

        # Define the body names based on the URDF structure
        self.commands.motion.body_names = [
            "pelvis",
            "hip_pitch_l_link",
            "hip_roll_l_link",
            "hip_yaw_l_link",
            "knee_pitch_l_link",
            "ankle_pitch_l_link",
            "ankle_roll_l_link",
            "hip_pitch_r_link",
            "hip_roll_r_link",
            "hip_yaw_r_link",
            "knee_pitch_r_link",
            "ankle_pitch_r_link",
            "ankle_roll_r_link",
            "waist_yaw_link",
            "waist_roll_link",
            "waist_pitch_link",
            "shoulder_pitch_l_link",
            "shoulder_roll_l_link",
            "shoulder_yaw_l_link",
            "elbow_pitch_l_link",
            # "elbow_yaw_l_link",
            # "wrist_pitch_l_link",
            # "wrist_roll_l_link",
            "shoulder_pitch_r_link",
            "shoulder_roll_r_link",
            "shoulder_yaw_r_link",
            "elbow_pitch_r_link",
            # "elbow_yaw_r_link",
            # "wrist_pitch_r_link",
            # "wrist_roll_r_link"
        ]


@configclass
class DexEVTFlatWoStateEstimationEnvCfg(DexEVTFlatEnvConfig):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class DexEVTFlatWoStateLongHorizonEnvCfg(DexEVTFlatWoStateEstimationEnvCfg):
    """Wo-State refinement task for faithful long-horizon root-path tracking."""

    def __post_init__(self):
        super().__post_init__()
        _configure_long_horizon_tracking(self)


@configclass
class DexEVTFlatStateFeedbackLongHorizonEnvCfg(DexEVTFlatEnvConfig):
    """Long-horizon task with closed-loop pelvis position/velocity feedback."""

    def __post_init__(self):
        super().__post_init__()
        _configure_long_horizon_tracking(self)


@configclass
class DexEVTFlatStateFeedbackLaunchRecoveryEnvCfg(DexEVTFlatStateFeedbackLongHorizonEnvCfg):
    """Startup curriculum that permits recoverable ankle-height errors.

    The reference pelvis path and every reward remain unchanged.  Only the
    training-time ankle Z termination is widened so a policy can observe and
    recover from the difficult launch instead of being reset immediately.
    Final playback still uses the standard 0.25 m acceptance threshold.
    """

    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_pos.params["threshold"] = 0.5
        self.terminations.ee_body_pos.params["threshold"] = 0.5
        self.rewards.motion_joint_pos.weight = 1.0
        self.rewards.motion_body_pos.weight = 1.5


@configclass
class DexEVTFlatStateFeedbackLaunchAnneal040EnvCfg(DexEVTFlatStateFeedbackLaunchRecoveryEnvCfg):
    """Intermediate launch curriculum with 0.40 m Z recovery gates."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_pos.params["threshold"] = 0.4
        self.terminations.ee_body_pos.params["threshold"] = 0.4


@configclass
class DexEVTFlatStateFeedbackLaunchAnneal030EnvCfg(DexEVTFlatStateFeedbackLaunchRecoveryEnvCfg):
    """Intermediate launch curriculum with 0.30 m Z recovery gates."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_pos.params["threshold"] = 0.3
        self.terminations.ee_body_pos.params["threshold"] = 0.3


@configclass
class DexEVTFlatStateFeedbackPrecisionEnvCfg(DexEVTFlatStateFeedbackLongHorizonEnvCfg):
    """Precision refinement objective for accurate full-trajectory recovery.

    Every global-position term remains relative to the time-varying source
    pelvis path.  There is deliberately no terrain-origin or platform-center
    attraction in this configuration.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_global_anchor_pos.weight = 2.0
        self.rewards.motion_global_anchor_pos.params["std"] = 0.25
        self.rewards.motion_global_anchor_xy_coarse.weight = 2.0
        self.rewards.motion_global_anchor_xy_coarse.params["std"] = 0.75
        self.rewards.motion_global_anchor_xy_vel.weight = 1.0
        self.rewards.motion_joint_pos.weight = 1.0
        self.rewards.motion_joint_pos.params["std"] = 0.8
        self.rewards.motion_global_anchor_ori.weight = 1.0
        self.rewards.motion_global_anchor_ori.params["std"] = 0.3
        self.rewards.motion_body_pos.weight = 1.25
        self.rewards.motion_body_pos.params["std"] = 0.25


@configclass
class DexEVTFlatStateFeedbackStrictPrecisionEnvCfg(DexEVTFlatStateFeedbackPrecisionEnvCfg):
    """Recovery objective for policies that are stable but miss strict pose gates.

    This keeps the original time-varying motion reference and only sharpens
    pose/orientation tracking.  It does not add a platform-center attraction.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_joint_pos.weight = 1.5
        self.rewards.motion_joint_pos.params["std"] = 0.6
        self.rewards.motion_global_anchor_ori.weight = 2.0
        self.rewards.motion_global_anchor_ori.params["std"] = 0.2
        self.rewards.motion_body_pos.weight = 3.0
        self.rewards.motion_body_pos.params["std"] = 0.15
        # Direct mean and worst-quartile distance penalties align the training
        # signal with the strict playback body-position P95 gate.  Both remain
        # relative to the time-varying source pose; neither attracts the robot
        # to the terrain origin or platform center.
        self.rewards.motion_body_pos_l1 = RewTerm(
            func=mdp.motion_relative_body_position_error_l1,
            weight=-2.0,
            params={"command_name": "motion"},
        )
        self.rewards.motion_body_pos_worst_quartile = RewTerm(
            func=mdp.motion_relative_body_position_worst_quartile_error,
            weight=-4.0,
            params={"command_name": "motion"},
        )


@configclass
class DexEVTFlatStateFeedbackBodyPrecisionEnvCfg(DexEVTFlatStateFeedbackStrictPrecisionEnvCfg):
    """Final refinement focused on the remaining relative-body pose error.

    The policy already satisfies the joint, orientation, and global XY gates.
    This stage therefore tightens only the body-relative objective.  All terms
    still follow the moving reference pose and do not attract the robot to the
    platform center.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_body_pos.weight = 6.0
        self.rewards.motion_body_pos.params["std"] = 0.10
        self.rewards.motion_body_pos_l1.weight = -4.0
        self.rewards.motion_body_pos_worst_quartile.weight = -8.0


@configclass
class DexEVTFlatStateFeedbackBalancedRecoveryEnvCfg(DexEVTFlatStateFeedbackBodyPrecisionEnvCfg):
    """Recover path and joint fidelity while retaining learned body precision."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_global_anchor_pos.weight = 3.0
        self.rewards.motion_global_anchor_pos.params["std"] = 0.20
        self.rewards.motion_global_anchor_xy_coarse.weight = 3.0
        self.rewards.motion_global_anchor_xy_coarse.params["std"] = 0.50
        self.rewards.motion_global_anchor_xy_vel.weight = 1.5
        self.rewards.motion_joint_pos.weight = 3.0
        self.rewards.motion_joint_pos.params["std"] = 0.50
        self.rewards.motion_body_pos.weight = 4.0
        self.rewards.motion_body_pos.params["std"] = 0.12
        self.rewards.motion_body_pos_l1.weight = -3.0
        self.rewards.motion_body_pos_worst_quartile.weight = -5.0


@configclass
class DexEVTFlatStateFeedbackFinalBalanceEnvCfg(DexEVTFlatStateFeedbackBalancedRecoveryEnvCfg):
    """Joint/body precision finish after path recovery has converged."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_joint_pos.weight = 5.0
        self.rewards.motion_joint_pos.params["std"] = 0.45
        self.rewards.motion_body_pos.weight = 5.0
        self.rewards.motion_body_pos.params["std"] = 0.11
        self.rewards.motion_body_pos_l1.weight = -4.0
        self.rewards.motion_body_pos_worst_quartile.weight = -6.0


@configclass
class DexEVTFlatStateFeedbackDance2PrecisionEnvCfg(DexEVTFlatStateFeedbackPrecisionEnvCfg):
    """Backward-compatible alias for the original Dance2 precision task."""

    pass

@configclass
class DexEVTFlatLowFreqEnvCfg(DexEVTFlatEnvConfig):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
