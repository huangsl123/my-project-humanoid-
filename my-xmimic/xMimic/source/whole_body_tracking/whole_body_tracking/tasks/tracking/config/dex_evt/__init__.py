import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments for Dex-V3.
##

gym.register(
    id="Tracking-Flat-DexEVT-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatEnvConfig,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-Wo-State-Long-Horizon-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatWoStateLongHorizonEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Long-Horizon-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackLongHorizonEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Launch-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackLaunchRecoveryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Launch-Anneal-040-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackLaunchAnneal040EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Launch-Anneal-030-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackLaunchAnneal030EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Dance2-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackDance2PrecisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackPrecisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Strict-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackStrictPrecisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Body-Precision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackBodyPrecisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Balanced-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackBalancedRecoveryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-State-Feedback-Final-Balance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatStateFeedbackFinalBalanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-DexEVT-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DexEVTFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexEVTFlatLowFreqPPORunnerCfg",
    },
)
