"""
FSM Implementation
Complete FSM implementation with state management
"""
from typing import Dict
from .fsm_base import RobotFSM, FSMStateName
# from .fsm_states import FSMStateStop, FSMStateZero, FSMStateMLP
# from .fsm_states import FSMStateStop
from policy.mlp.fsm_mlp import FSMStateMLP
from policy.zero.fsm_zero import FSMStateZero
from policy.stop.fsm_stop import FSMStateStop
from policy.dh.fsm_dh import FSMStateDH
from policy.dh_zero.fsm_dhzero import FSMStateDHZero
from policy.pbhc.fsmstate_pbhc import FSMStatePBHC
from policy.pbhc_zero.fsm_pbhczero import FSMStatePBHCZero
from policy.beyond_mimic.fsm_beyond_mimic import FSMStateBeyondMimic
from policy.beyondzero.fsm_beyondzero import FSMStateBeyondZero
from policy.getup_hhd.fsm_getup import FSMSGetup
from common.robot_data import RobotData
from common.joystick import ControlFlag
import functools
import time

def timing_decorator(func):
    """
    装饰器：记录函数执行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper

class RobotFSMImpl(RobotFSM):
    """机器人FSM具体实现"""

    def __init__(self, robot_data: RobotData, config: Dict):
        super().__init__(robot_data)
        self.config = config

        # 当前状态
        self.current_state = FSMStateName.STOP
        self.state_objects = {}
        self._init_beyondmimic_variants()

        # 初始化所有状态对象
        self._init_states()

        # 进入初始状态
        self.state_objects[self.current_state].on_enter()

    def _init_beyondmimic_variants(self):
        """预初始化所有 BeyondMimic 策略实例，支持多策略共存切换。"""
        config_dict = self.config if isinstance(self.config, dict) else {}
        variants_cfg = config_dict.get("beyond_mimic_variants", [])
        variants = []
        if isinstance(variants_cfg, list):
            for idx, item in enumerate(variants_cfg):
                if isinstance(item, dict):
                    name = item.get("name") or f"bm{idx}"
                    cfg_path = item.get("config_path")
                else:
                    name = f"bm{idx}"
                    cfg_path = item
                variants.append((name, cfg_path))
        if not variants:
            variants = [("default", None)]

        self.beyondmimic_states = {}
        for name, cfg_path in variants:
            if name in self.beyondmimic_states:
                continue
            try:
                self.beyondmimic_states[name] = FSMStateBeyondMimic(
                    self.robot_data_, config_path=cfg_path, variant_name=name
                )
            except Exception as exc:
                print(f"[RobotFSM] Failed to init BeyondMimic variant '{name}' ({cfg_path}): {exc}")
        if not self.beyondmimic_states:
            # 强制至少有一个默认实例，避免后续空指针
            self.beyondmimic_states["default"] = FSMStateBeyondMimic(self.robot_data_)

        self.default_beyondmimic_variant = (
            "default" if "default" in self.beyondmimic_states else next(iter(self.beyondmimic_states))
        )
        self.current_beyondmimic_variant = self.default_beyondmimic_variant

    def _init_states(self):
        """初始化所有状态对象"""
        self.state_objects[FSMStateName.STOP] = FSMStateStop(self.robot_data_)
        self.state_objects[FSMStateName.ZERO] = FSMStateZero(self.robot_data_)
        self.state_objects[FSMStateName.MLP] = FSMStateMLP(self.robot_data_)
        self.state_objects[FSMStateName.DH] = FSMStateDH(self.robot_data_)
        self.state_objects[FSMStateName.DHZERO] = FSMStateDHZero(self.robot_data_)
        self.state_objects[FSMStateName.PBHC] = FSMStatePBHC(self.robot_data_)
        self.state_objects[FSMStateName.PBHCZERO] = FSMStatePBHCZero(self.robot_data_)
        self.state_objects[FSMStateName.BEYONDMIMIC] = self.beyondmimic_states[self.default_beyondmimic_variant]
        self.state_objects[FSMStateName.BEYONDZERO] = FSMStateBeyondZero(self.robot_data_)
        self.state_objects[FSMStateName.HhdGetUp] = FSMSGetup(self.robot_data_)

        # TODO: 添加其他状态对象
    @timing_decorator
    def run_fsm(self, flag: ControlFlag):
        """运行FSM"""
        desired_bm_variant = self._normalize_beyondmimic_command(flag)
        if desired_bm_variant is None:
            desired_bm_variant = getattr(flag, "beyondmimic_variant", None) or self.current_beyondmimic_variant

        # 检查状态转换
        current_state_obj = self.state_objects[self.current_state]
        next_state = current_state_obj.check_transition(flag)

        change_bm_variant = (
            flag.fsm_state_command == "gotoBEYONDMIMIC"
            and desired_bm_variant is not None
            and desired_bm_variant != self.current_beyondmimic_variant
        )

        # 如果需要状态转换
        if next_state is not None and (next_state != self.current_state or change_bm_variant):
            if next_state in self.state_objects:
                print(f"FSM transition: {self.current_state.name} -> {next_state.name}")

                # 退出当前状态
                current_state_obj.on_exit()

                # 切换到新状态
                self.current_state = next_state
                if self.current_state == FSMStateName.BEYONDMIMIC:
                    target_variant = desired_bm_variant or self.default_beyondmimic_variant
                    self._select_beyondmimic_state(target_variant)
                self.state_objects[self.current_state].on_enter()
            else:
                print(f"Warning: State {next_state.name} not implemented")
        elif change_bm_variant and self.current_state == FSMStateName.BEYONDMIMIC:
            # 同一状态下更换 BeyondMimic 变体，需手动重进一次
            current_state_obj.on_exit()
            target_variant = desired_bm_variant or self.default_beyondmimic_variant
            self._select_beyondmimic_state(target_variant)
            self.state_objects[self.current_state].on_enter()

        # 运行当前状态
        self.state_objects[self.current_state].run(flag)

    def get_current_state(self) -> FSMStateName:
        """获取当前FSM状态"""
        return self.current_state

    def _select_beyondmimic_state(self, variant: str):
        """根据变体名选择 BeyondMimic 策略实例，找不到则回退到默认。"""
        if variant not in self.beyondmimic_states:
            print(f"[RobotFSM] BeyondMimic variant '{variant}' not found, fallback to '{self.default_beyondmimic_variant}'.")
            variant = self.default_beyondmimic_variant
        self.state_objects[FSMStateName.BEYONDMIMIC] = self.beyondmimic_states[variant]
        self.current_beyondmimic_variant = variant

    def _normalize_beyondmimic_command(self, flag: ControlFlag):
        """
        解析类似 'gotoBEYONDMIMIC:variant' 或 'gotoBEYONDMIMIC_variant' 的命令，
        把 flag.fsm_state_command 规范化为 gotoBEYONDMIMIC，并返回目标变体名。
        """
        cmd = getattr(flag, "fsm_state_command", "") or ""
        base_cmd = "gotoBEYONDMIMIC"
        desired_variant = getattr(flag, "beyondmimic_variant", None)

        if cmd.startswith(base_cmd):
            suffix = cmd[len(base_cmd):]
            candidate = None
            if suffix.startswith(("_", ":", "-")):
                candidate = suffix[1:] or None
            elif suffix:
                candidate = suffix
            if candidate:
                desired_variant = candidate
                flag.fsm_state_command = base_cmd
                flag.beyondmimic_variant = candidate
        return desired_variant


def get_robot_fsm(robot_data: RobotData, config: Dict) -> RobotFSM:
    """工厂函数，返回机器人FSM实例"""
    return RobotFSMImpl(robot_data, config)
