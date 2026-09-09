"""
Keyboard Control Module for SSH and Local Environments
Keyboard input handling for robot state management without external libraries
"""
import threading
import sys
import select
import termios
import tty
import os
import yaml
from typing import Optional
from .joystick import ControlFlag
import signal

class KeyboardFlag(ControlFlag):  # 继承ControlFlag
    def __init__(self):
        super().__init__()  # 调用父类初始化
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.height_cmd: float = 0.89


class KeyboardController:
    """键盘控制器，纯Python实现，不依赖外部库"""

    def __init__(self):
        print("Keyboard Control Started (Pure Python Implementation)")
        
        # 初始化成员变量
        self.keyboard_flag = KeyboardFlag()
        self.data_mutex = threading.Lock()
        
        # 状态追踪变量
        self.current_height = 0.89
        self.target_height = 0.89
        self.height_step = 0.05
        
        # 配置参数
        self.initial_height = 0.89
        self.forward_command_offset = 0.0
        self.lateral_command_offset = 0.0
        self.rotation_command_offset = 0.0
        self.max_forward_speed = 1.0
        self.max_lateral_speed = 0.5
        self.max_rotation_speed = 0.5
        
        # 控制标志
        self.running = False
        self.input_thread = None
        self.original_terminal_settings = None
        
        # 加载配置文件
        self._load_config()
        
        print("Available keyboard commands:")
        print("  z - Goto ZERO state")
        print("  c - Goto STOP state")
        print("  m - Goto MLP state")
        print("  h - Goto MLPH state")
        print("  v/1/2/3/4/5/6/7/8/9/0/t/l/u - Goto BeyondMimic variants")
        print("    down/up/dance/kick/jumpover_29dof/sit_stand/sit_chair/squart/stairs/jump_box/climb_stairs")
        print("    l - dance1_link24 (10000 iterations, 130-D state-feedback fallback)")
        print("    u - weibst_hard (best validated 50000-iteration pipeline policy)")
        print("    ; - automation-only variant from XMIGCS_BEYONDMIMIC_VARIANT")
        print("  Left/Right arrows - Adjust height")
        print("  w/a/s/d - Movement controls")
        print("  q/e - Rotation controls (turn left/right)")
        print("  r - Reset all movement commands to zero")
        print("  x - Quit")
        print("  Ctrl+C - Emergency stop")
        
    def _load_config(self):
        """加载YAML配置文件"""
        try:
            config_path = os.path.join('.', "config", "dex_config.yaml")
            
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            if not config:
                print("[Keyboard_controller] Failed to load config file")
                return
                
            keyboard_cfg = config.get("keyboard", {})
            
            # 加载配置参数
            self.initial_height = keyboard_cfg.get("initial_height", 0.89)
            self.forward_command_offset = keyboard_cfg.get("forward_command_offset", 0.0)
            self.lateral_command_offset = keyboard_cfg.get("lateral_command_offset", 0.0)
            self.rotation_command_offset = keyboard_cfg.get("rotation_command_offset", 0.0)
            self.height_step = keyboard_cfg.get("height_step", 0.05)
            self.max_forward_speed = keyboard_cfg.get("max_forward_speed", 1.0)
            self.max_lateral_speed = keyboard_cfg.get("max_lateral_speed", 0.5)
            self.max_rotation_speed = keyboard_cfg.get("max_rotation_speed", 0.5)
            
            print(f"Loaded keyboard config:")
            print(f"  Initial height: {self.initial_height}")
            print(f"  Height step: {self.height_step}")
            print(f"  Max forward speed: {self.max_forward_speed}")
            print(f"  Max lateral speed: {self.max_lateral_speed}")
            print(f"  Max rotation speed: {self.max_rotation_speed}")
            
            self.current_height = self.initial_height
            self.target_height = self.initial_height
            self.keyboard_flag.height_cmd = self.current_height
            
        except Exception as e:
            print(f"[Keyboard_controller] YAML load error: {e}")
    
    def start(self):
        """启动键盘监听线程"""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        print("Keyboard input thread started")
        
    def stop(self):
        """停止键盘监听"""
        self.running = False

        # 等待线程结束（但不要无限等待）
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
            if self.input_thread.is_alive():
                print("Warning: Keyboard thread did not exit cleanly")

        # # 恢复终端设置（重要：这会让Ctrl+C重新工作）
        if self.original_terminal_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_terminal_settings)
                self.original_terminal_settings = None
            except Exception as e:
                print(f"Error restoring terminal settings: {e}")
            
        print("Keyboard input thread stopped")
        
    def _input_loop(self):
        """主输入循环"""
        # 保存原始终端设置
        self.original_terminal_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # 设置终端为原始模式，支持即时按键检测
            tty.setraw(sys.stdin.fileno())
            
            print("Keyboard listener ready. Press keys to control.")
            print("Press 'x' to quit or Ctrl+C for emergency stop.")
            
            while self.running:
                # 检查输入，100ms超时避免占用太多CPU
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    self._process_key(key)
                    
        except KeyboardInterrupt:
            print("\nEmergency stop detected! Stopping keyboard controller.")
            self._emergency_stop()
        except Exception as e:
            print(f"Input loop error: {e}")
        finally:
            # 确保终端设置被恢复
            if self.original_terminal_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_terminal_settings)
                self.original_terminal_settings = None
    
    def _process_key(self, key):
        """处理按键输入"""
        if key == 'w':
            self._on_w_key()
        elif key == 's':
            self._on_s_key()
        elif key == 'a':
            self._on_a_key()
        elif key == 'd':
            self._on_d_key()
        elif key == 'q':
            self._on_q_key()
        elif key == 'e':
            self._on_e_key()
        elif key == 'z':
            self._on_z_key()
        elif key == 'c':
            self._on_c_key()
        elif key == 'm':
            self._on_m_key()
        elif key == 'h':
            self._on_h_key()
        elif key == 'r':
            self._on_r_key()
        elif key == 'x':
            self._on_x_key()
        elif key == 'g':
            self._on_g_key()
        elif key == 'p':
            self._on_p_key()
        elif key == 'o':
            self._on_o_key()
        elif key == 'v':
            self._on_v_key()
        elif key == '1':
            self._on_v1_key()
        elif key == '2':
            self._on_v2_key()
        elif key == '3':
            self._on_v3_key()
        elif key == '4':
            self._on_v4_key()
        elif key == '5':
            self._on_v5_key()
        elif key == '6':
            self._on_v6_key()
        elif key == '7':
            self._on_v7_key()
        elif key == '8':
            self._on_v8_key()
        elif key == '9':
            self._on_v9_key()
        elif key == '0':
            self._on_v0_key()
        elif key == 't':
            self._on_t_key()
        elif key == 'l':
            self._on_l_key()
        elif key == 'u':
            self._on_u_key()
        elif key == ';':
            self._on_automation_variant_key()
        elif key == 'b':
            self._on_b_key()
        elif key == '\x03':  # Ctrl+C
            print("\nCtrl+C detected - sending interrupt signal")
            self._handle_ctrl_c()
        elif key == '\x1b':  # ESC键，可能是方向键
            self._handle_arrow_key()
        else:
            # 忽略其他按键
            pass
    
    def _handle_arrow_key(self):
        """处理方向键序列"""
        # 方向键序列: ESC + [ + A/B/C/D
        if select.select([sys.stdin], [], [], 0.1)[0]:
            key2 = sys.stdin.read(1)
            if key2 == '[':
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key3 = sys.stdin.read(1)
                    if key3 == 'D':  # 左箭头
                        self._on_left_arrow()
                    elif key3 == 'C':  # 右箭头
                        self._on_right_arrow()
                    elif key3 == 'A':  # 上箭头
                        self._on_up_arrow()
                    elif key3 == 'B':  # 下箭头
                        self._on_down_arrow()
        
    def _on_left_arrow(self):
        """处理左箭头键 - 增加高度"""
        with self.data_mutex:
            self._increase_height()
                
    def _on_right_arrow(self):
        """处理右箭头键 - 降低高度"""
        with self.data_mutex:
            self._decrease_height()
    
    def _on_up_arrow(self):
        """处理上箭头键（备用功能）"""
        print("Up arrow pressed - available for additional functions")
    
    def _on_down_arrow(self):
        """处理下箭头键（备用功能）"""
        print("Down arrow pressed - available for additional functions")
    
    def _on_w_key(self):
        """处理w键 - 前进"""
        with self.data_mutex:
            self.keyboard_flag.x_speed_command += 0.1
            if self.keyboard_flag.x_speed_command > self.max_forward_speed:
                self.keyboard_flag.x_speed_command = self.max_forward_speed
            print(f"Moving forward (speed: {self.keyboard_flag.x_speed_command:.2f})")
            
    def _on_s_key(self):
        """处理s键 - 后退"""
        with self.data_mutex:
            self.keyboard_flag.x_speed_command -= 0.1
            if self.keyboard_flag.x_speed_command < -self.max_forward_speed:
                self.keyboard_flag.x_speed_command = -self.max_forward_speed
            print(f"Moving backward (speed: {self.keyboard_flag.x_speed_command:.2f})")
            
    def _on_a_key(self):
        """处理a键 - 左移"""
        with self.data_mutex:
            self.keyboard_flag.y_speed_command -= 0.1
            if self.keyboard_flag.y_speed_command < -self.max_lateral_speed:
                self.keyboard_flag.y_speed_command = -self.max_lateral_speed
            print(f"Moving left (speed: {self.keyboard_flag.y_speed_command:.2f})")
            
    def _on_d_key(self):
        """处理d键 - 右移"""
        with self.data_mutex:
            self.keyboard_flag.y_speed_command += 0.1
            if self.keyboard_flag.y_speed_command > self.max_lateral_speed:
                self.keyboard_flag.y_speed_command = self.max_lateral_speed
            print(f"Moving right (speed: {self.keyboard_flag.y_speed_command:.2f})")
            
    def _on_q_key(self):
        """处理q键 - 左转"""
        with self.data_mutex:
            self.keyboard_flag.yaw_speed_command -= 0.1
            if self.keyboard_flag.yaw_speed_command < -self.max_rotation_speed:
                self.keyboard_flag.yaw_speed_command = -self.max_rotation_speed
            print(f"Turning left (speed: {self.keyboard_flag.yaw_speed_command:.2f})")
            
    def _on_e_key(self):
        """处理e键 - 右转"""
        with self.data_mutex:
            self.keyboard_flag.yaw_speed_command += 0.1
            if self.keyboard_flag.yaw_speed_command > self.max_rotation_speed:
                self.keyboard_flag.yaw_speed_command = self.max_rotation_speed
            print(f"Turning right (speed: {self.keyboard_flag.yaw_speed_command:.2f})")
            
    def _on_z_key(self):
        """处理z键 - 切换到ZERO状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoZERO"
            print("Command: gotoZERO")
    
    def _on_g_key(self):
        """处理g键 - 切换到DHZERO状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoDHZERO"
            print("Command: gotoDHZERO")

    def _on_p_key(self):
        """处理p键 - 切换到PBHC状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoPBHC"
            print("Command: gotoPBHC")
    def _on_o_key(self):
        """处理o键 - 切换到PBHCZERO状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoPBHCZERO"
            print("Command: gotoPBHCZERO")
    
    def _on_v_key(self):
        """处理v键 - 切换到BeyondMimic down变体"""
        self._set_beyondmimic_variant("down")

    def _on_v1_key(self):
        """处理1键 - 切换到BeyondMimic down变体"""
        self._set_beyondmimic_variant("down")

    def _on_v2_key(self):
        """处理2键 - 切换到BeyondMimic up变体"""
        self._set_beyondmimic_variant("up")

    def _on_v3_key(self):
        """处理3键 - 切换到BeyondMimic dance变体"""
        self._set_beyondmimic_variant("dance")

    def _on_v4_key(self):
        """处理4键 - 切换到BeyondMimic kick变体"""
        self._set_beyondmimic_variant("kick")

    def _on_v5_key(self):
        """处理5键 - 切换到BeyondMimic 29DoF 跳越变体"""
        self._set_beyondmimic_variant("jumpover_29dof")

    def _on_v6_key(self):
        """处理6键 - 切换到BeyondMimic sit_stand变体"""
        self._set_beyondmimic_variant("sit_stand")

    def _on_v7_key(self):
        """处理7键 - 切换到BeyondMimic sit_chair变体"""
        self._set_beyondmimic_variant("sit_chair")

    def _on_v8_key(self):
        """处理8键 - 切换到BeyondMimic squart变体"""
        self._set_beyondmimic_variant("squart")

    def _on_v9_key(self):
        """处理9键 - 切换到BeyondMimic stairs变体"""
        self._set_beyondmimic_variant("stairs")

    def _on_v0_key(self):
        """处理0键 - 切换到BeyondMimic jump_box变体"""
        self._set_beyondmimic_variant("jump_box")

    def _on_t_key(self):
        """处理t键 - 切换到BeyondMimic climb_stairs变体"""
        self._set_beyondmimic_variant("climb_stairs")

    def _on_l_key(self):
        """处理l键 - 切换到dance1_link24 10000轮策略。"""
        self._set_beyondmimic_variant("dance1_link24")

    def _on_u_key(self):
        """处理u键 - 切换到weibst_hard最终验收策略。"""
        self._set_beyondmimic_variant("weibst_hard")

    def _on_automation_variant_key(self):
        """Select the variant requested by the isolated sim2sim validator."""
        variant = os.environ.get("XMIGCS_BEYONDMIMIC_VARIANT", "").strip()
        if not variant:
            print("Automation variant ignored: XMIGCS_BEYONDMIMIC_VARIANT is empty")
            return
        self._set_beyondmimic_variant(variant)

    def _set_beyondmimic_variant(self, variant: str):
        """统一设置BeyondMimic变体并下发指令。"""
        with self.data_mutex:
            self.keyboard_flag.beyondmimic_variant = variant
            self.keyboard_flag.fsm_state_command = f"gotoBEYONDMIMIC:{variant}"
            print(f"Command: gotoBEYONDMIMIC ({variant})")
    def _on_b_key(self):
        """处理b键 - 切换到BEYONDZERO状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoBEYONDZERO"
            print("Command: gotoBEYONDZERO")
    def _on_c_key(self):
        """处理c键 - 切换到STOP状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoSTOP"
            print("Command: gotoSTOP")
            
    def _on_m_key(self):
        """处理m键 - 切换到MLP状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoMLP"
            print("Command: gotoMLP")
            
    def _on_h_key(self):
        """处理h键 - 切换到DH状态"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoDH"
            print("Command: gotoDH")
            
    def _on_r_key(self):
        """处理r键 - 重置移动命令"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoHhdGetUp"
            print("All movement commands reset to zero")
            
    def _on_x_key(self):
        """处理x键 - 退出"""
        with self.data_mutex:
            self.running = False
            print("Quit command received")
    
    def _handle_ctrl_c(self):
        """处理Ctrl+C - 发送SIGINT信号给主进程"""
        # 先停止键盘控制器
        self.running = False
        # 恢复终端设置，让信号处理正常工作
        if self.original_terminal_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_terminal_settings)
                self.original_terminal_settings = None
            except:
                pass
        # 发送SIGINT信号给当前进程
        os.kill(os.getpid(), signal.SIGINT)
    
    def _increase_height(self):
        """增加机器人高度"""
        new_target = self.target_height + self.height_step
        if new_target <= 0.90:
            self.target_height = new_target
            print(f"Height increased to {self.target_height:.2f}m")
        else:
            print("Maximum height reached (0.90m)")
            
    def _decrease_height(self):
        """降低机器人高度"""
        new_target = self.target_height - self.height_step
        if new_target >= 0.65:
            self.target_height = new_target
            print(f"Height decreased to {self.target_height:.2f}m")
        else:
            print("Minimum height reached (0.65m)")
            
    def update_flag(self):
        """更新控制标志"""
        with self.data_mutex:
            # 平滑高度调节
            if abs(self.current_height - self.target_height) > 0.0001:
                if self.current_height < self.target_height:
                    self.current_height += 0.0001
                else:
                    self.current_height -= 0.0001
            else:
                self.current_height = self.target_height
                
            self.keyboard_flag.height_cmd = self.current_height
            
    def get_keyboard_flag(self) -> KeyboardFlag:
        """获取当前键盘标志的副本"""
        with self.data_mutex:
            flag_copy = KeyboardFlag()
            flag_copy.__dict__.update(self.keyboard_flag.__dict__)
            return flag_copy
            
    def init(self) -> int:
        """初始化键盘控制器"""
        print("Keyboard controller initialized (Pure Python)")
        return 0


# 测试代码
if __name__ == "__main__":
    controller = KeyboardController()
    controller.init()
    controller.start()
    
    try:
        # 主循环
        while controller.running:
            controller.update_flag()
            import time
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nMain loop interrupted")
    finally:
        controller.stop()
