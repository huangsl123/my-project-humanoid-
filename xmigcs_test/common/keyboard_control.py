"""
Keyboard Control Module
Keyboard input handling for robot state management
"""
import threading
import os
import yaml
from pynput import keyboard
from  .joystick import ControlFlag

# class KeyboardFlag:
#     """键盘控制标志，模仿ControlFlag的接口"""
#     def __init__(self):
#         self.is_disable: bool = False
#         self.fsm_state_command: str = "gotoZERO"
#         self.command_gait: float = 0.0
#         self.x_speed_command: float = 0.0
#         self.y_speed_command: float = 0.0
#         self.yaw_speed_command: float = 0.0
#         self.x_speed_offset: float = 0.0
#         self.y_speed_offset: float = 0.0
#         self.motion_number: int = 0
#         self.motion_state: bool = False
#         self.height_cmd: float = 0.89
#         self.arm_control: bool = False
#         self.waist_control: bool = False
#         self.height_control: bool = False

class KeyboardFlag(ControlFlag):  # 继承ControlFlag
    def __init__(self):
        super().__init__()  # 调用父类初始化
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.height_cmd: float = 0.89


class KeyboardController:
    """键盘控制器，用于机器人状态管理"""

    def __init__(self):
        print("Keyboard Control Started")
        
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
        self.max_forward_speed = 1.0  # 默认最大前进速度
        self.max_lateral_speed = 0.5   # 默认最大侧向速度
        self.max_rotation_speed = 0.5  # 默认最大旋转速度
        
        # 加载配置文件
        self._load_config()
        
        # 控制标志
        self.running = False
        self.input_thread = None
        self.listener = None
        
        print("Available keyboard commands:")
        print("  z - Goto ZERO state")
        print("  c - Goto STOP state")
        print("  m - Goto MLP state")
        print("  h - Goto MLPH state")
        print("  Left/Right arrows - Adjust height")
        print("  w/a/s/d - Movement controls")
        print("  q/e - Rotation controls (turn left/right)")
        print("  r - Reset all movement commands to zero")
        print("  x - Quit")
        
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
            
            print(f"Loaded keyboard config - initial_height: {self.initial_height}, "
                  f"forward_command_offset: {self.forward_command_offset}, "
                  f"lateral_command_offset: {self.lateral_command_offset}, "
                  f"rotation_command_offset: {self.rotation_command_offset}, "
                  f"height_step: {self.height_step}")
            print(f"Speed limits - max_forward_speed: {self.max_forward_speed}, "
                  f"max_lateral_speed: {self.max_lateral_speed}, "
                  f"max_rotation_speed: {self.max_rotation_speed}")
            
            self.current_height = self.initial_height
            self.target_height = self.initial_height
            self.keyboard_flag.height_cmd = self.current_height
            
        except Exception as e:
            print(f"[Keyboard_controller] YAML load error: {e}")
    
    def start(self):
        """启动键盘监听线程"""
        self.running = True
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()
        print("Keyboard input thread started")
        
    def stop(self):
        """停止键盘监听"""
        self.running = False
        if self.listener:
            self.listener.stop()
        print("Keyboard input thread stopped")
        
    def _on_key_press(self, key):
        """处理按键按下事件"""
        print("\n")
        try:
            # 处理字符键
            char = key.char.lower()
            if char == 'w':
                self._on_w_key(None)
            elif char == 's':
                self._on_s_key(None)
            elif char == 'a':
                self._on_a_key(None)
            elif char == 'd':
                self._on_d_key(None)
            elif char == 'q':
                self._on_q_key(None)
            elif char == 'e':
                self._on_e_key(None)
            elif char == 'z':
                self._on_z_key(None)
            elif char == 'c':
                self._on_c_key(None)
            elif char == 'm':
                self._on_m_key(None)
            elif char == 'h':
                self._on_h_key(None)
            elif char == 'r':
                self._on_r_key(None)
            elif char == 'x':
                self._on_x_key(None)
        except AttributeError:
            # 处理特殊键
            if key == keyboard.Key.left:
                self._on_left_arrow(None)
            elif key == keyboard.Key.right:
                self._on_right_arrow(None)
    
    def _on_up_arrow(self, event):
        """处理上箭头键 （已禁用"""
        pass
    
    def _on_down_arrow(self, event):
        """处理下箭头键 （已禁用"""
        pass
    def _on_left_arrow(self, event):
        """处理左箭头键（已禁用）"""
        with self.data_mutex:
            self._increase_height()
                
    def _on_right_arrow(self, event):
        """处理右箭头键）"""
        with self.data_mutex:
            self._decrease_height()
    
    def _on_w_key(self, event):
        """处理w键"""
        with self.data_mutex:
            self.keyboard_flag.x_speed_command += 0.1
            if self.keyboard_flag.x_speed_command > self.max_forward_speed:
                self.keyboard_flag.x_speed_command = self.max_forward_speed
            print(f"Moving forward (speed: {self.keyboard_flag.x_speed_command})")
            
    def _on_s_key(self, event):
        """处理s键"""
        with self.data_mutex:
            self.keyboard_flag.x_speed_command -= 0.1
            if self.keyboard_flag.x_speed_command < -self.max_forward_speed:
                self.keyboard_flag.x_speed_command = -self.max_forward_speed
            print(f"Moving backward (speed: {self.keyboard_flag.x_speed_command})")
            
    def _on_a_key(self, event):
        """处理a键"""
        with self.data_mutex:
            self.keyboard_flag.y_speed_command -= 0.1
            if self.keyboard_flag.y_speed_command < -self.max_lateral_speed:
                self.keyboard_flag.y_speed_command = -self.max_lateral_speed
            print(f"Moving left (speed: {self.keyboard_flag.y_speed_command})")
            
    def _on_d_key(self, event):
        """处理d键"""
        with self.data_mutex:
            self.keyboard_flag.y_speed_command += 0.1
            if self.keyboard_flag.y_speed_command > self.max_lateral_speed:
                self.keyboard_flag.y_speed_command = self.max_lateral_speed
            print(f"Moving right (speed: {self.keyboard_flag.y_speed_command})")
            
    def _on_q_key(self, event):
        """处理q键"""
        with self.data_mutex:
            self.keyboard_flag.yaw_speed_command -= 0.1
            if self.keyboard_flag.yaw_speed_command < -self.max_rotation_speed:
                self.keyboard_flag.yaw_speed_command = -self.max_rotation_speed
            print(f"Turning left (speed: {self.keyboard_flag.yaw_speed_command})")
            
    def _on_e_key(self, event):
        """处理e键"""
        with self.data_mutex:
            self.keyboard_flag.yaw_speed_command += 0.1
            if self.keyboard_flag.yaw_speed_command > self.max_rotation_speed:
                self.keyboard_flag.yaw_speed_command = self.max_rotation_speed
            print(f"Turning right (speed: {self.keyboard_flag.yaw_speed_command})")
            
    def _on_z_key(self, event):
        """处理z键"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoZERO"
            print("Command: gotoZERO")
            
    def _on_c_key(self, event):
        """处理c键"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoSTOP"
            print("Command: gotoSTOP")
            
    def _on_m_key(self, event):
        """处理m键"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoMLP"
            print("Command: gotoMLP")
            
    def _on_h_key(self, event):
        """处理h键"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command = "gotoMLPH"
            print("Command: gotoMLPH")
            
    def _on_r_key(self, event):
        """处理r键"""
        with self.data_mutex:
            self.keyboard_flag.fsm_state_command == "gotoHhdGetUp"
            print("Command: gotoHhdGetUp")
            
    def _on_x_key(self, event):
        """处理x键"""
        with self.data_mutex:
            self.running = False
            print("Quit command received")
    
    def _increase_height(self):
        """增加机器人高度"""
        new_target = self.target_height + self.height_step
        if new_target <= 0.90:  # 最大高度限制
            self.target_height = new_target
            print(f"Height increased to {self.target_height:.2f}")
        else:
            print("Maximum height reached (0.90m)")
            
    def _decrease_height(self):
        """降低机器人高度"""
        new_target = self.target_height - self.height_step
        if new_target >= 0.65:  # 最小高度限制
            self.target_height = new_target
            print(f"Height decreased to {self.target_height:.2f}")
        else:
            print("Minimum height reached (0.65m)")
            
    def update_flag(self):
        """更新控制标志（模拟手柄的更新方法）"""
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
            
            # 模拟摇杆回中，重置移动命令
            # 注意：现在通过'r'键手动重置，而不是自动重置
            
    def get_keyboard_flag(self) -> KeyboardFlag:
        """获取当前键盘标志的副本"""
        with self.data_mutex:
            # 创建一个新的 KeyboardFlag 实例并复制所有属性
            flag_copy = KeyboardFlag()
            flag_copy.__dict__.update(self.keyboard_flag.__dict__)
            return flag_copy
            
    def init(self) -> int:
        """初始化键盘控制器"""
        print("Keyboard controller initialized")
        return 0