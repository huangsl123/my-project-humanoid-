# xMIGCS

xMIGCS (eXtensible Motion Intelligence Group Control System) 是一个用于机器人控制的软件系统，专注于通过有限状态机（FSM）和策略模块实现对机器人的灵活控制。项目服务于运动智能领域的研究与开发，支持多种控制模式和外部输入方式（如键盘、手柄等）。

## 功能特性

- **状态机管理**: 基于 FSM 模块实现机器人行为的状态流转控制
- **多策略支持**: 提供多种控制策略（如 dh, mlp, pbhc, zero 等）
- **人机交互控制**: 支持键盘、手柄等外设进行机器人实时操控
- **配置驱动**: 使用 YAML 文件进行参数配置，支持不同场景下的快速部署
- **模块化设计**: 各策略独立封装，便于扩展与维护

## 安装与运行

### 环境要求

- Python 3.7+
- Git (用于版本控制)
- bodyctrl_msgs 下载地址：http://10.20.0.252:5302/installation.html

### 安装步骤

```bash
cd your_project_folder
git clone https://git.x-humanoid-cloud.com/motion-intelligence-group/xmigcs.git .
```

### 运行项目

```
# 启动XBOX手柄数据节点
export ROS_DOMAIN_ID=YOUR_DOMAIN_ID
source /opt/ros/humble/setup.bash
ros2 run joy joy_node --ros-args --remap joy:=xbox_data

# 启动主控制节点
python3 rl_control_node.py
```

## 控制器使用说明

### XBOX手柄键位映射

xMIGCS支持标准XBOX手柄控制，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换

| 按钮 | 对应状态 | 功能说明 |
|------|----------|----------|
| A | gotoDHZERO | DH零位状态 |
| B | gotoPBHCZERO | PBHC零位状态 |
| X | gotoZERO | 回到零位状态 |
| Y | gotoSTOP | 停止状态 |

##### 组合按钮状态切换

| 按钮组合 | 对应状态 | 功能说明 |
|----------|----------|----------|
| 左触发键(LT) + A | gotoDH | DH策略状态 |
| 左触发键(LT) + B | gotoPBHC | PBHC策略状态 |
| 左触发键(LT) + X | gotoMLP | MLP策略状态 |
| 左触发键(LT) + HOME | gotoBEYONDMIMIC | BOYONDMIMIC策略状态 |
| LB + X | gotoMLPH | MLP+H策略状态（增强模式） |

##### 基础运动控制

| 控制方式 | 功能 |
|----------|------|
| 左摇杆Y轴 | 前后移动控制（正向为前进） |
| 左摇杆X轴 | 左右移动控制 |
| 右摇杆X轴 | 机身旋转控制 |

##### 辅助功能控制

| 控制方式 | 功能说明 |
|----------|----------|
| 方向键水平(D-pad H) | 增加/减少机身高度（左键增加，右键减少） |
| Start按钮 | 若持续按下，可重置移动速度为0 |
| 左触发键(LT) / 右触发键(RT) | 作为组合键使用，需配合其他按钮激活特殊状态 |

#### 高度调整机制

高度调整通过以下方式实现：
- D-pad水平方向键：增加/减少高度（左键增加+0.05，右键减少-0.05）
- 高度值限制在0.65到0.90之间，并通过平滑步进（0.01）逐步调整到目标值。

### 云卓手柄键位映射

xMIGCS支持标准云卓手柄控制，开始使用前先确保所有键都回中，以下是详细键位映射关系：

#### 状态映射关系

##### 单按钮状态切换

| 按钮 | 对应状态 | 功能说明 |
|------|----------|----------|
| C | gotoSTOP | 停止状态 |

##### 组合按钮状态切换

| 切入策略按钮组合| 策略内使用按键 | 对应状态 | 功能说明 |
|------------|----------------|--------------|-------------|
|   H(拨中)   |         D      | gotoZERO     | MLP零位状态  |
|   H(拨中)   |         A      | gotoMLP      | MLP策略状态  |
|   H(左拨)   |         D      | gotoPBHCZERO | PBHC零位状态 |
|   H(左拨)   |         A      | gotoPBHC     | PBHC策略状态 |
|   H(右拨)   |         D      | gotoDHZERO   | DH零位状态   |
|   H(右拨)   |         A      | gotoDH       | DH策略状态   |
|   E(上拨)   |         A      | gotoBEYONDMIMIC   | BEYONDMIMIC策略状态   |
|   F(上拨)   |         无      | 手柄控制失能，只有停止键可用      | 用于切换话题控制   |

##### 基础运动控制

| 控制方式 | 功能 |
|----------|------|
| 左摇杆Y1轴 | 前后移动控制（正向为前进） |
| 左摇杆X1轴 | 左右移动控制 |
| 右摇杆X2轴 | 机身旋转控制 |

#### 高度调整机制

高度调整通过以下方式实现：
- E键：每上/下拨一次再回中，增加/减少高度（步长0.05m）
- 高度值限制在min_height到max_height之间，并通过平滑步进（0.01）逐步调整到目标值。

## 项目结构

```
.
├── FSM                 # 有限状态机模块
├── common              # 通用功能模块
├── config              # 配置文件
├── policy              # 控制策略模块
├── test                # 测试文件
└── rl_control_node.py  # 主控制节点
```

## 如何添加新的控制策略
1. 在 policy 目录下创建新的策略文件夹，例如 my_new_policy

2. 在新文件夹中创建以下文件：
   - fsm_mypolicy.py - 实现具体的FSM状态类
   - config/mypolicy.yaml - 策略配置文件（可选）
3. 在 fsm_mypolicy.py 中实现 FSMState 类：
```python
    from FSM.fsm_base import FSMState, FSMStateName, ControlFlag
    from common.robot_data import RobotData

    class FSMStateMyPolicy(FSMState):
        def __init__(self, robot_data: RobotData):
            super().__init__(robot_data)
            # 初始化策略特定变量
        
        def on_enter(self):
            # 进入状态时的初始化操作
            pass
        
        def run(self, flag: ControlFlag):
            # 策略的主要运行逻辑
            pass
        
        def on_exit(self):
            # 退出状态时的清理操作
            pass
        
        def check_transition(self, flag: ControlFlag) -> FSMStateName:
            # 检查是否需要转换到其他状态
            pass
```
1. 在 FSM/robot_fsm.py 中注册新状态：
   - 导入新策略类
   - 在 _init_states() 方法中初始化状态对象
   - 在 FSMStateName 枚举中添加新状态
2. 在控制接口（如 common/xbox_control.py中添加对应的状态映射
3. 更新配置文件dex_config.yaml 以支持新策略的相关参数
## 开发与贡献

欢迎对项目进行贡献，开发前请确保：

1. 遵循项目代码规范
2. 添加适当的测试用例
3. 提交前运行所有测试确保无误

## 许可证

本项目仅供内部使用。

## 项目状态

项目正在积极开发中。
