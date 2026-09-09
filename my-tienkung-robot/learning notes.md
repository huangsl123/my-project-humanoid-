下面按你这次实际训练配置来画。假设：

```text
并行环境数：1024
每个环境 rollout 长度：24 个控制步
物理步长：0.005 秒
decimation：4
控制周期：0.02 秒
动作维度：20
训练迭代：50000
mini-batch：4 个
learning epoch：5 个
```

当前 TienKung `walk` 配置确实使用 24 步 rollout、20 个动作、`[512,256,128]` 的 Actor/Critic、5 个学习 epoch、4 个 mini-batch、PPO clip 0.2、`γ=0.99`、`λ=0.95`。([GitHub][1])

# 一、完整训练总流程图

```text
┌─────────────────────────────────────────────────────────────┐
│ 0. 启动 train.py                                            │
│                                                             │
│ 读取 walk 配置                                               │
│ 创建 1024 个 Isaac Sim 环境                                  │
│ 创建 Actor、Critic、AMP Discriminator                        │
│ 创建 PPO RolloutStorage、AMP ReplayBuffer                    │
│ 载入 200000 条 AMP 专家动作 transition                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 获取 1024 个机器人的初始观测                              │
│                                                             │
│ Actor observation: [1024, 750]                              │
│ Critic observation: [1024, 800]                             │
│ AMP observation: [1024, 52]                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
╔═════════════════════════════════════════════════════════════╗
║ 2. 外层训练循环：iteration = 0 ... 49999                    ║
╚══════════════════════════╤══════════════════════════════════╝
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Rollout 数据采集：重复 24 个控制步                        │
│                                                             │
│ 每一个控制步：                                               │
│                                                             │
│ ① Actor 输入 [1024,750]                                     │
│    输出 mean [1024,20]                                      │
│    使用 std [1024,20] 构造高斯分布                           │
│    采样 action [1024,20]                                    │
│                                                             │
│ ② Critic 输入 [1024,800]                                    │
│    输出 value [1024,1]                                      │
│                                                             │
│ ③ 计算旧策略 log_prob [1024]                                │
│                                                             │
│ ④ 暂存 obs、action、value、log_prob、mean、std               │
│                                                             │
│ ⑤ env.step(action)                                          │
│    action → 关节目标位置                                     │
│    同一个 action 执行 4 个 PhysX 步                           │
│    总共推进 20 ms                                            │
│                                                             │
│ ⑥ 环境计算 task reward、done、下一 observation               │
│                                                             │
│ ⑦ AMP 判别器计算 style reward                               │
│    与 task reward 混合成最终 reward                          │
│                                                             │
│ ⑧ 把完整 transition 存进 RolloutStorage                     │
│    AMP 状态对存进 AMP ReplayBuffer                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ 24 次结束
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 得到一批 rollout 数据                                    │
│                                                             │
│ 样本数 = 24 × 1024 = 24576                                  │
│                                                             │
│ observation: [24,1024,750]                                  │
│ critic_obs:  [24,1024,800]                                  │
│ action:      [24,1024,20]                                   │
│ reward:      [24,1024,1]                                    │
│ done:        [24,1024,1]                                    │
│ value:       [24,1024,1]                                    │
│ old_logprob: [24,1024,1]                                    │
│ old_mean:    [24,1024,20]                                   │
│ old_std:     [24,1024,20]                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 对 rollout 最后状态计算 last_value                       │
│                                                             │
│ Critic(s_24) → [1024,1]                                     │
│                                                             │
│ 用于估计第 24 步以后尚未采集的未来奖励                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 从 t=23 倒着算到 t=0                                    │
│                                                             │
│ TD error:                                                   │
│ δ_t = r_t + γ(1-d_t)V_{t+1} - V_t                          │
│                                                             │
│ GAE Advantage:                                              │
│ A_t = δ_t + γλ(1-d_t)A_{t+1}                               │
│                                                             │
│ Return target:                                              │
│ R_t = A_t + V_t                                             │
│                                                             │
│ 得到 24576 个 Advantage 和 24576 个 Return                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Advantage 整批归一化                                    │
│                                                             │
│ Â = (A - mean(A)) / (std(A) + 1e-8)                        │
│                                                             │
│ 平均约为 0，标准差约为 1                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. 展平时间轴和环境轴                                       │
│                                                             │
│ [24,1024,...] → [24576,...]                                │
│                                                             │
│ 随机生成 0～24575 的排列                                    │
│ 分为 4 个 mini-batch                                        │
│ 每个 mini-batch = 6144 条                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
╔═════════════════════════════════════════════════════════════╗
║ 9. 使用同一批 rollout 数据训练 5 个 epoch                   ║
║                                                             ║
║ 每个 epoch 有 4 个 mini-batch                               ║
║ 共执行 5 × 4 = 20 次参数更新                                ║
╚══════════════════════════╤══════════════════════════════════╝
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. 每个 mini-batch 的计算                                  │
│                                                             │
│ ① 当前 Actor 重新计算 new_mean、new_std、new_logprob         │
│ ② 当前 Critic 重新计算 new_value                            │
│ ③ 计算新旧策略概率比 ratio                                  │
│ ④ 计算 PPO surrogate loss                                  │
│ ⑤ 计算 clipped value loss                                  │
│ ⑥ 计算 entropy                                             │
│ ⑦ 抽取 AMP policy 样本和 expert 样本                        │
│ ⑧ 计算 AMP discriminator loss 和 gradient penalty           │
│ ⑨ 所有 loss 相加成一个标量 total_loss                       │
│ ⑩ loss.backward() 反向传播                                  │
│ ⑪ Adam optimizer.step() 修改网络参数                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ 20 次更新结束
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 11. 清空当前 PPO RolloutStorage                             │
│                                                             │
│ 保留更新后的：                                               │
│ Actor 参数                                                  │
│ Critic 参数                                                 │
│ 动作标准差参数                                              │
│ AMP Discriminator 参数                                      │
│ Adam 优化器状态                                              │
│ AMP ReplayBuffer                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 12. 打印日志、写 TensorBoard、定期保存 checkpoint            │
│                                                             │
│ 每 100 iteration 保存一次 model_xxx.pt                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           └────→ 回到第 3 步，使用更新后的策略
```

这个“采集一批数据，再用多轮 mini-batch 更新”的结构就是 PPO 的核心设计。PPO 论文明确采用“环境交互采样”和“多 epoch mini-batch 优化”交替进行。([arXiv][2])

---

# 二、训练中有四层不同的循环

这是理解程序最重要的结构。

```text
外层：
50000 个 learning iteration

每个 iteration 内：
├── 采集循环：24 个控制步
│   └── 每个控制步内部：4 个物理步
│
└── 学习循环：5 个 epoch
    └── 每个 epoch：4 个 mini-batch
```

换成数字：

| 层级                 | 每次做什么             |        次数 |
| ------------------ | ----------------- | --------: |
| Learning iteration | 采集一次、更新一次         |     50000 |
| Rollout step       | Actor 控制机器人一次     |     每轮 24 |
| Physics substep    | PhysX 推进 0.005 秒  |    每控制步 4 |
| Learning epoch     | 完整使用一次 rollout 数据 |      每轮 5 |
| Mini-batch update  | 执行一次反向传播和 Adam 更新 | 每 epoch 4 |

因此一次 iteration：

[
24\times4=96
]

个物理子步/环境。

一次 iteration 参数更新：

[
5\times4=20
]

次。

整个 50000 iteration 理论上执行：

[
50000\times20=1,000,000
]

次 `optimizer.step()`。

---

# 三、初始化阶段创建了什么

## 1. 创建1024个并行机器人环境

每个环境有自己的：

* 机器人姿态；
* 关节位置和速度；
* 地形位置；
* 速度命令；
* episode 长度；
* 接触状态；
* reward；
* done；
* gait phase。

但所有机器人共享同一个 Actor、Critic 和 AMP 判别器。

也就是：

```text
1024 个不同机器人状态
            │
            ▼
同一个 Actor 神经网络
            │
            ▼
输出 1024 组不同的20维动作
```

---

## 2. Actor 网络

当前一帧 Actor observation 包含：

| 信息                   |     维度 |
| -------------------- | -----: |
| 机身角速度                |      3 |
| 投影重力                 |      3 |
| 速度命令                 |      3 |
| 20个关节相对位置            |     20 |
| 20个关节速度              |     20 |
| 上一时刻20维动作            |     20 |
| 左右脚 gait phase 的 sin |      2 |
| 左右脚 gait phase 的 cos |      2 |
| 左右脚 phase ratio      |      2 |
| **合计**               | **75** |

最近 10 帧拼接：

[
75\times10=750
]

所以 Actor 结构是：

```text
750
 ↓
Linear 750→512
 ↓
ELU
 ↓
Linear 512→256
 ↓
ELU
 ↓
Linear 256→128
 ↓
ELU
 ↓
Linear 128→20
 ↓
20维动作均值
```

`TienKungEnv` 的实际 observation 比基础环境多了 6 维 gait phase 信息；历史长度为10，所以实际 Actor 输入是750维。([GitHub][3])

Actor 的可学习参数包括：

```text
各层权重 W
各层偏置 b
20 个动作标准差 std
```

大约有：

[
551316+20=551336
]

个参数。

---

## 3. Critic 网络

Critic 当前帧在 Actor 的75维基础上增加：

| 特权信息    | 维度 |
| ------- | -: |
| 机身真实线速度 |  3 |
| 左右脚接触状态 |  2 |

因此一帧：

[
75+3+2=80
]

历史10帧：

[
80\times10=800
]

Critic 网络：

```text
800
 ↓
512
 ↓
256
 ↓
128
 ↓
1个 Value
```

它约有574465个参数。

Actor、Critic和20个标准差加起来约：

[
1,125,801
]

个可训练参数。

---

## 4. AMP observation

AMP 不使用 Actor 的750维 observation。

它单独构造动作风格状态：

| AMP 信息   |     维度 |
| -------- | -----: |
| 左右手臂关节位置 |      8 |
| 左右腿关节位置  |     12 |
| 左右手臂关节速度 |      8 |
| 左右腿关节速度  |     12 |
| 左右手相对位置  |      6 |
| 左右脚相对位置  |      6 |
| **总计**   | **52** |

AMP 判别器看的是一对状态：

[
(z_t,z_{t+1})
]

所以输入维度：

[
52+52=104
]

判别器结构：

```text
104
 ↓
1024
 ↓
512
 ↓
256
 ↓
1个判别值 D
```

项目的 AMP observation 由20维关节位置、20维关节速度和12维手脚位置组成；判别器将当前状态和下一状态拼接后输入网络。([GitHub][3])

---

# 四、一次 rollout 控制步具体发生什么

以下过程会同时处理1024个机器人。

## 第1步：Actor根据观测生成动作分布

输入：

```text
obs.shape = [1024, 750]
```

Actor输出：

```text
mean.shape = [1024, 20]
```

系统还有20个可学习标准差，例如后期：

```text
std ≈ [
  0.18, 0.23, 0.31, ... 共20维
]
```

它们被扩展成：

```text
std.shape = [1024, 20]
```

构造高斯分布：

[
a_{e,i}\sim\mathcal N(\mu_{e,i},\sigma_i^2)
]

其中：

* (e)：第几个环境；
* (i)：第几个关节动作。

训练时：

[
a=\mu+\sigma\epsilon,\qquad\epsilon\sim\mathcal N(0,1)
]

得到：

```text
actions.shape = [1024, 20]
```

Actor 代码输出20维均值，使用20个可学习标准差构造正态分布；训练调用采样，播放则直接返回均值。([GitHub][4])

---

## 第2步：Critic评价当前状态

输入：

```text
critic_obs.shape = [1024, 800]
```

输出：

```text
values.shape = [1024, 1]
```

例如：

```text
env 0 → V=12.3
env 1 → V=8.7
env 2 → V=-2.1
...
```

Value表示：

> 从当前状态出发，后面继续按照当前Actor策略运行，预计可以获得多少折扣累计奖励。

---

## 第3步：计算旧动作概率

对于每个环境的一组20维动作，代码计算：

[
\log \pi_{\text{old}}(\mathbf a_t|\mathbf o_t)
==============================================

\sum_{i=1}^{20}
\log\mathcal N(a_{t,i};\mu_{t,i},\sigma_i^2)
]

所以每个环境得到一个联合 log probability：

```text
old_log_prob.shape = [1024]
```

不是20个独立 PPO loss，而是整组20维联合动作对应一个概率和一个 Advantage。

---

## 第4步：先暂存旧策略数据

此时先保存：

```text
Actor observation
Critic observation
20维 action
旧 Value
旧 log probability
旧 mean
旧 std
当前 AMP observation
```

注意，必须保存“旧”的 mean、std、log probability，因为之后网络参数会改变，PPO要比较新旧策略。

当前 `AMPPPO.act()` 确实保存动作、Value、动作log probability、均值、标准差、Actor/Critic observation和当前AMP observation。([GitHub][5])

---

# 五、20维动作怎样真正控制机器人

环境收到：

```text
action.shape = [1024,20]
```

先裁剪：

[
a_{\text{clip}}
===============

\operatorname{clip}(a,-100,100)
]

当前范围很宽，正常训练动作通常不会触碰这个边界。

然后：

[
q_{\text{target}}
=================

q_{\text{default}}
+
0.25a_{\text{clip}}
]

例如某个膝关节：

```text
默认角度 = 1.00 rad
Actor动作 = 0.60
action_scale = 0.25
```

目标角度：

[
q_{\text{target}}
=================

# 1.00+0.25\times0.60

1.15\text{ rad}
]

然后：

```text
这20个目标角度
       ↓
关节位置执行器/PD控制器
       ↓
产生关节力矩
       ↓
PhysX计算运动
```

当前环境确实将动作乘以0.25后加到默认关节位置，并将结果设置为关节位置目标。([GitHub][3])

---

# 六、为什么一次 action 内有4个物理步

当前：

[
dt_{\text{physics}}=0.005\text{秒}
]

[
decimation=4
]

同一个20维目标保持4次：

```text
Actor计算 action_t
       │
       ├─ Physics step 1：5 ms
       ├─ Physics step 2：5 ms
       ├─ Physics step 3：5 ms
       └─ Physics step 4：5 ms
```

总时间：

[
4\times0.005=0.02\text{秒}
]

之后Actor才根据新的 observation 再输出一组动作。

因此：

```text
PhysX频率：200 Hz
Actor控制频率：50 Hz
```

每个 rollout 24个控制步：

[
24\times0.02=0.48\text{秒}
]

当前环境的 `step()` 内部确实用同一组位置目标执行4次物理仿真，然后才计算奖励、终止和下一观测。([GitHub][3])

---

# 七、环境计算哪些结果

4个物理步完成后，环境得到新的：

```text
机器人关节位置
机器人关节速度
机身姿态
机身角速度
机身线速度
接触力
足部速度
步态相位
```

然后依次计算：

## 1. 更新速度命令和步态相位

步态相位大致按：

[
\text{phase}
============

\left(
\frac{\text{episode step}\times0.02}
{\text{gait cycle}}
+\text{offset}
\right)\bmod1
]

当前 gait cycle：

[
0.85\text{秒}
]

这些相位信息既参与周期步态奖励，也作为Actor observation的一部分。

---

## 2. 检查是否终止

主要包括：

* 膝盖、骨盆、手臂等不应触地部位发生接触；
* episode 达到最大1000个控制步。

因为：

[
20\text{秒}/0.02=1000
]

如果某环境 done：

```text
done[e] = True
```

环境会重置该机器人，但其他机器人继续运行。

---

## 3. 计算 task reward

RewardManager计算许多奖励项，例如：

```text
速度跟踪
角速度跟踪
垂直速度惩罚
姿态惩罚
能耗
关节加速度
动作变化
脚滑
足部冲击力
步态周期
非法接触
终止惩罚
```

所有项相加形成：

[
r_{\text{task}}
]

例如：

[
r_{\text{task}}=0.82
]

配置中的每个奖励函数及权重都定义在 `walk_cfg.py` 中。([GitHub][1])

---

# 八、AMP奖励怎么加入

环境先返回 task reward。

Runner随后取得：

```text
当前 AMP 状态 z_t：[1024,52]
下一 AMP 状态 z_{t+1}：[1024,52]
```

拼接：

[
x_t=[z_t,z_{t+1}]
]

形状：

```text
[1024,104]
```

判别器输出：

[
D(x_t)
]

例如：

```text
D = 0.6
```

风格奖励：

[
r_{\text{style}}
================

0.3\cdot
\max
\left(
1-\frac14(D-1)^2,
0
\right)
]

代入：

[
r_{\text{style}}
================

0.3
\left(
1-\frac14(0.6-1)^2
\right)
]

# [

# 0.3(1-0.04)

0.288
]

然后和task reward混合：

[
r_{\text{final}}
================

(1-0.7)r_{\text{style}}
+
0.7r_{\text{task}}
]

如果：

[
r_{\text{task}}=0.82
]

那么：

[
r_{\text{final}}
================

0.3\times0.288+0.7\times0.82
]

# [

# 0.0864+0.574

0.6604
]

最后存入 PPO 的是：

[
\boxed{r_{\text{final}}}
]

不是原始 task reward。

判别器代码使用二次形式把输出转成风格奖励，并按 `task_reward_lerp=0.7` 与任务奖励混合；AMP论文的核心思想也是由动作数据训练运动先验，再将其产生的style reward用于强化学习。([GitHub][6])

---

# 九、一条transition最终保存什么

一次控制步、一个环境对应一条transition：

[
\tau_t=
(
o_t^A,
o_t^C,
a_t,
r_t,
d_t,
V_t,
\log\pi_{\text{old}},
\mu_{\text{old}},
\sigma_{\text{old}}
)
]

具体包括：

| 数据                   | 作用                       |
| -------------------- | ------------------------ |
| Actor observation    | 更新时重新运行Actor             |
| Critic observation   | 更新时重新运行Critic            |
| 20维action            | 对同一个旧动作重新计算新概率           |
| 最终reward             | 计算GAE和Return             |
| done                 | 截断不同episode之间的信息传播       |
| old value            | 计算GAE和clipped value loss |
| old log probability  | 计算PPO probability ratio  |
| old mean、old std     | 计算KL散度和自适应学习率            |
| AMP state、next state | 训练AMP判别器                 |

1024个环境一次控制步产生1024条。

24步产生：

[
1024\times24=24576
]

条。

---

# 十、24步采集完成后的Tensor

采集期间必须保留时间轴：

```text
时间轴 × 环境轴 × 特征轴
```

因此：

```text
Actor observations  [24,1024,750]
Critic observations [24,1024,800]
Actions             [24,1024,20]
Rewards             [24,1024,1]
Dones               [24,1024,1]
Values              [24,1024,1]
Old log probability [24,1024,1]
Old means           [24,1024,20]
Old stds            [24,1024,20]
```

时间轴不能马上打乱，因为下一阶段要计算：

[
V_{t+1}
]

以及：

[
A_{t+1}
]

---

# 十一、计算最后状态的Value

rollout只采集24步，但episode可能有1000步。

因此对第24步后的状态计算：

[
V_{24}=V(s_{24})
]

形状：

```text
last_values = [1024,1]
```

它估计：

> 从第24步状态开始，后面虽然没有在本次rollout继续采集，但预计还有多少未来奖励。

这叫 bootstrap。

Runner采集完成后调用 `compute_returns()`，算法先对最后Critic observation计算Value，再交给RolloutStorage倒序计算。([GitHub][7])

---

# 十二、倒序计算GAE和Return

对每一个环境独立计算。

从：

```text
t=23
```

倒着算到：

```text
t=0
```

## 1. 选择下一状态Value

第23步：

[
V_{t+1}=V_{24}
]

其他步骤：

[
V_{t+1}=\text{storage.values}[t+1]
]

---

## 2. 计算TD error

[
\delta_t
========

r_t+\gamma(1-d_t)V_{t+1}-V_t
]

当前：

[
\gamma=0.99
]

它回答：

> 当前奖励加上下一状态价值，相比Critic原来对当前状态的预测，好了多少或差了多少？

---

## 3. 计算GAE Advantage

[
A_t
===

\delta_t
+\gamma\lambda(1-d_t)A_{t+1}
]

当前：

[
\lambda=0.95
]

所以：

[
\gamma\lambda=0.9405
]

未来的好坏会逐渐衰减地传回当前动作。

---

## 4. 计算Return目标

[
R_t=A_t+V_t
]

Return用于训练Critic。

Advantage用于训练Actor。

```text
Advantage：
这次动作相对当前状态平均水平好还是差

Return：
Critic在这个状态应该预测成多少
```

当前代码就是从最后一步反向计算TD误差、GAE Advantage及`return=advantage+value`；GAE用Value函数降低策略梯度方差，同时通过λ控制偏差—方差折中。([GitHub][8])

---

# 十三、done在计算中怎样作用

如果某一步真正终止：

[
d_t=1
]

那么：

[
1-d_t=0
]

于是：

[
\delta_t=r_t-V_t
]

[
A_t=\delta_t
]

不会将reset后新episode的价值接到旧episode上。

对于人为时间上限，代码会在存储前额外进行Value bootstrap修正，避免把“正常跑满20秒”当成未来价值突然归零。([GitHub][5])

---

# 十四、为什么要归一化Advantage

24576个Advantage可能是：

```text
-50、-10、-2、0.3、5、30……
```

代码变换：

[
\hat A
======

\frac{A-\operatorname{mean}(A)}
{\operatorname{std}(A)+10^{-8}}
]

得到：

```text
平均值约0
标准差约1
```

它不会改变大致排序：

```text
表现好的仍然是正向样本
表现差的仍然是负向样本
```

但会让梯度尺度更稳定。

当前配置设置 `normalize_advantage_per_mini_batch=False`，所以归一化在整个24576条rollout上完成，而不是每个6144样本的mini-batch单独完成。([GitHub][1])

---

# 十五、什么叫展平

原来：

```text
[24,1024,750]
```

表示：

```text
24个时间步
每步1024个环境
每个Actor observation为750维
```

展平前：

```text
t=0：
    env0
    env1
    ...
    env1023

t=1：
    env0
    env1
    ...
```

展平后：

```text
[24576,750]
```

索引关系：

[
\text{row}=t\times1024+\text{env id}
]

例如：

```text
row 0    = t0, env0
row 1    = t0, env1
...
row 1023 = t0, env1023
row 1024 = t1, env0
```

所有Tensor采用相同索引，所以obs、action、Advantage、Return仍然一一对应。

---

# 十六、为什么计算GAE后可以打乱

在GAE计算前，时间关系必须保留，因为要使用：

[
V_{t+1},A_{t+1}
]

GAE完成后，第 (t) 步未来的影响已经压缩进：

```text
Advantage[t]
Return[t]
```

当前网络又是前馈MLP，不需要沿mini-batch保存隐藏状态，所以可以随机混合不同时间和不同环境的样本。

当前RolloutStorage先将时间维和环境维展平，再使用一次随机排列生成mini-batch；循环网络则会使用另外的轨迹生成器保留时间结构。([GitHub][8])

---

# 十七、4个mini-batch和5个epoch

总样本：

[
24576
]

分4批：

[
24576/4=6144
]

一次epoch：

```text
mini-batch 1：6144条 → 更新一次
mini-batch 2：6144条 → 更新一次
mini-batch 3：6144条 → 更新一次
mini-batch 4：6144条 → 更新一次
```

5个epoch：

```text
Epoch 1：A → B → C → D
Epoch 2：A → B → C → D
Epoch 3：A → B → C → D
Epoch 4：A → B → C → D
Epoch 5：A → B → C → D
```

当前实现的随机索引在epoch循环外创建，所以同一次update中的5个epoch复用同一组mini-batch划分；但每次处理时网络参数已经变化，因此loss会重新计算，不是重复同一个固定结果。([GitHub][8])

---

# 十八、一个6144样本mini-batch具体计算什么

## 1. 当前Actor重新运行

输入：

```text
obs_batch [6144,750]
```

得到：

```text
new_mean [6144,20]
new_std  [6144,20]
entropy  [6144]
```

注意：

```text
action_batch仍然是rollout时采样的旧动作
```

不会重新采样一组动作来算PPO loss。

---

## 2. 计算旧动作在新策略下的概率

[
\log\pi_{\text{new}}(a_t|o_t)
]

形状：

```text
[6144]
```

旧的：

```text
old_log_prob [6144]
```

---

## 3. 概率比

[
\rho_t
======

\exp
\left(
\log\pi_{\text{new}}
--------------------

\log\pi_{\text{old}}
\right)
]

例如：

```text
old log probability = 4.50
new log probability = 4.60
```

则：

[
\rho=e^{0.1}\approx1.105
]

表示新策略让整组20维动作的联合概率密度提高约10.5%。

---

# 十九、Surrogate loss训练Actor

未裁剪部分：

[
L_1=-\hat A_t\rho_t
]

裁剪概率比：

[
\rho_{\text{clip}}
==================

\operatorname{clip}(\rho_t,0.8,1.2)
]

裁剪部分：

[
L_2=-\hat A_t\rho_{\text{clip}}
]

代码取：

[
L_{\text{surrogate}}
====================

\operatorname{mean}
\left[
\max(L_1,L_2)
\right]
]

## Advantage为正

```text
这组动作比预期好
→ 希望 ratio > 1
→ 提高该20维联合动作的概率
```

但ratio超过1.2后，不继续奖励过大的提高。

## Advantage为负

```text
这组动作比预期差
→ 希望 ratio < 1
→ 降低该动作概率
```

但ratio低于0.8后，不继续奖励过大的下降。

PPO实现用新旧log probability计算ratio，并使用`[0.8,1.2]`裁剪范围形成surrogate loss。([GitHub][5])

---

# 二十、Value loss训练Critic

当前Critic重新输出：

```text
new_value [6144,1]
```

目标：

```text
return [6144,1]
```

旧Value：

```text
old_value [6144,1]
```

裁剪Value：

[
V_{\text{clip}}
===============

V_{\text{old}}
+
\operatorname{clip}
(V_{\text{new}}-V_{\text{old}},-0.2,0.2)
]

普通误差：

[
L_{\text{normal}}
=================

(V_{\text{new}}-R)^2
]

裁剪误差：

[
L_{\text{clip}}
===============

(V_{\text{clip}}-R)^2
]

最终：

[
L_{\text{value}}
================

\operatorname{mean}
\left[
\max(L_{\text{normal}},L_{\text{clip}})
\right]
]

作用是避免Critic在同一批rollout上变化过猛。

---

# 二十一、Entropy控制探索

每条样本的20维高斯熵：

[
H_t
===

\sum_{i=1}^{20}
\frac12\log(2\pi e\sigma_i^2)
]

然后：

[
H=\operatorname{mean}(H_t)
]

总loss中是：

[
-0.005H
]

由于优化器最小化loss，这项会阻止标准差过快变小。

```text
std大：
动作在mean周围变化大
探索较强

std小：
动作接近mean
策略更确定
```

Actor代码对20维正态分布的entropy求和，总loss使用`-entropy_coef × entropy`，当前系数为0.005。([GitHub][4])

---

# 二十二、自适应学习率

代码还比较新旧高斯策略的KL散度：

[
KL(\pi_{\text{old}}|\pi_{\text{new}})
]

目标：

[
desired_kl=0.01
]

如果：

[
KL>0.02
]

说明更新太大：

[
lr\leftarrow lr/1.5
]

如果：

[
0<KL<0.005
]

说明更新太保守：

[
lr\leftarrow1.5lr
]

学习率限制在：

[
10^{-5}\sim10^{-2}
]

这进一步控制策略每次更新的幅度。当前实现按mini-batch计算高斯KL并自适应调整所有优化器参数组的学习率。([GitHub][5])

---

# 二十三、AMP判别器怎么训练

每个PPO mini-batch同时取得：

```text
6144条policy AMP transition
6144条expert AMP transition
```

每条状态52维：

```text
policy_state      [6144,52]
policy_next_state [6144,52]
expert_state      [6144,52]
expert_next_state [6144,52]
```

拼接后：

```text
[6144,104]
```

判别器输出：

```text
policy_D [6144,1]
expert_D [6144,1]
```

目标：

[
D(\text{expert})\rightarrow+1
]

[
D(\text{policy})\rightarrow-1
]

损失：

[
L_{\text{AMP}}
==============

\frac12
\left[
\operatorname{MSE}(D_E,1)
+
\operatorname{MSE}(D_\pi,-1)
\right]
]

另外对专家数据计算梯度惩罚：

[
L_{\text{GP}}
=============

10\cdot
\mathbb E
\left[
|\nabla_xD(x)|_2^2
\right]
]

AMP判别器通过专家样本和策略样本的MSE分类损失训练，并对专家输入使用零中心梯度惩罚。([GitHub][5])

---

# 二十四、最终总Loss

当前未启用RND和symmetry，所以：

[
\boxed{
L_{\text{total}}
================

L_{\text{surrogate}}
+
1.0L_{\text{value}}
-------------------

0.005H
+
L_{\text{AMP}}
+
L_{\text{GP}}
}
]

它是一个标量，例如：

```text
surrogate loss     = -0.030
value loss         =  0.100
entropy            =  5.000
AMP loss           =  0.300
gradient penalty   =  0.100
```

总loss：

[
L
=

-0.030
+0.100
-0.005\times5
+0.300
+0.100
]

# [

0.445
]

---

# 二十五、反向传播到底更新谁

执行：

```python
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(policy.parameters(), 1.0)
optimizer.step()
```

梯度路径如下：

```text
Surrogate loss
      │
      ├──→ Actor MLP的W和b
      └──→ 20个动作std

Entropy
      │
      └──→ 20个动作std

Value loss
      │
      └──→ Critic MLP的W和b

AMP loss + gradient penalty
      │
      └──→ AMP Discriminator的W和b
```

特别注意：

> AMP loss不会直接把梯度传给Actor。

Actor从AMP得到的影响是：

```text
Discriminator产生style reward
        ↓
与task reward混合
        ↓
计算GAE Advantage
        ↓
进入PPO surrogate loss
        ↓
更新Actor
```

AMP的训练样本在rollout时已经脱离梯度图，所以判别器loss直接训练判别器；Actor通过AMP奖励间接学习风格。当前代码将policy和discriminator参数放入同一Adam优化器，调用一次`loss.backward()`后更新；梯度裁剪只应用于policy参数。([GitHub][5])

---

# 二十六、`loss.backward()`计算什么

它计算每个参数对总loss的偏导数：

[
\frac{\partial L}{\partial\theta_i}
]

例如Actor最后一层：

[
\mu=W_4h+b_4
]

如果某个正Advantage动作高于当前均值，梯度会让相关均值向该动作靠近。

链式法则：

[
\frac{\partial L}{\partial W_4}
===============================

\frac{\partial L}{\partial\log\pi}
\frac{\partial\log\pi}{\partial\mu}
\frac{\partial\mu}{\partial W_4}
]

然后继续传播到：

[
W_3,W_2,W_1
]

最终Actor的几十万个参数都会获得梯度。

---

# 二十七、Adam怎样修改参数

最简化的梯度下降是：

[
\theta_{\text{new}}
===================

\theta_{\text{old}}
-\alpha g
]

其中：

[
g=\frac{\partial L}{\partial\theta}
]

Adam还会记录：

* 梯度的一阶移动平均；
* 梯度平方的移动平均；
* 对不同参数自动调整实际步长。

因此不是所有权重都简单乘同一个梯度值，但核心仍是：

```text
能降低总loss的方向
→ 参数移动一点
```

每个mini-batch执行一次，所以每iteration共20次。

---

# 二十八、5个epoch为什么每次结果不同

第一个mini-batch前：

```text
Actor参数 = θ0
Critic参数 = φ0
```

更新后：

```text
θ0 → θ1
φ0 → φ1
```

第二次再看到同一条旧样本时：

```text
旧action不变
旧log probability不变
Advantage不变
Return不变
```

但是：

```text
当前mean变了
当前std变了
当前new_log_prob变了
当前ratio变了
当前Value变了
当前loss变了
```

因此5个epoch是：

> 用逐渐变化的新网络，反复从同一批旧经验中提取有限的学习信号。

当概率比超过clip范围，许多Actor样本会停止提供继续扩大变化的收益，从而限制过拟合。

---

# 二十九、一次iteration结束得到什么

经过24步采集和20次参数更新，得到：

## 更新后的Actor

[
\pi_{\theta_{\text{new}}}
]

下一次遇到类似状态时：

* 输出的20维mean略有改变；
* 20维std可能略有改变；
* 好动作更容易出现；
* 差动作更不容易出现。

## 更新后的Critic

[
V_{\phi_{\text{new}}}
]

对未来累计奖励的预测更接近本批Return target。

## 更新后的AMP判别器

更能区分：

```text
专家动作transition
策略动作transition
```

从而给下一批策略轨迹产生新的style reward。

## 被丢弃的数据

当前PPO rollout的：

```text
24576条obs/action/reward/value/advantage/return
```

使用完后清空。

PPO是on-policy算法，下一轮要使用更新后的Actor重新采样。

## 被保留的数据

```text
Actor/Critic参数
动作std
AMP判别器参数
Adam状态
AMP replay buffer
AMP normalizer
训练迭代编号
```

---

# 三十、整个50000轮的总量

你实际使用1024个环境：

[
1024\times24=24576
]

条transition/iteration。

总transition：

[
24576\times50000
================

1,228,800,000
]

与你日志完全一致。

每个transition对应4个物理子步，因此总环境物理子步约：

[
1,228,800,000\times4
====================

4,915,200,000
]

参数更新次数：

[
50000\times5\times4
===================

1,000,000
]

最终训练可以概括为：

```text
12.288亿条机器人控制transition
        ↓
反复计算Reward、GAE、Return、Advantage
        ↓
100万次mini-batch参数更新
        ↓
Actor从随机动作逐渐学会稳定行走
```

# 最核心的记忆框架

```text
采集阶段回答：
“当前策略真实做了什么，结果怎么样？”

Critic和GAE回答：
“这次结果比当前状态的平均预期好还是差？”

PPO loss回答：
“好动作应该增加多少概率，坏动作应该降低多少概率，
而且一次不能改变得太猛。”

反向传播回答：
“具体修改哪些神经网络权重，才能改变这些动作概率？”

Adam回答：
“按照梯度，以稳定的小步幅真正修改参数。”

下一次rollout回答：
“修改后的策略，在真实物理仿真中是否真的更好？”
```

[1]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/legged_lab/envs/tienkung/walk_cfg.py "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/1707.06347 "[1707.06347] Proximal Policy Optimization Algorithms"
[3]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/legged_lab/envs/tienkung/tienkung_env.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/modules/actor_critic.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/algorithms/amp_ppo.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/modules/discriminator.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/runners/amp_on_policy_runner.py "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/storage/rollout_storage.py "raw.githubusercontent.com"











可以。下面不再只用“一维动作”讲，而是用**接近你这个 TienKung 项目真实结构的 20 维动作**，完整算一遍：

[
\text{20维均值}
\rightarrow
\text{采样20维动作}
\rightarrow
\text{计算旧概率}
\rightarrow
\text{计算Advantage}
\rightarrow
\text{重新计算新概率}
\rightarrow
\text{PPO loss}
\rightarrow
\text{反向传播}
\rightarrow
\text{修改Actor参数}
]

先说明：下面的数字是我按照你项目的真实网络结构、20 个关节、`action_scale=0.25` 和后期约 `std=0.27` 构造的一条完整演算样本，**不是从你的 checkpoint 中直接读取的真实 Tensor**。

你的 Actor 是 `[512,256,128]` 的 MLP，输出 20 个动作均值；另外还有 20 个可学习标准差，并构造 `Normal(mean, std)`。项目会把 20 维动作乘以 `0.25`，再加到默认关节位置上，作为关节位置目标。([GitHub][1])

# 1. 当前观察输入 Actor

假设某一个机器人当前的 Actor observation 是：

```text
obs.shape = [690]
```

它包括最近 10 帧的：

* 身体角速度；
* 重力方向；
* 速度命令；
* 20 个关节位置；
* 20 个关节速度；
* 上一时刻 20 个动作。

Actor 执行：

```python
mean = self.actor(observation)
```

输出：

```text
mean.shape = [20]
```

为了演算，假设 Actor 输出的旧动作均值为：

```python
mu_old = [
  0.05, -0.30,  0.02,  0.55, -0.20,  0.03,
 -0.04,  0.18, -0.01,  0.35, -0.10, -0.02,
  0.12,  0.05, -0.03, -0.20,
 -0.10, -0.04,  0.02, -0.18
]
```

这 20 个数不是关节绝对角度，而是动作分布的均值。

我们用下面的关节顺序解释：

| 维度 | 关节       |
| -: | -------- |
|  0 | 左髋 Roll  |
|  1 | 左髋 Pitch |
|  2 | 左髋 Yaw   |
|  3 | 左膝 Pitch |
|  4 | 左踝 Pitch |
|  5 | 左踝 Roll  |
|  6 | 右髋 Roll  |
|  7 | 右髋 Pitch |
|  8 | 右髋 Yaw   |
|  9 | 右膝 Pitch |
| 10 | 右踝 Pitch |
| 11 | 右踝 Roll  |
| 12 | 左肩 Pitch |
| 13 | 左肩 Roll  |
| 14 | 左肩 Yaw   |
| 15 | 左肘 Pitch |
| 16 | 右肩 Pitch |
| 17 | 右肩 Roll  |
| 18 | 右肩 Yaw   |
| 19 | 右肘 Pitch |

这些是当前轻量版模型配置中的 20 个受控关节；表格顺序是为了方便演算，运行时最准确的 Tensor 顺序应通过 `robot.joint_names` 确认。([GitHub][2])

你可以在环境创建后打印：

```python
print(env.unwrapped.scene["robot"].joint_names)
```

---

# 2. Actor 同时还有 20 个标准差

假设现在训练后期每个维度的标准差暂时都取：

```python
sigma = [
    0.27, 0.27, 0.27, 0.27, 0.27,
    0.27, 0.27, 0.27, 0.27, 0.27,
    0.27, 0.27, 0.27, 0.27, 0.27,
    0.27, 0.27, 0.27, 0.27, 0.27
]
```

于是策略不是直接说：

```text
动作就是 mu_old
```

而是定义 20 个高斯分布：

[
a_i\sim\mathcal N(\mu_i,\sigma_i^2)
]

整体可以写成：

[
\mathbf a
\sim
\mathcal N
\left(
\boldsymbol\mu,
\operatorname{diag}(\boldsymbol\sigma^2)
\right)
]

代码中正是先创建 `Normal(mean, std)`，训练时调用 `.sample()`，再对 20 维动作的 log probability 求和。([GitHub][3])

---

# 3. 从分布中采样出一组真实动作

这一次 rollout 假设采样得到：

```python
action = [
  0.09, -0.25,  0.00,  0.62, -0.17,  0.01,
 -0.08,  0.22,  0.03,  0.31, -0.14,  0.01,
  0.15,  0.02, -0.01, -0.25,
 -0.06, -0.07,  0.05, -0.14
]
```

每一维都可以理解成：

[
a_i=\mu_i+\sigma_i\epsilon_i
]

其中：

[
\epsilon_i\sim\mathcal N(0,1)
]

例如左膝：

[
\mu_{\text{左膝}}=0.55
]

采样结果：

[
a_{\text{左膝}}=0.62
]

也就是这一次随机探索让左膝动作比均值大了：

[
0.62-0.55=0.07
]

---

# 4. 这20维动作变成实际关节目标

环境代码执行：

[
q_{\text{target}}
=================

q_{\text{default}}
+
0.25a
]

因为当前：

```python
action_scale = 0.25
```

项目中就是把裁剪后的动作乘以 `action_scale`，再加到默认关节位置，之后调用关节位置目标接口。([GitHub][1])

完整演算如下：

| 维度 | 关节       | 均值 (\mu_{\rm old}) | 采样动作 (a) |  默认角度 |  最终目标角度 |
| -: | -------- | -----------------: | -------: | ----: | ------: |
|  0 | 左髋 Roll  |               0.05 |     0.09 |  0.00 |  0.0225 |
|  1 | 左髋 Pitch |              -0.30 |    -0.25 | -0.50 | -0.5625 |
|  2 | 左髋 Yaw   |               0.02 |     0.00 |  0.00 |  0.0000 |
|  3 | 左膝 Pitch |               0.55 |     0.62 |  1.00 |  1.1550 |
|  4 | 左踝 Pitch |              -0.20 |    -0.17 | -0.50 | -0.5425 |
|  5 | 左踝 Roll  |               0.03 |     0.01 |  0.00 |  0.0025 |
|  6 | 右髋 Roll  |              -0.04 |    -0.08 |  0.00 | -0.0200 |
|  7 | 右髋 Pitch |               0.18 |     0.22 | -0.50 | -0.4450 |
|  8 | 右髋 Yaw   |              -0.01 |     0.03 |  0.00 |  0.0075 |
|  9 | 右膝 Pitch |               0.35 |     0.31 |  1.00 |  1.0775 |
| 10 | 右踝 Pitch |              -0.10 |    -0.14 | -0.50 | -0.5350 |
| 11 | 右踝 Roll  |              -0.02 |     0.01 |  0.00 |  0.0025 |
| 12 | 左肩 Pitch |               0.12 |     0.15 |  0.00 |  0.0375 |
| 13 | 左肩 Roll  |               0.05 |     0.02 |  0.10 |  0.1050 |
| 14 | 左肩 Yaw   |              -0.03 |    -0.01 |  0.00 | -0.0025 |
| 15 | 左肘 Pitch |              -0.20 |    -0.25 | -0.30 | -0.3625 |
| 16 | 右肩 Pitch |              -0.10 |    -0.06 |  0.00 | -0.0150 |
| 17 | 右肩 Roll  |              -0.04 |    -0.07 | -0.10 | -0.1175 |
| 18 | 右肩 Yaw   |               0.02 |     0.05 |  0.00 |  0.0125 |
| 19 | 右肘 Pitch |              -0.18 |    -0.14 | -0.30 | -0.3350 |

以左膝为例：

[
q_{\text{target}}
=================

1.0+0.25\times0.62
]

[
=1.155\text{ rad}
]

所以 Actor 的概率分布最终确实变成了实际机器人关节目标。

---

# 5. Rollout 时保存旧策略概率

对于一维高斯分布，动作 (a_i) 的 log probability 为：

[
\log\pi_i
=========

-\frac12
\left(\frac{a_i-\mu_i}{\sigma_i}\right)^2
-\log\sigma_i
-\frac12\log(2\pi)
]

20 维联合 log probability 是：

[
\log\pi(\mathbf a|\mathbf o)
============================

\sum_{i=1}^{20}\log\pi_i(a_i|\mu_i,\sigma_i)
]

因为代码执行：

```python
distribution.log_prob(actions).sum(dim=-1)
```

不是得到 20 个独立 PPO ratio，而是先把 20 维的 log probability 相加，形成**整个20维联合动作的一个 log probability**。([GitHub][3])

在这组数字中，旧策略总 log probability 大约为：

[
\boxed{
\log\pi_{\rm old}(\mathbf a|\mathbf o)
======================================

7.60694
}
]

这里 log probability 为正并不是错误。连续分布使用的是概率密度，密度可以大于 1。

Rollout storage 保存：

```text
observation
20维 action
old_log_prob = 7.60694
old mean
old std
old value
reward
done
```

---

# 6. 这个20维动作只有一个 Advantage

这是非常重要的一点。

不会给每个关节分别计算：

```text
左膝 Advantage
右膝 Advantage
左髋 Advantage
……
```

而是这一次完整 20 维联合动作：

[
\mathbf a=
[a_1,a_2,\ldots,a_{20}]
]

对应一个标量 Advantage：

[
A_t
]

因为机器人获得的结果是 20 个关节共同作用造成的。

假设经过后续奖励、Critic 和 GAE 计算后，这条样本得到：

[
\boxed{A_t=+1.40}
]

含义是：

> 这整组 20 维动作产生的后续结果，比 Critic 原本预计的平均表现好 1.40。

这不表示每一个关节动作都一定单独正确，而是表示整个联合动作总体表现较好。

---

# 7. 第一次更新时重新计算概率

刚开始更新时，新策略参数还没变化：

[
\mu_{\rm new}=\mu_{\rm old}
]

因此：

[
\log\pi_{\rm new}=7.60694
]

概率比：

[
\rho
====

\exp
\left(
\log\pi_{\rm new}
-----------------

\log\pi_{\rm old}
\right)
]

代入：

[
\rho
====

\exp(7.60694-7.60694)
=1
]

PPO surrogate loss：

[
L_{\rm surrogate}
=================

-A\rho
]

因为还没有超过 clip：

[
L=-1.40\times1=-1.40
]

代码中正是用新旧 log probability 相减后取指数获得 ratio，再乘 Advantage，并与裁剪版本取较保守的一项。([GitHub][4])

---

# 8. 如何算出20维均值应该往哪里移动

暂时假设标准差保持为：

[
\sigma_i=0.27
]

当 ratio 尚未被 clip 时，一条正 Advantage 样本对某个动作均值的梯度为：

[
\frac{\partial L}{\partial\mu_i}
================================

-A\rho
\frac{a_i-\mu_i}{\sigma_i^2}
]

现在：

[
A=1.4,\qquad
\rho=1,\qquad
\sigma^2=0.27^2=0.0729
]

因此：

[
\frac{\partial L}{\partial\mu_i}
================================

-1.4
\frac{a_i-\mu_i}{0.0729}
]

完整 20 维结果大致如下：

| 关节       | (a-\mu) | (\partial L/\partial\mu) | 梯度下降后均值方向 |
| -------- | ------: | -----------------------: | --------- |
| 左髋 Roll  |   +0.04 |                   -0.768 | 增大        |
| 左髋 Pitch |   +0.05 |                   -0.960 | 增大        |
| 左髋 Yaw   |   -0.02 |                   +0.384 | 减小        |
| 左膝 Pitch |   +0.07 |                   -1.344 | 增大        |
| 左踝 Pitch |   +0.03 |                   -0.576 | 增大        |
| 左踝 Roll  |   -0.02 |                   +0.384 | 减小        |
| 右髋 Roll  |   -0.04 |                   +0.768 | 减小        |
| 右髋 Pitch |   +0.04 |                   -0.768 | 增大        |
| 右髋 Yaw   |   +0.04 |                   -0.768 | 增大        |
| 右膝 Pitch |   -0.04 |                   +0.768 | 减小        |
| 右踝 Pitch |   -0.04 |                   +0.768 | 减小        |
| 右踝 Roll  |   +0.03 |                   -0.576 | 增大        |
| 左肩 Pitch |   +0.03 |                   -0.576 | 增大        |
| 左肩 Roll  |   -0.03 |                   +0.576 | 减小        |
| 左肩 Yaw   |   +0.02 |                   -0.384 | 增大        |
| 左肘 Pitch |   -0.05 |                   +0.960 | 减小        |
| 右肩 Pitch |   +0.04 |                   -0.768 | 增大        |
| 右肩 Roll  |   -0.03 |                   +0.576 | 减小        |
| 右肩 Yaw   |   +0.03 |                   -0.576 | 增大        |
| 右肘 Pitch |   +0.04 |                   -0.768 | 增大        |

梯度下降更新规则是：

[
\theta_{\rm new}
================

\theta_{\rm old}
-\alpha\nabla_\theta L
]

所以：

* 梯度为负，减去负数，输出倾向增大；
* 梯度为正，减去正数，输出倾向减小。

例如左膝：

[
\frac{\partial L}{\partial\mu_{\text{左膝}}}
=-1.344
]

梯度下降会让左膝均值从：

[
0.55
]

向实际采样到的好动作：

[
0.62
]

靠近。

右膝则是：

[
a=0.31,\qquad\mu=0.35
]

因为好动作比均值小，所以更新会让右膝均值下降。

---

# 9. 但均值 (\mu) 本身不是直接保存的参数

这是理解神经网络更新的关键。

Actor 的最后一层是：

[
\boldsymbol\mu=W_{\rm out}\mathbf h+\mathbf b
]

其中：

* (\mathbf h)：上一隐藏层的 128 维输出；
* (W_{\rm out})：形状 `[20,128]`；
* (\mathbf b)：20 维偏置；
* (\boldsymbol\mu)：20 维动作均值。

例如左膝是第 3 维：

[
\mu_3
=====

\sum_{j=1}^{128}
W_{3j}h_j+b_3
]

假设隐藏神经元第 17 维：

[
h_{17}=0.30
]

刚才算出：

[
\frac{\partial L}{\partial\mu_3}
=-1.344
]

根据链式法则：

[
\frac{\partial L}{\partial W_{3,17}}
====================================

\frac{\partial L}{\partial\mu_3}
\frac{\partial\mu_3}{\partial W_{3,17}}
]

而：

[
\frac{\partial\mu_3}{\partial W_{3,17}}
=======================================

h_{17}=0.30
]

所以：

[
\frac{\partial L}{\partial W_{3,17}}
====================================

-1.344\times0.30
]

[
=-0.4032
]

假设只是最简单的 SGD，学习率：

[
\alpha=0.001
]

那么：

[
W_{\rm new}
===========

W_{\rm old}
-0.001\times(-0.4032)
]

# [

W_{\rm old}+0.0004032
]

左膝对应的这个权重会略微增大。

实际代码使用的是 Adam，不是简单 SGD，所以还会综合：

* 当前梯度；
* 历史梯度平均；
* 历史平方梯度；
* 数值稳定项。

但核心方向不变：

> 改变 Actor 的 (W) 和 (b)，让相似 observation 输入时，20 维均值更偏向这组高 Advantage 动作。

---

# 10. 梯度还会继续传到前面的层

Actor 不只有最后一层。

它是：

```text
690
↓
Linear 690→512
↓
ELU
↓
Linear 512→256
↓
ELU
↓
Linear 256→128
↓
ELU
↓
Linear 128→20
↓
20维 mean
```

所以梯度路径是：

[
L
\rightarrow
\boldsymbol\mu
\rightarrow
W_4,b_4
\rightarrow
h_3
\rightarrow
W_3,b_3
\rightarrow
h_2
\rightarrow
W_2,b_2
\rightarrow
h_1
\rightarrow
W_1,b_1
]

`loss.backward()` 会自动计算：

```text
输出层每个权重应该变化多少
第三隐藏层每个权重应该变化多少
第二隐藏层每个权重应该变化多少
第一隐藏层每个权重应该变化多少
20个标准差应该变化多少
```

Actor 的全部几十万个参数都会根据这 20 维联合动作和 Advantage 获得相应梯度。

---

# 11. 第一个 mini-batch 更新后会怎样

经过许多样本共同更新后，假设对于这条样本，Actor 输出均值大致向采样动作靠近了一半：

```python
mu_new = [
  0.070, -0.275,  0.010,  0.585, -0.185,  0.020,
 -0.060,  0.200,  0.010,  0.330, -0.120, -0.005,
  0.135,  0.035, -0.020, -0.225,
 -0.080, -0.055,  0.035, -0.160
]
```

注意，采样动作仍然是原来那组：

```python
action = [
  0.09, -0.25, 0.00, 0.62, ...
]
```

重新计算这组旧动作在新策略下的 log probability：

[
\log\pi_{\rm new}
\approx7.75766
]

旧的是：

[
\log\pi_{\rm old}
\approx7.60694
]

概率比：

[
\rho
====

\exp(7.75766-7.60694)
]

[
\boxed{\rho\approx1.1627}
]

含义是：

> 新策略对这整个20维联合动作的概率密度，大约提高了16.27%。

因为：

[
1.1627<1.2
]

还没有超过 PPO clip 上限。

Surrogate loss：

[
L=-1.4\times1.1627
]

[
\boxed{L\approx-1.6277}
]

继续最小化这个 loss，还会推动该动作概率上升。

---

# 12. 重复训练后 PPO Clip 开始生效

假设再经过几次 mini-batch/epoch 更新，新均值更靠近采样动作，计算得到：

[
\rho=1.2128
]

但 PPO 的 clip 范围是：

[
[0.8,1.2]
]

因此：

[
\operatorname{clip}(1.2128,0.8,1.2)=1.2
]

未裁剪 loss：

[
L_1=-1.4\times1.2128=-1.6979
]

裁剪 loss：

[
L_2=-1.4\times1.2=-1.68
]

代码取：

[
\max(L_1,L_2)
]

所以：

[
L_{\rm surrogate}=-1.68
]

由于裁剪项中的 `1.2` 已经是常数，这条正 Advantage 样本不会再继续推动概率超过合理范围。

这就是 PPO Clip 的作用：

```text
这组20维动作很好
→ 可以提高它的联合概率

但提高大约20%以后
→ 暂时不要再猛烈提高
→ 等下一次用新策略重新采样
```

---

# 13. 负 Advantage 时方向相反

假设完全相同的20维动作，最终得到：

[
A=-1.4
]

含义是这整组动作比平均表现差。

第一次更新时：

[
\rho=1
]

PPO 会倾向于降低这组动作的概率。

对均值的梯度方向会反转：

* 原来正 Advantage 时均值向采样动作靠近；
* 负 Advantage 时均值远离采样动作。

例如左膝：

```text
均值 = 0.55
坏的采样动作 = 0.62
```

更新后可能变成：

```text
0.55 → 0.54
```

让以后再次采样到 `0.62` 的概率降低。

如果概率比已经降到：

[
\rho<0.8
]

PPO 也会裁剪，避免一次把坏动作概率降得过猛。

---

# 14. 实际不是一条样本单独更新

上面为了看清计算，只计算了一条 transition。

你的实际 mini-batch 有：

[
6144
]

条样本。

每条样本都有：

```text
690维 observation
20维 action
1个 old log probability
1个 Advantage
1个 Return
1个 old Value
```

mini-batch 的 Actor loss 是：

[
L_{\rm actor}
=============

\frac1{6144}
\sum_{n=1}^{6144}
L_{\rm surrogate}^{(n)}
]

因此最终梯度是：

[
\nabla_\theta L
===============

\frac1{6144}
\sum_{n=1}^{6144}
\nabla_\theta L^{(n)}
]

其中可能有：

```text
样本1：要求左膝均值增大
样本2：要求左膝均值减小
样本3：要求右髋增大
样本4：要求左踝减小
……
```

最后取平均形成总体更新方向。

这就是为什么网络不会简单死记一条动作，而是从大量状态中学习总体规律。

---

# 15. 为什么一个参数更新会影响其他状态

假设左膝输出：

[
\mu_{\rm knee}=W_{\rm knee}h(o)+b_{\rm knee}
]

参数 (W_{\rm knee}) 是所有 observation 共享的。

修改它以后，不只是当前 observation 的输出改变。

以后遇到类似状态：

```text
身体前倾
左脚支撑
右脚摆动
目标速度向前
```

隐藏层会产生类似的特征 (\mathbf h)，于是左膝输出也会产生类似变化。

所以 PPO 学到的不是：

```text
这一条 observation → 固定动作
```

而是：

```text
这一类状态特征
→ 应该产生怎样的20维关节配合
```

---

# 16. 一次真实 update 的完整代码关系

对应你项目的代码，可以概念化成：

```python
# 同一个旧 observation
policy.act(obs_batch)

# 生成当前的新 mean 和 std
new_mean = policy.action_mean
new_std = policy.action_std

# 对 rollout 保存的同一个20维旧 action 重新算概率
new_log_prob = policy.get_actions_log_prob(actions_batch)

# 新旧联合动作概率比
ratio = torch.exp(new_log_prob - old_log_prob)

# PPO clip actor loss
surrogate = -advantage * ratio
surrogate_clipped = -advantage * torch.clamp(
    ratio, 0.8, 1.2
)
surrogate_loss = torch.max(
    surrogate,
    surrogate_clipped
).mean()

# 加上 Critic 和 entropy
loss = (
    surrogate_loss
    + value_loss
    - 0.005 * entropy
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

当前仓库的核心实现就是这个过程：重新计算当前策略下旧动作的 log probability，计算 ratio、clipped surrogate、value loss 和 entropy，再执行反向传播及优化器更新。([GitHub][4])

# 最关键的结论

对于一条20维机器人动作，PPO实际做的是：

```text
Actor输出20个均值和20个标准差
↓
采样一组20维联合动作
↓
20个关节共同产生机器人运动
↓
整组动作得到一个Advantage
↓
正Advantage：
让20维均值总体向这组动作靠近

负Advantage：
让20维均值总体远离这组动作
↓
计算整个20维联合动作的新旧概率比
↓
PPO Clip限制联合概率变化不要超过太多
↓
反向传播把梯度传给Actor所有层的W和b
↓
Adam微调几十万个神经网络参数
↓
下一次遇到类似状态时，生成稍微更好的20维动作
```

并不是 PPO 直接修改关节角度，而是：

[
\boxed{
\text{修改神经网络参数}
\Rightarrow
\text{改变20维动作均值与标准差}
\Rightarrow
\text{改变以后实际采样或推理得到的关节动作}
}
]

[1]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/legged_lab/envs/tienkung/walk_cfg.py "raw.githubusercontent.com"
[2]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/main/legged_lab/assets/tienkung2_lite/tienkung.py "TienKung-Lab/legged_lab/assets/tienkung2_lite/tienkung.py at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[3]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/modules/actor_critic.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/algorithms/amp_ppo.py "raw.githubusercontent.com"

















先纠正一个容易混淆的说法：

> AMP 在这个项目里不是拿“策略的20维 action”与“专家20维 action”直接比较，而是比较**策略产生的状态转移**与**专家运动数据中的状态转移**。

也就是比较：

[
\text{策略数据：}(s_t^{\pi},s_{t+1}^{\pi})
]

和：

[
\text{专家数据：}(s_t^{E},s_{t+1}^{E})
]

判别器学习判断：

> 这段从当前姿态到下一姿态的运动，看起来更像专家，还是更像当前策略？

---

# 一、AMP整体在做什么

完整对抗关系是：

```text
专家运动文件
   ↓
专家状态对 (sE_t, sE_t+1)
   ↓
                       ┌──────────────┐
                       │ AMP判别器 D  │
                       └──────────────┘
                       ↑
策略控制机器人
   ↓
策略状态对 (sπ_t, sπ_t+1)
```

判别器的目标：

```text
专家状态对 → 输出接近 +1
策略状态对 → 输出接近 -1
```

而Actor的目标不是直接修改判别器，而是：

```text
让自己产生的运动越来越像专家
↓
判别器对策略运动的输出越来越接近 +1
↓
策略获得更高 AMP style reward
↓
通过 PPO 更新 Actor
```

这就是“Adversarial Motion Prior”中的“Adversarial”，即判别器与策略形成对抗学习。AMP论文使用运动数据集训练运动先验，先验给策略提供风格奖励，而任务奖励仍负责速度跟踪等目标。([arXiv][1])

---

# 二、为什么不是只输入一个 state

你当前每个AMP状态是52维：

[
s_t=
[
q_t,\dot q_t,p_t^{ee}
]
]

其中：

| 信息                 | 维数 |
| ------------------ | -: |
| 20个关节位置 (q_t)      | 20 |
| 20个关节速度 (\dot q_t) | 20 |
| 左右手、左右脚相对位置        | 12 |
| 合计                 | 52 |

当前环境按右臂、左臂、右腿、左腿的关节位置和速度，以及双手双脚相对根部的位置，拼成52维AMP observation。([GitHub][2])

## 单个state只能说明“现在是什么姿势”

假设某一时刻机器人姿势是：

```text
左腿在后
右腿在前
左膝较直
右膝弯曲
```

只看这一帧，可能存在很多不同情况：

```text
情况1：机器人正在向前走
情况2：机器人正在向后走
情况3：机器人正在停止
情况4：机器人刚刚失去平衡
情况5：机器人只是保持一个静态姿势
```

它们在某一瞬间的关节位置可能很接近。

但下一状态不同：

```text
向前走：
右腿继续落地，左腿开始摆动

向后走：
右腿往后移动

静止：
关节基本不变

摔倒：
身体和足部位置快速偏离
```

所以加入下一状态后：

[
(s_t,s_{t+1})
]

判别器不仅看到“姿势”，还能看到：

> 这个姿势正在如何变化。

---

# 三、既然state里有速度，为什么还需要下一state

虽然52维状态中已经有20维关节速度，但状态对仍然能提供更多信息。

## 1. 检查位置和速度是否一致

假设当前关节速度声称：

[
\dot q_t=1.0\ {\rm rad/s}
]

经过20 ms，合理的位置变化大约是：

[
q_{t+1}-q_t
\approx
\dot q_t\times0.02
==================

0.02\ {\rm rad}
]

如果实际位置变化是：

[
0.15\ {\rm rad}
]

判别器可以判断这段运动与正常专家运动不一致。

## 2. 看到手脚轨迹

状态中有手脚位置：

[
p_t^{hand},p_t^{foot}
]

状态对可以得到：

[
p_{t+1}^{foot}-p_t^{foot}
]

从而判断：

* 摆动脚是否自然向前；
* 支撑脚是否滑动；
* 手臂摆动方向是否合理；
* 落脚是否突然跳变。

## 3. 表示局部运动片段

一个状态更像一张照片。

一个状态对则像一个非常短的两帧视频：

```text
第1帧 → 第2帧
```

AMP要学习的是运动风格，而不是静态姿势风格，所以状态转移通常比单帧更合适。AMP论文的style reward本身就是依据相邻状态转移计算的。([arXiv][3])

---

# 四、为什么要把两个state拼接

代码执行：

```python
discriminator_input = torch.cat(
    [state, next_state],
    dim=-1,
)
```

原来：

```text
state.shape      = [batch, 52]
next_state.shape = [batch, 52]
```

拼接后：

```text
input.shape = [batch, 104]
```

结构是：

[
x=
[
s_t^{(1)},\ldots,s_t^{(52)},
s_{t+1}^{(1)},\ldots,s_{t+1}^{(52)}
]
]

当前判别器构造时，输入维度就是 `amp_data.observation_dim * 2`；训练时对策略和专家都将当前状态、下一状态沿最后一维拼接。([GitHub][4])

## 为什么不只输入差值 (s_{t+1}-s_t)

因为只输入差值会丢失绝对姿态。

例如两个情况都可能有：

[
q_{t+1}-q_t=0.05
]

但：

```text
情况A：
膝盖从0.2变到0.25

情况B：
膝盖从1.5变到1.55
```

变化量一样，但绝对姿态完全不同。

拼接能同时保留：

```text
当前是什么姿态
下一步是什么姿态
变化方向和大小
```

而且神经网络自己可以学习差值。

例如第一层某个神经元设置权重：

[
[-1,,+1]
]

就可以计算：

[
-s_t+s_{t+1}
============

s_{t+1}-s_t
]

所以拼接比只提供差值包含的信息更多。

---

# 五、专家风格数据是怎么来的

专家风格不是训练过程中由Actor自己生成的。

它来自配置中的专家运动文件。你当前训练使用的是仓库配置指定的 `motion_amp_expert` 目录下的行走文件；如果不替换它，风格就是仓库作者提供的行走运动风格。

项目README给出的完整生成流程是：

```text
人类动作数据
AMASS / SMPLX
      ↓
GMR运动重定向
把人的骨骼动作映射到TienKung机器人
      ↓
生成机器人运动数据.pkl
      ↓
gmr_data_conversion.py
生成用于动画播放的motion_visualization文件
      ↓
play_amp_animation.py
在Isaac Sim中按动作数据驱动机器人
      ↓
从机器人状态中提取52维AMP observation
      ↓
保存为motion_amp_expert/motion.txt
      ↓
训练时作为专家数据
```

仓库说明，原始动作可以来自SMPLX类型数据，例如AMASS；先经过GMR重定向，再生成可视化文件，最后由 `play_amp_animation.py` 播放并保存训练需要的专家文件。专家文件字段是关节位置、关节速度和末端执行器位置。([GitHub][5])

---

## 专家文件不是电机action文件

它保存的是每一帧机器人的状态：

```text
20维关节位置
20维关节速度
12维手脚位置
----------------
每帧52维
```

并没有保存：

```text
Actor输出的20维归一化action
PD控制目标
电机力矩
PPO动作概率
```

所以AMP可以只使用动作演示，而不要求知道专家当时使用了什么控制器、什么电机指令。

这是AMP的一大特点：

> 只需要“专家最终怎么动”，不一定需要“专家是怎么控制出来的”。

---

## `play_amp_animation.py`怎样保存专家文件

脚本以指定帧率播放机器人动作，例如：

```bash
python legged_lab/scripts/play_amp_animation.py \
  --task=walk \
  --num_envs=1 \
  --save_path=.../motion.txt \
  --fps=30
```

播放过程中每一帧调用：

```python
frame = env.visualize_motion(time)
```

将得到的机器人AMP状态存入文件，并写入：

```text
FrameDuration
MotionWeight
Frames
```

30 fps时：

[
FrameDuration\approx\frac1{30}=0.033\text{秒}
]

当前脚本确实逐帧保存数据，并将其包装成包含 `FrameDuration`、`MotionWeight` 和 `Frames` 的JSON格式文件。([GitHub][6])

---

# 六、训练时怎样从专家动画提取状态对

训练初始化时：

```python
amp_data = AMPLoader(
    time_between_frames=env.step_dt,
    preload_transitions=True,
    num_preload_transitions=200000,
    motion_files=...,
)
```

你的控制周期：

[
env.step_dt=0.02\text{秒}
]

所以Loader随机选择专家动作中的一个时间 (t)，然后提取：

[
s_t^E
]

和：

[
s_{t+0.02}^E
]

形成一条专家transition：

[
(s_t^E,s_{t+0.02}^E)
]

初始化时会预加载20万条这样的专家transition。当前Runner将环境控制周期作为专家状态对的时间间隔；MotionLoader随机选择动作片段和时间，并获取该时间及 `time+step_dt` 的两帧。([GitHub][4])

## 专家文件是30 fps，间隔0.033秒，怎么取0.02秒

MotionLoader会在相邻文件帧之间做线性插值。

例如文件中：

```text
frame 10：时间0.330秒
frame 11：时间0.363秒
```

需要取：

```text
时间0.350秒
```

它会计算一个混合比例：

[
\beta
=====

\frac{0.350-0.330}{0.363-0.330}
\approx0.606
]

然后：

[
s(0.350)
========

(1-\beta)s_{10}
+
\beta s_{11}
]

所以专家文件帧率不需要和Actor控制频率完全一样。MotionLoader根据随机时间定位前后帧，并使用线性混合生成对应时间的AMP状态。([GitHub][7])

---

# 七、判别器也是神经网络吗

是的。

当前判别器是一个MLP：

```text
输入：104
  ↓
Linear(104 → 1024)
  ↓
ReLU
  ↓
Linear(1024 → 512)
  ↓
ReLU
  ↓
Linear(512 → 256)
  ↓
ReLU
  ↓
Linear(256 → 1)
  ↓
输出一个数 D
```

数学上：

[
h_1=\operatorname{ReLU}(W_1x+b_1)
]

[
h_2=\operatorname{ReLU}(W_2h_1+b_2)
]

[
h_3=\operatorname{ReLU}(W_3h_2+b_3)
]

[
D(x)=W_4h_3+b_4
]

判别器最后没有Sigmoid，因此输出不是被限制在0到1之间的概率，可以输出：

```text
-1.2
-0.4
0
0.8
1.5
```

当前代码使用ReLU隐藏层和一个线性输出头，输入维度是拼接后的104维，输出维度是1。([GitHub][8])

---

# 八、为什么专家目标是+1，策略目标是-1

这是当前AMP使用的**最小二乘判别器**设计。

定义：

[
D(x_E)\rightarrow+1
]

[
D(x_\pi)\rightarrow-1
]

于是：

```text
D接近+1：更像专家
D接近-1：更像当前策略/非专家
D接近0：判别器不确定
```

选择 `+1/-1` 的好处是标签以0为中心，非常对称：

```text
策略 ←──── -1 ──── 0 ──── +1 ────→ 专家
```

理论上也可以设计成专家1、策略0，但对应的loss和风格奖励公式也要一起改变。当前代码和AMP奖励公式都是围绕专家目标 `+1`、策略目标 `-1` 写的。([GitHub][9])

---

# 九、MSE是什么

MSE全称：

> Mean Squared Error，均方误差。

公式：

[
\operatorname{MSE}
==================

\frac1N
\sum_{i=1}^{N}
(\text{预测}_i-\text{目标}_i)^2
]

例如专家样本目标是：

[
y_E=+1
]

判别器输出：

[
D_E=0.6
]

误差：

[
0.6-1=-0.4
]

平方误差：

[
(-0.4)^2=0.16
]

如果策略样本目标为：

[
y_\pi=-1
]

判别器输出：

[
D_\pi=-0.2
]

误差：

[
-0.2-(-1)=0.8
]

平方：

[
0.8^2=0.64
]

当前AMP判别器损失：

[
L_{\rm AMP}
===========

\frac12
\left[
\operatorname{MSE}(D_E,1)
+
\operatorname{MSE}(D_\pi,-1)
\right]
]

代入：

[
L_{\rm AMP}
===========

# \frac12(0.16+0.64)

0.40
]

代码正是分别对专家输出和全1标签、策略输出和全负1标签计算MSE，再取两者平均。([GitHub][9])

---

# 十、MSE怎样推动判别器更新

专家部分：

[
L_E=(D_E-1)^2
]

对输出 (D_E) 求导：

[
\frac{\partial L_E}{\partial D_E}
=================================

2(D_E-1)
]

若：

[
D_E=0.6
]

则：

[
\frac{\partial L_E}{\partial D_E}
=================================

2(-0.4)=-0.8
]

梯度下降会增加 (D_E)，让它靠近1。

策略部分：

[
L_\pi=(D_\pi+1)^2
]

若：

[
D_\pi=-0.2
]

则：

[
\frac{\partial L_\pi}{\partial D_\pi}
=====================================

# 2(-0.2+1)

1.6
]

梯度下降会降低 (D_\pi)，让它靠近-1。

因此：

```text
专家输出0.6 → 往1推
策略输出-0.2 → 往-1推
```

再通过反向传播修改判别器MLP所有层的权重和偏置。

---

# 十一、什么是梯度惩罚

判别器输入是104维状态对：

[
x=[s_t,s_{t+1}]
]

输出：

[
D(x)
]

输入梯度是：

[
\nabla_xD(x)
============

\left[
\frac{\partial D}{\partial x_1},
\frac{\partial D}{\partial x_2},
\ldots,
\frac{\partial D}{\partial x_{104}}
\right]
]

它表示：

> 输入的某一个关节角度、速度或手脚位置稍微变化时，判别器输出会变化多快。

例如：

[
\frac{\partial D}{\partial q_{\rm knee}}=8
]

表示膝关节输入只改变：

[
0.01
]

判别器输出就可能改变约：

[
8\times0.01=0.08
]

如果许多梯度非常大，判别器可能形成非常尖锐的判断边界：

```text
姿态只差一点点
D从+1突然跳到-1
```

这会导致：

* 判别器过拟合有限专家帧；
* 专家附近的合理变化也被判成非专家；
* style reward剧烈跳动；
* Actor收到不稳定奖励；
* 对抗训练更容易震荡。

梯度惩罚就是限制这种过度敏感。AMP论文使用专家数据上的梯度惩罚改善判别器和style reward的稳定性。([arXiv][3])

---

# 十二、梯度惩罚代码具体做了什么

当前代码大致是：

```python
expert_data = torch.cat(
    [expert_state, expert_next_state],
    dim=-1,
)

expert_data.requires_grad = True

disc = discriminator(expert_data)

grad = autograd.grad(
    outputs=disc,
    inputs=expert_data,
    grad_outputs=torch.ones_like(disc),
    create_graph=True,
)[0]

grad_penalty = 10 * (
    grad.norm(2, dim=1) - 0
).pow(2).mean()
```

每一步含义如下。

## 第1步：拼成104维输入

```text
expert_state      [6144,52]
expert_next_state [6144,52]
```

变成：

```text
expert_data [6144,104]
```

## 第2步：允许计算输入梯度

```python
expert_data.requires_grad = True
```

平时训练主要对网络参数求梯度。

这里还要对输入 (x) 求：

[
\frac{\partial D}{\partial x}
]

所以必须打开输入梯度。

## 第3步：判别器前向计算

```python
disc = D(expert_data)
```

输出：

```text
[6144,1]
```

## 第4步：对输入求导

得到：

```text
grad.shape = [6144,104]
```

每一行是一条专家状态对的104维梯度。

## 第5步：计算每条样本的L2范数

[
|g_i|_2
=======

\sqrt{
g_{i,1}^2+
g_{i,2}^2+
\cdots+
g_{i,104}^2
}
]

## 第6步：平方、平均并乘10

[
L_{\rm GP}
==========

10
\cdot
\frac1B
\sum_{i=1}^{B}
|g_i|_2^2
]

当前实现确实对拼接后的专家状态对求判别器输入梯度，并惩罚该梯度L2范数与0之间的平方距离。([GitHub][8])

---

# 十三、什么叫“零中心梯度惩罚”

公式是：

[
L_{\rm GP}
==========

10\cdot
\mathbb E
\left[
(|\nabla_xD(x_E)|_2-0)^2
\right]
]

因为目标是：

[
0
]

所以叫：

> Zero-centered gradient penalty，零中心梯度惩罚。

它希望在专家样本附近：

[
\nabla_xD(x_E)\approx0
]

直观上就是：

> 专家运动附近，判别器输出应该比较平坦，不应该因为输入出现非常小的噪声就剧烈变化。

例如专家样本：

```text
左膝角度 = 1.000 rad
判别器输出 = 0.80
```

稍微扰动：

```text
左膝角度 = 1.001 rad
```

希望输出还是接近：

```text
0.80
```

而不是突然变成：

```text
-0.50
```

---

## 零中心不是让判别器输出为0

这是最容易误解的地方。

零中心惩罚要求的是：

[
\nabla_xD(x_E)\rightarrow0
]

不是：

[
D(x_E)\rightarrow0
]

专家分类loss仍要求：

[
D(x_E)\rightarrow+1
]

两者同时作用：

```text
专家点上的输出：接近+1
专家点附近的斜率：接近0
```

可以想象成：

```text
D
1.0 ───────────── 专家附近较平坦
             \
              \
-1.0           ───── 策略区域
```

不是在专家点上形成尖锐的针：

```text
D
1.0       /\
         /  \
-1.0 ───/    \────
```

---

# 十四、如果梯度全部趋近0，判别器不就无法分类了吗

如果整个空间所有地方梯度都严格为0，那么判别器确实只能输出常数，无法分类。

但实际总loss同时存在两个相反要求：

```text
MSE分类loss：
必须把专家推向+1、策略推向-1

梯度惩罚：
不要在专家附近变化得过于尖锐
```

最终形成折中：

```text
判别器仍然能区分专家与策略
但边界更加平滑
不过度记忆每个专家样本
```

梯度惩罚不是唯一目标，只是一个正则化项。

---

# 十五、当前代码的一个具体细节

分类loss使用了经过AMP Normalizer归一化的策略和专家状态：

```python
policy_state = normalizer.normalize(...)
expert_state = normalizer.normalize(...)
```

但当前代码调用梯度惩罚时传入的是：

```python
compute_grad_pen(*sample_amp_expert)
```

也就是generator原始取出的专家样本，而不是上面重新赋值后的归一化变量。

所以当前实现中：

```text
MSE分类：
使用归一化后的AMP状态

Gradient penalty：
使用原始专家AMP状态
```

这是当前代码的具体行为，而不是所有AMP实现都必须如此。([GitHub][9])

---

# 十六、完整的实际AMP计算例子

下面使用一条接近真实结构的专家transition和一条策略transition。

实际每个state有52维。为了不把104个数字全部铺开，只展示有代表性的几维，其他维度同样参与网络计算。

## 1. 专家状态对

当前专家状态：

```text
expert s_t：

左髋Pitch       -0.30
左膝Pitch        0.58
右髋Pitch        0.17
右膝Pitch        0.34
左脚相对x       -0.12
右脚相对x        0.20
……共52维
```

20 ms后的专家状态：

```text
expert s_t+1：

左髋Pitch       -0.28
左膝Pitch        0.55
右髋Pitch        0.20
右膝Pitch        0.37
左脚相对x       -0.11
右脚相对x        0.22
……共52维
```

拼接：

```text
expert_x.shape = [104]
```

## 2. 策略状态对

```text
policy s_t：

左髋Pitch       -0.30
左膝Pitch        0.58
右髋Pitch        0.17
右膝Pitch        0.34
左脚相对x       -0.12
右脚相对x        0.20
……
```

当前姿态看起来与专家很像。

但20 ms后：

```text
policy s_t+1：

左髋Pitch       -0.38
左膝Pitch        0.70
右髋Pitch        0.12
右膝Pitch        0.28
左脚相对x       -0.18
右脚相对x        0.19
……
```

它变化太猛，可能有抖动或不自然落脚。

如果只看当前 (s_t)，两者很相似。

看状态对后，判别器可以发现：

```text
专家：平滑地向下一步移动
策略：关节和足部出现不自然跳变
```

---

## 3. 判别器前向输出

假设当前判别器得到：

[
D_E=0.60
]

[
D_\pi=-0.20
]

含义：

```text
专家样本：
偏向专家，但还没有达到目标+1

策略样本：
偏向策略，但还没有达到目标-1
```

---

## 4. 计算AMP分类loss

专家loss：

[
L_E=(0.60-1)^2
]

[
L_E=0.16
]

策略loss：

[
L_\pi=(-0.20-(-1))^2
]

[
L_\pi=0.8^2=0.64
]

AMP loss：

[
L_{\rm AMP}
===========

\frac12(0.16+0.64)
]

[
\boxed{L_{\rm AMP}=0.40}
]

---

## 5. 计算梯度惩罚

假设对这条专家104维输入求导后，得到的完整梯度向量范数是：

[
|\nabla_xD(x_E)|_2=0.12
]

那么：

[
L_{\rm GP}
==========

10(0.12-0)^2
]

# [

10\times0.0144
]

[
\boxed{L_{\rm GP}=0.144}
]

判别器本次加入总loss的部分：

[
L_D
===

L_{\rm AMP}+L_{\rm GP}
]

[
=0.40+0.144
]

[
\boxed{L_D=0.544}
]

反向传播以后：

```text
专家输出0.60 → 向+1移动
策略输出-0.20 → 向-1移动
专家附近输入梯度0.12 → 向0缩小
```

---

# 十七、判别器输出如何变成Actor的style reward

策略的判别器输出：

[
D_\pi=-0.20
]

当前风格奖励公式：

[
r_{\rm style}
=============

0.3
\cdot
\max
\left(
1-\frac14(D_\pi-1)^2,
0
\right)
]

代入：

[
D_\pi-1=-1.2
]

[
(-1.2)^2=1.44
]

[
\frac14\times1.44=0.36
]

[
1-0.36=0.64
]

因此：

[
r_{\rm style}
=============

0.3\times0.64
]

[
\boxed{r_{\rm style}=0.192}
]

当前代码就是以专家目标 `D=1` 为中心构造二次奖励；越接近1，style reward越高。([GitHub][8])

---

## 判别器不同输出对应的style reward

|  (D) | 风格含义     | (r_{\rm style}) |
| ---: | -------- | --------------: |
| -1.0 | 很像策略/非专家 |               0 |
| -0.5 | 较不像专家    |         0.13125 |
|  0.0 | 判别器不确定   |           0.225 |
|  0.5 | 较像专家     |         0.28125 |
|  1.0 | 最像专家目标   |           0.300 |

所以Actor希望让自己产生的运动满足：

[
D(s_t^\pi,s_{t+1}^\pi)\rightarrow1
]

---

# 十八、和task reward混合

假设当前任务奖励：

[
r_{\rm task}=0.8
]

当前：

[
amp_task_reward_lerp=0.7
]

最终奖励：

[
r_{\rm final}
=============

0.3r_{\rm style}
+
0.7r_{\rm task}
]

代入：

[
r_{\rm final}
=============

0.3\times0.192
+
0.7\times0.8
]

# [

0.0576+0.56
]

[
\boxed{r_{\rm final}=0.6176}
]

这个最终奖励进入：

```text
Reward
↓
Return
↓
Advantage
↓
PPO surrogate loss
↓
Actor反向传播
```

注意Actor不是直接通过：

[
L_{\rm AMP}
]

得到梯度。

真实路径是：

```text
Actor产生运动
↓
判别器给style reward
↓
style reward影响Advantage
↓
PPO修改Actor动作概率
```

Runner确实先用判别器把环境任务奖励转换为混合奖励，再把奖励存入PPO rollout；判别器分类loss本身则直接更新判别器参数。([GitHub][4])

---

# 十九、结合你最终日志理解

你的最终日志：

```text
Mean amp_policy_pred: -0.4092
Mean amp_expert_pred:  0.4079
Mean amp loss:         0.3716
Mean amp_grad_pen:     0.2153
```

使用平均输出粗略估算：

专家误差：

[
(0.4079-1)^2
\approx0.3506
]

策略误差：

[
(-0.4092+1)^2
\approx0.3490
]

粗略AMP loss：

[
\frac12(0.3506+0.3490)
\approx0.3498
]

日志是：

[
0.3716
]

之所以不完全相等，是因为日志loss计算的是：

[
\operatorname{mean}((D-目标)^2)
]

而我们使用的是：

[
(\operatorname{mean}(D)-目标)^2
]

两者在batch有方差时不会相等。

---

## 你的梯度惩罚意味着什么

日志：

[
L_{\rm GP}=0.2153
]

而：

[
L_{\rm GP}
==========

10\mathbb E[|g|^2]
]

所以：

[
\mathbb E[|g|^2]
================

0.02153
]

对应均方根梯度范数约：

[
\sqrt{0.02153}
\approx0.147
]

也就是专家样本附近，判别器输入梯度的典型量级大约在0.15左右，并没有出现非常大的尖锐梯度。

---

# 二十、完整AMP训练闭环

```text
1. 专家文件提供52维运动帧
             ↓
2. 随机取专家时间t和t+0.02
             ↓
3. 得到专家状态对 [52]+[52]=[104]
             ↓
4. Actor控制机器人运行20ms
             ↓
5. 得到策略状态对 [52]+[52]=[104]
             ↓
6. 两类状态对输入同一个判别器
             ↓
7. 判别器学习：
   专家 → +1
   策略 → -1
             ↓
8. 梯度惩罚让专家附近的判断更平滑
             ↓
9. 用判别器输出计算style reward
             ↓
10. style reward与task reward混合
             ↓
11. 计算Return和Advantage
             ↓
12. PPO更新Actor
             ↓
13. Actor运动越来越像专家
             ↓
14. 判别器重新学习区分新的策略运动
             ↓
不断对抗迭代
```

最核心的理解是：

> 专家数据告诉AMP“自然行走的状态变化长什么样”；判别器通过状态对学习这种局部运动规律；策略并不逐帧照抄专家，而是在完成速度跟踪任务的同时，尽量产生让判别器认为像专家的运动。

[1]: https://arxiv.org/abs/2104.02180 "[2104.02180] AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
[2]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/legged_lab/envs/tienkung/tienkung_env.py "raw.githubusercontent.com"
[3]: https://arxiv.org/abs/2104.02180?utm_source=chatgpt.com "AMP: Adversarial Motion Priors for Stylized Physics-Based ..."
[4]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/runners/amp_on_policy_runner.py "raw.githubusercontent.com"
[5]: https://github.com/Open-X-Humanoid/TienKung-Lab "GitHub - Open-X-Humanoid/TienKung-Lab: Tien Kung-Lab: Direct IsaacLab Workflow for Legged Robots · GitHub"
[6]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/legged_lab/scripts/play_amp_animation.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/utils/motion_loader.py "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/modules/discriminator.py "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/main/rsl_rl/rsl_rl/algorithms/amp_ppo.py "raw.githubusercontent.com"

你理解得已经很接近了，真正混淆的是这一点：

> **专家数据是“标准样本”，但判别器一开始并不知道什么样的104维数字代表专家。**

专家文件只提供：

[
x_E=(s_t^E,s_{t+1}^E)
]

也就是一串104维数值。文件本身不会自动告诉随机初始化的神经网络：

```text
哪些关节组合重要
什么样的脚步变化自然
什么样的速度变化像专家
什么样的状态转移不自然
```

所以必须训练判别器，让它学习一个评分函数：

[
D_\psi(x)
]

这里 (\psi) 是判别器神经网络的参数。

---

# 一、最关键的纠正

AMP中同时存在两个不同的学习目标。

## 判别器的目标

判别器想把两类数据分开：

[
D(x_E)\rightarrow+1
]

[
D(x_\pi)\rightarrow-1
]

这里：

* (x_E)：专家文件中的状态对；
* (x_\pi)：当前Actor实际产生的状态对。

## Actor策略的目标

Actor不想让自己的状态对保持在 (-1)。

Actor希望：

[
D(x_\pi)\ \text{尽可能高}
]

也就是让策略运动看起来越来越像专家。

所以这两个目标是对抗的：

```text
判别器：
“我要认出这是策略产生的，把它判成-1。”

Actor：
“我要改变运动，让判别器把我看成专家。”
```

`-1`是**判别器训练策略样本时的标签**，不是要求Actor最终变成 (-1)。

---

# 二、两条更新路径必须分开看

完整结构是：

```text
                    专家状态对 xE
                         │
                         ▼
                  ┌──────────────┐
                  │ 判别器 Dψ(x) │
                  └──────────────┘
                         │
                    希望输出 +1
                         │
                更新判别器参数 ψ


Actor产生策略状态对 xπ
                         │
                         ▼
                  ┌──────────────┐
                  │ 判别器 Dψ(x) │
                  └──────────────┘
                    │          │
        判别器训练时│          │Actor训练时
                    │          │
              希望输出-1       输出越高
              更新判别器ψ      style reward越高
                               更新Actor参数θ
```

所以：

[
\boxed{
D(x_\pi)\rightarrow-1
}
]

只属于判别器的分类任务。

而Actor的目标是：

[
\boxed{
x_\pi\text{变得像}x_E，使D(x_\pi)\text{提高}
}
]

AMP就是利用这种相互对抗，让判别器不断提出更严格的“像专家”标准，而策略不断改进运动来通过这个标准。AMP论文中，运动数据用于训练对抗式运动先验，判别器产生的风格奖励再用于强化学习策略。([arXiv][1])

---

# 三、为什么专家文件还要经过判别器计算 (D_E)

你可以把专家文件理解为：

> 标准答案的样本集合。

但它不是现成的“评分公式”。

例如专家文件给出一条104维状态对：

[
x_E=
[
-0.30,\ 0.58,\ 0.17,\ldots
]
]

随机初始化的判别器只是一个随机函数。

一开始它可能输出：

[
D(x_E)=-0.35
]

虽然我们人类知道这是专家数据，但判别器不知道。

所以要告诉判别器：

```text
这条输入来自专家文件
它的标签是+1
你的输出应该向+1靠近
```

计算专家误差：

[
L_E=(D_E-1)^2
]

然后反向传播修改判别器参数。

因此，计算 (D_E) 的目的不是检查专家文件“够不够专家”，而是：

> 检查当前判别器是否已经学会对专家状态对输出高分，并用误差训练判别器。

---

# 四、(D_E=0.60)代表什么

假设：

[
D_E=0.60
]

这不表示：

> 这条专家数据只有60%是专家。

也不表示：

> 专家文件可能不标准。

它只表示：

> 当前这套判别器参数，对这条专家状态对输出了0.60，而训练目标是1，因此判别器还需要继续学习。

专家loss：

[
L_E=(0.60-1)^2=0.16
]

这个误差会让判别器下一次倾向于输出更高，例如：

```text
0.60 → 0.65 → 0.72 → 0.80
```

---

# 五、为什么必须给策略样本一个 (-1) 标签

这是核心。

假设我们只给判别器专家数据，并要求：

[
D(x_E)\rightarrow1
]

却不提供策略数据。

那么判别器最简单的做法是什么？

它可以无论输入什么，都输出：

[
D(x)=1
]

也就是：

```python
def discriminator(any_input):
    return 1.0
```

这时专家loss为：

[
(1-1)^2=0
]

看起来训练完美。

但问题是：

```text
专家走路 → D=1
正常策略走路 → D=1
乱踢腿 → D=1
躺在地上 → D=1
摔倒 → D=1
```

判别器完全没有区分能力。

style reward对所有动作都一样，Actor就无法知道怎样才更像专家。

因此必须同时提供负样本：

[
x_\pi
]

并告诉判别器：

[
D(x_\pi)\rightarrow-1
]

这样判别器才必须学习：

> 专家状态对和当前策略状态对究竟有什么区别。

判别器本质上是一个分类器，需要正样本和负样本共同学习分类边界。普通GAN中的判别器同样使用真实数据作为正样本、生成数据作为负样本。([Google for Developers][2])

---

# 六、没有策略负样本会发生什么

只有专家数据时：

```text
专家样本 → +1
```

可能得到：

```text
所有输入 → +1
```

这也是零loss。

加入策略数据后：

```text
专家样本 → +1
策略样本 → -1
```

判别器不能再输出常数。

因为如果全部输出 (+1)：

专家loss：

[
(1-1)^2=0
]

策略loss：

[
(1-(-1))^2=4
]

总loss很大。

如果全部输出 (-1)：

专家loss：

[
(-1-1)^2=4
]

策略loss：

[
(-1-(-1))^2=0
]

同样不行。

它必须真正从输入特征中找差异。

---

# 七、用老师批改作文来理解

假设专家文件是优秀作文。

你要训练一个“作文评分老师”。

## 只有优秀作文

你告诉老师：

```text
这些都是优秀作文，打满分。
```

老师最简单的学习结果可能是：

```text
无论看到什么作文，都打满分。
```

因为他没有见过差作文，不知道什么特征意味着差。

## 同时给优秀和当前学生作文

现在告诉老师：

```text
专家作文：+1
当前学生作文：-1
```

老师会开始学习：

```text
专家作文：
结构完整、语言自然、论证清楚

当前学生作文：
语句跳跃、逻辑断裂、结构混乱
```

然后学生根据老师评分改作文。

学生的目标不是：

```text
让我永远得-1
```

而是：

```text
虽然老师目前把我标记为-1，
但我要改得越来越像优秀作文，
最终让老师无法区分。
```

---

# 八、判别器和Actor在“拔河”

假设当前策略走路很差。

## 第一次判别器训练

专家：

[
D_E=0.6
]

策略：

[
D_\pi=-0.2
]

判别器的目标：

```text
专家0.6 → 推向+1
策略-0.2 → 推向-1
```

更新后可能变成：

[
D_E=0.75
]

[
D_\pi=-0.45
]

---

## Actor看到style reward

Actor产生的状态对得到：

[
D_\pi=-0.45
]

风格奖励较低。

PPO因此修改Actor，让机器人产生更像专家的动作。

更新后，新策略运动可能更加自然。

---

## 第二次判别器观察新策略

新的策略状态对不再像以前那么差。

判别器可能输出：

[
D_\pi=0.1
]

说明它开始觉得策略运动具有部分专家特征。

但判别器训练时仍会把这条数据标记为策略样本：

[
目标=-1
]

于是判别器会进一步寻找更细致的区别：

```text
脚步虽然像了
但手臂摆动还不够自然

关节轨迹虽然像了
但落脚速度仍有差异
```

然后Actor再继续改进。

因此二者不断升级：

```text
判别器更会找区别
        ↕
Actor更会模仿专家
```

---

# 九、Actor究竟有没有直接接受“目标+1”

严格地说，Actor没有像判别器一样直接计算：

[
(D_\pi-1)^2
]

并反向传播穿过物理仿真。

当前项目是先把 (D_\pi) 转成style reward：

[
r_{\text{style}}
================

0.3\max
\left[
1-\frac14(D_\pi-1)^2,\ 0
\right]
]

然后：

```text
Dπ越接近+1
→ style reward越高
→ Return越高
→ Advantage越高
→ PPO提高相关动作概率
```

所以从目标效果上看，Actor希望自己的状态对得到更高的 (D_\pi)。

但更新路径是：

[
D_\pi
\rightarrow
r_{\text{style}}
\rightarrow
A_t
\rightarrow
L_{\text{PPO}}
\rightarrow
\text{Actor参数}
]

不是直接：

[
(D_\pi-1)^2
\rightarrow
\text{Actor反向传播}
]

---

# 十、判别器训练时不会把Actor推向 (-1)

这是你最担心的地方。

判别器训练策略样本时计算：

[
L_\pi=(D(x_\pi)+1)^2
]

然后反向传播。

但此时更新的是：

[
\psi
]

也就是判别器参数。

不是Actor参数：

[
\theta
]

用数学表示：

[
D_\psi(x_\pi)
]

判别器loss对判别器参数求导：

[
\frac{\partial L_D}{\partial\psi}
]

而不会在这里更新：

[
\frac{\partial L_D}{\partial\theta}
]

因为策略状态对已经是从rollout/replay buffer取出的数据，不保留穿过PhysX和Actor的梯度计算图。

所以：

```text
策略目标为-1
```

更准确应该说成：

```text
判别器在训练时，
把“当前策略产生的数据”当作负样本，
用-1作为分类标签。
```

不是：

```text
Actor被训练成让自己输出-1。
```

---

# 十一、用一次完整数值计算区分两种更新

假设：

[
D_E=0.60
]

[
D_\pi=-0.20
]

## 判别器更新

专家loss：

[
L_E=(0.60-1)^2=0.16
]

策略loss：

[
L_\pi=(-0.20+1)^2=0.64
]

总分类loss：

[
L_D=\frac12(0.16+0.64)=0.40
]

反向传播：

```text
更新判别器参数ψ：

专家输出0.60 → 更接近+1
策略输出-0.20 → 更接近-1
```

Actor参数在这里不变。

---

## Actor更新

策略输出：

[
D_\pi=-0.20
]

style reward：

[
r_{\text{style}}
================

0.3
\left[
1-\frac14(-0.20-1)^2
\right]
]

# [

0.3
\left[
1-\frac14(1.44)
\right]
]

[
=0.3(0.64)=0.192
]

Actor发现这段运动只能获得0.192的风格奖励。

假设后来Actor产生更像专家的新运动，判别器输出：

[
D_\pi=0.50
]

style reward变成：

[
r_{\text{style}}
================

0.3
\left[
1-\frac14(0.5-1)^2
\right]
]

# [

0.3(1-0.0625)
]

[
=0.28125
]

显然：

[
0.28125>0.192
]

所以PPO会倾向于提高产生第二种运动的动作概率。

---

# 十二、为什么判别器不能固定训练一次

可能会想到：

> 先用专家和早期乱动的数据训练好判别器，然后固定它不就行了？

问题是Actor一直在变化。

训练早期策略可能是：

```text
乱动、摔倒、抖腿
```

判别器很容易区分。

后来策略可能已经能够正常走路：

```text
姿态接近专家
但脚步节奏略有区别
```

如果判别器停留在早期水平，它可能对所有“能走路”的动作都给高分，无法继续提供细致指导。

所以判别器需要不断看：

```text
最新的策略运动
+
专家运动
```

然后重新学习当前最关键的区别。

这就是为什么叫对抗训练：

```text
策略改进
→ 判别器面对更难的负样本
→ 判别器改进
→ 策略需要进一步改进
```

---

# 十三、真正训练平衡时，判别器一定输出+1吗

不一定。

这是一个更深的结论。

假设专家状态对的概率密度为：

[
p_E(x)
]

策略状态对的概率密度为：

[
p_\pi(x)
]

对于当前这种 (+1/-1) 的MSE判别器，某个输入位置 (x) 上的理想判别器输出是：

[
\boxed{
D^*(x)
======

\frac{p_E(x)-p_\pi(x)}
{p_E(x)+p_\pi(x)}
}
]

这个式子可以直接看出：

## 只有专家数据出现的位置

如果：

[
p_E(x)>0,\qquad p_\pi(x)=0
]

那么：

[
D^*(x)=+1
]

## 只有策略数据出现的位置

如果：

[
p_E(x)=0,\qquad p_\pi(x)>0
]

那么：

[
D^*(x)=-1
]

## 专家和策略分布完全一样

如果：

[
p_E(x)=p_\pi(x)
]

那么：

[
D^*(x)=0
]

因为判别器已经无法区分这条状态对来自专家还是策略。

因此理想对抗平衡并不是：

[
D_\pi=+1
]

而往往是：

[
D_E\approx D_\pi\approx0
]

这表示：

> 策略状态分布已经和专家状态分布非常相似，判别器只能猜测，无法可靠区分。

---

# 十四、这个理想判别器公式怎么来的

在某个固定状态对 (x) 上，判别器要最小化：

[
L(D)
====

p_E(x)(D-1)^2
+
p_\pi(x)(D+1)^2
]

对 (D) 求导：

[
\frac{dL}{dD}
=============

2p_E(D-1)
+
2p_\pi(D+1)
]

令导数为0：

[
p_E(D-1)+p_\pi(D+1)=0
]

展开：

[
p_ED-p_E+p_\pi D+p_\pi=0
]

整理：

[
D(p_E+p_\pi)=p_E-p_\pi
]

所以：

[
D^*
===

\frac{p_E-p_\pi}{p_E+p_\pi}
]

这个公式说明，判别器输出不是一个固定的“专家相似度百分比”，而是在比较：

> 这个状态转移在专家数据中常见，还是在策略数据中更常见。

---

# 十五、一个分布变化的实际例子

考虑某种自然的落脚状态对 (x_A)。

## 训练早期

专家中经常出现：

[
p_E(x_A)=0.8
]

策略中很少出现：

[
p_\pi(x_A)=0.2
]

理想判别器：

[
D^*(x_A)
========

# \frac{0.8-0.2}{0.8+0.2}

0.6
]

所以即使这条状态对来自专家，输出也可能是：

[
D_E\approx0.6
]

因为当前判别器是在综合比较两个数据分布，而不是机械读取文件标签。

---

## 策略逐渐模仿成功

策略也经常产生这种自然落脚：

[
p_\pi(x_A)=0.8
]

此时：

[
D^*(x_A)
========

\frac{0.8-0.8}{0.8+0.8}
=0
]

判别器不能区分。

这不是失败，反而意味着策略成功进入了专家运动分布。

---

# 十六、既然最终可能是0，为什么style reward以+1为最高

当前风格奖励：

[
r_{\text{style}}
================

0.3\max
\left[
1-\frac14(D-1)^2,\ 0
\right]
]

不同输出：

| (D_\pi) | style reward |
| ------: | -----------: |
|    -1.0 |            0 |
|    -0.5 |      0.13125 |
|     0.0 |        0.225 |
|     0.5 |      0.28125 |
|     1.0 |        0.300 |

所以即使对抗平衡时：

[
D_\pi\approx0
]

仍然能获得：

[
r_{\text{style}}=0.225
]

已经是最大值0.3的75%。

真正明显的差别是：

```text
不像专家：D接近-1 → 奖励接近0
进入专家分布：D接近0 → 得到较高奖励
被判别器高度认为像专家：D接近1 → 奖励最高
```

---

# 十七、把你的理解改写成准确版本

你原来的理解是：

> 专家对是标准，策略对通过判别器算D，然后让它目标为1。

更精确地说应该是：

```text
1. 专家对是“标准样本”，不是现成的评分函数。

2. 判别器需要同时看到：
   专家正样本，标签+1；
   当前策略负样本，标签-1。

3. 判别器通过两类样本学习：
   什么状态变化更常见于专家，
   什么状态变化更常见于当前策略。

4. -1只用于训练判别器识别当前策略数据，
   不用于把Actor训练成-1。

5. Actor通过style reward反向获得信号：
   判别器输出越高，运动越值得强化。

6. Actor不断改变策略运动，使策略分布靠近专家分布。

7. 当两种分布非常接近时，
   判别器难以区分，输出可能接近0。
```

最核心的一句话是：

[
\boxed{
\text{判别器努力把专家和策略分开；Actor努力让两者无法被分开。}
}
]

如果没有策略样本的 (-1) 标签，判别器可以对所有输入直接输出 (+1)，style reward就失去任何指导意义。

[1]: https://arxiv.org/abs/2104.02180?utm_source=chatgpt.com "AMP: Adversarial Motion Priors for Stylized Physics-Based ..."
[2]: https://developers.google.com/machine-learning/gan/discriminator?utm_source=chatgpt.com "The Discriminator | Machine Learning"



下面按当前公开仓库 `main` 分支，把 **TienKung-Lab 的源代码、配置、模型资源和数据文件**全部归类说明。

先说明范围：

> 我会逐个覆盖仓库中的 Python 源代码和主要配置文件；`meshes/`、`usd/`、动捕数据等包含大量非代码资源，不逐个列每个网格文件，而是按目录解释。

你本地训练后生成的：

```text
logs/
local_deploy/
__pycache__/
*.egg-info/
```

不属于 GitHub 原始仓库，是运行或安装时自动生成的。

---

# 一、整个仓库的分层结构

可以先把仓库理解成五层：

```text
TienKung-Lab
│
├── 1. 机器人和任务层：legged_lab/
│      机器人模型、环境、观测、奖励、任务配置、传感器
│
├── 2. 强化学习算法层：rsl_rl/
│      PPO、AMP、Actor-Critic、Runner、Storage
│
├── 3. 程序入口层：legged_lab/scripts/
│      train、play、动作数据处理、Sim2Sim
│
├── 4. 数据和模型资源层
│      assets、datasets、Exported_policy、docs
│
└── 5. 工程管理层
       setup.py、LICENSE、pre-commit、flake8、GitHub配置
```

项目本身定位是基于 Isaac Lab 的天工人形机器人强化学习运动控制框架，包含 AMP、周期步态奖励、传感器以及向 MuJoCo 的 Sim2Sim 验证。([GitHub][1])

---

# 二、根目录

```text
TienKung-Lab/
├── .github/
├── Exported_policy/
├── docs/
├── legged_lab/
├── rsl_rl/
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
└── setup.py
```

## 根目录文件作用

| 文件或目录                     | 所属部分   | 作用                                                  |
| ------------------------- | ------ | --------------------------------------------------- |
| `legged_lab/`             | 机器人业务层 | 环境、机器人、奖励、传感器、训练和仿真脚本                               |
| `rsl_rl/`                 | 强化学习层  | PPO、AMP、神经网络、Runner、经验存储                            |
| `Exported_policy/`        | 预训练模型  | 作者提供的可直接用于 Sim2Sim 的策略，例如 `walk.pt`                 |
| `docs/`                   | 文档资源   | README 中的图片、动画和说明材料                                 |
| `.github/`                | 工程管理   | GitHub Actions、Issue 模板等                            |
| `setup.py`                | 安装配置   | 将 `legged_lab` 安装为 Python 包，并声明 IsaacLab、MuJoCo 等依赖 |
| `README.md`               | 使用说明   | 安装、动作数据转换、训练、播放、Sim2Sim 和部署说明                       |
| `.flake8`                 | 代码检查   | flake8 检查规则、最大行长、复杂度等                               |
| `.pre-commit-config.yaml` | 代码格式化  | 提交前自动执行 black、flake8 等工具                            |
| `.gitignore`              | Git配置  | 忽略日志、缓存、模型和临时文件                                     |
| `LICENSE`                 | 许可证    | 仓库代码使用许可                                            |

根目录 `setup.py` 安装的是 `legged_lab`，而 `rsl_rl` 具有自己独立的 `setup.py`，因此之前需要分别执行两次 `pip install -e .`。([Gitee][2])

---

# 三、`legged_lab/`：机器人和仿真业务层

完整主结构：

```text
legged_lab/
├── assets/
├── envs/
├── mdp/
├── scripts/
├── sensors/
├── terrains/
├── utils/
└── __init__.py
```

这部分回答：

* 天工机器人长什么样；
* 有哪些关节；
* 环境如何运行；
* Actor 看见什么；
* 动作如何控制关节；
* reward 如何计算；
* 什么时候 reset；
* 如何训练、播放和进入 MuJoCo。

([GitHub][3])

---

# 四、`legged_lab/assets/`：机器人模型与执行器配置

```text
assets/
├── __init__.py
└── tienkung2_lite/
    ├── __init__.py
    ├── tienkung.py
    ├── meshes/
    ├── mjcf/
    ├── urdf/
    └── usd/
```

## `tienkung.py`

这是机器人资产的 Python 配置入口，主要定义：

* Isaac Lab 使用哪个 USD 模型；
* 机器人的初始根部位置；
* 各个关节默认角度；
* 关节执行器分组；
* stiffness、damping、最大力矩等执行器参数；
* 哪些关节属于腿、手臂；
* 机器人生成到场景中的配置。

它解决的是：

```text
“在Isaac Sim中创建一台什么样的天工机器人？”
```

环境配置中的：

```python
robot = TIENKUNG_CFG
```

最终就会引用这里定义的机器人。

## 非Python资源目录

| 目录            | 作用                                      |
| ------------- | --------------------------------------- |
| `usd/`        | Isaac Sim/Isaac Lab 直接加载的机器人模型          |
| `mjcf/`       | MuJoCo 使用的 XML/MJCF 模型，`sim2sim.py` 会加载 |
| `urdf/`       | ROS、可视化、机器人描述和其他仿真工具使用                  |
| `meshes/`     | 机器人各连杆的三维几何、碰撞或视觉网格                     |
| `__init__.py` | 将资产配置导出为可导入模块                           |

因此同一台机器人有多种描述：

```text
Isaac Lab → USD
MuJoCo    → MJCF/XML
ROS等     → URDF
外观      → Mesh
```

([GitHub][4])

---

# 五、`legged_lab/envs/base/`：通用强化学习环境

```text
envs/base/
├── base_config.py
├── base_env.py
└── base_env_config.py
```

这三个文件是整个机器人环境的底座。

## 1. `base_config.py`

定义大量“小配置块”。

例如：

```text
RewardCfg
BaseSceneCfg
RobotCfg
NormalizationCfg
CommandsCfg
NoiseCfg
DomainRandCfg
SimCfg
```

分别管理：

| 配置类                | 管理内容                            |
| ------------------ | ------------------------------- |
| `BaseSceneCfg`     | 并行环境数、间距、地形、机器人、传感器             |
| `RobotCfg`         | 历史帧数、动作缩放、脚和终止接触部位              |
| `ObsScalesCfg`     | 角速度、关节位置、关节速度等观测缩放              |
| `NormalizationCfg` | 观测和动作裁剪                         |
| `CommandsCfg`      | 前进、横移、转向命令以及重采样                 |
| `NoiseCfg`         | 观测噪声                            |
| `EventCfg`         | 摩擦、质量、初始状态、推力随机化                |
| `ActionDelayCfg`   | 动作延迟                            |
| `SimCfg`           | `dt=0.005`、`decimation=4` 等物理参数 |

这里是所有具体任务共享的默认参数库。([GitHub][5])

## 2. `base_env_config.py`

把 `base_config.py` 中分散的配置块组合成一个完整环境配置。

概念上类似：

```python
class BaseEnvCfg:
    scene = BaseSceneCfg()
    robot = RobotCfg()
    rewards = RewardCfg()
    commands = CommandsCfg()
    noise = NoiseCfg()
    domain_rand = DomainRandCfg()
    sim = SimCfg()
```

它还负责：

* 根据配置计算 observation/action 维数；
* 检查配置是否合理；
* 将 scene、robot、reward、simulation 等部分组合起来；
* 为 `walk_cfg.py`、`run_cfg.py` 提供父类。

## 3. `base_env.py`

这是环境真正运行的核心。

它实现类似 Gym 的：

```python
reset()
step(actions)
get_observations()
```

每一次 `step()` 大致执行：

```text
收到Actor动作
    ↓
动作延迟与裁剪
    ↓
action × scale + default_joint_pos
    ↓
设置关节目标
    ↓
执行decimation次物理仿真
    ↓
读取机器人状态和接触力
    ↓
计算reward和done
    ↓
重置结束环境
    ↓
构造下一帧Actor/Critic observation
```

它还管理：

* 1024/4096 个并行环境；
* terrain；
* contact sensor；
* reward manager；
* event/domain randomization；
* observation history；
* 噪声；
* episode长度；
* 命令重采样；
* action buffer；
* timeout和终止。

可以把它理解成：

> 强化学习算法和 Isaac Sim 物理世界之间的适配器。

([GitHub][6])

---

# 六、`legged_lab/envs/tienkung/`：天工具体任务

```text
envs/tienkung/
├── datasets/
│   ├── motion_amp_expert/
│   └── motion_visualization/
├── tienkung_env.py
├── walk_cfg.py
├── run_cfg.py
├── walk_with_sensor_cfg.py
└── run_with_sensor_cfg.py
```

## 1. `tienkung_env.py`

它继承或扩展 `BaseEnv`，加入天工特有逻辑：

* 周期步态相位；
* 左右腿 gait phase；
* `sin/cos` 相位观测；
* 天工专用 Actor/Critic observation；
* 52维 AMP observation；
* 双手和双脚相对位置；
* 动作演示的可视化；
* 专家运动帧写入；
* 天工特有 reset 和运动逻辑。

关系是：

```text
BaseEnv
  提供通用环境循环
      ↓
TienKungEnv
  增加天工步态和AMP状态
```

## 2. `walk_cfg.py`

你这次训练最核心的配置文件。

它同时包含两类配置。

### 环境配置

定义：

* 使用哪个天工资产；
* 环境数量；
* 训练地形；
* 速度命令范围；
* gait cycle；
* 默认姿态；
* 观测历史；
* reward项及其权重；
* domain randomization；
* 哪些接触导致终止；
* AMP专家文件路径。

### 强化学习配置

定义：

* Actor/Critic隐藏层；
* 初始动作噪声；
* PPO学习率；
* `gamma`、`lambda`；
* clip参数；
* mini-batch数量；
* epoch数量；
* entropy系数；
* AMP判别器结构；
* 保存频率；
* `AmpOnPolicyRunner`；
* `AMPPPO`。

因此 `walk_cfg.py` 横跨：

```text
环境设计 + 奖励设计 + 强化学习超参数
```

## 3. `run_cfg.py`

跑步任务对应配置。

总体代码结构与 `walk_cfg.py` 相同，但会改变：

* 速度范围；
* gait周期；
* 奖励权重；
* 初始姿态或动作约束；
* 专家动作文件；
* 可能的episode与仿真参数。

它需要搭配单独训练的 `run` checkpoint，不能把 walk 策略仅通过修改 task 就变成跑步策略。

## 4. `walk_with_sensor_cfg.py`

在 walk 任务上增加传感器配置，例如：

* 深度相机；
* tiled camera；
* LiDAR；
* ray casting；
* sensor输出。

普通 `walk` 训练没有使用这个文件。

## 5. `run_with_sensor_cfg.py`

对应跑步任务的传感器版本。

这两个 `_with_sensor_cfg.py` 目前主要用于动作和传感器可视化，也为后续感知控制预留接口。([GitHub][7])

---

# 七、`datasets/`：AMP动作数据

```text
datasets/
├── motion_visualization/
└── motion_amp_expert/
```

## `motion_visualization/`

保存完整动画播放数据，包含：

```text
root position
root rotation
joint position
root linear velocity
root angular velocity
joint velocity
```

用途：

```text
play_amp_animation.py
```

它可以直接把机器人根部和关节设置到参考运动帧，用来检查动作重定向是否正确。

## `motion_amp_expert/`

保存训练AMP判别器所需的精简数据：

```text
joint position
joint velocity
hand/foot end-effector position
```

用途：

```text
motion_loader.py
      ↓
AMP discriminator
```

它不是 Actor 的20维action，也不是力矩，而是专家机器人运动状态。README明确把两类数据分别用于动画播放和AMP训练。([GitHub][1])

---

# 八、`legged_lab/mdp/`：奖励函数

```text
mdp/
├── __init__.py
└── rewards.py
```

## `rewards.py`

定义项目自有的 locomotion reward 函数。

典型内容包括：

* 线速度跟踪；
* 转向速度跟踪；
* 垂直速度惩罚；
* roll/pitch角速度惩罚；
* 姿态稳定；
* 关节偏移；
* 关节速度和加速度；
* torque/能耗；
* action变化率；
* 脚滑；
* 脚部接触力；
* 腾空时间；
* gait相位；
* 左右腿协调；
* 非法身体接触；
* 终止惩罚。

`walk_cfg.py` 负责指定：

```text
使用哪个reward函数
权重是多少
传入什么参数
```

`rewards.py` 负责真正计算数值。

## `__init__.py`

作用是：

* 导出本地 `rewards.py` 函数；
* 统一暴露 Isaac Lab 自带的 MDP/event 函数；
* 使配置文件可以统一使用：

```python
import legged_lab.mdp as mdp
```

所以你在配置里看到的一些函数实际来自 Isaac Lab，并非全部写在这个仓库。([GitHub][8])

---

# 九、`legged_lab/scripts/`：程序入口

```text
scripts/
├── train.py
├── play.py
├── sim2sim.py
├── play_amp_animation.py
└── gmr_data_conversion.py
```

## 1. `train.py`

训练总入口。

执行：

```bash
python legged_lab/scripts/train.py --task=walk
```

主要流程：

```text
解析命令行参数
    ↓
启动Isaac Sim AppLauncher
    ↓
从task_registry获取walk配置
    ↓
创建TienKungEnv
    ↓
创建AmpOnPolicyRunner
    ↓
加载checkpoint（如果resume）
    ↓
runner.learn()
    ↓
保存日志和checkpoint
```

它自己不实现PPO公式，只负责把环境、配置和Runner连接起来。

## 2. `play.py`

加载训练checkpoint并进行Isaac Lab推理。

作用：

* 创建1个或多个环境；
* 关闭训练噪声和随机推力；
* 加载 `model_xxx.pt`；
* 只使用Actor均值；
* 在Isaac Sim中显示机器人；
* 导出：

```text
exported/policy.pt
exported/policy.onnx
```

## 3. `sim2sim.py`

完全独立的 MuJoCo 跨仿真验证程序。

它不再运行：

* PPO；
* Critic；
* AMP判别器；
* Isaac Lab环境；
* 专家文件。

它只使用：

```text
MuJoCo MJCF机器人
+
导出的policy.pt
```

程序负责：

* 读取MuJoCo关节和IMU；
* 构造75维单帧观测；
* 堆叠10帧为750维；
* 调用Actor；
* 转换Isaac/MuJoCo关节顺序；
* 计算目标关节位置；
* 执行MuJoCo；
* 监听键盘速度命令。

## 4. `gmr_data_conversion.py`

将 GMR 重定向得到的 `.pkl` 动作数据转换成：

```text
motion_visualization/*.txt
```

也就是：

```text
人体SMPLX动作
    ↓ GMR
机器人pkl
    ↓ gmr_data_conversion.py
可视化动作txt
```

## 5. `play_amp_animation.py`

读取 `motion_visualization` 数据并在Isaac Sim中逐帧播放。

它有两个用途：

1. 检查动作重定向是否正常；
2. 从Isaac中的机器人状态提取AMP observation，并保存到：

```text
motion_amp_expert/*.txt
```

所以动作数据生成链路是：

```text
GMR pkl
    ↓
gmr_data_conversion.py
    ↓
motion_visualization
    ↓
play_amp_animation.py
    ↓
motion_amp_expert
```

官方 README 的训练、播放、数据生成和Sim2Sim命令都对应这五个脚本。([GitHub][9])

---

# 十、`legged_lab/sensors/`：相机和LiDAR

```text
sensors/
├── camera/
│   ├── __init__.py
│   ├── camera.py
│   ├── camera_cfg.py
│   ├── tiled_camera.py
│   ├── tiled_camera_cfg.py
│   └── camera_cfgs/
│       ├── __init__.py
│       ├── d455_depth_config.py
│       └── luxonis_oak_d_config.py
└── lidar/
    ├── __init__.py
    └── lidat_cfg.py
```

注意仓库中的文件名确实写作：

```text
lidat_cfg.py
```

不是常见的 `lidar_cfg.py`。

## 相机文件

| 文件                        | 作用                            |
| ------------------------- | ----------------------------- |
| `camera.py`               | 自定义或封装Isaac Lab相机的数据获取逻辑      |
| `camera_cfg.py`           | 普通相机的配置类                      |
| `tiled_camera.py`         | 面向多个并行环境的批量/tiled渲染相机         |
| `tiled_camera_cfg.py`     | TiledCamera参数配置               |
| `d455_depth_config.py`    | Intel RealSense D455风格的深度相机预设 |
| `luxonis_oak_d_config.py` | Luxonis OAK-D风格相机预设           |
| 各级`__init__.py`           | 导出相机类和配置                      |

`tiled_camera` 的意义是避免为数千个环境分别创建完全独立的渲染流程，适合并行仿真。

## LiDAR文件

| 文件             | 作用                    |
| -------------- | --------------------- |
| `lidat_cfg.py` | LiDAR/ray sensor的参数配置 |
| `__init__.py`  | 导出LiDAR配置             |

这些传感器主要由 `_with_sensor_cfg.py` 使用，普通 walk 训练并未使用图像或LiDAR输入。([GitHub][10])

---

# 十一、`legged_lab/terrains/`：地形与射线扫描

```text
terrains/
├── __init__.py
├── ray_caster.py
├── ray_caster_cfg.py
└── terrain_generator_cfg.py
```

| 文件                         | 作用                        |
| -------------------------- | ------------------------- |
| `terrain_generator_cfg.py` | 定义平地、粗糙地形、坡面、台阶等地形生成配置    |
| `ray_caster.py`            | 从机器人或传感器向地形发射射线，计算命中位置/高度 |
| `ray_caster_cfg.py`        | 射线数量、方向、范围、分辨率、挂载部位等配置    |
| `__init__.py`              | 导出地形和RayCaster相关类         |

RayCaster可以用于：

* 机器人脚下高度扫描；
* 地形感知；
* LiDAR式检测；
* 生成高度观测。

你当前 walk 配置关闭了 height scan，因此训练主路径没有使用这些高度数据。([GitHub][11])

---

# 十二、`legged_lab/utils/`：业务层工具

```text
utils/
├── __init__.py
├── cli_args.py
├── keyboard.py
├── task_registry.py
└── env_utils/
    ├── __init__.py
    └── scene.py
```

## `cli_args.py`

统一定义命令行参数，例如：

```text
--task
--num_envs
--seed
--device
--resume
--load_run
--checkpoint
--logger
--headless
```

并把命令行参数覆盖到环境配置和RSL-RL配置中。

## `keyboard.py`

Isaac Lab播放阶段的键盘控制器。

它可以把按键转换成：

```text
前进速度
横向速度
转向角速度
```

与 MuJoCo `sim2sim.py` 内部的键盘逻辑用途类似，但运行平台不同。

## `task_registry.py`

任务注册中心，是整个项目非常重要的“路由器”。

它保存类似：

```text
walk
  → TienKungEnv
  → WalkEnvCfg
  → WalkAgentCfg

run
  → TienKungEnv
  → RunEnvCfg
  → RunAgentCfg
```

`train.py` 和 `play.py` 并不会直接硬编码 `walk_cfg.py`，而是根据：

```bash
--task=walk
```

让 registry 找到对应环境和配置。

它还负责：

* 创建环境；
* 创建Runner；
* 查找checkpoint；
* 处理resume；
* 生成log目录。

## `env_utils/scene.py`

对Isaac Lab场景类进行项目级封装或扩展，例如：

* 访问机器人和传感器；
* 创建多个并行scene；
* 场景实体管理；
* 为环境层提供统一接口。

## `__init__.py`

主要负责导出常用工具。([GitHub][12])

---

# 十三、`rsl_rl/`：强化学习算法库

外层结构：

```text
rsl_rl/
├── rsl_rl/
├── config/
├── licenses/dependencies/
├── setup.py
├── pyproject.toml
├── readme.md
├── LICENSE
├── .flake8
└── .pre-commit-config.yaml
```

这是一个基于 RSL-RL 修改的独立Python包，加入了AMP功能。它具有自己的安装和代码检查配置。([GitHub][13])

真正的Python包在：

```text
rsl_rl/rsl_rl/
```

结构为：

```text
rsl_rl/rsl_rl/
├── algorithms/
├── env/
├── modules/
├── networks/
├── runners/
├── storage/
├── utils/
└── __init__.py
```

([GitHub][14])

---

# 十四、`rsl_rl/rsl_rl/algorithms/`：优化算法

```text
algorithms/
├── __init__.py
├── ppo.py
├── amp_ppo.py
└── distillation.py
```

## `ppo.py`

通用PPO算法。

负责：

* Actor/Critic rollout；
* ratio；
* clipped surrogate loss；
* value loss；
* entropy；
* KL自适应学习率；
* gradient clipping；
* Adam更新。

它不包含AMP判别器。

## `amp_ppo.py`

你当前训练实际使用的算法：

```python
class_name = "AMPPPO"
```

它在普通PPO上加入：

* AMP策略状态存储；
* AMP replay buffer；
* 专家样本；
* discriminator预测；
* expert目标 `+1`；
* policy目标 `-1`；
* MSE判别器loss；
* gradient penalty；
* AMP normalizer；
* AMP loss和PPO loss联合更新。

你之前研究的：

```text
ratio
surrogate loss
value loss
entropy
AMP loss
gradient penalty
optimizer.step
```

都在这个文件中。

## `distillation.py`

教师—学生蒸馏算法。

作用：

```text
Teacher策略
    ↓ 产生目标动作
Student策略
    ↓ 学习模仿
```

用于将一个较复杂或具有特权观测的策略蒸馏为另一个策略。

你当前 walk 训练没有使用。

## `__init__.py`

导出 `PPO`、`AMPPPO`、`Distillation` 等算法类。([GitHub][15])

---

# 十五、`rsl_rl/rsl_rl/env/`：环境接口

```text
env/
├── __init__.py
└── vec_env.py
```

## `vec_env.py`

定义强化学习Runner要求环境实现的接口。

例如：

```python
num_envs
num_actions
step()
reset()
get_observations()
episode_length_buf
```

它不负责Isaac物理仿真，而是规定：

> 任何环境只要实现这个接口，就可以被RSL-RL Runner训练。

TienKung的 `BaseEnv/TienKungEnv` 提供这些实际功能。

## `__init__.py`

导出 `VecEnv`。([GitHub][16])

---

# 十六、`rsl_rl/rsl_rl/modules/`：神经网络模型

```text
modules/
├── __init__.py
├── actor_critic.py
├── actor_critic_recurrent.py
├── discriminator.py
├── normalizer.py
├── rnd.py
├── student_teacher.py
└── student_teacher_recurrent.py
```

## `actor_critic.py`

你当前策略的核心网络。

包含：

```text
Actor MLP
Critic MLP
可学习动作std
Normal分布
动作采样
动作log probability
entropy
Value估计
推理均值输出
```

当前训练：

```text
750维 Actor observation
    ↓
Actor [512,256,128]
    ↓
20维 mean
```

Critic则输出一个Value。

训练调用：

```python
act()
```

从高斯分布采样。

部署调用：

```python
act_inference()
```

直接返回均值。

## `actor_critic_recurrent.py`

RNN/GRU/LSTM版本Actor-Critic。

适合需要显式隐藏状态的任务。训练时不能像普通MLP那样完全随机打乱时间步。

你当前使用的是普通 `ActorCritic`，没有使用这个文件。

## `discriminator.py`

AMP判别器神经网络。

负责：

* 接收拼接后的两个AMP state；
* 输出一个判别值；
* 计算style reward；
* 对专家输入计算gradient penalty。

## `normalizer.py`

维护数据的运行均值和方差：

[
\hat x=\frac{x-\mu}{\sigma+\epsilon}
]

主要用于AMP observation归一化，使不同量纲的：

```text
关节角
关节速度
手脚位置
```

处于更稳定的数值范围。

## `rnd.py`

Random Network Distillation。

用于产生好奇心或内在奖励：

```text
固定随机网络target
+
可训练predictor
+
预测误差作为新奇度
```

当前配置没有启用。

## `student_teacher.py`

前馈Teacher–Student网络结构，服务于 `distillation.py`。

## `student_teacher_recurrent.py`

带RNN隐藏状态的Teacher–Student结构。

## `__init__.py`

统一导出这些模块。([GitHub][17])

---

# 十七、`rsl_rl/rsl_rl/networks/`

```text
networks/
├── __init__.py
└── memory.py
```

## `memory.py`

封装循环网络记忆模块，例如GRU或LSTM。

处理：

* hidden state；
* episode结束后的隐藏状态清零；
* mask；
* padded trajectory；
* 训练和推理的不同隐藏状态处理。

主要被：

```text
actor_critic_recurrent.py
student_teacher_recurrent.py
```

使用。

当前前馈 walk 策略不会使用。([GitHub][18])

---

# 十八、`rsl_rl/rsl_rl/runners/`：训练总调度器

```text
runners/
├── __init__.py
├── on_policy_runner.py
└── amp_on_policy_runner.py
```

Runner位于环境和算法之间：

```text
Environment
    ↕
Runner
    ↕
Algorithm
```

## `on_policy_runner.py`

普通PPO/蒸馏训练调度器。

负责：

* 创建Actor-Critic；
* 创建PPO；
* 收集rollout；
* 调用algorithm update；
* 输出日志；
* 保存和加载checkpoint；
* TensorBoard/WandB/Neptune；
* 导出推理策略。

## `amp_on_policy_runner.py`

你当前使用：

```python
runner_class_name = "AmpOnPolicyRunner"
```

除了普通Runner功能，它还负责：

* 创建AMP MotionLoader；
* 预加载专家transition；
* 创建AMP discriminator；
* 获取策略当前和下一AMP state；
* 用判别器计算style reward；
* 混合task reward和style reward；
* 创建AMPPPO；
* 记录AMP相关日志。

训练中每轮执行的高层循环就在这里：

```text
24步环境采集
    ↓
计算最后Value和GAE
    ↓
AMPPPO.update()
    ↓
日志与保存
```

## `__init__.py`

导出两种Runner。([GitHub][19])

---

# 十九、`rsl_rl/rsl_rl/storage/`：训练数据存储

```text
storage/
├── __init__.py
├── rollout_storage.py
└── replay_buffer.py
```

## `rollout_storage.py`

保存PPO当前rollout：

```text
observations
critic observations
actions
rewards
dones
values
returns
advantages
old log probabilities
old mean/std
hidden states
```

当前真实形状类似：

```text
[24, 1024, ...]
```

它还负责：

* 倒序计算GAE；
* Return；
* Advantage归一化；
* `[时间,环境]` 展平；
* 随机mini-batch；
* recurrent trajectory padding。

你之前问的“展平、随机打乱、5个epoch”，核心数据生成就在这里。

## `replay_buffer.py`

AMP策略状态转移的循环缓冲区。

存储：

[
(s_t^{policy},s_{t+1}^{policy})
]

判别器训练时，从这里抽取策略样本，再从专家MotionLoader抽取专家样本。

PPO本身是on-policy，所以PPO transition使用 `RolloutStorage`；AMP判别器允许从ReplayBuffer中重复抽取近期策略运动。

## `__init__.py`

导出Storage类。([GitHub][20])

---

# 二十、`rsl_rl/rsl_rl/utils/`：算法工具

```text
utils/
├── __init__.py
├── motion_loader.py
├── motion_loader_for_display.py
├── utils.py
├── wandb_utils.py
└── neptune_utils.py
```

## `motion_loader.py`

加载 `motion_amp_expert/*.txt`。

负责：

* 读取专家动作；
* 解析FrameDuration；
* 计算动作长度；
* 随机选择动作文件；
* 随机选择时间点；
* 在相邻帧之间插值；
* 获取 (s_t) 和 (s_{t+\Delta t})；
* 预加载20万条专家transition；
* 提供AMP专家mini-batch。

## `motion_loader_for_display.py`

加载 `motion_visualization/*.txt`。

与前一个不同，它保留：

* root position；
* root rotation；
* root velocity；
* joint data。

主要供 `play_amp_animation.py` 逐帧设置机器人姿态。

## `utils.py`

通用算法工具，例如：

* 激活函数解析；
* trajectory拆分和padding；
* 策略导出JIT；
* 策略导出ONNX；
* checkpoint辅助；
* 配置字典处理。

## `wandb_utils.py`

Weights & Biases日志上传和配置记录。

## `neptune_utils.py`

Neptune日志集成。

你使用：

```bash
--logger=tensorboard
```

时，这两个外部日志文件通常不会参与主流程。

## `__init__.py`

统一导出常用工具。([GitHub][21])

---

# 二十一、当前walk训练真正经过哪些文件

虽然仓库文件很多，你这次训练的主调用链其实集中在以下文件：

```text
legged_lab/scripts/train.py
          │
          ▼
legged_lab/utils/cli_args.py
legged_lab/utils/task_registry.py
          │
          ▼
legged_lab/envs/tienkung/walk_cfg.py
legged_lab/envs/tienkung/tienkung_env.py
          │
          ▼
legged_lab/envs/base/base_env_config.py
legged_lab/envs/base/base_config.py
legged_lab/envs/base/base_env.py
          │
          ├── legged_lab/assets/tienkung2_lite/tienkung.py
          ├── legged_lab/mdp/rewards.py
          └── motion_amp_expert/walk.txt
          │
          ▼
rsl_rl/runners/amp_on_policy_runner.py
          │
          ├── rsl_rl/modules/actor_critic.py
          ├── rsl_rl/modules/discriminator.py
          ├── rsl_rl/modules/normalizer.py
          ├── rsl_rl/utils/motion_loader.py
          └── rsl_rl/algorithms/amp_ppo.py
                         │
                         ├── rollout_storage.py
                         └── replay_buffer.py
```

这条链就是：

[
\boxed{
\text{任务配置}
\rightarrow
\text{物理环境}
\rightarrow
\text{数据采集}
\rightarrow
\text{PPO/AMP计算}
\rightarrow
\text{参数更新}
}
]

---

# 二十二、Play和Sim2Sim经过哪些文件

## Isaac Lab Play

```text
play.py
  ↓
task_registry.py
  ↓
walk_cfg.py
  ↓
TienKungEnv / BaseEnv
  ↓
加载model_49999.pt
  ↓
ActorCritic.act_inference()
  ↓
导出policy.pt / policy.onnx
```

仍然使用原来的Isaac环境代码。

## MuJoCo Sim2Sim

```text
sim2sim.py
  ├── exported/policy.pt
  └── assets/tienkung2_lite/mjcf/*.xml
```

它不经过：

```text
walk_cfg.py的大部分训练配置
BaseEnv
AMPPPO
Critic
Discriminator
RolloutStorage
MotionLoader
专家文件
```

所需的部署参数，例如：

```text
75维观测结构
10帧历史
默认关节位置
动作缩放
关节映射
gait周期
```

都在 `sim2sim.py` 中重新实现了一遍。

---

# 二十三、哪些文件与你当前任务无关

你现在普通walk训练没有直接使用：

```text
run_cfg.py
run_with_sensor_cfg.py
walk_with_sensor_cfg.py
camera/*
lidar/*
terrains/ray_caster*
actor_critic_recurrent.py
student_teacher*.py
distillation.py
rnd.py
networks/memory.py
on_policy_runner.py
ppo.py
wandb_utils.py
neptune_utils.py
```

它们不是没用，而是服务于：

```text
跑步
视觉和LiDAR
复杂地形
RNN策略
策略蒸馏
好奇心奖励
普通非AMP PPO
其他日志平台
```

---

# 二十四、推荐的最终代码阅读顺序

根据你已经掌握的内容，最合理的顺序是：

```text
1. train.py
2. task_registry.py
3. walk_cfg.py
4. base_config.py
5. base_env_config.py
6. tienkung_env.py
7. base_env.py
8. rewards.py
9. tienkung.py
10. amp_on_policy_runner.py
11. actor_critic.py
12. rollout_storage.py
13. amp_ppo.py
14. discriminator.py
15. motion_loader.py
16. play.py
17. sim2sim.py
```

这样阅读时能始终沿着一条完整数据流：

```text
命令行参数
→ 创建任务
→ 创建机器人和环境
→ 构造观测
→ Actor输出动作
→ 物理仿真
→ reward
→ storage
→ GAE
→ PPO/AMP loss
→ backward
→ checkpoint
→ 导出
→ MuJoCo部署
```

整个仓库最核心的划分可以记成：

[
\boxed{
\texttt{legged_lab}
===================

\text{机器人、环境、任务和仿真}
}
]

[
\boxed{
\texttt{rsl_rl}
===============

\text{神经网络、PPO、AMP和训练数据}
}
]

[
\boxed{
\texttt{scripts}
================

\text{把上述模块组织成可运行程序}
}
]

[1]: https://github.com/Open-X-Humanoid/TienKung-Lab?utm_source=chatgpt.com "GitHub - Open-X-Humanoid/TienKung-Lab: Tien Kung-Lab: Direct IsaacLab Workflow for Legged Robots · GitHub"
[2]: https://gitee.com/open_x_humanoid/TienKung-Lab/blob/main/setup.py "setup.py · Open X-Humanoid/TienKung-Lab - Gitee"
[3]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab "TienKung-Lab/legged_lab at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[4]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/assets/tienkung2_lite "TienKung-Lab/legged_lab/assets/tienkung2_lite at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[5]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/main/legged_lab/envs/base/base_config.py "TienKung-Lab/legged_lab/envs/base/base_config.py at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[6]: https://github.com/Open-X-Humanoid/TienKung-Lab/blob/main/legged_lab/envs/base/base_env.py "TienKung-Lab/legged_lab/envs/base/base_env.py at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[7]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/envs/tienkung "TienKung-Lab/legged_lab/envs/tienkung at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[8]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/mdp "TienKung-Lab/legged_lab/mdp at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[9]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/scripts "TienKung-Lab/legged_lab/scripts at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[10]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/sensors/camera "TienKung-Lab/legged_lab/sensors/camera at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[11]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/terrains "TienKung-Lab/legged_lab/terrains at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[12]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/legged_lab/utils "TienKung-Lab/legged_lab/utils at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[13]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl "TienKung-Lab/rsl_rl at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[14]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl "TienKung-Lab/rsl_rl/rsl_rl at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[15]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/algorithms "TienKung-Lab/rsl_rl/rsl_rl/algorithms at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[16]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/env "TienKung-Lab/rsl_rl/rsl_rl/env at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[17]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/modules "TienKung-Lab/rsl_rl/rsl_rl/modules at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[18]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/networks "TienKung-Lab/rsl_rl/rsl_rl/networks at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[19]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/runners "TienKung-Lab/rsl_rl/rsl_rl/runners at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[20]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/storage "TienKung-Lab/rsl_rl/rsl_rl/storage at main · Open-X-Humanoid/TienKung-Lab · GitHub"
[21]: https://github.com/Open-X-Humanoid/TienKung-Lab/tree/main/rsl_rl/rsl_rl/utils "TienKung-Lab/rsl_rl/rsl_rl/utils at main · Open-X-Humanoid/TienKung-Lab · GitHub"
