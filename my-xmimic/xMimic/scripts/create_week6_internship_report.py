#!/usr/bin/env python3
"""Create the sixth-week internship report from the fifth-week DOCX template."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


TEMPLATE = Path("/home/eai/Downloads/2023010366_5_黄诗琅.docx")
OUTPUT = Path("/home/eai/Downloads/2023010366_6_黄诗琅.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

TITLE = "0810—0816 实习周报"
INFO = (
    "姓名：黄诗琅",
    "实习单位：北京人形机器人创新中心有限公司",
    "实习时间：2026 年 8 月 10 日—2026 年 8 月 16 日",
)
BODY = (
    "本周为实习第六周，工作重点是在第五周已建立的单动作训练与部署流程基础上，进一步完成多类动作的批量化训练、严格评估和 sim2sim 验证，并尝试处理持续时间超过两分钟的高复杂度动作。我围绕 LAFAN1 动作数据、xGMR 动作重映射、Isaac Lab 强化学习训练、ONNX 策略导出、xMIGCS 控制以及 xSIM_MUJOCO 物理仿真开展工作。与此前十几秒或单一舞蹈动作相比，本周更关注长时程动作中的累计漂移、局部高难度姿态、检查点选择和数据质量问题，并将“能够播放动作”与“完整、稳定、可部署地执行动作”区分开来，逐步完善自动化证据链。",
    "多动作训练方面，我继续推进 FallAndGetUp、FightAndSports、Jumps、Ground、PushAndStumble 和 Fight 六类 LAFAN1 动作的全流程处理。各动作均先经过源 BVH 片段筛选和平台位移检查，再由 xGMR 重映射为 DexEVT2 机器人关节数据，随后转换为 100 Hz、23 个策略关节和 39 个具名刚体的 NPZ 文件。正式训练采用 4096 个并行环境，并根据动作难度组合长时程状态反馈训练与精度训练。训练结束后，不直接使用最后一个 checkpoint，而是对后期多个候选模型进行完整轨迹回放，在名义 seed 42 和随机动力学种子下比较动作完成率、骨盆位置、刚体位置、关节姿态、平台边界和终止原因，以实际回放结果选择模型。六类动作随后均完成了候选筛选、ONNX 导出和对应证据整理。",
    "策略部署与仿真方面，我为上述动作分别生成 xMIGCS 的 BeyondMimic 配置和 ONNX 模型安装项，并逐项完成 MuJoCo sim2sim 操作验证。部署时继续使用 130 维状态反馈观测、23 维动作输出和 29 个 MuJoCo 电机映射，确保 YAML 中的 num_obs、num_actions、locked_joint_map 与 ONNX 网络输入保持一致。我还整理了六个动作各自的启动命令和策略选择方式，检查 ROS 2 domain、Python 路径、动态库路径、仿真器与控制节点的启动顺序，并确认 ESC 只负责释放机器人约束，真正的运动由策略控制节点持续输出。通过分动作验证，我进一步熟悉了从训练 checkpoint 到 ONNX、部署配置、ROS 通信和 MuJoCo 闭环控制之间的完整数据流。",
    "长动作筛选方面，为寻找比此前动作更复杂且持续时间更长的新任务，我对 LAFAN1 中多个 BVH 文件进行了定量扫描，综合比较片段时长、根节点平面位移、姿态变化速度和平台可用范围。最终选取 fightAndSports1_subject4.bvh 的第 2760—7260 帧，原始长度约 150 秒，包含连续格斗、快速转身、下蹲、重心切换和多种运动姿态。该片段的根节点单轴最大相对位移约 3.68 m，能够保留原动作的空间运动，而不通过强制固定机器人位置来规避平台问题。动作经重映射和 30 Hz 到 100 Hz 插值后，最初生成 15787 帧、约 157.86 秒的训练数据，并通过 ZIP 完整性、数值有限性、四元数归一化、关节维度、身体名称和平台范围等检查。",
    "长动作训练与评估方面，我设计了由 20 秒、60 秒、100 秒到完整 157.86 秒的课程学习方案，在 4096 环境下依次训练 20000、15000、15000 和 10000 次迭代，总计 60000 次。训练完成后评估了 model_57500 至 model_59996 的六个后期 checkpoint，综合评分选择 model_58000，并成功验证 checkpoint 的 130 维 Actor 输入、346 维 Critic 输入、23 维输出以及 ONNX 权重一致性。但严格的完整轨迹测试显示，所有名义和随机工况都在约 3.44—3.59 秒因骨盆或末端刚体位置误差而终止，动作完成率仅约 2.24%。因此我没有把“训练轮数完成”和“ONNX 能导出”误判为策略通过，也没有部署该不合格模型，而是保留失败视频、指标曲线和终止证据继续定位问题。",
    "问题排查发现，失败时刻与动作开头过渡结束的时间高度一致。进一步检查参考数据后发现，首次转换使用了 frame_range 1 -1，而项目 README 示例要求从第 10 帧开始；xGMR 最初若干帧的逆运动学尚未稳定，在约 3.45 秒处形成单帧约 0.99 rad 的关节变化、约 0.50 m 的刚体位移和接近 99 rad/s 的关节速度峰值。按 README 改为 frame_range 10 -1 后，我重新生成了 15757 帧、157.56 秒的 V2 数据，开头刚体单帧位移峰值下降至约 0.037 m，并完成 64 环境冒烟测试。目前已从 model_58000 暖启动新的恢复训练，按照 20 秒、60 秒、120 秒和完整时长逐级扩展；后续仍将以完整轨迹和多随机种子结果作为准入条件，通过后才进行视频、ONNX 和 MuJoCo 部署。本周的主要收获是建立了对长动作数据异常、训练指标和真实部署有效性进行分层判断的方法。",
)


def update_document(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    paragraphs = root.findall(f".//{W}body/{W}p")
    if len(paragraphs) != 8:
        raise ValueError(f"Expected 8 template paragraphs, found {len(paragraphs)}")

    title_nodes = paragraphs[0].findall(f".//{W}t")
    if len(title_nodes) != 1:
        raise ValueError("Unexpected title structure")
    title_nodes[0].text = TITLE

    info_nodes = paragraphs[1].findall(f".//{W}t")
    if len(info_nodes) != 3:
        raise ValueError("Unexpected information block structure")
    for node, value in zip(info_nodes, INFO, strict=True):
        node.text = value

    for paragraph, value in zip(paragraphs[2:], BODY, strict=True):
        nodes = paragraph.findall(f".//{W}t")
        if len(nodes) != 1:
            raise ValueError("Unexpected body paragraph structure")
        nodes[0].text = value

    ET.register_namespace("w", "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(TEMPLATE, "r") as source, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as target:
            for item in source.infolist():
                payload = source.read(item.filename)
                if item.filename == "word/document.xml":
                    payload = update_document(payload)
                target.writestr(item, payload)
        with zipfile.ZipFile(temporary) as result:
            if result.testzip() is not None:
                raise ValueError("Generated DOCX failed ZIP CRC validation")
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
