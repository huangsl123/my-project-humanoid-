#!/usr/bin/env python3
"""Create the seventh-week internship report from the sixth-week DOCX template."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


TEMPLATE = Path("/home/eai/Downloads/2023010366_6_黄诗琅.docx")
OUTPUT = Path("/home/eai/Downloads/2023010366_7_黄诗琅.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

TITLE = "0817—0823 实习周报"
INFO = (
    "姓名：黄诗琅",
    "实习单位：北京人形机器人创新中心有限公司",
    "实习时间：2026 年 8 月 17 日—2026 年 8 月 23 日",
)
BODY = (
    "本周为实习第七周，主要工作是在前期单动作训练和长时程动作实验的基础上，进一步探索复杂、持续时间更长且包含多种运动技能的机器人动作数据，并完善从数据生成、物理一致性检查、强化学习训练到严格回放评估的自动化流程。工作继续围绕 xMimic、Isaac Lab、RSL-RL、DexEVT2 机器人模型以及后续 xMIGCS 和 MuJoCo sim2sim 部署展开。本周尤其强调实验可验证性：不以训练轮数完成、训练窗口中的 timeout 比例或 ONNX 文件能够生成作为最终成功依据，而是要求策略在名义物理参数下完整执行全部参考帧，再比较根节点、刚体、关节和朝向误差，并为后续随机扰动及跨仿真器验证保留清晰的模型来源和证据文件。",
    "新动作筛选方面，我继续检查 LAFAN1 中持续时间较长的多动作和格斗运动数据，并对候选片段的时长、根节点位移、关节速度、落地姿态以及机器人平台可用范围进行量化分析。multiple_actions1_subject1 的约 60 秒候选虽然内容丰富，但多条处理路线均在约 20.3—20.4 秒附近出现重复的物理失败，表明问题来自原始动作中的高风险地面或过渡姿态，而不是简单增加训练次数即可解决。因此我没有继续无依据追加训练，也没有把短暂正常播放误判为可用数据。随后改变思路，使用此前已经完成严格回放的 FightAndSports、Jumps、PushAndStumble 和 Fight 四类机器人动作作为基础素材，创建一条新的复杂组合动作，以兼顾拳击、跳跃、受推恢复、转身和连续格斗等内容。",
    "复合动作生成方面，我编写了可复现的数据编排脚本。脚本首先去除各源动作末尾多余的静止保持帧，再将下一段动作的根节点位置与航向对齐到上一段结束位置，同时保留源动作在水平面上的真实位移，不增加把机器人强制拉回平台中心的控制项。相邻动作之间插入 20 帧、约 0.2 秒的五次平滑过渡，对根节点位置、四元数和关节角分别进行连续插值。最终组合轨迹约 73.19 秒，覆盖约 2.40 m×2.20 m 的水平范围，包含四个主要动作片段和三个短过渡段。完成 qpos 生成后，再通过 Isaac Lab 正向运动学重新计算 39 个刚体的位置、姿态及速度，并补充具名 body_names，使数据能够适配当前 Isaac Sim 版本中的刚体映射。",
    "数据验证过程中发现并纠正了一个关键问题。初版脚本把训练 NPZ 中按照 Isaac Articulation 顺序排列的 23 维 joint_pos，直接写入了转换器所要求的 qpos 关节顺序；两套顺序并不相同，导致往返转换后第一段动作的关节位置最大偏差达到约 3.08 rad。虽然骨架预览仍呈现为人形，这类错误仅靠肉眼很难识别。我根据 ONNX 元数据和 qpos 转换器的显式关节列表建立完整映射，立即停止了错误数据上的 4096 环境训练，并将错误数据、训练日志和诊断结果隔离保存，防止后续误选。修正后重新生成数据，四个源片段的逐帧关节最大差异下降到约 1.6×10⁻⁴ rad，99% 的关节差异低于 2.6×10⁻⁵ rad，99% 的刚体相对位置差异低于 0.26 mm，同时通过数值有限性、四元数归一化、边界速度和 64 环境训练入口检查。",
    "策略训练方面，我先用已训练的 FightAndSports 策略测试修正版复合数据。该策略能够执行到约 22.99 秒，顺利通过第一段动作和第一个拼接点，比从零策略约 5.15 秒的结果明显更好，因此选择其 model_48000 作为迁移初始化。随后在 4096 个并行环境中，以 10 秒随机相位方式训练 5000 次迭代，使 73 秒轨迹的不同阶段都能得到采样。训练完成后评估 model_50500 至 model_52999 的六个候选，其中前五个均完成全部 7319 帧，而最后的 model_52999 在约 4.67 秒发生回退并倒地。综合完整性和误差评分后选择 model_52000，而不是机械采用最后一个 checkpoint。该模型虽然能够连续完成动作，但身体位置、关节姿态和朝向 P95 仍未全部达到严格精度门槛，因此继续进行了整段 73.18 秒的严格精度续训。",
    "严格精度续训从 model_52000 开始追加 5000 次迭代，训练记录连续完整，后期 model_54500、55000、55500、56000、56500 和 56999 六个 checkpoint 均能够完整执行 73.18 秒且没有终止。综合排名选择 model_56999 后，朝向误差 P95 已降至约 0.081 rad，满足 0.150 rad 的门槛；身体位置 P95 约为 0.102 m，关节位置 P95 约为 0.936，距离 0.080 m 和 0.900 的目标仍有小幅差距。分段分析表明三个拼接区和跳跃段表现良好，主要误差集中在 PushAndStumble 以及前后两个格斗片段。因此目前已从 model_56999 启动 4096 环境、10 秒严格随机相位的针对性续训，为难段提供更均衡的训练样本。下一步将在保留现有完整模型作为回退的前提下重新比较候选，只有名义和随机扰动完整回放均满足门槛后，才进行最终视频、ONNX 一致性验证和 MuJoCo sim2sim 部署。本周最大的收获是进一步认识到复杂动作项目中，数据关节顺序、完整轨迹证据和检查点选择的重要性往往高于单纯增加训练轮数。",
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
