# -*- coding: utf-8 -*-
"""
build_ppt.py
生成《sEMG 手势识别系统：数据采集 · 模型训练 · 设备控制》演示文稿。

设计约束：
  - 仅使用文本框、基础形状、表格，不使用动画/切换/SmartArt/外部图片，
    以保证 Microsoft Office 与 WPS 均可正常放映。
  - 字体统一为「微软雅黑」（WPS/Office 均自带），同时写入 latin 与 eastAsian 字体属性。

运行（Windows，emgControl 环境）：
  C:\\Users\\lixin\\anaconda3\\envs\\emgControl\\python.exe build_ppt.py
"""

import os

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ===================== 视觉规范 =====================
FONT = "微软雅黑"

INK = RGBColor(0x1B, 0x24, 0x30)
MUTED = RGBColor(0x5B, 0x6B, 0x7C)
ACCENT = RGBColor(0x0F, 0x76, 0x6E)      # 主色：青绿
ACCENT2 = RGBColor(0x25, 0x63, 0xEB)     # 辅色：蓝
ACCENT3 = RGBColor(0xB4, 0x53, 0x09)     # 强调：琥珀
LIGHT = RGBColor(0xE8, 0xF5, 0xF2)       # 浅青底
LIGHT2 = RGBColor(0xEE, 0xF3, 0xFD)      # 浅蓝底
LIGHT3 = RGBColor(0xFD, 0xF3, 0xE6)      # 浅橙底
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xD3, 0xDD, 0xE6)
DARK = RGBColor(0x0F, 0x1B, 0x24)

SLIDE_W = 13.333
SLIDE_H = 7.5
MARGIN = 0.72
CONTENT_W = SLIDE_W - MARGIN * 2

prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)
BLANK = prs.slide_layouts[6]


# ===================== 基础工具 =====================
def new_slide(bg=WHITE):
    slide = prs.slides.add_slide(BLANK)
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide


def _apply_font(run, size, bold, color, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rpr = run._r.get_or_add_rPr()
    ea = rpr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rpr, qn("a:ea"))
    ea.set("typeface", font)


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tf


def add_line(tf, text, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
             space_before=0, line_spacing=1.25, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = line_spacing
    p.space_before = Pt(space_before)
    run = p.add_run()
    run.text = text
    _apply_font(run, size, bold, color)
    return p


def rect(slide, x, y, w, h, fill, line=None, radius=0.08, line_w=1.0):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    shape.shadow.inherit = False
    shape.text_frame.word_wrap = True
    shape.text_frame.margin_left = Inches(0.08)
    shape.text_frame.margin_right = Inches(0.08)
    shape.text_frame.margin_top = Inches(0.04)
    shape.text_frame.margin_bottom = Inches(0.04)
    return shape


def shape_text(shape, lines, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = shape.text_frame
    tf.vertical_anchor = anchor
    for i, (text, size, bold, color) in enumerate(lines):
        add_line(tf, text, size=size, bold=bold, color=color, align=align, first=(i == 0))


def arrow(slide, x, y, w=0.26, h=0.22):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BORDER
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def slide_header(slide, tag, title, subtitle=""):
    tf = textbox(slide, MARGIN, 0.46, CONTENT_W, 0.3)
    add_line(tf, tag, size=11.5, bold=True, color=ACCENT, first=True)

    tf = textbox(slide, MARGIN, 0.78, CONTENT_W, 0.6)
    add_line(tf, title, size=27, bold=True, color=INK, first=True)

    if subtitle:
        tf = textbox(slide, MARGIN, 1.36, CONTENT_W, 0.3)
        add_line(tf, subtitle, size=12.5, bold=False, color=MUTED, first=True)

    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(1.66),
                                  Inches(CONTENT_W), Inches(0.015))
    rule.fill.solid()
    rule.fill.fore_color.rgb = BORDER
    rule.line.fill.background()
    rule.shadow.inherit = False


def footer(slide, page):
    tf = textbox(slide, MARGIN, SLIDE_H - 0.46, 6.0, 0.25)
    add_line(tf, "sEMG 手势识别系统 · 数据采集 / 训练 / 控制", size=8.5, color=MUTED, first=True)
    tf = textbox(slide, SLIDE_W - MARGIN - 0.8, SLIDE_H - 0.46, 0.8, 0.25)
    add_line(tf, f"{page:02d}", size=9, bold=True, color=MUTED, align=PP_ALIGN.RIGHT, first=True)


# ===================== 01 封面 =====================
s = new_slide(DARK)
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(1.85), Inches(1.5), Inches(0.06))
bar.fill.solid()
bar.fill.fore_color.rgb = ACCENT
bar.line.fill.background()
bar.shadow.inherit = False

tf = textbox(s, MARGIN, 2.15, 11.0, 1.6)
add_line(tf, "sEMG 手势识别系统", size=44, bold=True, color=WHITE, first=True)
add_line(tf, "数据采集 · 模型训练 · 设备控制", size=24, bold=False, color=RGBColor(0x8F, 0xE3, 0xD8), space_before=8)

tf = textbox(s, MARGIN, 4.15, 8.6, 1.3)
add_line(tf, "基于 8 通道表面肌电臂环的端到端方案", size=15, color=RGBColor(0xC7, 0xD2, 0xDD), first=True)
add_line(tf, "Android 采集端  |  本地与云端双训练路径  |  多类被控设备", size=16, bold=True,
         color=WHITE, space_before=8)

tf = textbox(s, MARGIN, 6.25, 10.0, 0.5)
add_line(tf, "技术汇报 · 采集 / 训练 / 控制全链路", size=11, color=RGBColor(0x7E, 0x8C, 0x9A), first=True)


# ===================== 02 系统总览 =====================
s = new_slide()
slide_header(s, "00 · 总览", "系统架构：一条从肌电信号到设备动作的链路")

steps = [
    ("sEMG 臂环", "8 通道 · 500 Hz"),
    ("蓝牙链路", "BLE 低功耗\n帧头 + CRC"),
    ("Android App", "解析 / 录制\n实时推理"),
    ("手势命令", "标签 → 指令\n置信度过滤"),
    ("UDP 中转", "树莓派\n热点 8888"),
    ("被控设备", "小车 / Go2 /\nGo2-W / 侦察球"),
]
bw, bh, gap = 1.62, 1.12, 0.42
x = MARGIN
for i, (head, sub) in enumerate(steps):
    box = rect(s, x, 2.0, bw, bh, LIGHT if i % 2 == 0 else LIGHT2, BORDER)
    shape_text(box, [(head, 12, True, INK), (sub, 9, False, MUTED)])
    if i < len(steps) - 1:
        arrow(s, x + bw + 0.08, 2.44, 0.26, 0.24)
    x += bw + gap

# 三条并行路径
cards = [
    ("数据采集", "CSV 原始波形 + 标签 / 手别 / 用户元数据\n历史数据管理可回看与导出", LIGHT, ACCENT),
    ("模型训练", "本地：质心 / SVM，端侧秒级\n云端：CNN-Transformer，两阶段 + 零样本", LIGHT2, ACCENT2),
    ("设备控制", "UDP 指令持续下发 + 中立锁存\n超时自动停止，支持 5 类设备", LIGHT3, ACCENT3),
]
cw = (CONTENT_W - 0.36 * 2) / 3
for i, (head, body, fill, color) in enumerate(cards):
    box = rect(s, MARGIN + i * (cw + 0.36), 3.95, cw, 1.55, fill, BORDER)
    shape_text(box, [(head, 14, True, color), (body, 10, False, INK)], align=PP_ALIGN.LEFT,
               anchor=MSO_ANCHOR.TOP)

tf = textbox(s, MARGIN, 5.85, CONTENT_W, 0.8)
add_line(tf, "关键设计：训练与推理共用同一套预处理（带通滤波 + 100 点窗口 + 均值/方差归一化），"
             "保证端侧与云端输入分布一致。", size=11.5, color=MUTED, first=True)
footer(s, 2)


# ===================== 03 数据采集 · 硬件 =====================
s = new_slide()
slide_header(s, "01 · 数据采集", "硬件与数据链路：8 通道、500 Hz、带 CRC 的帧协议",
             "采样率与通道数决定了后续所有窗口与模型输入的形状")

items = [
    ("信号源", "8 通道表面肌电（sEMG）\nSTM32F411 采集板\n500 Hz 采样率"),
    ("传输", "BLE 低功耗（主链路）\nApp 内可切换备选方式\n断连自动重连提示"),
    ("帧结构", "帧头 + 样本数 + N×8×int16 + CRC\n解析后校验 CRC\n统计丢帧与错误数"),
    ("实时观察", "采样率 / 帧数 / 丢帧 / CRC\n8 通道原始波形\n末帧通道数值"),
]
cw = (CONTENT_W - 0.3 * 3) / 4
for i, (head, body) in enumerate(items):
    box = rect(s, MARGIN + i * (cw + 0.3), 2.0, cw, 1.85, WHITE, BORDER)
    shape_text(box, [(head, 13.5, True, ACCENT), (body, 10, False, INK)],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(4.15),
                          Inches(CONTENT_W), Inches(0.012))
band.fill.solid()
band.fill.fore_color.rgb = BORDER
band.line.fill.background()
band.shadow.inherit = False

tf = textbox(s, MARGIN, 4.38, CONTENT_W, 2.0)
add_line(tf, "为什么这些参数重要", size=14, bold=True, color=INK, first=True)
add_line(tf, "• 500 Hz：手势动作的主要能量集中在 20–250 Hz，500 Hz 采样满足奈奎斯特要求，同时保证 200 ms 窗口内足够样本。",
         size=12, color=INK, space_before=8)
add_line(tf, "• 8 通道：前臂不同肌群的空间分布提供可分性，是区分「手掌 / 握拳 / 左右」的关键。",
         size=12, color=INK, space_before=6)
add_line(tf, "• CRC 与丢帧统计：实时控制对数据完整性敏感，坏帧必须在进入模型前丢弃。",
         size=12, color=INK, space_before=6)
footer(s, 3)


# ===================== 04 数据采集 · 录制 =====================
s = new_slide()
slide_header(s, "01 · 数据采集", "录制与标注：标签、手别、用户元数据一起落盘",
             "采集时记录的信息越完整，后面能做的算法约束就越多")

left = rect(s, MARGIN, 2.0, 6.0, 3.55, WHITE, BORDER)
shape_text(left, [
    ("CSV 文件结构", 13.5, True, ACCENT),
    ("# label=手掌        # user=李鑫", 9.5, False, INK),
    ("# session=李鑫_session_001", 9.5, False, INK),
    ("# protocol=STM32    # hand=right", 9.5, False, INK),
    ("sample_index,ch1,ch2,...,ch8", 9.5, False, INK),
    ("0,2048,2031,...,1997", 9.5, False, MUTED),
    ("1,2051,2028,...,2004", 9.5, False, MUTED),
    ("…", 9.5, False, MUTED),
], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

x2 = MARGIN + 6.3
w2 = CONTENT_W - 6.3
rows = [
    ("动作标签", "休息 / 握拳 / 手掌 / 向右 / 向左 + 自定义"),
    ("采集手别", "右手 / 左手，每次采集可选并写入 CSV"),
    ("用户元数据", "性别、体重(kg)、惯用手，注册时录入"),
    ("两种入口", "「开始采集」带标签；「记录数据」纯原始数据"),
    ("历史管理", "波形回看、导出 CSV、批量删除、备份恢复"),
]
y = 2.0
for head, body in rows:
    box = rect(s, x2, y, w2, 0.6, LIGHT, BORDER)
    shape_text(box, [(head, 11, True, ACCENT)], align=PP_ALIGN.LEFT)
    tf = textbox(s, x2 + 1.25, y + 0.13, w2 - 1.35, 0.4)
    add_line(tf, body, size=10.2, color=INK, first=True)
    y += 0.71

tf = textbox(s, MARGIN, 5.78, CONTENT_W, 1.0)
add_line(tf, "为什么要记「手别」：左手做「向右」与右手做「向右」在肌电上是相反的运动，"
             "若不记录，同一标签会混入两种语义，模型无法收敛。", size=11.5, bold=True,
         color=ACCENT3, first=True)
add_line(tf, "为什么要记「性别 / 体重」：它们影响肌肉量与阻抗，作为域标签可用于对抗训练，"
             "降低模型对用户属性的依赖。", size=11.5, color=MUTED, space_before=6)
footer(s, 4)


# ===================== 05 预处理 =====================
s = new_slide()
slide_header(s, "02 · 预处理", "统一预处理：滤波、滑窗、归一化、标签语义对齐",
             "训练端与 Android 端必须完全一致，否则精度会在部署时丢失")

pipeline = [
    ("原始 8 通道", "ADC 计数"),
    ("带通滤波", "Butterworth 4 阶\n20–249.999 Hz"),
    ("滑窗切分", "100 点 = 200 ms\n步长 20（训练）"),
    ("归一化", "逐维均值 / 标准差\n随模型下发"),
    ("模型输入", "8 × 100"),
]
bw, gap = 1.95, 0.42
x = MARGIN
for i, (head, sub) in enumerate(pipeline):
    box = rect(s, x, 2.05, bw, 1.15, LIGHT2 if i < 4 else LIGHT, BORDER)
    shape_text(box, [(head, 12.5, True, INK), (sub, 9.5, False, MUTED)])
    if i < len(pipeline) - 1:
        arrow(s, x + bw + 0.08, 2.5, 0.26, 0.24)
    x += bw + gap

tf = textbox(s, MARGIN, 3.5, CONTENT_W, 0.4)
add_line(tf, "两个容易忽略的语义处理", size=14, bold=True, color=INK, first=True)

cards = [
    ("左手标签交换", "左手用户的「向右(3)」与「向左(4)」在训练时互换，\n"
                    "使「向左 / 向右」在整个数据集上语义一致。\n"
                    "支持记录级手别，同一用户混采左右手也能正确处理。", ACCENT),
    ("性别 / 体重分桶", "性别映射为 2 类；体重按 <55 / <70 / <85 / ≥85 kg\n"
                     "分为 4 个桶，作为对抗训练的域标签，\n"
                     "让特征对这些属性不敏感。", ACCENT2),
]
cw = (CONTENT_W - 0.34) / 2
for i, (head, body, color) in enumerate(cards):
    box = rect(s, MARGIN + i * (cw + 0.34), 3.98, cw, 1.75, WHITE, BORDER)
    shape_text(box, [(head, 13, True, color), (body, 10.5, False, INK)],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

tf = textbox(s, MARGIN, 5.98, CONTENT_W, 0.6)
add_line(tf, "部署一致性：Android 端 PyTorch 推理同样执行 4 阶带通滤波与逐维标准化，"
             "窗口 100、步长 20，与训练脚本参数一一对应。", size=11.5, color=MUTED, first=True)
footer(s, 5)


# ===================== 06 本地训练 =====================
s = new_slide()
slide_header(s, "03 · 模型训练", "本地轻量模型：RBF SVM 端侧秒级训练",
             "演示统一使用 RBF SVM（径向基核），不依赖网络，其余算法仅作对比")

algos = [
    ("质心分类", "每类特征均值作为模板\n按标准距离取最近类", "最快，适合快速试验"),
    ("Linear SVM", "线性核，端侧 libsvm\n内存占用小", "小样本稳定"),
    ("RBF SVM", "径向基核，非线性边界\n对类间非线性更友好", "精度通常更好"),
]
cw = (CONTENT_W - 0.32 * 2) / 3
for i, (head, body, note) in enumerate(algos):
    box = rect(s, MARGIN + i * (cw + 0.32), 2.0, cw, 1.55, WHITE, BORDER)
    shape_text(box, [(head, 14, True, ACCENT), (body, 10.5, False, INK), (note, 9.5, False, MUTED)],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

tf = textbox(s, MARGIN, 3.78, CONTENT_W, 0.4)
add_line(tf, "特征工程：每通道 7 维（共 56 维），全部采用增益容忍的时域量", size=14, bold=True,
         color=INK, first=True)

feats = [
    ("log(MAV)", "平均绝对值"), ("log(RMS)", "均方根"), ("log(WL)", "波形长度"),
    ("log(能量)", "均方值"), ("过零率", "ZC"), ("斜率变号率", "SSC"), ("RMS 占比", "通道能量比"),
]
fw = (CONTENT_W - 0.18 * 6) / 7
for i, (a, b) in enumerate(feats):
    box = rect(s, MARGIN + i * (fw + 0.18), 4.28, fw, 0.85, LIGHT, BORDER)
    shape_text(box, [(a, 11, True, INK), (b, 9, False, MUTED)])

tf = textbox(s, MARGIN, 5.42, CONTENT_W, 1.1)
add_line(tf, "训练流程：读取 CSV → 8 通道特征 → 逐维标准化 → 训练 → 保存 JSON 模型并自动启用。",
         size=12, color=INK, first=True)
add_line(tf, "适配场景：单用户、少量录制、现场快速标定；模型完全保存在手机本地。",
         size=12, color=MUTED, space_before=6)
footer(s, 6)


# ===================== 07 云端模型 =====================
s = new_slide()
slide_header(s, "03 · 模型训练", "云端 CNN-Transformer：面向跨用户的深度模型",
             "卷积提取局部肌电模式，Transformer 建模窗口内长程依赖")

blocks = [
    ("输入", "8 × 100\n（8 通道 × 200 ms）"),
    ("CNN 特征提取", "Conv1d 8→64 (k10)\n→ 64→96 (k5)\n→ 96→128 (k3)"),
    ("序列建模", "位置编码\nTransformerEncoder\n2 层 · 4 头 · d=128"),
    ("池化", "时间维均值\n→ 128 维表征"),
    ("分类", "FC 128→5\n休息/握拳/手掌\n/向右/向左"),
]
bw, gap = 2.1, 0.36
x = MARGIN
for i, (head, sub) in enumerate(blocks):
    box = rect(s, x, 2.02, bw, 1.4, LIGHT2 if i in (1, 2) else WHITE, BORDER)
    shape_text(box, [(head, 12.5, True, ACCENT2 if i in (1, 2) else INK), (sub, 9.3, False, MUTED)])
    if i < len(blocks) - 1:
        arrow(s, x + bw + 0.05, 2.55, 0.26, 0.24)
    x += bw + gap

tf = textbox(s, MARGIN, 3.68, CONTENT_W, 0.4)
add_line(tf, "每个卷积后接 BatchNorm + ReLU，前两层带 Dropout 0.3", size=11, color=MUTED, first=True)

# 辅助头
tf = textbox(s, MARGIN, 4.12, CONTENT_W, 0.4)
add_line(tf, "训练期挂载的辅助头（导出时剔除，不进入手机端）", size=14, bold=True, color=INK, first=True)

heads = [
    ("用户域判别头", "Linear 128→128→N用户", "GRL 反转梯度"),
    ("对比投影头", "Linear 128→128→64", "跨用户/会话对比"),
    ("元数据判别头", "Linear 128→64→6", "性别2 + 体重4"),
]
cw = (CONTENT_W - 0.3 * 2) / 3
for i, (head, arch, note) in enumerate(heads):
    box = rect(s, MARGIN + i * (cw + 0.3), 4.62, cw, 1.1, LIGHT3, BORDER)
    shape_text(box, [(head, 12, True, ACCENT3), (arch, 9.5, False, INK), (note, 9.5, False, MUTED)],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

tf = textbox(s, MARGIN, 6.0, CONTENT_W, 0.6)
add_line(tf, "导出：剔除辅助头 → 包装为仅返回 logits 张量的模块 → TorchScript，供 Android PyTorch 接口直接加载。",
         size=11.5, color=MUTED, first=True)
footer(s, 7)


# ===================== 08 不变性约束 =====================
s = new_slide()
slide_header(s, "03 · 模型训练", "三种不变性约束：让特征对「人」不敏感",
             "这是跨用户与零样本能够成立的核心")

constraints = [
    ("① GRL 用户对抗", "梯度反转层", "域判别器尝试区分「这是哪个用户」，\n"
                                     "GRL 把梯度取反回传到编码器，\n"
                                     "迫使特征无法区分用户。", "λ = 0.3\n前 5 epoch 预热"),
    ("② 跨用户对比学习", "同手势拉近", "同一手势在不同用户 / 不同会话间\n"
                                    "在投影空间中拉近，\n"
                                    "不同手势推开。", "温度 0.1\n会话权重 2.0"),
    ("③ Meta GRL 对抗", "性别 / 体重不变", "以性别（2 类）与体重桶（4 类）\n"
                                         "作为额外域标签做对抗，\n"
                                         "削弱生理属性带来的分布差异。", "权重 0.15"),
]
cw = (CONTENT_W - 0.3 * 2) / 3
for i, (head, sub, body, param) in enumerate(constraints):
    box = rect(s, MARGIN + i * (cw + 0.3), 2.02, cw, 2.55, WHITE, BORDER)
    shape_text(box, [(head, 13, True, ACCENT), (sub, 10, True, MUTED),
                     (body, 10, False, INK), (param, 9.5, False, ACCENT3)],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

# 总损失
box = rect(s, MARGIN, 4.82, CONTENT_W, 1.05, LIGHT, BORDER)
shape_text(box, [
    ("总损失 = 手势分类损失 + 0.5 × 对比损失 + 0.3 × 用户对抗损失 + 0.15 × 元数据对抗损失", 13, True, INK),
    ("分类损失保证可区分性；三项辅助损失分别约束用户、会话、生理属性带来的分布偏移。", 10.5, False, MUTED),
], align=PP_ALIGN.LEFT)

tf = textbox(s, MARGIN, 6.1, CONTENT_W, 0.6)
add_line(tf, "目标：新用户即使不提供数据，基础模型的特征分布也已尽量贴近其生理特征。",
         size=11.5, color=MUTED, first=True)
footer(s, 8)


# ===================== 09 两阶段 + 零样本 =====================
s = new_slide()
slide_header(s, "03 · 模型训练", "两阶段训练与零样本部署",
             "先在全体用户上训练通用基础模型，再按需个性化")

# 阶段 1
box = rect(s, MARGIN, 2.02, 4.35, 2.25, LIGHT2, BORDER)
shape_text(box, [
    ("阶段 1 · 基础模型", 15, True, ACCENT2),
    ("数据：全体用户", 10.5, False, INK),
    ("训练：100 epoch，patience 15", 10.5, False, INK),
    ("学习率 1e-3 · batch 64", 10.5, False, INK),
    ("开启 GRL / 对比 / Meta", 10.5, False, INK),
    ("产出：base_model_transformer.pth", 9.5, False, MUTED),
], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

arrow(s, MARGIN + 4.55, 2.95, 0.42, 0.3)

# 阶段 2
box = rect(s, MARGIN + 5.2, 2.02, 4.35, 2.25, LIGHT, BORDER)
shape_text(box, [
    ("阶段 2 · 用户微调", 15, True, ACCENT),
    ("数据：当前用户", 10.5, False, INK),
    ("训练：50 epoch，patience 10", 10.5, False, INK),
    ("学习率 1e-4（更小）", 10.5, False, INK),
    ("加载基础权重作为起点", 10.5, False, INK),
    ("产出：个性化模型（更高精度）", 9.5, False, MUTED),
], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

# 零样本
box = rect(s, MARGIN, 4.62, CONTENT_W, 1.35, LIGHT3, BORDER)
shape_text(box, [
    ("零样本路径：跳过采集与微调，直接使用基础模型", 14, True, ACCENT3),
    ("App「云端训练」面板 →「零样本下载（基础模型）」→ 下载 .pt 与标签/归一化/滤波元数据 → 直接启用识别",
     10.5, False, INK),
    ("服务端接口：GET /api/download_base_model（需 Token），返回 TorchScript 模型与 X-Model-* 头信息",
     10, False, MUTED),
], align=PP_ALIGN.LEFT)

tf = textbox(s, MARGIN, 6.18, CONTENT_W, 0.6)
add_line(tf, "新用户的取舍：零样本立即可用但精度略低；采集少量数据微调后精度显著提升。",
         size=11.5, color=MUTED, first=True)
footer(s, 9)


# ===================== 10 实时识别 =====================
s = new_slide()
slide_header(s, "04 · 手势识别", "实时推理：滑窗、投票与置信度过滤",
             "在数据回调线程中完成，不阻塞采集与绘制")

chain = [
    ("数据回调", "每帧 8 通道\n写入滑窗缓冲"),
    ("凑满窗口", "100 点\n步长 20"),
    ("模型推理", "TorchScript\n或本地模型"),
    ("投票", "最近 5 帧\n同标签 ≥3 票"),
    ("输出", "标签 + 置信度\n→ 控制层"),
]
bw, gap = 2.0, 0.4
x = MARGIN
for i, (head, sub) in enumerate(chain):
    box = rect(s, x, 2.02, bw, 1.05, LIGHT if i == 3 else WHITE, BORDER)
    shape_text(box, [(head, 12.5, True, ACCENT if i == 3 else INK), (sub, 9.3, False, MUTED)])
    if i < len(chain) - 1:
        arrow(s, x + bw + 0.07, 2.42, 0.26, 0.24)
    x += bw + gap

notes = [
    ("置信度阈值 0.48", "低于阈值时显示「不确定」，并作为中立状态处理，避免误触发。"),
    ("投票 7 帧 / 需 5 票", "同一标签在最近 7 帧中至少 5 票才输出，抑制单帧抖动。"),
    ("命令二次确认", "控制层要求新命令连续 2 次一致，才真正驱动设备。"),
    ("异常隔离", "推理异常被捕获并清空滑窗，数据线程不会崩溃。"),
]
cw = (CONTENT_W - 0.28) / 2
for i, (head, body) in enumerate(notes):
    box = rect(s, MARGIN + (i % 2) * (cw + 0.28), 3.45 + (i // 2) * 1.05, cw, 0.9, WHITE, BORDER)
    shape_text(box, [(head, 11.5, True, ACCENT2), (body, 9.8, False, INK)],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

tf = textbox(s, MARGIN, 5.72, CONTENT_W, 0.9)
add_line(tf, "推理节奏：每 20 个新样本推进一次（约 40 ms），单次推理耗时远小于采样间隔，可稳定实时运行。",
         size=11.5, color=MUTED, first=True)
footer(s, 10)


# ===================== 11 手势映射与锁存 =====================
s = new_slide()
slide_header(s, "05 · 设备控制", "手势到命令：默认映射 + 中立锁存状态机",
             "锁存机制解决「动作收回时被误判为反向手势」的问题")

mapping = [
    ("休息", "停止", "中立"),
    ("握拳", "后退", "动作"),
    ("手掌", "前进", "动作"),
    ("向左", "左转", "动作"),
    ("向右", "右转", "动作"),
]
cw = (CONTENT_W - 0.22 * 4) / 5
for i, (gesture, cmd, kind) in enumerate(mapping):
    fill = LIGHT if kind == "中立" else WHITE
    box = rect(s, MARGIN + i * (cw + 0.22), 2.02, cw, 1.0, fill, BORDER)
    shape_text(box, [(gesture, 13, True, ACCENT3 if kind == "中立" else INK),
                     ("→ " + cmd, 11, True, ACCENT if kind == "动作" else MUTED)])

tf = textbox(s, MARGIN, 3.32, 6.1, 0.4)
add_line(tf, "中立锁存规则", size=14, bold=True, color=INK, first=True)

rules = rect(s, MARGIN, 3.78, 6.1, 2.15, WHITE, BORDER)
shape_text(rules, [
    ("① 任何动作开始前，必须先识别到「休息 / 不确定」", 10.5, False, INK),
    ("② 从动作 A 直接切到动作 B 会被忽略（防止抖动与误判）", 10.5, False, INK),
    ("③ 同一动作持续时不断续期，不会误报中断", 10.5, False, INK),
    ("④ 识别到停止或回到中立后，才解锁下一次切换", 10.5, False, INK),
], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

x2 = MARGIN + 6.4
box = rect(s, x2, 3.32, CONTENT_W - 6.4, 2.6, LIGHT2, BORDER)
shape_text(box, [
    ("为什么要锁存", 13.5, True, ACCENT2),
    ("做「手掌」后收手时，肌肉信号常被模型判成「握拳」，", 10.2, False, INK),
    ("若直接执行就会产生一次本不该有的后退。", 10.2, False, INK),
    ("中立锁存把「动作之间的过渡」定义为必须经过休息，", 10.2, False, INK),
    ("用一点操作习惯换取控制的确定性。", 10.2, False, INK),
], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

tf = textbox(s, MARGIN, 6.15, CONTENT_W, 0.6)
add_line(tf, "超时保护：3 秒无新的有效识别结果 → 停止动作但保持控制开启；30 秒完全没有识别回调 → 自动关闭控制。",
         size=11.5, color=MUTED, first=True)
footer(s, 11)


# ===================== 12 通信链路与设备 =====================
s = new_slide()
slide_header(s, "05 · 设备控制", "通信链路：App → 树莓派 → 五类被控设备",
             "统一走 UDP，按设备类型选择不同的载荷协议")

devices = [
    ("机械车", "字符串指令", "forward / backward / left / right / stop\n支持夹爪 open / close", ACCENT),
    ("智能车", "数字指令", "1 前进 · 2 后退 · 3 左 · 4 右\n5 停止 · 6 / 7 夹爪", ACCENT),
    ("Go2 机械狗", "字符串指令", "forward / backward / left / right / stop\n树莓派转 SportClient.Move", ACCENT2),
    ("Go2-W 轮式", "go2w: 前缀", "go2w:forward / … / go2w:stop\n先 BalanceStand（等效按 9）再 Move", ACCENT3),
    ("侦察球", "11 字节二进制", "球 ID + 控制字 + steer/drive(int16 LE)\n+ servo + turn_speed + CRC16/MAXIM", ACCENT3),
]
y = 1.98
for name, proto, detail, color in devices:
    rect(s, MARGIN, y, 1.95, 0.78, WHITE, BORDER)
    tf = textbox(s, MARGIN + 0.1, y + 0.19, 1.75, 0.4)
    add_line(tf, name, size=12, bold=True, color=color, first=True)

    rect(s, MARGIN + 2.05, y, 1.55, 0.78, LIGHT, BORDER)
    tf = textbox(s, MARGIN + 2.15, y + 0.19, 1.35, 0.4)
    add_line(tf, proto, size=10, bold=True, color=MUTED, first=True)

    tf = textbox(s, MARGIN + 3.75, y + 0.06, CONTENT_W - 3.75, 0.7, anchor=MSO_ANCHOR.MIDDLE)
    add_line(tf, detail, size=9.5, color=INK, first=True)
    y += 0.88

tf = textbox(s, MARGIN, 6.35, CONTENT_W, 0.6)
add_line(tf, "发送节奏：启用后每 50 ms 持续重发当前动作；手动模式下不受识别超时限制，点「停止」或切回手势即退出。",
         size=11, color=MUTED, first=True)
footer(s, 12)


# ===================== 13 非线性 =====================
s = new_slide()
slide_header(s, "06 · 小结", "「非线性」体现在链路的哪些环节",
             "从信号处理到控制决策，非线性并不是单点，而是贯穿全链路")

layers = [
    ("信号层", "4 阶 IIR 带通滤波\n幅频与相频均为非线性响应", ACCENT),
    ("特征层", "log / 比值 / 过零率 / 斜率变号率\n对幅值非线性压缩", ACCENT),
    ("模型层", "ReLU、LeakyReLU、Softmax\n自注意力与 RBF 核映射", ACCENT2),
    ("对抗层", "梯度反转（GRL）\n训练动力学本身是非线性的", ACCENT2),
    ("决策层", "5 帧投票 + 置信度阈值\n中立锁存为离散状态机", ACCENT3),
]
y = 2.0
for name, body, color in layers:
    rect(s, MARGIN, y, 1.75, 0.72, WHITE, BORDER)
    tf = textbox(s, MARGIN + 0.12, y + 0.18, 1.5, 0.4)
    add_line(tf, name, size=11.5, bold=True, color=color, first=True)

    tf = textbox(s, MARGIN + 1.95, y + 0.08, CONTENT_W - 1.95, 0.6, anchor=MSO_ANCHOR.MIDDLE)
    add_line(tf, body, size=10.5, color=INK, first=True)
    y += 0.82

box = rect(s, MARGIN, 6.18, CONTENT_W, 0.62, LIGHT, BORDER)
shape_text(box, [("因此系统不是「线性阈值触发」，而是特征学习 + 概率决策 + 状态约束的组合。", 11.5, True, INK)],
           align=PP_ALIGN.LEFT)
footer(s, 13)


# ===================== 14 参数速查 =====================
s = new_slide()
slide_header(s, "07 · 附录", "关键参数速查",
             "以下取值均可通过配置或脚本参数调整")

rows = [
    ("采集", "采样率 / 通道数", "500 Hz / 8", "STM32 协议"),
    ("采集", "窗口 / 步长", "100 点 / 20 点", "200 ms @ 500 Hz"),
    ("预处理", "带通滤波", "20–249.999 Hz，4 阶", "Butterworth"),
    ("训练", "类别数", "5", "休息/握拳/手掌/向右/向左"),
    ("训练", "batch / 学习率", "64 / 1e-3", "微调学习率 1e-4"),
    ("训练", "基础 / 微调 epoch", "100 / 50", "patience 15 / 10"),
    ("训练", "GRL λ / 温度", "0.3 / 0.1", "预热 5 epoch"),
    ("训练", "对比权重 / Meta 权重", "0.5 / 0.15", "会话权重 2.0"),
    ("识别", "置信度阈值", "0.48", "低于则判为不确定"),
    ("识别", "投票 / 命令确认", "7 帧需 5 票 / 连续 2 次", "演示稳定性优先"),
    ("控制", "发送周期", "50 ms", "持续重发当前动作"),
    ("控制", "识别超时 / 控制超时", "3 s / 30 s", "前者停动作，后者关控制"),
    ("侦察球", "速度上限 / 端口", "±1000 / 3020", "CRC16-MAXIM"),
]
col_w = [1.15, 3.35, 3.0, 3.6]
tbl_left = MARGIN
tbl_top = 1.98
tbl_w = sum(col_w)
tbl_h = 0.34 * (len(rows) + 1)
table = s.shapes.add_table(len(rows) + 1, 4, Inches(tbl_left), Inches(tbl_top),
                           Inches(tbl_w), Inches(tbl_h)).table
for i, w in enumerate(col_w):
    table.columns[i].width = Inches(w)
for i in range(len(rows) + 1):
    table.rows[i].height = Inches(0.34)

headers = ["环节", "参数", "取值", "备注"]
for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = ACCENT
    cell.margin_left = Inches(0.08)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for p in cell.text_frame.paragraphs:
        for r in p.runs:
            _apply_font(r, 10.5, True, WHITE)

for i, row in enumerate(rows, start=1):
    for j, value in enumerate(row):
        cell = table.cell(i, j)
        cell.text = value
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
        cell.margin_left = Inches(0.08)
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                _apply_font(r, 9.5, j == 0, INK if j != 0 else ACCENT)

footer(s, 14)


# ===================== 保存 =====================
prs.core_properties.title = "sEMG 手势识别系统：数据采集 · 模型训练 · 设备控制"
prs.core_properties.author = "sEMG 项目组"
prs.core_properties.subject = "数据采集 / 模型训练 / 设备控制"

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "sEMG手势识别系统_采集训练控制.pptx")
prs.save(out_path)
print("已生成:", out_path)
print("幻灯片数量:", len(prs.slides))
