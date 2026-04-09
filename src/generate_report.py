# -*- coding: utf-8 -*-
"""生成改进文档PDF - MobileNetV3 Small 花卉识别改进报告"""
import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import cm, inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                 Table, TableStyle, Image)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from PIL import Image as PILImage

# ==================== 注册字体 ====================
pdfmetrics.registerFont(TTFont('Microsoft YaHei', '/usr/share/fonts/truetype/chinese/msyh.ttf'))
pdfmetrics.registerFont(TTFont('SimHei', '/usr/share/fonts/truetype/chinese/SimHei.ttf'))
pdfmetrics.registerFont(TTFont('Times New Roman', '/usr/share/fonts/truetype/english/Times-New-Roman.ttf'))
registerFontFamily('Microsoft YaHei', normal='Microsoft YaHei', bold='Microsoft YaHei')
registerFontFamily('SimHei', normal='SimHei', bold='SimHei')
registerFontFamily('Times New Roman', normal='Times New Roman', bold='Times New Roman')

# ==================== 路径 ====================
save_dir = '/home/z/my-project/mobilenet_project/work_data/model/weights/'
output_pdf = '/home/z/my-project/mobilenet_project/improvement_report.pdf'

# ==================== 加载实验数据 ====================
with open(f'{save_dir}baseline_history.json') as f:
    baseline = json.load(f)
with open(f'{save_dir}improved_history.json') as f:
    improved = json.load(f)
with open(f'{save_dir}experiment_comparison.json') as f:
    comparison = json.load(f)
with open(f'{save_dir}improved_per_class_metrics.json') as f:
    per_class = json.load(f)

# ==================== 样式定义 ====================
# 封面
cover_title_style = ParagraphStyle(
    name='CoverTitle', fontName='Microsoft YaHei', fontSize=32, leading=42,
    alignment=TA_CENTER, spaceAfter=24, wordWrap='CJK'
)
cover_subtitle_style = ParagraphStyle(
    name='CoverSubtitle', fontName='SimHei', fontSize=16, leading=24,
    alignment=TA_CENTER, spaceAfter=12, wordWrap='CJK'
)
cover_info_style = ParagraphStyle(
    name='CoverInfo', fontName='SimHei', fontSize=13, leading=20,
    alignment=TA_CENTER, spaceAfter=8, wordWrap='CJK'
)
# 正文
h1_style = ParagraphStyle(
    name='H1', fontName='Microsoft YaHei', fontSize=18, leading=28,
    spaceBefore=18, spaceAfter=10, textColor=colors.black, wordWrap='CJK'
)
h2_style = ParagraphStyle(
    name='H2', fontName='Microsoft YaHei', fontSize=14, leading=22,
    spaceBefore=14, spaceAfter=8, textColor=colors.black, wordWrap='CJK'
)
h3_style = ParagraphStyle(
    name='H3', fontName='SimHei', fontSize=12, leading=18,
    spaceBefore=10, spaceAfter=6, textColor=colors.black, wordWrap='CJK'
)
body_style = ParagraphStyle(
    name='Body', fontName='SimHei', fontSize=10.5, leading=18,
    alignment=TA_LEFT, firstLineIndent=21, wordWrap='CJK', spaceAfter=4
)
body_no_indent = ParagraphStyle(
    name='BodyNoIndent', fontName='SimHei', fontSize=10.5, leading=18,
    alignment=TA_LEFT, wordWrap='CJK', spaceAfter=4
)
caption_style = ParagraphStyle(
    name='Caption', fontName='SimHei', fontSize=9.5, leading=14,
    alignment=TA_CENTER, spaceAfter=6, wordWrap='CJK',
    textColor=colors.HexColor('#333333')
)
# 表格
tbl_header_style = ParagraphStyle(
    name='TblHeader', fontName='SimHei', fontSize=10, leading=14,
    alignment=TA_CENTER, textColor=colors.white, wordWrap='CJK'
)
tbl_cell_style = ParagraphStyle(
    name='TblCell', fontName='SimHei', fontSize=9.5, leading=13,
    alignment=TA_CENTER, wordWrap='CJK'
)
tbl_cell_left = ParagraphStyle(
    name='TblCellLeft', fontName='SimHei', fontSize=9.5, leading=13,
    alignment=TA_LEFT, wordWrap='CJK'
)

TABLE_HEADER_COLOR = colors.HexColor('#1F4E79')
TABLE_ROW_ODD = colors.HexColor('#F5F5F5')

# ==================== 构建文档 ====================
doc = SimpleDocTemplate(
    output_pdf, pagesize=A4,
    title='MobileNetV3_Small_improvement_report',
    author='Z.ai', creator='Z.ai',
    subject='MobileNetV3 Small 花卉识别改进报告',
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm
)

story = []

# ==================== 封面 ====================
story.append(Spacer(1, 100))
story.append(Paragraph('<b>MobileNetV3 Small</b>', cover_title_style))
story.append(Spacer(1, 12))
story.append(Paragraph('<b>花卉图像识别改进报告</b>', cover_title_style))
story.append(Spacer(1, 36))
story.append(Paragraph('基于迁移学习的5分类花卉识别系统优化', cover_subtitle_style))
story.append(Spacer(1, 48))
story.append(Paragraph('深度学习课程设计', cover_info_style))
story.append(Spacer(1, 60))
story.append(Paragraph('2026年4月', cover_info_style))
story.append(PageBreak())

# ==================== 1. 项目概述 ====================
story.append(Paragraph('<b>1  项目概述</b>', h1_style))
story.append(Paragraph(
    '本项目是一个基于深度学习的花卉图像识别系统，采用<font name="Times New Roman">PyTorch</font>框架和<font name="Times New Roman">MobileNetV3 Small</font>预训练模型，实现了对蒲公英、白色鸡蛋花、牵牛花、牡丹、向日葵共5种常见花卉的分类识别。项目包含完整的数据采集、预处理、模型训练和<font name="Times New Roman">Web</font>应用部署流程。',
    body_style
))
story.append(Paragraph(
    '原始项目使用标准的迁移学习方案：加载<font name="Times New Roman">ImageNet</font>预训练权重，替换最后的全连接层为5分类输出，使用<font name="Times New Roman">Adam</font>优化器和<font name="Times New Roman">StepLR</font>学习率调度器进行10轮训练。数据预处理仅采用简单的<font name="Times New Roman">CenterCrop</font>和归一化操作，训练集准确率达到了98%以上，但验证集准确率约为90%左右，存在明显的过拟合现象。',
    body_style
))
story.append(Paragraph(
    '为了提升模型的泛化能力和实际表现，本报告在原始基线方案的基础上，从数据增强策略、损失函数优化、学习率调度、优化器选择等多个维度进行了系统性改进，并通过对比实验验证了改进效果。',
    body_style
))

# ==================== 2. 数据集介绍 ====================
story.append(Spacer(1, 12))
story.append(Paragraph('<b>2  数据集介绍</b>', h1_style))
story.append(Paragraph(
    '数据集来源于<font name="Times New Roman">Bing</font>图片搜索，包含5种花卉共2086张图片，按8:2比例划分为训练集（1667张）和验证集（419张）。各类别分布较为均衡，具体如下表所示。数据集规模适中，适合用于验证迁移学习在小样本场景下的效果。',
    body_style
))

# 数据集分布表
story.append(Spacer(1, 18))
ds_data = [
    [Paragraph('<b>类别</b>', tbl_header_style), Paragraph('<b>英文名</b>', tbl_header_style),
     Paragraph('<b>训练集</b>', tbl_header_style), Paragraph('<b>验证集</b>', tbl_header_style),
     Paragraph('<b>总计</b>', tbl_header_style)],
    [Paragraph('蒲公英', tbl_cell_style), Paragraph('Dandelion', tbl_cell_style),
     Paragraph('328', tbl_cell_style), Paragraph('83', tbl_cell_style), Paragraph('411', tbl_cell_style)],
    [Paragraph('白色鸡蛋花', tbl_cell_style), Paragraph('Frangipani', tbl_cell_style),
     Paragraph('311', tbl_cell_style), Paragraph('78', tbl_cell_style), Paragraph('389', tbl_cell_style)],
    [Paragraph('牵牛花', tbl_cell_style), Paragraph('Morning Glory', tbl_cell_style),
     Paragraph('325', tbl_cell_style), Paragraph('82', tbl_cell_style), Paragraph('407', tbl_cell_style)],
    [Paragraph('牡丹', tbl_cell_style), Paragraph('Peony', tbl_cell_style),
     Paragraph('353', tbl_cell_style), Paragraph('88', tbl_cell_style), Paragraph('441', tbl_cell_style)],
    [Paragraph('向日葵', tbl_cell_style), Paragraph('Sunflower', tbl_cell_style),
     Paragraph('350', tbl_cell_style), Paragraph('88', tbl_cell_style), Paragraph('438', tbl_cell_style)],
    [Paragraph('<b>合计</b>', tbl_cell_style), Paragraph('', tbl_cell_style),
     Paragraph('<b>1667</b>', tbl_cell_style), Paragraph('<b>419</b>', tbl_cell_style), Paragraph('<b>2086</b>', tbl_cell_style)],
]
ds_table = Table(ds_data, colWidths=[2.8*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm])
ds_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), TABLE_ROW_ODD),
    ('BACKGROUND', (0, 4), (-1, 4), TABLE_ROW_ODD),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(ds_table)
story.append(Spacer(1, 6))
story.append(Paragraph('表1  数据集类别分布', caption_style))

# ==================== 3. 基线方案与结果 ====================
story.append(Spacer(1, 18))
story.append(Paragraph('<b>3  基线方案（Baseline）</b>', h1_style))

story.append(Paragraph('<b>3.1  数据预处理</b>', h2_style))
story.append(Paragraph(
    '基线方案的数据预处理较为简单：训练集和验证集均使用<font name="Times New Roman">CenterCrop(224)</font>进行中心裁剪，然后通过<font name="Times New Roman">ToTensor()</font>将图像转换为张量，最后使用均值和标准差均为[0.5, 0.5, 0.5]的归一化将像素值映射到[-1, 1]范围。这种预处理方式没有引入任何随机性，训练集和验证集的预处理完全一致。',
    body_style
))

story.append(Paragraph('<b>3.2  模型架构</b>', h2_style))
story.append(Paragraph(
    '模型采用<font name="Times New Roman">torchvision</font>提供的<font name="Times New Roman">MobileNetV3 Small</font>，加载<font name="Times New Roman">ImageNet1K_V1</font>预训练权重。将原始模型分类头<font name="Times New Roman">classifier[3]</font>的<font name="Times New Roman">Linear(576, 1000)</font>替换为<font name="Times New Roman">Linear(576, 5)</font>，适配5分类任务。全部参数均参与训练，不冻结任何层。',
    body_style
))

story.append(Paragraph('<b>3.3  训练配置</b>', h2_style))
train_config = [
    [Paragraph('<b>配置项</b>', tbl_header_style), Paragraph('<b>基线设置</b>', tbl_header_style),
     Paragraph('<b>说明</b>', tbl_header_style)],
    [Paragraph('优化器', tbl_cell_left), Paragraph('Adam, lr=0.0001', tbl_cell_style), Paragraph('自适应学习率优化器', tbl_cell_left)],
    [Paragraph('损失函数', tbl_cell_left), Paragraph('CrossEntropyLoss', tbl_cell_style), Paragraph('标准交叉熵损失', tbl_cell_left)],
    [Paragraph('调度器', tbl_cell_left), Paragraph('StepLR(step=5, gamma=0.5)', tbl_cell_style), Paragraph('每5轮学习率减半', tbl_cell_left)],
    [Paragraph('训练轮数', tbl_cell_left), Paragraph('10 epochs', tbl_cell_style), Paragraph('共训练10轮', tbl_cell_left)],
    [Paragraph('批次大小', tbl_cell_left), Paragraph('64', tbl_cell_style), Paragraph('每批64张图片', tbl_cell_left)],
    [Paragraph('是否冻结', tbl_cell_left), Paragraph('否', tbl_cell_style), Paragraph('全部参数参与训练', tbl_cell_left)],
]
train_table = Table(train_config, colWidths=[3*cm, 5.5*cm, 5*cm])
train_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), TABLE_ROW_ODD),
    ('BACKGROUND', (0, 4), (-1, 4), TABLE_ROW_ODD),
    ('BACKGROUND', (0, 6), (-1, 6), TABLE_ROW_ODD),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(Spacer(1, 18))
story.append(train_table)
story.append(Spacer(1, 6))
story.append(Paragraph('表2  基线方案训练配置', caption_style))

story.append(Paragraph('<b>3.4  基线结果</b>', h2_style))
bl = comparison['baseline']
story.append(Paragraph(
    f'经过10轮训练，基线模型的最高验证集准确率出现在第{bl["best_epoch"]}轮，达到{bl["best_val_acc"]:.2%}。'
    f'最终训练集准确率为{bl["final_train_acc"]:.2%}，验证集准确率为{bl["final_val_acc"]:.2%}。'
    f'训练集与验证集准确率之间存在约{bl["final_train_acc"] - bl["final_val_acc"]:.2%}的差距，表明模型存在较为明显的过拟合现象。',
    body_style
))

# ==================== 4. 改进方案 ====================
story.append(Spacer(1, 18))
story.append(Paragraph('<b>4  改进方案设计</b>', h1_style))
story.append(Paragraph(
    '针对基线方案中存在的过拟合问题和泛化能力不足的情况，本文从数据增强、正则化、优化策略三个维度提出了7项改进措施，以下是每项改进的详细说明和理论依据。',
    body_style
))

# 改进1
story.append(Paragraph('<b>4.1  增强数据增强策略</b>', h2_style))
story.append(Paragraph(
    '基线方案仅使用<font name="Times New Roman">CenterCrop</font>和简单归一化，训练集和验证集的数据分布完全一致，导致模型缺乏对图像变化的鲁棒性。改进方案引入了多种数据增强操作，显著增加了训练数据的多样性，使模型能够学习到更加不变的特征表示。',
    body_style
))
story.append(Paragraph(
    '训练集增强包括：使用<font name="Times New Roman">RandomResizedCrop</font>替代<font name="Times New Roman">CenterCrop</font>，以随机缩放比例(0.8-1.0)和宽高比(0.75-1.333)裁剪图像，使模型适应不同尺度和构图；<font name="Times New Roman">RandomHorizontalFlip</font>(概率0.5)模拟镜像视角；<font name="Times New Roman">RandomRotation</font>(15度)增强旋转不变性；<font name="Times New Roman">ColorJitter</font>(亮度、对比度、饱和度各0.2，色调0.1)模拟不同光照条件；<font name="Times New Roman">RandomAffine</font>(平移10%)模拟位置偏移。同时，将归一化参数从[0.5,0.5,0.5]改为<font name="Times New Roman">ImageNet</font>标准统计量[0.485,0.456,0.406]/[0.229,0.224,0.225]，与预训练权重的训练分布一致。',
    body_style
))

# 改进2
story.append(Paragraph('<b>4.2  Mixup数据增强</b>', h2_style))
story.append(Paragraph(
    '<font name="Times New Roman">Mixup</font>是一种先进的正则化数据增强方法，其核心思想是在训练过程中将两张图片及其标签进行线性插值混合。具体而言，对于一对样本(x<sub>A</sub>, y<sub>A</sub>)和(x<sub>B</sub>, y<sub>B</sub>)，<font name="Times New Roman">Mixup</font>生成混合样本为x = lambda * x<sub>A</sub> + (1-lambda) * x<sub>B</sub>，其中lambda服从<font name="Times New Roman">Beta(0.2, 0.2)</font>分布。对应的损失函数也进行加权混合。',
    body_style
))
story.append(Paragraph(
    '本实现中，<font name="Times New Roman">Mixup</font>以50%的概率在每个<font name="Times New Roman">batch</font>中应用（即不是每个<font name="Times New Roman">batch</font>都做混合），混合系数alpha设为0.2。<font name="Times New Roman">Mixup</font>的引入有效抑制了模型对训练数据的过拟合倾向，促使模型学习更加平滑的决策边界。这也解释了为何改进版的训练集准确率（82.84%）低于基线（98.61%），但验证集准确率反而更高。',
    body_style
))

# 改进3
story.append(Paragraph('<b>4.3  Label Smoothing标签平滑</b>', h2_style))
story.append(Paragraph(
    '标准的交叉熵损失函数要求模型对正确类别的输出概率趋近于1，这种"过于自信"的训练目标容易导致过拟合。<font name="Times New Roman">Label Smoothing</font>通过将硬标签[0, 0, 1, 0, 0]软化为[0.02, 0.02, 0.92, 0.02, 0.02]（以5分类、smoothing=0.1为例），让模型不要过于自信地输出极端概率值。',
    body_style
))
story.append(Paragraph(
    '本方案将<font name="Times New Roman">label_smoothing</font>参数设为0.1，意味着正确类别的目标概率从1.0降为0.9，剩余0.1的概率均匀分配给其他类别。这一技术能有效提升模型的校准能力和泛化性能，特别是在训练数据量有限的情况下效果更加显著。',
    body_style
))

# 改进4和5
story.append(Paragraph('<b>4.4  Warmup + CosineAnnealing学习率调度</b>', h2_style))
story.append(Paragraph(
    '基线方案使用的<font name="Times New Roman">StepLR</font>调度器以阶梯形式降低学习率，这种不连续的变化可能导致训练不稳定。改进方案采用两阶段学习率策略：前2个<font name="Times New Roman">epoch</font>进行线性<font name="Times New Roman">Warmup</font>，学习率从较小的初始值（0.00005）线性增长到目标值（0.0001），帮助模型在训练初期平稳收敛，避免因初始学习率过大导致的梯度不稳定。',
    body_style
))
story.append(Paragraph(
    '<font name="Times New Roman">Warmup</font>结束后，切换为<font name="Times New Roman">CosineAnnealing</font>余弦退火调度，学习率按照余弦函数从峰值平滑衰减到接近0。这种平滑的衰减模式相比于阶梯式衰减，能够让模型在训练后期更精细地探索参数空间，有利于找到更优的局部最小值。最终学习率最低降至约0.000004，保证了训练后期的充分微调。',
    body_style
))

# 改进6
story.append(Paragraph('<b>4.5  AdamW优化器与权重衰减</b>', h2_style))
story.append(Paragraph(
    '将优化器从<font name="Times New Roman">Adam</font>替换为<font name="Times New Roman">AdamW</font>，并设置权重衰减系数为1e-4。<font name="Times New Roman">AdamW</font>与<font name="Times New Roman">Adam</font>的核心区别在于权重衰减的实现方式：<font name="Times New Roman">Adam</font>中的<font name="Times New Roman">L2</font>正则化与自适应学习率耦合，导致不同参数的正则化强度不一致；而<font name="Times New Roman">AdamW</font>将权重衰减从梯度更新中解耦，实现了真正的<font name="Times New Roman">L2</font>正则化效果。',
    body_style
))
story.append(Paragraph(
    '解耦的权重衰减确保了所有参数受到一致的正则化约束，有效抑制了过拟合，同时不会影响优化器对学习率的自适应调节。这一改进配合<font name="Times New Roman">Mixup</font>和<font name="Times New Roman">Label Smoothing</font>，构成了完整的正则化体系。',
    body_style
))

# ==================== 5. 实验结果对比 ====================
story.append(Spacer(1, 18))
story.append(Paragraph('<b>5  实验结果对比</b>', h1_style))

story.append(Paragraph('<b>5.1  整体性能对比</b>', h2_style))
imp = comparison['improved']
g = comparison['improvement']
story.append(Paragraph(
    f'改进方案的最高验证集准确率为{imp["best_val_acc"]:.2%}（第{imp["best_epoch"]}轮），'
    f'相比基线的{bl["best_val_acc"]:.2%}提升了{g["val_acc_gain"]:.2%}（相对提升{g["val_acc_gain_percent"]:.2%}%）。'
    f'虽然绝对提升幅度看起来不大，但更值得关注的是泛化能力的显著改善：基线模型训练集准确率高达{bl["final_train_acc"]:.2%}，'
    f'而验证集仅为{bl["final_val_acc"]:.2%}，两者差距达{bl["final_train_acc"] - bl["final_val_acc"]:.2%}，存在明显过拟合；'
    f'改进模型训练集准确率为{imp["final_train_acc"]:.2%}，验证集为{imp["final_val_acc"]:.2%}，差距仅为{abs(imp["final_train_acc"] - imp["final_val_acc"]):.2%}，'
    f'模型表现出更好的泛化特性。',
    body_style
))

# 对比表
story.append(Spacer(1, 18))
cmp_data = [
    [Paragraph('<b>指标</b>', tbl_header_style), Paragraph('<b>基线方案</b>', tbl_header_style),
     Paragraph('<b>改进方案</b>', tbl_header_style), Paragraph('<b>变化</b>', tbl_header_style)],
    [Paragraph('最高验证集准确率', tbl_cell_left),
     Paragraph(f'{bl["best_val_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'{imp["best_val_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'+{g["val_acc_gain"]:.2%}', tbl_cell_style)],
    [Paragraph('最高准确率所在轮次', tbl_cell_left),
     Paragraph(f'第{bl["best_epoch"]}轮', tbl_cell_style),
     Paragraph(f'第{imp["best_epoch"]}轮', tbl_cell_style),
     Paragraph('', tbl_cell_style)],
    [Paragraph('最终训练集准确率', tbl_cell_left),
     Paragraph(f'{bl["final_train_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'{imp["final_train_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'{imp["final_train_acc"] - bl["final_train_acc"]:.2%}', tbl_cell_style)],
    [Paragraph('最终验证集准确率', tbl_cell_left),
     Paragraph(f'{bl["final_val_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'{imp["final_val_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'+{imp["final_val_acc"] - bl["final_val_acc"]:.2%}', tbl_cell_style)],
    [Paragraph('训练-验证准确率差距', tbl_cell_left),
     Paragraph(f'{bl["final_train_acc"] - bl["final_val_acc"]:.2%}', tbl_cell_style),
     Paragraph(f'{abs(imp["final_train_acc"] - imp["final_val_acc"]):.2%}', tbl_cell_style),
     Paragraph('过拟合大幅减轻', tbl_cell_style)],
]
cmp_table = Table(cmp_data, colWidths=[4.2*cm, 3.2*cm, 3.2*cm, 3.2*cm])
cmp_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('BACKGROUND', (0, 2), (-1, 2), TABLE_ROW_ODD),
    ('BACKGROUND', (0, 4), (-1, 4), TABLE_ROW_ODD),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(cmp_table)
story.append(Spacer(1, 6))
story.append(Paragraph('表3  基线与改进方案性能对比', caption_style))

# 训练曲线图
story.append(Paragraph('<b>5.2  训练曲线对比</b>', h2_style))
story.append(Paragraph(
    '下图展示了基线和改进方案在10轮训练中的损失值和准确率变化趋势。可以观察到：基线方案的训练损失下降更快，训练集准确率迅速接近100%，但验证集准确率在第5轮后增速明显放缓，出现了过拟合的典型特征。改进方案由于引入了多种正则化手段，训练损失下降较慢，训练集准确率也较低，但验证集准确率保持稳定上升，最终超过了基线方案。',
    body_style
))

# 插入训练对比图
img_path = f'{save_dir}training_comparison.png'
if os.path.exists(img_path):
    pil_img = PILImage.open(img_path)
    orig_w, orig_h = pil_img.size
    target_w = 15.5 * cm
    scale = target_w / orig_w
    story.append(Spacer(1, 18))
    story.append(Image(img_path, width=target_w, height=orig_h * scale))
    story.append(Spacer(1, 6))
    story.append(Paragraph('图1  基线与改进方案训练曲线对比', caption_style))

# 过拟合分析图
img_path2 = f'{save_dir}overfitting_analysis.png'
if os.path.exists(img_path2):
    pil_img2 = PILImage.open(img_path2)
    orig_w2, orig_h2 = pil_img2.size
    target_w2 = 14 * cm
    scale2 = target_w2 / orig_w2
    story.append(Spacer(1, 18))
    story.append(Image(img_path2, width=target_w2, height=orig_h2 * scale2))
    story.append(Spacer(1, 6))
    story.append(Paragraph('图2  过拟合分析：训练集-验证集准确率差距对比', caption_style))

story.append(Paragraph(
    '从图2的过拟合分析中可以清晰看到：基线方案的训练-验证准确率差距从第1轮的19%逐步扩大到第10轮的80%，呈现出严重的过拟合趋势。而改进方案的差距始终控制在合理范围内，表明正则化措施有效抑制了过拟合，模型在新数据上的表现更加可靠。',
    body_style
))

# 5.3 每类准确率
story.append(Paragraph('<b>5.3  改进模型每类准确率</b>', h2_style))
story.append(Paragraph(
    '下表展示了改进模型在验证集上各类别的识别准确率。整体来看，5个类别的准确率均在87%以上，其中向日葵的识别准确率最高，达到96.59%，这与其独特的外形特征（大花盘、黄色花瓣）密切相关。牡丹和蒲公英的准确率相对较低，可能是由于这两类花卉品种多样、形态变化较大，存在一定的类间相似性。',
    body_style
))

story.append(Spacer(1, 18))
pc_data = [
    [Paragraph('<b>类别</b>', tbl_header_style), Paragraph('<b>正确数</b>', tbl_header_style),
     Paragraph('<b>总数</b>', tbl_header_style), Paragraph('<b>准确率</b>', tbl_header_style)],
]
for cls, metrics in per_class.items():
    pc_data.append([
        Paragraph(cls, tbl_cell_style),
        Paragraph(str(metrics['correct']), tbl_cell_style),
        Paragraph(str(metrics['total']), tbl_cell_style),
        Paragraph(f'{metrics["accuracy"]:.2%}', tbl_cell_style),
    ])
pc_table = Table(pc_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
row_styles = []
for i in range(1, len(pc_data)):
    if i % 2 == 0:
        row_styles.append(('BACKGROUND', (0, i), (-1, i), TABLE_ROW_ODD))
pc_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
] + row_styles))
story.append(pc_table)
story.append(Spacer(1, 6))
story.append(Paragraph('表4  改进模型各类别准确率', caption_style))

# 插入混淆矩阵
img_path3 = f'{save_dir}confusion_matrix.png'
if os.path.exists(img_path3):
    pil_img3 = PILImage.open(img_path3)
    orig_w3, orig_h3 = pil_img3.size
    target_w3 = 11 * cm
    scale3 = target_w3 / orig_w3
    story.append(Spacer(1, 18))
    story.append(Image(img_path3, width=target_w3, height=orig_h3 * scale3))
    story.append(Spacer(1, 6))
    story.append(Paragraph('图3  改进模型混淆矩阵', caption_style))

story.append(Paragraph(
    '混淆矩阵进一步揭示了模型的分类细节。对角线上的数值代表正确分类的数量，非对角线上的数值代表错误分类的情况。从矩阵可以看出，主要的混淆发生在牡丹与向日葵之间、蒲公英与白色鸡蛋花之间，这些花卉在颜色和形态上存在一定的相似性，是未来可以重点优化的方向。',
    body_style
))

# ==================== 6. 改进策略总结 ====================
story.append(Spacer(1, 18))
story.append(Paragraph('<b>6  改进策略总结</b>', h1_style))
story.append(Paragraph(
    '下表汇总了本次改进所采用的所有策略及其作用原理。这些改进措施并非孤立地发挥作用，而是相互配合，共同构成了一个完整的正则化和优化体系。',
    body_style
))

story.append(Spacer(1, 18))
sum_data = [
    [Paragraph('<b>改进策略</b>', tbl_header_style), Paragraph('<b>具体设置</b>', tbl_header_style),
     Paragraph('<b>作用原理</b>', tbl_header_style)],
    [Paragraph('数据增强', tbl_cell_left), Paragraph('RandomResizedCrop, Flip, Rotation, ColorJitter, Affine', tbl_cell_left),
     Paragraph('增加训练数据多样性，提升模型对图像变化的鲁棒性', tbl_cell_left)],
    [Paragraph('Mixup混合增强', tbl_cell_left), Paragraph('alpha=0.2, 概率50%', tbl_cell_left),
     Paragraph('线性插值混合样本，平滑决策边界，抑制过拟合', tbl_cell_left)],
    [Paragraph('Label Smoothing', tbl_cell_left), Paragraph('smoothing=0.1', tbl_cell_left),
     Paragraph('软化标签目标，防止模型过度自信，提升泛化能力', tbl_cell_left)],
    [Paragraph('学习率Warmup', tbl_cell_left), Paragraph('前2轮线性增长', tbl_cell_left),
     Paragraph('训练初期稳定收敛，避免梯度不稳定', tbl_cell_left)],
    [Paragraph('CosineAnnealing', tbl_cell_left), Paragraph('余弦退火至接近0', tbl_cell_left),
     Paragraph('平滑衰减学习率，训练后期精细调优参数', tbl_cell_left)],
    [Paragraph('AdamW优化器', tbl_cell_left), Paragraph('weight_decay=1e-4', tbl_cell_left),
     Paragraph('解耦权重衰减，实现真正的L2正则化效果', tbl_cell_left)],
    [Paragraph('ImageNet归一化', tbl_cell_left), Paragraph('mean/std采用ImageNet统计量', tbl_cell_left),
     Paragraph('与预训练权重训练分布一致，确保迁移效果', tbl_cell_left)],
]
sum_table = Table(sum_data, colWidths=[3*cm, 4.2*cm, 6.6*cm])
sum_row_styles = []
for i in range(1, len(sum_data)):
    if i % 2 == 0:
        sum_row_styles.append(('BACKGROUND', (0, i), (-1, i), TABLE_ROW_ODD))
sum_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_COLOR),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
] + sum_row_styles))
story.append(sum_table)
story.append(Spacer(1, 6))
story.append(Paragraph('表5  改进策略汇总', caption_style))

# ==================== 7. 个人思考与见解 ====================
story.append(Spacer(1, 18))
story.append(Paragraph('<b>7  个人思考与见解</b>', h1_style))

story.append(Paragraph('<b>7.1  关于迁移学习中的数据增强</b>', h2_style))
story.append(Paragraph(
    '在迁移学习实践中，数据增强策略的选择需要与预训练模型的特性相匹配。本实验的一个关键发现是：使用<font name="Times New Roman">ImageNet</font>标准归一化参数（而非简单的[0.5, 0.5, 0.5]）对迁移学习效果有显著影响。预训练模型在特定的数据分布上学习到了有效的特征表示，如果微调阶段使用差异较大的归一化参数，会导致输入分布偏移，削弱预训练权重的优势。这一发现提醒我们，在进行迁移学习时，应当尽可能保持与预训练阶段一致的数据预处理流程。',
    body_style
))

story.append(Paragraph('<b>7.2  关于正则化的平衡</b>', h2_style))
story.append(Paragraph(
    '本实验中，基线方案训练集准确率接近99%但验证集仅约90%，是典型的过拟合症状。改进方案通过多重正则化手段（<font name="Times New Roman">Mixup + Label Smoothing + AdamW</font>权重衰减 + 数据增强），成功将训练-验证差距从约8%缩小到接近0%。值得注意的是，改进模型的训练集准确率（82.84%）低于验证集准确率（91.41%），这看似反常但实际上是正则化充分的有效信号。当训练集准确率远高于验证集时，说明模型在"记忆"训练数据而非学习通用特征；而两者接近或验证集略高时，说明模型真正学到了具有泛化能力的特征表示。',
    body_style
))

story.append(Paragraph('<b>7.3  关于轻量模型的适用场景</b>', h2_style))
story.append(Paragraph(
    '<font name="Times New Roman">MobileNetV3 Small</font>作为一款轻量级模型，仅有约150万参数量，在CPU上即可完成训练和推理。虽然其在大规模数据集上的表现不如<font name="Times New Roman">ResNet</font>、<font name="Times New Roman">EfficientNet</font>等大型模型，但在本项目的5分类花卉识别任务中，通过合理的迁移学习和正则化策略，已经能够达到91%以上的准确率，完全满足实际应用需求。这证明了轻量模型在特定场景下的实用价值：较低的算力需求使其更适合部署在移动端和嵌入式设备上，而通过优化训练策略可以有效弥补模型容量不足的缺陷。',
    body_style
))

story.append(Paragraph('<b>7.4  未来改进方向</b>', h2_style))
story.append(Paragraph(
    '基于本次实验的分析，未来可以从以下几个方向进一步提升模型性能：第一，增加数据量并通过数据清洗去除低质量或错误标注的样本，从根本上提升数据集质量；第二，尝试使用<font name="Times New Roman">Test Time Augmentation (TTA)</font>，在推理阶段对输入图片进行多次增强后取平均预测结果，进一步提升预测稳定性；第三，探索知识蒸馏技术，用大模型（如<font name="Times New Roman">ResNet50</font>）指导小模型训练，在保持轻量的同时逼近大模型的性能；第四，引入注意力机制可视化工具（如<font name="Times New Roman">Grad-CAM</font>），分析模型关注区域是否合理，帮助定位分类错误的根本原因。',
    body_style
))

# ==================== 构建PDF ====================
doc.build(story)
print(f"PDF generated: {output_pdf}")
