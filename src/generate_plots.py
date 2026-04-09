"""生成可视化图表：训练曲线对比 + 混淆矩阵"""
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

save_dir = '/home/z/my-project/mobilenet_project/work_data/model/weights/'

# ==================== 加载数据 ====================
with open(f'{save_dir}baseline_history.json') as f:
    baseline = json.load(f)
with open(f'{save_dir}improved_history.json') as f:
    improved = json.load(f)

epochs_range = range(1, len(baseline['train_loss']) + 1)

# ==================== 图1: Loss曲线对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1 = axes[0]
ax1.plot(epochs_range, baseline['train_loss'], 'b-o', markersize=5, label='Baseline Train Loss')
ax1.plot(epochs_range, baseline['val_loss'], 'b--s', markersize=5, label='Baseline Val Loss')
ax1.plot(epochs_range, improved['train_loss'], 'r-o', markersize=5, label='Improved Train Loss')
ax1.plot(epochs_range, improved['val_loss'], 'r--s', markersize=5, label='Improved Val Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Curve Comparison', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy
ax2 = axes[1]
ax2.plot(epochs_range, baseline['train_acc'], 'b-o', markersize=5, label='Baseline Train Acc')
ax2.plot(epochs_range, baseline['val_acc'], 'b--s', markersize=5, label='Baseline Val Acc')
ax2.plot(epochs_range, improved['train_acc'], 'r-o', markersize=5, label='Improved Train Acc')
ax2.plot(epochs_range, improved['val_acc'], 'r--s', markersize=5, label='Improved Val Acc')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy Curve Comparison', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}training_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_comparison.png")

# ==================== 图2: 改进版混淆矩阵 ====================
classes = ['dandelion', 'frangipani', 'morning_glory', 'peony', 'sunflower']
classes_cn = ['Dandelion', 'Frangipani', 'Morning Glory', 'Peony', 'Sunflower']

cm = np.zeros((5, 5), dtype=int)
with open(f'{save_dir}improved_confusion_matrix.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for i, row in enumerate(reader):
        for j, val in enumerate(row[1:]):
            cm[i][j] = int(val)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes_cn,
       yticklabels=classes_cn,
       title='Improved Model - Confusion Matrix',
       ylabel='True Label',
       xlabel='Predicted Label')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 在每个格子上标注数字
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color, fontsize=12)

plt.tight_layout()
plt.savefig(f'{save_dir}confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# ==================== 图3: 每类准确率柱状图 ====================
with open(f'{save_dir}improved_per_class_metrics.json') as f:
    per_class = json.load(f)

# Baseline per-class (从confusion matrix推导)
# 简单方式：用baseline的history中最好的epoch来评估
# 直接用已有的per_class对比
class_names = list(per_class.keys())
accs = [per_class[c]['accuracy'] for c in class_names]
totals = [per_class[c]['total'] for c in class_names]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(class_names)), accs, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'])
ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(['Dandelion', 'Frangipani', 'Morning\nGlory', 'Peony', 'Sunflower'], fontsize=11)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Improved Model - Per-class Accuracy', fontsize=14)
ax.set_ylim(0.8, 1.0)
ax.axhline(y=np.mean(accs), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(accs):.4f}')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# 在柱子上标注数值
for idx, (bar, acc, total) in enumerate(zip(bars, accs, totals)):
    cls_name = class_names[idx]
    correct = per_class[cls_name]['correct']
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
            f'{acc:.2%}\n({correct}/{total})',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{save_dir}per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: per_class_accuracy.png")

# ==================== 图4: 训练验证差距(过拟合分析) ====================
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.35
x = np.arange(len(epochs_range))

# 计算train-val gap
baseline_gap = [t - v for t, v in zip(baseline['train_acc'], baseline['val_acc'])]
improved_gap = [t - v for t, v in zip(improved['train_acc'], improved['val_acc'])]

ax.bar(x - width/2, baseline_gap, width, label='Baseline (Train-Val)', color='#2196F3', alpha=0.8)
ax.bar(x + width/2, improved_gap, width, label='Improved (Train-Val)', color='#F44336', alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy Gap (Train - Val)', fontsize=12)
ax.set_title('Overfitting Analysis: Train-Val Accuracy Gap', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(epochs_range)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: overfitting_analysis.png")

print("\nAll visualizations generated!")
