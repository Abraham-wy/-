"""
改进版训练脚本 - 多策略优化
改进点:
1. 增强数据增强 (RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter, Affine)
2. Mixup数据增强 (alpha=0.2)
3. Label Smoothing (smoothing=0.1)
4. CosineAnnealingLR + Warmup学习率调度
5. AdamW优化器 + 权重衰减
6. ImageNet标准化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
7. 完整评估 (混淆矩阵、每类准确率)
"""
import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# ==================== 可复现性 ====================
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==================== 路径 ====================
data_path = '/home/z/my-project/mobilenet_project/work_data/flower_split/'
save_dir = '/home/z/my-project/mobilenet_project/work_data/model/weights/'
os.makedirs(save_dir, exist_ok=True)

# ==================== 数据预处理 (改进版) ====================
# ImageNet标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 训练集: 丰富的数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# 验证集: 只做尺寸调整和标准化
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Classes: {train_dataset.classes}")
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# ==================== Mixup ====================
def mixup_data(x, y, alpha=0.2):
    """Mixup: 对batch内样本进行线性插值混合"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失: 两标签的加权混合"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== 模型构建 ====================
def create_model(classes_num=5):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, classes_num)
    return model.to(device)

model = create_model(classes_num=5)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}")

# ==================== 训练参数 (改进版) ====================
lr = 0.0001
epochs = 10
# 改进1: Label Smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
# 改进2: AdamW + weight_decay
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# 改进3: Warmup + CosineAnnealing
def get_lr_lambda(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性warmup: 从1e-6到lr
            return (epoch + 1) / warmup_epochs
        else:
            # CosineAnnealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_lambda

scheduler = LambdaLR(optimizer, lr_lambda=get_lr_lambda(warmup_epochs=2, total_epochs=epochs))

# ==================== 训练函数 ====================
def train_one_epoch(model, train_loader, loss_fn, optimizer, use_mixup=True, mixup_alpha=0.2, mixup_prob=0.5):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Mixup: 以一定概率应用
        if use_mixup and random.random() < mixup_prob:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
            outputs = model(mixed_images)
            loss = mixup_criterion(loss_fn, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc, all_preds, all_labels

# ==================== 开始训练 ====================
print("\n" + "=" * 60)
print("IMPROVED TRAINING STARTED")
print("Improvements: DataAug+Mixup+LabelSmooth+Warmup+CosineLR+AdamW")
print("=" * 60)

best_acc = 0.0
best_epoch = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

for epoch in range(epochs):
    current_lr = optimizer.param_groups[0]['lr']
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                       use_mixup=True, mixup_alpha=0.2, mixup_prob=0.5)
    scheduler.step()
    val_loss, val_acc, _, _ = evaluate(model, val_loader, loss_fn)

    history['train_loss'].append(round(tr_loss, 4))
    history['train_acc'].append(round(tr_acc, 4))
    history['val_loss'].append(round(val_loss, 4))
    history['val_acc'].append(round(val_acc, 4))
    history['lr'].append(round(current_lr, 6))

    print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(save_dir, 'improved_best.pth'))
        print(f"  >> Saved best model (val_acc={best_acc:.4f})")

# ==================== 最终评估 (加载最优模型) ====================
print("\n" + "-" * 60)
print("Loading best model for final evaluation...")
model.load_state_dict(torch.load(os.path.join(save_dir, 'improved_best.pth'), map_location=device, weights_only=True))
val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, nn.CrossEntropyLoss())
print(f"Best Model Val Acc: {val_acc:.4f}")

# 每类准确率
classes = val_dataset.classes
num_classes = len(classes)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
per_class_correct = np.zeros(num_classes)
per_class_total = np.zeros(num_classes)

for pred, label in zip(all_preds, all_labels):
    confusion_matrix[label][pred] += 1
    per_class_total[label] += 1
    if pred == label:
        per_class_correct[label] += 1

per_class_metrics = {}
print("\nPer-class Accuracy:")
for i, cls in enumerate(classes):
    acc = per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
    per_class_metrics[cls] = {
        'correct': int(per_class_correct[i]),
        'total': int(per_class_total[i]),
        'accuracy': round(float(acc), 4)
    }
    print(f"  {cls}: {acc:.4f} ({int(per_class_correct[i])}/{int(per_class_total[i])})")

# 保存混淆矩阵
cm_path = os.path.join(save_dir, 'improved_confusion_matrix.csv')
with open(cm_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([''] + classes)
    for i, cls in enumerate(classes):
        writer.writerow([cls] + confusion_matrix[i].tolist())
print(f"\nConfusion matrix saved to {cm_path}")

# 保存每类指标
pcm_path = os.path.join(save_dir, 'improved_per_class_metrics.json')
with open(pcm_path, 'w') as f:
    json.dump(per_class_metrics, f, indent=4)

# ==================== 保存历史 ====================
with open(os.path.join(save_dir, 'improved_history.json'), 'w') as f:
    json.dump(history, f, indent=4)

print("\n" + "=" * 60)
print("IMPROVED TRAINING COMPLETED")
print(f"Best Epoch: {best_epoch}")
print(f"Best Val Acc: {best_acc:.4f}")
print(f"Final Train Acc: {history['train_acc'][-1]:.4f}")
print(f"Final Val Acc: {history['val_acc'][-1]:.4f}")
print("=" * 60)

# ==================== 对比 ====================
with open(os.path.join(save_dir, 'baseline_history.json'), 'r') as f:
    baseline = json.load(f)

baseline_best = max(baseline['val_acc'])
comparison = {
    "baseline": {
        "best_val_acc": baseline_best,
        "best_epoch": baseline['val_acc'].index(baseline_best) + 1,
        "final_train_acc": baseline['train_acc'][-1],
        "final_val_acc": baseline['val_acc'][-1]
    },
    "improved": {
        "best_val_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "final_train_acc": history['train_acc'][-1],
        "final_val_acc": history['val_acc'][-1],
        "per_class_accuracy": per_class_metrics
    },
    "improvement": {
        "val_acc_gain": round(best_acc - baseline_best, 4),
        "val_acc_gain_percent": round((best_acc - baseline_best) / baseline_best * 100, 2)
    }
}

with open(os.path.join(save_dir, 'experiment_comparison.json'), 'w') as f:
    json.dump(comparison, f, indent=4)

print(f"\nComparison: Baseline {baseline_best:.4f} -> Improved {best_acc:.4f} "
      f"(+{best_acc - baseline_best:.4f}, +{(best_acc - baseline_best)/baseline_best*100:.2f}%)")
