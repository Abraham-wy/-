"""
Baseline训练脚本 - 复现原项目训练流程
MobileNetV3 Small + 迁移学习 + 5分类花卉识别
"""
import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader

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

# ==================== 数据路径 ====================
data_path = '/home/z/my-project/mobilenet_project/work_data/flower_split/'
save_dir = '/home/z/my-project/mobilenet_project/work_data/model/weights/'
os.makedirs(save_dir, exist_ok=True)

# ==================== 数据预处理 (原始项目的简单方式) ====================
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载数据集
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"Classes: {train_dataset.classes}")
print(f"Class to idx: {train_dataset.class_to_idx}")
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# 保存类别映射
index_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
with open(os.path.join(save_dir, 'index_to_class.json'), 'w', encoding='utf-8') as f:
    json.dump(index_to_class, f, indent=4, ensure_ascii=False)

# ==================== 模型构建 ====================
def create_model(classes_num=5):
    """加载预训练MobileNetV3 Small并修改分类头"""
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, classes_num)
    return model.to(device)

model = create_model(classes_num=5)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}, Trainable params: {trainable_params:,}")

# ==================== 训练参数 (与原始项目一致) ====================
lr = 0.0001
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ==================== 训练函数 ====================
def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (torch.argmax(outputs, dim=1) == labels).sum().item() / len(labels)
        num_batches += 1

    return running_loss / num_batches, running_acc / num_batches

@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item()
        running_acc += (torch.argmax(outputs, dim=1) == labels).sum().item() / len(labels)
        num_batches += 1

    return running_loss / num_batches, running_acc / num_batches

# ==================== 开始训练 ====================
print("\n" + "=" * 60)
print("BASELINE TRAINING STARTED")
print("=" * 60)

best_acc = 0.0
best_epoch = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
    scheduler.step()
    val_loss, val_acc = evaluate(model, val_loader, loss_fn)

    history['train_loss'].append(round(tr_loss, 4))
    history['train_acc'].append(round(tr_acc, 4))
    history['val_loss'].append(round(val_loss, 4))
    history['val_acc'].append(round(val_acc, 4))

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(save_dir, 'baseline_best.pth'))
        print(f"  >> Saved best model (val_acc={best_acc:.4f})")

print("\n" + "=" * 60)
print("BASELINE TRAINING COMPLETED")
print(f"Best Epoch: {best_epoch}")
print(f"Best Val Acc: {best_acc:.4f}")
print(f"Final Train Acc: {history['train_acc'][-1]:.4f}")
print(f"Final Val Acc: {history['val_acc'][-1]:.4f}")
print("=" * 60)

# 保存训练历史
with open(os.path.join(save_dir, 'baseline_history.json'), 'w') as f:
    json.dump(history, f, indent=4)
print(f"History saved to baseline_history.json")
