import torch
from tqdm import tqdm
import sys
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import os, shutil
import os.path
import pickle
import json
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchsummary import summary
import pandas as pd

print(torch.__version__, torchvision.__version__)
# 数据转换
transform = transforms.Compose(
    [transforms.CenterCrop(224), # 图片大小为224*224
     transforms.ToTensor(), # 转换为0-1的张量
     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) # 标准化 -1到1
    ]
)
file_path = 'D:\\work_data\\flower_split\\'
Image_dataset = {
    x:datasets.ImageFolder(
        root = os.path.join(file_path, x),
        transform = transform
    )
    for x in ['train','val']
}
print(Image_dataset)
Image_data_loader = {
    x:torch.utils.data.DataLoader(
        dataset = Image_dataset[x],
        batch_size = 64, # 64张图一个批次读取
        shuffle = True
    )
    for x in ['train','val']
}
print(Image_data_loader['train'])
# 查看标签
classes = Image_dataset['train'].classes
print(classes)
classes_index = Image_dataset['train'].class_to_idx
print(classes_index)
print(len(Image_dataset['train']))
print(len(Image_dataset['val']))
# 查看一个批次的数据
imgs, labels = next(iter(Image_data_loader['train']))
print('labels=',labels,'\n'
     'labels.shape=', labels.shape, '\n'
     'imgs.shape=',imgs.shape)
print(imgs[0])
# 图像可视化
X_train, Y_train = next(iter(Image_data_loader['train']))
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
img = torchvision.utils.make_grid(X_train)
img = img.numpy().transpose((1,2,0))
img = img*std + mean
plt.imshow(img)
plt.show()

def mobilenet_v3_s_model(classes_num = 5, download = False,freeze = False, mode = 'train'):
    device = 'cpu'
    print('device:',device)

    if mode == 'train':
        if download:
            print('train transforms requires:\n',MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
            model = mobilenet_v3_small(weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            weights_path = 'D:/work_data/model/weights/mobilenet_v3_small_pretrain.pth'
            assert  os.path.exists(weights_path),'请将预训练权重放至当前工程的weights文件夹下，并将文件名改为mobilenet_v3_small_pretrain.pth'
            weights = torch.load(weights_path)
            model = mobilenet_v3_small()
            model.load_state_dict(state_dict=weights)
    if mode =='predict':
        model = mobilenet_v3_small()

    #修改模型的输出层
    in_channel = model.classifier[3].in_features
    # 直接将输出类别修改成这次项目的类别数，即classes_num=5
    model.classifier[3] = nn.Linear(in_features=in_channel, out_features=classes_num)

    if freeze:
        # 冻结权重，仅训练最后的fc层，模型效果一般；若训练全部层，需要的配置要求较高，效果较好
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    return model.to(device)
model = mobilenet_v3_s_model(classes_num = 5,
                            download=False,
                            freeze=False,
                            mode = 'train')
# 查看模型输出
x = torch.randn(1,3,224,224)
print(model(x))
# 查看模型架构
print(model)
print(model.features)


summary(model, input_size=(3,224,224), device='cpu')
train_dataloader = Image_data_loader['train']
val_dataloader = Image_data_loader['val']
classes_to_index = classes_index
print(classes_index)
# 保存数据
with open('D:/work_data/model/weights/train_dl.pkl','wb') as f:
    pickle.dump(train_dataloader, f)
with open('D:/work_data/model/weights/val_dl.pkl','wb') as f:
    pickle.dump(val_dataloader, f)
with open('D:/work_data/model/weights/index_to_class.json', mode='w', encoding='utf-8') as f:
    json.dump({v: k for k,v in classes_to_index.items()}, f, indent = 4)
lr = 0.0001
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 调度器


def train_one_epoch(model, train_dl, loss_fn, optimizer):
    model.train()  # 训练模式
    tr_acc = 0.
    tr_loss = 0.
    train_dl = tqdm(train_dl, file=sys.stdout)
    for x, y in train_dl:
        optimizer.zero_grad()  # 梯度清零
        output = model(x)  # 模型训练
        loss = loss_fn(output, y)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 梯度更新

        tr_loss += loss.item()  # torch的loss会自动计算一次平均
        tr_acc += (torch.eq(torch.argmax(output, dim=1), y).sum() / len(y)).item()
    return tr_loss / len(train_dl), tr_acc / len(train_dl)


lr = 0.0001
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 调度器


def train_one_epoch(model, train_dl, loss_fn, optimizer):
    model.train()  # 训练模式
    tr_acc = 0.
    tr_loss = 0.
    train_dl = tqdm(train_dl, file=sys.stdout)
    for x, y in train_dl:
        optimizer.zero_grad()  # 梯度清零
        output = model(x)  # 模型训练
        loss = loss_fn(output, y)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 梯度更新

        tr_loss += loss.item()  # torch的loss会自动计算一次平均
        tr_acc += (torch.eq(torch.argmax(output, dim=1), y).sum() / len(y)).item()
    return tr_loss / len(train_dl), tr_acc / len(train_dl)


def evaluate(model, val_dl, loss_fn):
    model.eval()  # 评估模式
    val_acc = 0.
    val_loss = 0.
    val_dl = tqdm(val_dl, file=sys.stdout)
    for x, y in val_dl:
        output = model(x)  # 模型训练
        loss = loss_fn(output, y)  # 计算损失

        val_loss += loss.item()  # torch的loss会自动计算一次平均
        val_acc += (torch.eq(torch.argmax(output, dim=1), y).sum() / len(y)).item()
    return val_loss / len(val_dl), val_acc / len(val_dl)


best_acc = 0.0
train_loss, train_acc = [], []
validation_loss, validation_acc = [], []

for epoch in range(epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_dataloader, loss_fn, optimizer)
    scheduler.step()
    val_loss, val_acc = evaluate(model, val_dataloader, loss_fn)

    train_loss.append(tr_loss)
    train_acc.append(tr_acc)
    validation_loss.append(val_loss)
    validation_acc.append(val_acc)

    print(f'epoch {epoch + 1}/{epochs} \n'
          f'train_loss {tr_loss:.3f} train_acc {tr_acc:.3f}' f'val_loss {val_loss:.3f} val_acc {val_acc:.3f}')

    # 保存权重（只保留验证集acc最大的）
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'D:/work_data/model/weights/mobilenet_v3_small_mytrain_1.pth')
        print(f'saved epochs:{epoch + 1} best validation acc:{best_acc:.3f}')
def plot(train_loss, validation_loss, train_acc, validation_acc, epochs=20, lr=0.001, save=False):
    his = pd.DataFrame({
        'train_loss': train_loss,
        'val_loss': validation_loss,
        'train_acc': train_acc,
        'val_acc': validation_acc,
    })

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(his['train_loss'], label='train_loss')
    plt.plot(his['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(f'epoch={epochs}, lr={lr}')

    plt.subplot(1, 2, 2)
    plt.plot(his['train_acc'], label='train_acc')
    plt.plot(his['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.title(f'epoch={epochs}, lr={lr}')
    if save:
        plt.savefig('./训练结果.jpg')
    plt.show()
# 可视化结果
plot(train_loss, validation_loss, train_acc, validation_acc,
     epochs=10, lr=lr,save=False)