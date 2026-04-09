# 载入与模型网络构建
import os
import json
import numpy as np
import glob
import cv2
import random
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas
from tqdm import tqdm
import sys
import pickle
import pandas as pd

def mobilenet_v3_s_model(classes_num=5, download=False, freeze=False, mode='train'):
    """
    :param classes_num: 数据的类别个数
    :param download: 是否从官网上下载预训练权重
    :param freeze: 是否冻结权重
    :param mode: {'train': 模型训练使用, 'predict': 模型预测使用}
    :return: efficient_v2_s_model
    """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print('device: ', device)

    if mode == 'train':
        if download:  # 若`download=True`, 则会自动将权重下载导本地用户缓存(会显示路径), 下载成功与否取决于当前网络
            # 第一次加载会自动下载权重
            # 查看该权重要求的输入, 主要查看的有：[crop_size, mean, std], 读取自己的数据时要用到
            print('train transforms requires: \n', MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms())
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            # 权重所在路径
            weights_path = 'D:/work_data/model/weights/mobilenet_v3_small_pretrain.pth'
            assert os.path.exists(weights_path), \
                '请将预训练权重放至当前工程的`weights`文件夹下, 并将文件名改为`mobilenet_v3_small_pretrain.pth`'
            weights = torch.load(weights_path)  # 读取权重, 这里加载到
            model = mobilenet_v3_small()         # 实例化模型
            model.load_state_dict(state_dict=weights)  # 模型加载权重
    if mode == 'predict':
        model = mobilenet_v3_small()  # 实例化模型

    # 修改模型的输出层(主要修改输出类别数量)
    in_channel = model.classifier[3].in_features
    # 直接将输出类别数改成我们自己数据集的类别数, 我们`classes_num=5`
    model.classifier[3] = nn.Linear(in_features=in_channel, out_features=classes_num)

    if freeze:
        # 冻结权重, 仅训练最后的`fc`层, 但模型效果较差; 若训练全部层, 效果较好, 不过电脑内存要求比较高
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    return model.to(device)

def train_one_epoch(model, train_dl, loss_fn, optimizer):
    model.train()  # 训练模式
    tr_acc = 0.
    tr_loss = 0.
    train_dl = tqdm(train_dl, file=sys.stdout)
    for x, y in train_dl:
        optimizer.zero_grad()      # 梯度清零
        output = model(x)          # 模型训练
        loss = loss_fn(output, y)  # 计算损失
        loss.backward()            # 反向传播计算梯度
        optimizer.step()           # 梯度更新

        tr_loss += loss.item()  # torch的loss会自动计算一次平均
        tr_acc += (torch.eq(torch.argmax(output, dim=1), y).sum() / len(y)).item()

    return tr_loss/len(train_dl), tr_acc/len(train_dl)


@torch.no_grad()
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

    return val_loss/len(val_dl), val_acc/len(val_dl)


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
        plt.savefig('D:/work_data/model/weights/训练结果.jpg')
    plt.show()


def model_train(train_dl, val_dl, epochs, num_classes):
    # ----------模型搭建-----------
    # 实例化模型
    model = mobilenet_v3_s_model(classes_num=num_classes)

    # 定义超参数
    lr = 0.0001  # 学习率
    # epochs = 10  # 训练周期
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器

    best_acc = 0.0
    train_loss, train_acc = [], []  # 存放训练结果用于可视化
    validation_loss, validation_acc = [], []

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, loss_fn, optimizer)  # 训练
        val_loss, val_acc = evaluate(model, val_dl, loss_fn)  # 评估

        train_loss.append(tr_loss), train_acc.append(tr_acc)
        validation_loss.append(val_loss), validation_acc.append(val_acc)

        print(f'epoch {epoch + 1}/{epochs}  \n'
              f'train_loss {tr_loss:.3f}  train_acc {tr_acc:.3f}  '
              f'test_loss {val_loss:.3f}  test_acc {val_acc:.3f}')

        # 保存权重(只保留验证集acc最大的)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'D:/work_data/model/weights/model.pth')
            print(f'saved epochs: {epoch + 1}  best validation acc: {best_acc:.3f}')

    # 可视化结果
    plot(train_loss, validation_loss, train_acc, validation_acc,
         epochs=epochs, lr=lr, save=True)

    print('训练完毕！'.center(30, '-'))
    # 保存网络
    torch.save(model.state_dict(), 'D:/work_data/model/weights/mobilenet_v3_small_mytrain.pth')
    print('模型保存成功！'.center(30, '-'))
    print(validation_acc)
    accuracy = max(validation_acc)
    print('模型准确率:', accuracy)
    return accuracy


@st.cache_data
def model_pred(img, img_name, weights='D:/work_data/model/weights/mobilenet_v3_small_mytrain.pth'):
    """
    模型预测
    :param img: 预测图片
    :param img_name: 预测图片的文件名
    :param weights: 模型所在路径
    :return:
    """
    # 读取索引文件
    with open('D:/work_data/model/weights/index_to_class.json') as f:
        index_to_class = {int(k): v for k, v in json.load(f).items()}

    # 实例化模型
    model = mobilenet_v3_s_model(classes_num=len(index_to_class), mode='predict')
    # model.state_dict()
    # 加载模型权重
    weight = torch.load(weights, map_location='cpu')
    model.load_state_dict(weight)

    # 读取一张图片
    # img = cv2.imread(predimg_path)
    h, w = img.shape[0], img.shape[1]
    img = img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9), :]
    img = cv2.resize(img, (224, 224))   # 缩放
    img = img / 255.0  # 标准化
    img = np.transpose(img, (2, 0, 1))  # 通道转换
    img = np.expand_dims(img, axis=0)   # 添加`batch`维度
    img = torch.as_tensor(img, dtype=torch.float32)  # 转换成`tensor`

    # 预测
    model.eval()  # 进入评估模式, 一定要进入评估模式
    op = torch.argmax(model(img), dim=1).item()
    print(f'` {img_name} `的预测结果是', index_to_class[op])

    return index_to_class[op]

def main(img_path, model_path, sidebarimg_path='D:/work_data/model/weights/image.jpg'):
    # -------------页面设置----------
    st.set_page_config(page_title='花种类识别器', layout="centered")
    st.title('基于PyTorch的图像识别器')

    # 导入数据
    with open('D:/work_data/model/weights/index_to_class.json', 'r', encoding='utf-8') as f:
        classes_to_index = json.load(f)
    with open('D:/work_data/model/weights/train_dl.pkl', 'rb') as f:
        train_dl = pickle.load(f)
    with open('D:/work_data/model/weights/val_dl.pkl', 'rb') as f:
        val_dl = pickle.load(f)

    # ----------模型训练-----------
    st.sidebar.image(sidebarimg_path, width=300)
    st.sidebar.write('# 选择一个模型？or 重新训练训练？📌')
    st.sidebar.write('提示：重新训练很耗时！')
    if_train = st.sidebar.selectbox('是否训练模型？', ['训练新的模型', '导入已有的模型'])
    if if_train == '训练新的模型':
        nepochs = int(st.sidebar.selectbox('选择训练次数', [5, 10, 20, 30, 40, 50]))
        if os.path.exists(model_path):
            st.sidebar.success('* 已有训练好的模型，请决定是否重新训练')
        else:
            st.sidebar.warning('* 未存在已训练好的模型，请先训练一个模型~')
        if st.sidebar.button('--> 点击开始训练🏃🏽‍♂ <--'):
            # 算法参数
            col1, col2 = st.columns(2)
            with col1:
                st.success('* 正在训练模型,速度慢请稍候...')
            # 模型训练
            accuracy = model_train(train_dl, val_dl, nepochs, num_classes=len(classes_to_index))
            with col2:
                st.success('* 模型训练保存完毕。')
            # 测试集准确率
            st.error('#### 模型准确率为📢:{}'.format(accuracy))
    else:
        # 提示模型文件是否存在
        if os.path.exists(model_path):
            st.sidebar.success('* 已有训练好的模型可导入')

        else:
            st.sidebar.warning('* 未存在已训练好的模型，请先训练一个模型~')
            st.error('模型还未存在，若要训练新模型，请单击左侧 "开始训练" ')

    st.write('## 从图片路径上传图片')
    content_file = st.file_uploader('')

    try:
        # 获取图片地址
        # style_file_path = os.path.join('D:/work_data/model/weights/test_images', content_file.name)
        # 获取二进制编码
        bytes_data = content_file.getvalue()
        # 将二进制编码转换成数组
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        # 显示输入的图片
        st.image(content_file, use_column_width=True)
    except:
        pass

    # 判断是否传入了图片，没有则提示，有则进行识别
    if content_file is not None:
        # 预测结果
        pred = model_pred(cv2_img, content_file.name, model_path)
        # 打印图片和预测结果
        st.success('我猜它是：{} 对吗？'.format(pred))
    else:
        st.error("还未上传图片！")


if __name__ == "__main__":
    main(img_path="D:/work_data/flower_split/flower/", 
         model_path="D:/work_data/model/weights/mobilenet_v3_small_mytrain.pth",
         sidebarimg_path='D:/work_data/model/weights/image.jpg')
