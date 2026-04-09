# 基于深度学习的花卉识别系统

## 项目简介

本项目是一个基于深度学习的花卉图像识别系统，使用PyTorch框架和MobileNetV3 Small预训练模型实现5种常见花卉的分类识别。项目包含完整的数据爬取、预处理、模型训练和Web应用部署流程。

## 识别的花卉类别

项目可以识别以下5种花卉：
1. **蒲公英** (dandelion)
2. **白色鸡蛋花** (frangipani)
3. **牵牛花** (morning_glory)
4. **牡丹** (peony)
5. **向日葵** (sunflower)

## 项目结构

```
基于深度学习的花卉识别/
├── src/                                # 主要代码文件夹
│   ├── 1爬取图片.py                     # 图片爬取脚本
│   ├── 2数据预处理.py                   # 数据预处理脚本
│   ├── 3数据集类 dataset与建模.py       # 模型训练脚本
│   └── 4app.py                          # Streamlit Web应用
├── work_data/                           # 数据文件夹
│   ├── 2408flower/                      # 原始图片数据
│   │   ├── dandelion/                   # 蒲公英图片
│   │   ├── peony/                       # 牡丹图片
│   │   ├── frangipani/                  # 鸡蛋花图片
│   │   ├── sunflower/                   # 向日葵图片
│   │   └── morning_glory/               # 牵牛花图片
│   ├── flower_split/                    # 划分后的数据集（运行后生成）
│   │   ├── train/                       # 训练集（80%）
│   │   └── val/                         # 验证集（20%）
│   └── model/                           # 模型文件夹
│       └── weights/                     # 模型权重
│           ├── index_to_class.json      # 类别索引映射
│           ├── train_dl.pkl             # 训练数据加载器
│           ├── val_dl.pkl               # 验证数据加载器
│           ├── mobilenet_v3_small_pretrain.pth  # 预训练权重
│           └── mobilenet_v3_small_mytrain.pth   # 训练后的模型权重
└── 运行前注意事项.txt                   # 路径修改提示
```

## 技术栈

### 深度学习框架
- **PyTorch**: 深度学习框架
- **torchvision**: 计算机视觉工具包，提供预训练模型和数据集处理
- **torchsummary**: 模型结构可视化

### 数据处理
- **Pandas**: 数据分析和处理
- **NumPy**: 数值计算
- **OpenCV (cv2)**: 图像处理
- **Matplotlib**: 数据可视化
- **scikit-learn**: 数据集划分（train_test_split）

### Web应用
- **Streamlit**: 快速构建机器学习Web应用

### 数据爬取
- **requests**: HTTP请求
- **lxml**: HTML解析
- **tqdm**: 进度条显示

## 项目流程

### 1. 数据爬取阶段（1爬取图片.py）

**功能**: 从Bing图片搜索爬取花卉图片

**技术要点**:
- 使用requests库发送HTTP请求
- 使用lxpath解析HTML获取图片URL
- 批量下载图片并保存到本地
- 每种花爬取约400张图片

**核心代码逻辑**:
```python
# 构造搜索URL，使用分页参数
for i in tqdm(range(1, 401, 10)):
    url = re.sub('first=1', f'first={i}', URL)
    res = requests.get(url, headers=header)
    # 解析图片地址
    pic_adress = f_adress.xpath('//*[@id="mmComponent_images_1"]/ul/li/div/div/a/@m')
```

**运行前需要修改**:
- 第41行：修改图片保存路径 `flower_path`

### 2. 数据预处理阶段（2数据预处理.py）

**功能**: 将原始图片按8:2比例划分为训练集和验证集

**技术要点**:
- 使用glob遍历所有图片
- 使用sklearn的train_test_split进行数据集划分
- 将图片复制到对应的train/val文件夹
- 统计各类别图片数量并可视化

**核心代码逻辑**:
```python
# 数据集划分
train, val = train_test_split(pic, train_size=0.8, shuffle=True, random_state=42)
# 复制图片到对应文件夹
shutil.copy(i, output_path + 'train' + '\\' + class_name + '\\' + pic_index_i)
```

**运行前需要修改**:
- 第6行：修改原始图片路径
- 第62行：修改输出路径

### 3. 模型训练阶段（3数据集类 dataset与建模.py）

**功能**: 使用MobileNetV3 Small进行迁移学习训练

**技术要点**:

#### 数据增强与预处理
```python
transform = transforms.Compose([
    transforms.CenterCrop(224),      # 中心裁剪到224x224
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # 标准化
])
```

#### 模型架构
- **基础模型**: MobileNetV3 Small（轻量级CNN）
- **预训练权重**: ImageNet1K_V1
- **迁移学习**: 修改最后一层全连接层，输出改为5类
- **是否冻结**: 可选择冻结特征提取层，仅训练分类器

**模型修改逻辑**:
```python
# 修改模型的输出层
in_channel = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features=in_channel, out_features=classes_num)
```

#### 训练参数
- **学习率**: 0.0001
- **批次大小**: 64
- **训练轮数**: 10 epochs
- **优化器**: Adam
- **学习率调度器**: StepLR（每5个epoch学习率乘以0.5）
- **损失函数**: CrossEntropyLoss

**训练流程**:
1. 加载预训练模型
2. 修改输出层为5类
3. 定义损失函数和优化器
4. 训练循环：前向传播→计算损失→反向传播→参数更新
5. 验证集评估
6. 保存验证集准确率最高的模型权重
7. 可视化训练过程（loss和acc曲线）

**运行前需要修改**:
- 第27行：修改数据集路径
- 第77行：修改预训练权重路径
- 第114-118行：修改数据保存路径
- 第202行：修改模型保存路径

### 4. Web应用阶段（4app.py）

**功能**: 使用Streamlit构建交互式Web应用

**功能模块**:

#### 模型训练模块
- 支持选择训练轮数（5/10/20/30/40/50）
- 实时显示训练进度
- 显示模型准确率
- 自动保存训练好的模型

#### 图像预测模块
- 支持上传图片进行识别
- 显示上传的图片
- 返回预测结果（花卉类别）

**核心代码逻辑**:
```python
# 预测函数
def model_pred(img, img_name, weights):
    # 加载模型和权重
    model = mobilenet_v3_s_model(classes_num=5, mode='predict')
    weight = torch.load(weights, map_location='cpu')
    model.load_state_dict(weight)
    
    # 图像预处理
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.as_tensor(img, dtype=torch.float32)
    
    # 预测
    model.eval()
    op = torch.argmax(model(img), dim=1).item()
    return index_to_class[op]
```

**运行前需要修改**:
- 第41行：修改预训练权重路径
- 第122行：修改模型路径
- 第155行：修改侧边栏图片路径
- 第164行：修改数据加载器路径
- 第173行：修改类别映射文件路径
- 第182行：修改模型保存路径
- 第209行：修改侧边栏图片路径
- 第215-219行：修改各类文件路径
- 第278行：修改主函数参数路径

## 环境配置

### 依赖库安装

```bash
# 核心深度学习库
pip install torch torchvision

# 数据处理库
pip install pandas numpy opencv-python matplotlib scikit-learn

# Web应用库
pip install streamlit

# 其他工具库
pip install requests lxml tqdm torchsummary
```

### 推荐版本
```
torch >= 1.9.0
torchvision >= 0.10.0
streamlit >= 1.0.0
```

## 运行指南

### 步骤1：修改路径配置

在运行任何脚本之前，需要根据你的实际路径修改以下文件中的路径：

**重要**: 打开`运行前注意事项.txt`文件，按照提示修改所有路径

### 步骤2：数据爬取（可选）

如果已经有图片数据，可以跳过此步骤。

```bash
cd "src"
python 1爬取图片.py
```

**说明**:
- 脚本会自动创建5个文件夹存放不同种类的花
- 每种花爬取约400张图片
- 爬取时间较长，请耐心等待

### 步骤3：数据预处理

```bash
python 2数据预处理.py
```

**说明**:
- 将原始图片按8:2划分为训练集和验证集
- 自动创建train和val文件夹
- 显示各类别图片数量分布饼图

### 步骤4：模型训练

```bash
python 3数据集类 dataset与建模.py
```

**说明**:
- 加载MobileNetV3 Small预训练模型
- 训练10个epoch（可在代码中修改）
- 自动保存验证集准确率最高的模型
- 显示训练过程和结果可视化

**训练输出**:
- 训练集和验证集的loss和acc
- 训练过程可视化图表
- 模型权重文件：`mobilenet_v3_small_mytrain.pth`

### 步骤5：启动Web应用

```bash
streamlit run 4app.py
```

**说明**:
- 自动在浏览器打开Web界面
- 左侧侧边栏可以选择训练新模型或使用已有模型
- 上传图片进行花卉识别

## 复试常见问题解答

### 1. 为什么选择MobileNetV3 Small？

**答案要点**:
- **轻量级**: 参数量少，推理速度快，适合移动端部署
- **高准确率**: 在ImageNet上达到67.4%的top-1准确率
- **适合小数据集**: 迁移学习效果好，不易过拟合
- **架构优化**: 使用了SE模块、h-swish激活函数等先进技术

### 2. 什么是迁移学习？为什么要用迁移学习？

**答案要点**:
- **定义**: 将在一个任务上学到的知识应用到另一个相关任务上
- **本项目应用**: 使用在ImageNet上预训练的MobileNetV3，只修改最后一层
- **优势**:
  - 减少训练时间
  - 提高模型在小数据集上的性能
  - 避免从头训练需要大量数据和计算资源
- **原理**: 底层特征（边缘、纹理）是通用的，高层特征需要针对特定任务微调

### 3. 数据增强有哪些作用？

**答案要点**:
- **增加数据多样性**: 防止过拟合
- **提高模型泛化能力**: 让模型对图像变换不敏感
- **本项目使用的增强**:
  - CenterCrop(224): 中心裁剪，统一输入尺寸
  - Normalize: 标准化，加速收敛
- **其他常见增强**: 随机翻转、旋转、颜色抖动、添加噪声等

### 4. 为什么使用Adam优化器而不是SGD？

**答案要点**:
- **Adam优点**:
  - 自适应学习率，对不同参数使用不同学习率
  - 结合了动量法和RMSProp的优点
  - 收敛速度快，对超参数不敏感
- **SGD优点**:
  - 泛化性能可能更好
  - 更适合大规模分布式训练
- **选择原因**: 本项目数据量不大，Adam收敛更快更稳定

### 5. 学习率调度器的作用是什么？

**答案要点**:
- **定义**: 在训练过程中动态调整学习率
- **本项目使用**: StepLR，每5个epoch学习率乘以0.5
- **作用**:
  - 训练初期使用较大学习率快速收敛
  - 训练后期使用较小学习率精细调优
  - 避免在最优解附近震荡
- **其他调度器**: CosineAnnealingLR、ReduceLROnPlateau等

### 6. 如何评估模型性能？

**答案要点**:
- **评估指标**:
  - 准确率（Accuracy）: 正确预测的样本占比
  - 损失值（Loss）: 交叉熵损失
- **评估方法**:
  - 使用验证集评估，避免过拟合
  - 保存验证集准确率最高的模型
  - 绘制loss和acc曲线观察训练过程
- **过拟合判断**: 训练集acc高但验证集acc低，两者差距大

### 7. 模型输入为什么要归一化？

**答案要点**:
- **目的**: 将像素值从[0, 255]映射到[-1, 1]
- **作用**:
  - 加速模型收敛
  - 避免梯度消失或爆炸
  - 符合预训练模型的输入要求
- **公式**: `normalized = (pixel / 255.0 - mean) / std`
- **本项目**: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

### 8. CrossEntropyLoss损失函数的原理？

**答案要点**:
- **适用场景**: 多分类问题
- **公式**: `L = -Σ y_i * log(p_i)`
  - y_i: 真实标签（one-hot编码）
  - p_i: 预测概率（经过softmax）
- **特点**:
  - 内部包含Softmax操作
  - 对预测错误的样本惩罚大
  - 概率分布越接近真实标签，损失越小
- **为什么不用MSE**: 分类问题预测的是概率，MSE不适合

### 9. 如何改进模型性能？

**答案要点**:
- **数据层面**:
  - 增加数据量
  - 使用更多数据增强
  - 清洗数据，去除噪声
- **模型层面**:
  - 尝试更大的模型（ResNet、EfficientNet）
  - 调整超参数（学习率、batch size）
  - 使用集成学习
- **训练策略**:
  - 增加训练轮数
  - 使用学习率预热
  - 尝试不同的优化器
- **正则化**:
  - Dropout
  - 权重衰减（L2正则）
  - 早停（Early Stopping）

### 10. Streamlit框架的特点？

**答案要点**:
- **优点**:
  - 纯Python编写，无需前端知识
  - 代码简洁，快速原型开发
  - 自动刷新，实时预览
  - 内置组件丰富（文件上传、按钮、图表等）
- **适用场景**: 数据分析、机器学习演示、快速原型
- **本项目应用**: 模型训练界面、图片上传和预测展示

## 核心技术点总结

### 1. 深度学习基础
- 卷积神经网络（CNN）
- 迁移学习
- 损失函数和优化器
- 前向传播和反向传播

### 2. 计算机视觉
- 图像预处理（裁剪、归一化）
- 数据增强
- 特征提取

### 3. 模型训练技巧
- 学习率调度
- 过拟合防止
- 模型保存和加载
- 训练过程可视化

### 4. 工程实践
- 数据集划分
- 批次训练
- 模型评估
- Web应用部署

## 常见问题

### Q1: 训练时显存不足怎么办？
**A**: 减小batch_size，或使用梯度累积

### Q2: 模型准确率很低怎么办？
**A**: 
- 检查数据质量和标注
- 调整学习率
- 增加训练轮数
- 尝试不同的模型

### Q3: 如何使用GPU加速训练？
**A**: 修改代码中的device设置：
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Q4: 如何添加新的花卉类别？
**A**: 
1. 在爬取脚本中添加新的URL和类别名
2. 重新爬取数据
3. 修改模型输出类别数 `classes_num`
4. 重新训练模型

### Q5: 预训练权重在哪里下载？
**A**: 
- 官方下载：PyTorch会自动下载到用户缓存目录
- 手动下载：从PyTorch官网下载后放到weights文件夹

## 项目亮点

1. **完整的端到端流程**: 从数据爬取到Web应用部署
2. **迁移学习实践**: 使用预训练模型提高性能
3. **可视化友好**: 训练过程和结果可视化
4. **交互式应用**: Streamlit构建的Web界面
5. **代码规范**: 模块化设计，易于理解和扩展

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/)
- [MobileNetV3论文](https://arxiv.org/abs/1905.02244)
- [Streamlit文档](https://docs.streamlit.io/)
- [ImageNet数据集](http://www.image-net.org/)

## 作者

本项目为深度学习课程设计作品，展示了从数据采集到模型部署的完整流程。

---

**祝复试顺利！**
