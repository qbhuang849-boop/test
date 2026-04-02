# 基于DeepLabV3+的二值语义分割模型优化

## 项目概述
本项目基于DeepLabV3+框架，针对**司风空拍模型数据**进行二值语义分割（前景/背景）任务的定制化开发与性能优化。通过多轮迭代训练和策略调优，实现了在复杂场景下的高精度分割效果。

## 核心成果

### 最终性能指标
- **整体准确率**: 95.29%
- **平均IoU (mIoU)**: 93.01%
- **前景IoU**: 86.72%
- **纯背景图像完美预测率**: 90.16% (110/122)
- **前景图像检测准确率**: 98.17% (214/218)

### 性能提升对比
相比上一版本，本次更新实现了显著性能提升：
- **误报率大幅降低**: 纯背景图像完美预测率提升至90.16%
- **前景检测精度提升**: 检测准确率达到98.17%，漏检问题显著改善
- **分割质量优化**: 前景IoU达到86.72%，分割边界更加精准

## 复现步骤

### 1. 环境准备
```bash
# 克隆项目
git clone <your-repo-url>
cd DeepLabv3+

# 创建虚拟环境（推荐）
conda create -n deeplab python=3.8
conda activate deeplab

# 安装依赖
pip install requirements.txt
```

### 2. 数据准备
#### 2.1 数据预处理（如果需要）
```bash
# 如果您有Label Studio JSON标注文件，转换为二值掩码
python _tool/json转掩码.py

# 划分数据集（如果需要）
python _tool/BinaryMask数据集划分训练集测试集.py
```

#### 2.2 数据集结构
确保您的数据集按照以下结构组织：
```
data_train/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```
### 3. 模型训练

#### 3.1 下载预训练权重
从官方提供的链接下载DeepLabV3+ ResNet101预训练权重：
- [Dropbox](https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0)
- [腾讯微云](https://share.weiyun.com/UNPZr3dk)

将权重文件保存为：`best_deeplabv3plus_resnet101_voc_os16.pth`

#### 3.2 初始训练阶段
```bash
python train_test.py \
    --data_root "E:\工作文件\20260327_司风空拍模型数据_2262\data_train" \
    --model_type resnet101 \
    --pretrained "best_deeplabv3plus_resnet101_voc_os16.pth" \
    --lr 4e-4 \
    --epochs 100 \
    --batch_size 4 \
    --save_dir "./models/train_initial"
```

#### 3.3 微调阶段
使用初始训练的最佳模型进行微调：
```bash
python train_test2.py \
    --data_root "E:\工作文件\20260327_司风空拍模型数据_2262\data_train" \
    --model_type resnet101 \
    --pretrained "models/train_initial/best_model.pth" \
    --lr 1e-5 \
    --epochs 10 \
    --batch_size 4 \
    --patience 5 \
    --save_dir "./models/train_final"
```

### 4. 模型测试

#### 4.1 配置测试参数
编辑 `_tool/test/test.py` 文件，修改以下配置参数（约444-463行）：

```python
# 模型配置
MODEL_PATH = "C:/Users/yj/Desktop/git/Deeplabv3+/models/train_final/best_model.pth"
BACKBONE = 'resnet101'  # 'resnet101' 或 'mobilenet'
OUTPUT_STRIDE = 16  # 8 或 16

# 测试数据路径
TEST_IMAGES_DIR = "E:/工作文件/20260327_司风空拍模型数据_2262/data_train/test/images"
TEST_MASKS_DIR = "E:/工作文件/20260327_司风空拍模型数据_2262/data_train/test/masks"
OUTPUT_DIR = "./test_results"

# 模型参数
IMG_SIZE = 960
THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PREDICTIONS = True
```

#### 4.2 运行测试
```bash
python _tool/test/test.py
```

测试脚本将：
- 自动评估整个测试集
- 生成三列并排的可视化结果（原始图像 + 预测Mask + 叠加结果）
- 输出详细的性能指标报告
- 保存评估结果到指定目录

## 核心优化策略

### 1. 数据集扩充
- **数据规模扩展**: 从1887张空拍样本扩充至2272张空拍数据
- **适配模型需求**: 满足ResNet101架构对大规模数据集的要求
- **提升泛化能力**: 更丰富的训练样本有效缓解过拟合问题

### 2. 分层学习率训练策略
采用差异化学习率策略，平衡特征保留与任务适配：

#### 初始训练阶段（100轮）
- **骨干网络学习率**: 4e-5（较小，保护预训练特征）
- **解码器学习率**: 4e-4（较大，快速适配分割任务）

#### 微调阶段（针对性优化）
- **初始学习率**: 1e-5（极小，避免破坏已有特征）
- **学习率调度器**: patience=5, factor=0.5

### 3. 三损失融合策略
针对空拍图像特点，设计多目标优化的损失函数组合：

#### 损失函数权重配置
- **Focal Loss (权重0.4)**: 聚焦难例样本，解决前景样本占比低导致的漏检问题
- **Dice Loss (权重0.4)**: 保障前景分割区域完整性，直接优化IoU指标
- **Cross Entropy Loss (权重0.2)**: 强化背景分类准确性，减少纯背景误判

## 技术架构

### 模型配置
- **骨干网络**: ResNet101 (ImageNet预训练)
- **分割头**: DeepLabV3+ ASPP模块
- **输出步长**: 16
- **输入尺寸**: 960×960
- **类别数**: 2 (背景 + 前景)

### 训练流程
1. **数据预处理**: Label Studio JSON标注转二值掩码
2. **两阶段训练**: 初始100轮 + 微调阶段
3. **验证与测试**: 验证集用于超参数调优，测试集仅用于最终评估

### 数据增强
- 随机水平翻转
- 小角度旋转（±10°）
- 图像尺寸统一调整为960×960

## 文件结构说明

### 核心训练脚本
- `train_test.py`: 主训练脚本，初始训练阶段
- `train_test2.py`: 微调模型阶段，包含三损失组合和分层学习率
- `_tool/test.py`: 推理预测脚本

### 工具脚本
- `utils_triple_loss.py`: 三损失组合实现（Focal+Dice+CE）
- `utils/loss.py`: Focal Loss等基础损失函数
- `_tool/json转掩码.py`: Label Studio JSON转二值掩码工具
- `_tool/BinaryMask数据集划分训练集测试集.py`: 数据集划分工具

### 模型权重管理
- `models/train*/`: 不同训练阶段的模型保存目录
- `best_model.pth`: 最佳性能模型权重
- 官方预训练权重: `best_deeplabv3plus_resnet101_voc_os16.pth`

## 优化历程

### 上一版本问题分析
- **数据规模不足**: 1887张样本难以支撑ResNet101大模型
- **学习率偏大**: 4e-4导致训练后期震荡和过拟合
- **性能缺陷**: 前景分割精度不足，纯背景误报率高

### 本次优化突破
1. **数据扩充**: 解决模型容量与数据规模不匹配问题
2. **学习率优化**: 两阶段学习率策略平衡收敛速度与稳定性
3. **损失函数创新**: 三损失融合实现多目标协同优化
4. **训练策略完善**: 分层训练 + 微调的完整pipeline

## 后续优化方向

### 短期重点
- **生产环境监控**: 持续观察司空生产环境的实际表现
- **困难样本分析**: 识别并针对性处理剩余的误报/漏检案例

### 中长期规划
1. **模型轻量化**: 探索MobileNet等轻量级骨干网络
2. **后处理优化**: 引入CRF或形态学操作细化分割边界
3. **主动学习**: 基于模型不确定性自动选择最有价值的样本进行标注

## 依赖环境
- Python 3.8+
- PyTorch 1.12+
- torchvision
- numpy, PIL, tqdm, matplotlib, opencv-python
- CUDA (可选，用于GPU加速)

## 参考文献
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)