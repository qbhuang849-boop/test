import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import warnings
from PIL import ImageFile

print("我很帅")

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略文件截断错误

# 导入DeepLabV3Plus模型
try:
    from network.modeling import (
        deeplabv3plus_mobilenet,
        deeplabv3plus_resnet50,
        deeplabv3plus_resnet101
    )

    print("成功导入所有模型")
except ImportError as e:
    print(f"警告: 无法从network.modeling导入模型: {e}")
    print("请确保network模块在Python路径中")
    exit(1)


# 自定义二值分割数据集类
class BinarySegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: 数据集根目录路径
            split: 'train' 或 'val'
            transform: 数据增强
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.images_dir = self.root_dir / split / 'images'
        self.masks_dir = self.root_dir / split / 'masks'

        # 检查目录是否存在
        if not self.images_dir.exists():
            raise ValueError(f"图像目录不存在: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"掩码目录不存在: {self.masks_dir}")

        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(self.images_dir)
                              if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))])

        # 创建图像-掩码对应关系
        self.samples = []

        for img_file in image_files:
            img_path = self.images_dir / img_file
            img_stem = Path(img_file).stem  # 获取不带扩展名的文件名

            # 尝试多种可能的掩码文件名
            possible_mask_names = [
                img_stem + '.png',  # 相同名称，.png扩展名
                img_stem + '.jpg',
                img_stem + '.PNG',
                img_stem + '.JPG',
                img_file,  # 完全相同文件名
                img_stem + '_mask.png',  # 添加后缀
                img_stem.replace('image', 'mask') + '.png',  # 替换关键词
            ]

            mask_path = None
            for mask_name in possible_mask_names:
                test_path = self.masks_dir / mask_name
                if test_path.exists():
                    mask_path = test_path
                    break

            if mask_path:
                self.samples.append((img_path, mask_path))
            else:
                print(f"警告: 找不到图像 {img_file} 对应的掩码文件，跳过此样本")

        if len(self.samples) == 0:
            raise ValueError(f"在 {self.images_dir} 中没有找到有效的图像-掩码对")

        print(f"找到 {len(self.samples)} 个 {split} 图像-掩码对")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, mask_path = self.samples[idx]

            # 读取图像
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as img_e:
                print(f"警告: 无法加载图像 {img_path}, 错误: {img_e}")
                image = Image.new('RGB', (512, 512), (0, 0, 0))

            # 读取mask
            try:
                mask = Image.open(mask_path).convert('L')  # 转为单通道
            except Exception as mask_e:
                print(f"警告: 无法加载掩码 {mask_path}, 错误: {mask_e}")
                # 创建占位符mask（全0，即背景）
                mask = Image.new('L', image.size, 0)

            # 应用数据增强
            if self.transform:
                image, mask = self.transform(image, mask)

            # 将mask从[0, 255]转换为[0, 1]
            mask_np = np.array(mask)

            # 二值化：>128为1，否则为0
            mask_binary = (mask_np > 128).astype(np.float32)
            mask = torch.from_numpy(mask_binary).long()

            # 图像归一化
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)

            return image, mask

        except Exception as e:
            print(f"严重错误: 处理索引 {idx} 时发生异常: {e}")
            print(f"图像路径: {img_path if 'img_path' in locals() else '未定义'}")
            print(f"掩码路径: {mask_path if 'mask_path' in locals() else '未定义'}")

            # 返回一个安全的占位符数据
            image = torch.zeros((3, 512, 512), dtype=torch.float32)
            mask = torch.zeros((512, 512), dtype=torch.long)

            return image, mask


# 数据增强类
class SegmentationTransform:
    def __init__(self, size=(512, 512), train=True):
        self.size = size
        self.train = train

    def __call__(self, image, mask):
        # Resize
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        if self.train:
            # 随机水平翻转
            if np.random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # 随机旋转（小角度）
            if np.random.random() > 0.5:
                angle = np.random.uniform(-10, 10)
                image = image.rotate(angle, Image.BILINEAR)
                mask = mask.rotate(angle, Image.NEAREST)

        return image, mask


# 计算IoU指标
def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return ious


# 训练一个epoch
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    mean_iou = 0.0

    pbar = tqdm(dataloader, desc=f'训练中 [{epoch}/{total_epochs}]')

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)  # 注意：对于DeepLabV3+，输出是直接的logits
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算IoU
        preds = torch.argmax(outputs, dim=1)
        ious = calculate_iou(preds, masks)
        batch_iou = np.nanmean(ious[1:])  # 忽略背景类，只计算前景IoU

        running_loss += loss.item()
        if not np.isnan(batch_iou):
            mean_iou += batch_iou

        # 更新进度条
        pbar.set_postfix({'损失': f'{loss.item():.4f}',
                          'IoU': f'{batch_iou:.4f}'})

    epoch_loss = running_loss / len(dataloader)
    epoch_iou = mean_iou / len(dataloader)

    return epoch_loss, epoch_iou


# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    mean_iou = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='验证中')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            ious = calculate_iou(preds, masks)
            batch_iou = np.nanmean(ious[1:])  # 忽略背景类

            running_loss += loss.item()
            if not np.isnan(batch_iou):
                mean_iou += batch_iou

            pbar.set_postfix({'损失': f'{loss.item():.4f}',
                              'IoU': f'{batch_iou:.4f}'})

    epoch_loss = running_loss / len(dataloader)
    epoch_iou = mean_iou / len(dataloader)

    return epoch_loss, epoch_iou


def load_pretrained_weights(model, pretrained_path, model_type='resnet101'):
    """
    安全加载预训练权重，处理PyTorch版本兼容性问题
    特别注意：您之前使用的是MobileNet模型，现在要加载ResNet101的预训练权重
    这意味着键名完全不匹配，我们需要重新设计加载策略
    """
    print(f'正在加载预训练权重从 {pretrained_path}')

    # 尝试不同的加载方式
    try:
        # 首先尝试最安全的方式（PyTorch 2.6+）
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        print("使用weights_only=True加载")
    except Exception as e1:
        try:
            # 如果失败，尝试添加安全的全局变量
            import torch.serialization
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            print("使用safe_globals加载")
        except Exception as e2:
            try:
                # 最后尝试不安全的方式（仅在信任源的情况下使用）
                warnings.warn("使用weights_only=False加载。仅在信任源的情况下使用！")
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                print("使用weights_only=False加载（不安全模式）")
            except Exception as e3:
                raise RuntimeError(f"加载预训练权重失败: {e3}")

    # 处理checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            pretrained_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            # 假设checkpoint本身就是state_dict
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint

    # 获取当前模型的state_dict
    model_dict = model.state_dict()

    # 对于从DeepLabV3转到DeepLabV3+的情况，需要特殊的键名处理
    print(f"预训练权重键数量: {len(pretrained_dict)}")
    print(f"当前模型键数量: {len(model_dict)}")

    # 显示前几个键名用于调试
    print("\n预训练权重前5个键:")
    for i, key in enumerate(list(pretrained_dict.keys())[:5]):
        print(f"  {key}: {pretrained_dict[key].shape}")

    print("\n当前模型前5个键:")
    for i, key in enumerate(list(model_dict.keys())[:5]):
        print(f"  {key}: {model_dict[key].shape}")

    # 修复：移除verbose参数
    # 我们将在后面手动添加打印

    # 对于ResNet模型，尝试匹配主干部分
    filtered_dict = {}
    ignored_layers = []
    matched_layers = 0

    # 尝试不同的键名匹配策略
    for pretrained_key, pretrained_value in pretrained_dict.items():
        # 1. 尝试直接匹配
        if pretrained_key in model_dict:
            if model_dict[pretrained_key].shape == pretrained_value.shape:
                filtered_dict[pretrained_key] = pretrained_value
                matched_layers += 1
                continue
            else:
                ignored_layers.append(f"形状不匹配: {pretrained_key}")
                continue

        # 2. 尝试移除'backbone.'前缀后再匹配
        # 从您的错误信息看，预训练权重的键名有'backbone.'前缀
        simple_key = pretrained_key
        if pretrained_key.startswith('backbone.'):
            simple_key = pretrained_key[9:]  # 移除'backbone.'

        if simple_key in model_dict:
            if model_dict[simple_key].shape == pretrained_value.shape:
                filtered_dict[simple_key] = pretrained_value
                matched_layers += 1
                continue
            else:
                ignored_layers.append(f"形状不匹配(简化键): {pretrained_key} -> {simple_key}")
                continue

        # 3. 对于ResNet模型，尝试处理特定的层名
        if model_type in ['resnet50', 'resnet101']:
            # ResNet的层名通常像这样：layer1.0.conv1.weight
            # 而预训练权重可能有：backbone.layer1.0.conv1.weight
            if 'layer' in pretrained_key and 'conv' in pretrained_key:
                # 尝试提取层名
                parts = pretrained_key.split('.')
                # 移除'backbone'部分
                if parts[0] == 'backbone':
                    new_key = '.'.join(parts[1:])
                    if new_key in model_dict and model_dict[new_key].shape == pretrained_value.shape:
                        filtered_dict[new_key] = pretrained_value
                        matched_layers += 1
                        continue

        ignored_layers.append(f"键不匹配: {pretrained_key}")

    print(f"\n成功匹配 {matched_layers}/{len(model_dict)} 层")

    if matched_layers > 0:
        # 更新模型权重
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"成功加载 {len(filtered_dict)} 层预训练权重")
    else:
        print("警告: 未能加载任何预训练层，模型将从头开始训练")

    if ignored_layers and len(ignored_layers) > 0:
        print(f"忽略了 {len(ignored_layers)} 层")
        if len(ignored_layers) <= 20:
            for layer in ignored_layers:
                print(f"  {layer}")
        else:
            print(f"前20个忽略的层:")
            for layer in ignored_layers[:20]:
                print(f"  {layer}")

    return model


def create_model(args):
    """
    根据参数创建模型
    """
    model_type = args.model_type.lower()

    if model_type == 'mobilenet':
        print("创建DeepLabV3+ with MobileNet骨干网络")
        model = deeplabv3plus_mobilenet(
            num_classes=args.num_classes,
            output_stride=args.output_stride,
            pretrained_backbone=False
        )
    elif model_type == 'resnet50':
        print("创建DeepLabV3+ with ResNet50骨干网络")
        model = deeplabv3plus_resnet50(
            num_classes=args.num_classes,
            output_stride=args.output_stride,
            pretrained_backbone=False
        )
    elif model_type == 'resnet101':
        print("创建DeepLabV3+ with ResNet101骨干网络")
        model = deeplabv3plus_resnet101(
            num_classes=args.num_classes,
            output_stride=args.output_stride,
            pretrained_backbone=False
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    return model


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 如果使用多GPU
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"使用 {torch.cuda.device_count()} 个GPU")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练参数
    with open(save_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 创建数据集和数据加载器
    print(f"\n从 {args.data_root} 加载数据集")
    train_transform = SegmentationTransform(size=(args.img_size, args.img_size), train=True)
    val_transform = SegmentationTransform(size=(args.img_size, args.img_size), train=False)

    train_dataset = BinarySegmentationDataset(args.data_root, split='train', transform=train_transform)
    val_dataset = BinarySegmentationDataset(args.data_root, split='val', transform=val_transform)

    # 根据模型大小调整batch_size
    if args.model_type in ['resnet50', 'resnet101'] and args.batch_size > 4:
        print(f"警告: 大模型 {args.model_type} 可能需要更小的批次大小")
        if torch.cuda.is_available():
            # 检查可用显存
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory

            print(f"GPU总显存: {total_memory / 1024 ** 3:.2f} GB")
            print(f"已分配显存: {allocated_memory / 1024 ** 3:.2f} GB")
            print(f"可用显存: {free_memory / 1024 ** 3:.2f} GB")

            if free_memory < 6 * 1024 ** 3:  # 小于6GB可用显存
                args.batch_size = max(2, args.batch_size // 2)
                print(f"由于GPU内存有限，将批次大小减少到 {args.batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # 创建模型
    model = create_model(args)

    # 加载预训练权重
    if args.pretrained and os.path.exists(args.pretrained):
        model = load_pretrained_weights(model, args.pretrained, args.model_type)
        print(f"从 {args.pretrained} 加载预训练权重")
    elif args.pretrained:
        print(f"警告: 未找到预训练权重文件: {args.pretrained}")
        print("从头开始训练...")

    # 多GPU支持
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 使用不同的学习率策略
    if args.pretrained and os.path.exists(args.pretrained):
        # 如果使用预训练模型，对backbone使用较小的学习率
        backbone_params = []
        decoder_params = []

        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': decoder_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器 - 修复：移除verbose参数
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=args.patience,
            factor=0.5
        )
        print(f"创建学习率调度器，耐心值={args.patience}")
    except TypeError:
        # 如果当前PyTorch版本不支持verbose参数，尝试不加verbose
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=args.patience,
            factor=0.5
        )
        print(f"创建学习率调度器(无verbose参数)，耐心值={args.patience}")

    # 训练循环
    best_iou = 0.0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []

    print('\n' + '=' * 60)
    print('开始训练...')
    print(f'模型: {args.model_type}')
    print(f'训练样本: {len(train_dataset)}')
    print(f'验证样本: {len(val_dataset)}')
    print(f'批次大小: {args.batch_size}')
    print('=' * 60 + '\n')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_iou = train_epoch(model, train_loader, criterion,
                                            optimizer, device, epoch, args.epochs)

        # 验证
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # 调整学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_iou)
        new_lr = optimizer.param_groups[0]['lr']

        # 手动检查学习率是否改变
        if new_lr != old_lr:
            print(f"Epoch {epoch}: 学习率从 {old_lr:.6f} 调整到 {new_lr:.6f}")

        # 保存指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f'\nEpoch [{epoch}/{args.epochs}]')
        print(f'训练损失: {train_loss:.4f}, 训练IoU: {train_iou:.4f}')
        print(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}')
        print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            # 如果是多GPU，保存module的state_dict
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            checkpoint = {
                'epoch': epoch,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict(),
                'best_iou': best_iou,
                'train_iou': train_iou,
                'val_iou': val_iou,
                'args': vars(args)
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f'✓ 保存最佳模型，验证IoU: {best_iou:.4f}')

        # 定期保存检查点
        if epoch % args.save_interval == 0:
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            checkpoint = {
                'epoch': epoch,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict(),
                'val_iou': val_iou,
                'args': vars(args)
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f'保存检查点 at epoch {epoch}')

        print('-' * 60)

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ious': train_ious,
        'val_ious': val_ious,
        'best_iou': best_iou,
        'args': vars(args)
    }

    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print('\n' + '=' * 60)
    print('训练完成!')
    print(f'最佳验证IoU: {best_iou:.4f}')
    print(f'模型保存在: {save_dir}')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练DeepLabV3+进行二分类分割')

    # 数据相关参数
    parser.add_argument('--data_root', type=str,
                        default='/media/ej/data/物种识别/空拍分割算法/_tool/1212_负样本混合训练/data_train',
                        help='数据集根目录路径')
    parser.add_argument('--img_size', type=int, default=960,
                        help='输入图像尺寸（默认：ResNet模型使用512）')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='resnet101',
                        choices=['mobilenet', 'resnet50', 'resnet101'],
                        help='选择使用的骨干网络类型')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='类别数量（背景+前景）')
    parser.add_argument('--output_stride', type=int, default=16,
                        help='DeepLabV3+的输出步长')
    parser.add_argument('--pretrained', type=str, default='C:\Users\yj\Desktop\git\Deeplabv3+\best_deeplabv3plus_resnet101_voc_os16.pth',
                        help='预训练模型权重路径（例如：best_deeplabv3plus_resnet101_voc_os16.pth）')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='训练批次大小（对于较大模型可能需要减小）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练总轮数')
    parser.add_argument('--lr', type=float, default=0.0004,#1e-3
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='优化器权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                        help='学习率调度器的耐心值（连续多少个epoch没有改进才降低学习率）')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='每N个epoch保存一次检查点')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='如果可用则使用多GPU')

    args = parser.parse_args()

    # 打印配置
    print('\n' + '=' * 60)
    print('训练配置:')
    for key, value in vars(args).items():
        print(f'{key:20}: {value}')
    print('=' * 60 + '\n')

    main(args)