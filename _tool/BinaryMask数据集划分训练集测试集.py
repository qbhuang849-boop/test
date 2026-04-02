import os
import random
from shutil import copy2
import sys


def split_dataset(input_root, output_root=None, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    划分数据集为训练集、验证集和测试集

    参数:
        input_root: 原始数据集根目录，包含images和masks子目录
        output_root: 输出目录，默认为input_root的父目录下创建'split_dataset'目录
        train_ratio: 训练集比例 (0.0-1.0)
        val_ratio: 验证集比例 (0.0-1.0)
        test_ratio: 测试集比例 (0.0-1.0)
        注意：train_ratio + val_ratio + test_ratio 必须等于 1

    目录结构:
        原始数据: input_root/
            ├── images/
            └── masks/

        输出结构: output_root/
            ├── train/
            │   ├── images/
            │   └── masks/
            ├── val/
            │   ├── images/
            │   └── masks/
            └── test/ (当test_ratio > 0时存在)
                ├── images/
                └── masks/
    """
    # 验证比例总和是否为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:  # 使用容差比较浮点数
        print(f"错误：比例总和应为1.0，实际为{total_ratio:.3f}")
        print(f"训练集: {train_ratio:.3f}, 验证集: {val_ratio:.3f}, 测试集: {test_ratio:.3f}")
        sys.exit(1)

    # 设置输出目录
    if output_root is None:
        # 在输入目录的父目录下创建split_dataset目录
        parent_dir = os.path.dirname(input_root.rstrip('/'))
        output_root = os.path.join(parent_dir, 'split_dataset')

    # 创建输出目录结构
    train_img_dir = os.path.join(output_root, 'train', 'images')
    train_mask_dir = os.path.join(output_root, 'train', 'masks')
    val_img_dir = os.path.join(output_root, 'val', 'images')
    val_mask_dir = os.path.join(output_root, 'val', 'masks')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    # 如果测试集比例大于0，创建测试目录
    if test_ratio > 0:
        test_img_dir = os.path.join(output_root, 'test', 'images')
        test_mask_dir = os.path.join(output_root, 'test', 'masks')
        os.makedirs(test_img_dir, exist_ok=True)
        os.makedirs(test_mask_dir, exist_ok=True)

    # 检查原始目录
    img_dir = os.path.join(input_root, 'images')
    mask_dir = os.path.join(input_root, 'masks')

    if not os.path.exists(img_dir):
        print(f"错误：图像目录不存在 - {img_dir}")
        sys.exit(1)
    if not os.path.exists(mask_dir):
        print(f"错误：掩码目录不存在 - {mask_dir}")
        sys.exit(1)

    # 获取所有图像文件（带扩展名）和对应的掩码文件（必须是png）
    all_files = []  # 元素为(img_path, mask_path)

    # 支持的图像扩展名
    img_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        if not os.path.isfile(img_path):
            continue

        # 检查扩展名
        base_name, ext = os.path.splitext(img_file)
        if ext.lower() not in img_exts:
            continue

        # 尝试查找对应的掩码文件
        mask_found = False
        # 先尝试相同的扩展名
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            all_files.append((img_path, mask_path))
            mask_found = True
        else:
            # 尝试png格式
            mask_path = os.path.join(mask_dir, base_name + '.png')
            if os.path.exists(mask_path):
                all_files.append((img_path, mask_path))
                mask_found = True

        if not mask_found:
            print(f"警告：图像 {img_file} 没有对应的掩码文件，跳过。")
            continue

    if len(all_files) == 0:
        print("错误：没有找到有效的图像-掩码对。")
        sys.exit(1)

    # 随机打乱
    random.shuffle(all_files)

    # 计算各集合的数量
    total_samples = len(all_files)
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)

    # 划分数据集
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:] if test_ratio > 0 else []

    # 打印统计信息
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_files)} ({train_ratio * 100:.1f}%)")
    print(f"验证集: {len(val_files)} ({val_ratio * 100:.1f}%)")
    if test_ratio > 0:
        print(f"测试集: {len(test_files)} ({test_ratio * 100:.1f}%)")

    # 复制训练集
    print("复制训练集...")
    for img_path, mask_path in train_files:
        # 复制图像
        img_filename = os.path.basename(img_path)
        copy2(img_path, os.path.join(train_img_dir, img_filename))
        # 复制掩码
        mask_filename = os.path.basename(mask_path)
        copy2(mask_path, os.path.join(train_mask_dir, mask_filename))

    # 复制验证集
    print("复制验证集...")
    for img_path, mask_path in val_files:
        img_filename = os.path.basename(img_path)
        copy2(img_path, os.path.join(val_img_dir, img_filename))
        mask_filename = os.path.basename(mask_path)
        copy2(mask_path, os.path.join(val_mask_dir, mask_filename))

    # 如果测试集比例大于0，复制测试集
    if test_ratio > 0 and test_files:
        print("复制测试集...")
        for img_path, mask_path in test_files:
            img_filename = os.path.basename(img_path)
            copy2(img_path, os.path.join(test_img_dir, img_filename))
            mask_filename = os.path.basename(mask_path)
            copy2(mask_path, os.path.join(test_mask_dir, mask_filename))

    print(f"数据集划分完成！输出目录: {output_root}")


# 使用示例
if __name__ == "__main__":
    # 原始数据集路径
    input_root = "/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/20160105_data/original_dataset"
    split_dataset(
        input_root=input_root,
        output_root="/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/20160105_data/data_train",
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10
    )