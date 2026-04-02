import argparse
from pathlib import Path
import cv2  # 添加OpenCV用于视频处理
import torch
import torch.serialization
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings
import shutil  # 用于复制文件
import os  # 添加os模块

# 和 train.py 保持一致的模型导入
from network.modeling import deeplabv3plus_mobilenet


def build_model(checkpoint_path, device, num_classes=2, output_stride=16):
    """
    构建模型并加载训练好的权重（兼容 PyTorch 2.6+ 的安全加载）
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = deeplabv3plus_mobilenet(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=False  # 推理阶段不需要再加载 backbone 预训练
    )

    # ---- 关键：安全加载 checkpoint，兼容 numpy 标量 & weights_only ----
    try:
        # 优先尝试最安全的方式
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print("Loaded checkpoint with weights_only=True")
    except Exception as e1:
        print(f"weights_only=True load failed once: {e1}")
        try:
            # 将 numpy 标量和 dtype 加入 safe_globals 再试一次（针对常见 NumPy 错误）
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            print("Loaded checkpoint with safe_globals + weights_only=True")
        except Exception as e2:
            print(f"safe_globals + weights_only=True still failed: {e2}")
            # 最后兜底：weights_only=False（前提是你信任 checkpoint 来源，这是你自己训练的，所以 OK）
            warnings.warn(
                "Loading checkpoint with weights_only=False. "
                "Only do this if you trust the source of the checkpoint!"
            )
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            print("Loaded checkpoint with weights_only=False (unsafe mode)")
    # ------------------------------------------------------------------

    # 兼容训练保存格式：{'model_state': ..., 'optimizer_state': ..., ...}
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # 有些情况下直接就是 state_dict
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode.")
    return model


def preprocess_image(img_pil, img_size):
    """
    与 train.py 一致的预处理：
    - resize 到 (img_size, img_size)
    - 转 tensor
    - ImageNet 标准化
    """
    orig_size = img_pil.size  # (W, H)

    # Resize
    img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)

    # ToTensor + Normalize（与训练完全一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img_resized)  # [3, H, W]

    return img_tensor.unsqueeze(0), orig_size  # [1, 3, H, W], (W, H)


def has_foreground(mask_tensor, min_foreground_ratio=0.01):
    """
    检查掩码中是否有足够的前景区域
    mask_tensor: [H, W] 的二值掩码 (0/1)
    min_foreground_ratio: 最小前景比例阈值
    """
    foreground_pixels = torch.sum(mask_tensor > 0).item()
    total_pixels = mask_tensor.numel()
    foreground_ratio = foreground_pixels / total_pixels

    return foreground_ratio >= min_foreground_ratio, foreground_ratio


def is_image_file(path):
    """
    判断是不是图片文件
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    return any(str(path).lower().endswith(ext) for ext in image_extensions)


def is_video_file(path):
    """
    判断是不是视频文件
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v", ".3gp"]
    return any(str(path).lower().endswith(ext) for ext in video_extensions)


def transfer_file(source_path, target_path, operation='copy'):
    """
    根据操作类型复制或移动文件
    operation: 'copy' 复制文件, 'move' 移动文件
    """
    try:
        if operation == 'copy':
            shutil.copy2(str(source_path), str(target_path))
            return True
        elif operation == 'move':
            # 如果目标文件已存在，则删除
            if target_path.exists():
                target_path.unlink()
            # 移动文件
            shutil.move(str(source_path), str(target_path))
            return True
        else:
            print(f"未知的操作类型: {operation}, 使用默认的复制操作")
            shutil.copy2(str(source_path), str(target_path))
            return True
    except Exception as e:
        print(f"文件操作失败: {e}")
        return False


def process_image(model, device, image_path, output_path, img_size, threshold=0.5, operation='copy'):
    """
    处理单张图片，分割到了就保存原图
    返回: (是否保存, 前景比例)
    operation: 'copy' 复制文件, 'move' 移动文件
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 读取图像
        img = Image.open(image_path).convert("RGB")
    except OSError as e:
        if "truncated" in str(e).lower():
            print(f"Skipping truncated image: {image_path} (error: {e}). Consider re-downloading.")
            return False, 0.0  # 跳过
        else:
            raise  # 其他错误正常抛出

    # 预处理
    img_tensor, orig_size = preprocess_image(img, img_size)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)  # 训练阶段已经改成直接 model(images)
        # outputs: [1, 2, H, W]，取 softmax 后前景通道 > threshold
        probs = torch.softmax(outputs, dim=1)  # [1, 2, H, W] 概率
        pred_mask = (probs[:, 1, :, :] > threshold).float()  # 前景通道 > 阈值 → [1, H, W] (0/1)
        pred_mask = pred_mask.squeeze(0)  # [H, W]

    # 检查是否有前景
    has_foreground_flag, foreground_ratio = has_foreground(pred_mask)

    if has_foreground_flag:
        # 根据操作类型复制或移动文件
        if transfer_file(image_path, output_path, operation):
            if operation == 'copy':
                print(f"Copied original image to: {output_path} (foreground ratio: {foreground_ratio:.4f})")
            else:  # move
                print(f"Moved original image to: {output_path} (foreground ratio: {foreground_ratio:.4f})")
            return True, foreground_ratio
        else:
            print(f"Failed to {operation} file from {image_path} to {output_path}")
            return False, foreground_ratio
    else:
        print(f"No foreground detected in {image_path}, skipping... (foreground ratio: {foreground_ratio:.4f})")
        return False, foreground_ratio


def process_video(model, device, video_path, output_dir, img_size, threshold=0.5, frame_interval=10, operation='copy'):
    """
    处理视频，每隔frame_interval帧取一帧进行推理
    分割到了就保存该帧的原图
    返回: (处理的帧数, 保存的帧数, 保存的帧信息列表)
    operation: 目前仅用于视频帧保存，原视频文件不会被移动
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(exist_ok=True)

    print(f"Processing video: {video_path}")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0, 0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")

    processed_frames = 0
    saved_frames = 0
    saved_frame_info = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔frame_interval帧处理一次
        if frame_index % frame_interval == 0:
            # 转换BGR到RGB用于模型推理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            # 预处理
            img_tensor, orig_size = preprocess_image(img_pil, img_size)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_mask = (probs[:, 1, :, :] > threshold).float()
                pred_mask = pred_mask.squeeze(0)

            # 检查是否有前景
            has_foreground_flag, foreground_ratio = has_foreground(pred_mask)

            if has_foreground_flag:
                # 保存原帧图片（BGR格式，因为OpenCV默认）
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_path = video_output_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                saved_frames += 1
                saved_frame_info.append({
                    'frame': frame_index,
                    'frame_path': frame_path,
                    'foreground_ratio': foreground_ratio
                })

                print(f"  Frame {frame_index}: Saved frame (foreground ratio: {foreground_ratio:.4f})")
            else:
                print(f"  Frame {frame_index}: No foreground detected (foreground ratio: {foreground_ratio:.4f})")

            processed_frames += 1

            # 显示进度
            if processed_frames % 10 == 0:
                print(f"  Processed {processed_frames}/{total_frames // frame_interval} frames...")

        frame_index += 1

    cap.release()

    print(
        f"Video processing complete: Processed {processed_frames} frames, saved {saved_frames} frames with foreground")
    return processed_frames, saved_frames, saved_frame_info


def batch_infer(model, device, input_path, output_dir, img_size, threshold=0.5, frame_interval=10, operation='copy'):
    """
    批量推理：处理文件夹中的图片和视频
    operation: 'copy' 复制文件, 'move' 移动文件
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # 单文件处理
        if is_image_file(input_path):
            # 对于单张图片，保持原文件名
            out_path = output_dir / input_path.name
            saved, ratio = process_image(model, device, input_path, out_path, img_size, threshold, operation)
            if saved:
                operation_text = "copied" if operation == 'copy' else "moved"
                print(f"Single image inference: 1 processed, 1 {operation_text} (foreground ratio: {ratio:.4f})")
            else:
                print(f"Single image inference: 1 processed, 0 saved (foreground ratio: {ratio:.4f})")
        elif is_video_file(input_path):
            processed, saved, saved_info = process_video(
                model, device, input_path, output_dir, img_size, threshold, frame_interval, operation
            )
            print(f"Single video inference: Processed {processed} frames, saved {saved} frames with foreground")
        else:
            print(f"Unsupported file type: {input_path}")
    else:
        # 文件夹处理
        image_files = []
        video_files = []

        # 收集所有图片和视频文件
        for ext in ["*", "*.*"]:
            for file_path in input_path.glob(f"**/{ext}"):
                if file_path.is_file():
                    if is_image_file(file_path):
                        image_files.append(file_path)
                    elif is_video_file(file_path):
                        video_files.append(file_path)

        print(f"Found {len(image_files)} images and {len(video_files)} videos in {input_path}")

        # 处理图片
        image_processed = 0
        image_saved = 0
        if image_files:
            print(f"Processing {len(image_files)} images...")
            for img_path in image_files:
                # 保持相对路径结构
                rel_path = img_path.relative_to(input_path)
                out_path = output_dir / "images" / rel_path.parent / img_path.name
                out_path.parent.mkdir(parents=True, exist_ok=True)

                saved, ratio = process_image(model, device, img_path, out_path, img_size, threshold, operation)
                image_processed += 1
                if saved:
                    image_saved += 1

        # 处理视频
        video_processed = 0
        video_frames_processed = 0
        video_frames_saved = 0
        if video_files:
            print(f"Processing {len(video_files)} videos...")
            for video_path in video_files:
                # 保持相对路径结构
                rel_path = video_path.relative_to(input_path)
                video_out_dir = output_dir / "videos" / rel_path.parent / video_path.stem

                processed, saved, saved_info = process_video(
                    model, device, video_path, video_out_dir, img_size, threshold, frame_interval, operation
                )
                video_processed += 1
                video_frames_processed += processed
                video_frames_saved += saved

        operation_text = "copied" if operation == 'copy' else "moved"
        print("\n" + "=" * 50)
        print("Batch inference summary:")
        print(f"  Operation type: {operation}")
        print(f"  Images: {image_processed} processed, {image_saved} {operation_text}")
        print(f"  Videos: {video_processed} processed")
        print(f"  Video frames: {video_frames_processed} processed, {video_frames_saved} saved with foreground")
        print("=" * 50)


def main():
    # =================== 直接在代码中设置参数 ===================
    checkpoint = "/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/models/checkpoint_epoch_160.pth"  # 训练好的权重文件路径
    input_path = "/media/ej/data/物种识别/备份数据/ims20251209全部数据/10000条图片一个/删除不需要标注的"  # 输入图片、视频或文件夹的路径
    output_dir = "/media/ej/data/物种识别/备份数据/ims20251209全部数据/10000条图片一个/去空拍图片"  # 输出目录
    img_size = 960  # 输入图像大小，与训练时一致
    threshold = 0.5  # 前景置信度阈值
    frame_interval = 10  # 视频处理时的帧间隔
    min_foreground_ratio = 0.01  # 最小前景比例阈值
    device_type = "cuda"  # 使用的设备，cuda或cpu
    operation = "move"  # 操作类型：'copy' 复制文件, 'move' 移动文件
    # ============================================================

    # 验证操作类型
    if operation not in ['copy', 'move']:
        print(f"警告: 未知的操作类型 '{operation}'，将使用默认的 'copy'")
        operation = 'copy'

    # 设置设备
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Operation type: {operation}")

    # 构建模型
    model = build_model(
        checkpoint,
        device=device,
        num_classes=2,
        output_stride=16
    )

    # 修改has_foreground函数使用传入的min_foreground_ratio参数
    import sys
    current_module = sys.modules[__name__]

    # 保存原始的has_foreground函数
    original_has_foreground = has_foreground

    # 创建新的has_foreground函数，使用固定的min_foreground_ratio
    def new_has_foreground(mask_tensor):
        return original_has_foreground(mask_tensor, min_foreground_ratio=min_foreground_ratio)

    # 替换模块中的has_foreground函数
    current_module.has_foreground = new_has_foreground

    # 执行批量推理
    batch_infer(
        model,
        device,
        input_path,
        output_dir,
        img_size,
        threshold,
        frame_interval,
        operation  # 传递操作类型参数
    )


if __name__ == "__main__":
    main()