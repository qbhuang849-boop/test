import argparse
from pathlib import Path

import torch
import torch.serialization
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings

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


def postprocess_mask(mask_tensor, orig_size):
    """
    将网络输出的 [H, W] mask 转成原图大小的 0/255 uint8 PNG 图
    """
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255  # 0/1 -> 0/255
    # 忽略 Pillow mode 警告（未来版本移除），直接用 fromarray
    mask_pil = Image.fromarray(mask_np)
    mask_pil = mask_pil.convert("L")  # 转为 L 模式，避免 mode 参数警告
    mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
    return mask_pil


def is_image_file(path):
    """
    简单判断是不是图片文件
    """
    return any(str(path).lower().endswith(ext)
               for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"])


def infer_single_image(model, device, image_path, output_path, img_size, threshold=0.5):
    """
    对单张图片做推理，生成二值 mask（添加阈值过滤）
    返回 True 如果成功处理，False 如果跳过
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
            return False  # 跳过
        else:
            raise  # 其他错误正常抛出

    # 预处理
    img_tensor, orig_size = preprocess_image(img, img_size)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)          # 训练阶段已经改成直接 model(images)
        # outputs: [1, 2, H, W]，取 softmax 后前景通道 > threshold
        probs = torch.softmax(outputs, dim=1)  # [1, 2, H, W] 概率
        pred_mask = (probs[:, 1, :, :] > threshold).float()  # 前景通道 > 阈值 → [1, H, W] (0/1)
        pred_mask = pred_mask.squeeze(0)     # [H, W]

    # 后处理成 PNG
    mask_pil = postprocess_mask(pred_mask, orig_size)
    mask_pil.save(output_path)
    print(f"Saved mask to: {output_path} (threshold: {threshold})")
    return True  # 成功


def batch_infer(model, device, input_path, output_dir, img_size, threshold=0.5):
    """
    如果 input 是文件夹，则遍历所有图片；
    如果是单个文件，则只处理这一张。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # 单张图片
        out_name = input_path.stem + "_mask.png"
        out_path = output_dir / out_name
        success = infer_single_image(model, device, input_path, out_path, img_size, threshold)
        if success:
            print("Batch inference complete: 1 processed, 0 skipped.")
        else:
            print("Batch inference complete: 0 processed, 1 skipped.")
    else:
        # 文件夹：遍历所有图片
        image_files = [p for p in input_path.glob("**/*") if is_image_file(p)]
        if len(image_files) == 0:
            print(f"No images found in {input_path}")
            return

        print(f"Found {len(image_files)} images in {input_path}")
        processed = 0
        skipped = 0
        for img_path in image_files:
            rel = img_path.relative_to(input_path)
            out_path = output_dir / rel.parent / (rel.stem + "_mask.png")
            success = infer_single_image(model, device, img_path, out_path, img_size, threshold)
            if success:
                processed += 1
            else:
                skipped += 1
        print(f"Batch inference complete: {processed} processed, {skipped} skipped.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary Segmentation Inference (DeepLabV3+ Mobilenet)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint, e.g. checkpoints/checkpoint_epoch_100.pth"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image or folder path"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save predicted masks"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=960,  # 与 train.py 默认一致
        help="Input size used in training (must match train.py)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for foreground mask (0.0-1.0)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on: cuda or cpu"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = build_model(
        args.checkpoint,
        device=device,
        num_classes=2,
        output_stride=16
    )

    batch_infer(model, device, args.input, args.output_dir, args.img_size, args.threshold)


if __name__ == "__main__":
    main()