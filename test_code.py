import argparse
from pathlib import Path
import torch
import torch.serialization
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

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


def load_gt_mask(mask_path, orig_size):
    """
    加载真实标签mask，并resize到原图大小
    """
    try:
        mask = Image.open(mask_path).convert("L")  # 转为灰度图
        mask = mask.resize(orig_size, Image.NEAREST)

        # 将mask转换为二值图 (0或1)
        mask_np = np.array(mask)

        # 处理不同类型的mask (0/255, 0/1, 或其他值)
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.uint8)  # 阈值化为0/1
        else:
            mask_np = mask_np.astype(np.uint8)

        return mask_np
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None


def postprocess_mask(mask_tensor, orig_size):
    """
    将网络输出的 [H, W] mask 转成原图大小的 0/1 uint8 数组
    """
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)  # 0/1
    mask_pil = Image.fromarray(mask_np * 255)  # 转为0/255显示
    mask_pil = mask_pil.convert("L")
    mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
    mask_np_resized = np.array(mask_pil)

    # 重新二值化
    mask_np_resized = (mask_np_resized > 127).astype(np.uint8)
    return mask_np_resized


def is_image_file(path):
    """
    简单判断是不是图片文件
    """
    return any(str(path).lower().endswith(ext)
               for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"])


def calculate_metrics(pred_mask, gt_mask):
    """
    计算评估指标
    """
    # 展平数组
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    # 确保两个数组形状相同
    if pred_flat.shape != gt_flat.shape:
        print(f"Warning: Shape mismatch! pred: {pred_flat.shape}, gt: {gt_flat.shape}")
        min_len = min(len(pred_flat), len(gt_flat))
        pred_flat = pred_flat[:min_len]
        gt_flat = gt_flat[:min_len]

    # 计算指标
    accuracy = accuracy_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)

    # IoU (Jaccard Score)
    iou = jaccard_score(gt_flat, pred_flat, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou
    }


def infer_single_image(model, device, image_path, mask_path, img_size, threshold=0.5):
    """
    对单张图片做推理，并与真实mask比较计算指标
    返回预测mask和评估指标
    """
    try:
        # 读取原图
        img = Image.open(image_path).convert("RGB")
    except OSError as e:
        if "truncated" in str(e).lower():
            print(f"Skipping truncated image: {image_path} (error: {e})")
            return None, None
        else:
            raise

    # 预处理
    img_tensor, orig_size = preprocess_image(img, img_size)
    img_tensor = img_tensor.to(device)

    # 加载真实mask
    gt_mask = load_gt_mask(mask_path, orig_size)
    if gt_mask is None:
        return None, None

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_mask = (probs[:, 1, :, :] > threshold).float()
        pred_mask = pred_mask.squeeze(0)  # [H, W]

    # 后处理
    pred_mask_resized = postprocess_mask(pred_mask, orig_size)

    # 计算指标
    metrics = calculate_metrics(pred_mask_resized, gt_mask)

    return pred_mask_resized, metrics


def batch_evaluate(model, device, image_dir, mask_dir, output_dir, img_size, threshold=0.5):
    """
    批量评估：读取原图和对应的mask，计算评估指标
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片文件
    image_files = []
    mask_files = []

    # 支持的图片格式
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    mask_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    # 收集图片
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return None

    print(f"Found {len(image_files)} images in {image_dir}")

    # 匹配对应的mask文件
    valid_pairs = []
    for img_path in image_files:
        img_name = img_path.stem  # 不带扩展名的文件名

        # 尝试多种可能的mask文件名格式
        possible_mask_names = [
            f"{img_name}_mask.png",
            f"{img_name}_mask.jpg",
            f"{img_name}_gt.png",
            f"{img_name}_gt.jpg",
            f"{img_name}.png",
            f"{img_name}.jpg",
            f"{img_name}_segmentation.png",
            f"{img_name}_label.png",
        ]

        found = False
        for mask_name in possible_mask_names:
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                valid_pairs.append((img_path, mask_path))
                found = True
                break

        # 如果上述格式都没找到，尝试同名的不同格式
        if not found:
            for ext in mask_extensions:
                mask_path = mask_dir / f"{img_name}{ext}"
                if mask_path.exists():
                    valid_pairs.append((img_path, mask_path))
                    found = True
                    break

        if not found:
            print(f"Warning: No corresponding mask found for {img_path.name}")

    if len(valid_pairs) == 0:
        print("No valid image-mask pairs found!")
        return None

    print(f"Found {len(valid_pairs)} valid image-mask pairs")

    # 逐对处理
    all_metrics = []
    all_pred_masks = []

    for i, (img_path, mask_path) in enumerate(valid_pairs):
        print(f"Processing {i + 1}/{len(valid_pairs)}: {img_path.name}")

        pred_mask, metrics = infer_single_image(
            model, device, img_path, mask_path, img_size, threshold
        )

        if pred_mask is not None and metrics is not None:
            all_metrics.append(metrics)
            all_pred_masks.append((img_path, pred_mask))

            # 保存预测结果（可选）
            pred_img = Image.fromarray(pred_mask * 255)
            pred_img.save(output_dir / f"{img_path.stem}_pred.png")

    # 计算总体指标
    if not all_metrics:
        print("No valid predictions to evaluate!")
        return None

    # 计算平均指标
    avg_metrics = {
        "accuracy": np.mean([m["accuracy"] for m in all_metrics]),
        "precision": np.mean([m["precision"] for m in all_metrics]),
        "recall": np.mean([m["recall"] for m in all_metrics]),
        "f1_score": np.mean([m["f1_score"] for m in all_metrics]),
        "iou": np.mean([m["iou"] for m in all_metrics]),
    }

    # 计算总体指标（将所有预测和真实值合并计算）
    if len(all_pred_masks) > 0:
        # 这里需要重新计算全局指标，但为了简单我们先使用平均值
        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print("=" * 50)
        print(f"Total Images Evaluated: {len(all_metrics)}")
        print(f"Average Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall:    {avg_metrics['recall']:.4f}")
        print(f"Average F1-Score:  {avg_metrics['f1_score']:.4f}")
        print(f"Average IoU:       {avg_metrics['iou']:.4f}")

        # 保存详细结果到JSON文件
        detailed_results = {
            "config": {
                "checkpoint": args.checkpoint,
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir),
                "img_size": img_size,
                "threshold": threshold,
                "device": str(device)
            },
            "per_image_metrics": all_metrics,
            "average_metrics": avg_metrics
        }

        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")

        return avg_metrics
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary Segmentation Evaluation (DeepLabV3+ Mobilenet)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint, e.g. checkpoints/checkpoint_epoch_100.pth"
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Directory containing ground truth mask images"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_outputs",
        help="Directory to save predicted masks and evaluation results"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=960,
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

    # 构建模型
    model = build_model(
        args.checkpoint,
        device=device,
        num_classes=2,
        output_stride=16
    )

    # 批量评估
    metrics = batch_evaluate(
        model=model,
        device=device,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        threshold=args.threshold
    )

    if metrics is not None:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed or no valid results!")


if __name__ == "__main__":
    main()