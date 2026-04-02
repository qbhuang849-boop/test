from __future__ import annotations

import os
import json
import threading
from typing import Optional, Tuple, Dict, Any
import warnings
import numpy as np
from PIL import Image
import torch
import torch.serialization
from torchvision import transforms

# 和 train.py 保持一致的模型导入
try:
    from network.modeling import deeplabv3plus_mobilenet
except ImportError as e:
    print(f"[WARNING] Failed to import model: {e}")
    print("[WARNING] Make sure 'network' module is in your Python path")
    # 定义占位符函数以避免完全失败
    deeplabv3plus_mobilenet = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models/20260112_model.pth")
CFG_PATH = os.path.join(SCRIPT_DIR, "configs/config.json")


def load_config(config_path: str = CFG_PATH) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证必要配置项
        required_keys = ['device', 'img_size', 'threshold', 'min_area_ratio', 'debug_mode']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件中缺少必要的键: {key}")

        print(f"[INFO] 从 {config_path} 加载配置成功")
        return config

    except FileNotFoundError:
        print(f"[WARNING] 配置文件未找到: {config_path}")
        print("[INFO] 使用默认配置")
        return {
            'device': "cuda",
            'img_size': 960,
            'threshold': 0.5,
            'min_area_ratio': 0.001,
            'debug_mode': True
        }
    except json.JSONDecodeError as e:
        print(f"[ERROR] 配置文件格式错误: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] 加载配置文件失败: {e}")
        raise


class BinarySegmentor:
    def __init__(
            self,
            checkpoint_path: str = None,
            config_path: str = None,
            device: str = None,
            img_size: int = None,
            threshold: float = None,
            min_area_ratio: float = None,
            debug_mode: bool = None
    ):
        """
        二值分割器初始化

        Args:
            checkpoint_path: 模型权重路径
            config_path: 配置文件路径
            device: 设备类型 ("cuda" 或 "cpu")
            img_size: 输入图像大小
            threshold: 分割阈值 (0.0-1.0)
            min_area_ratio: 最小有效区域比例
            debug_mode: 是否启用调试模式
        """
        if deeplabv3plus_mobilenet is None:
            raise ImportError("Failed to import model. Please check your 'network' module.")

        # 加载配置文件
        if config_path:
            config = load_config(config_path)
        else:
            config = load_config(CFG_PATH)

        # 使用配置文件中的值，允许参数覆盖
        self.config = config.copy()

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = DEFAULT_MODEL_PATH

        if device is not None:
            self.config['device'] = device
        if img_size is not None:
            self.config['img_size'] = img_size
        if threshold is not None:
            self.config['threshold'] = threshold
        if min_area_ratio is not None:
            self.config['min_area_ratio'] = min_area_ratio
        if debug_mode is not None:
            self.config['debug_mode'] = debug_mode

        self.debug_mode = self.config['debug_mode']
        self.img_size = self.config['img_size']
        self.threshold = self.config['threshold']

        # 设置设备
        if self.config['device'] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[INFO] 使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(f"[INFO] 使用CPU")

        # 构建预处理transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载模型
        self.model = self._load_model(self.checkpoint_path)

        if self.debug_mode:
            print(f"[DEBUG] 模型从 {self.checkpoint_path} 加载")
            print(f"[DEBUG] 输入大小: {self.img_size}x{self.img_size}")
            print(f"[DEBUG] 阈值: {self.threshold}")
            print(f"[DEBUG] 最小面积比例: {self.config['min_area_ratio']}")

    def _load_model(self, checkpoint_path: str):
        """加载模型权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点未找到: {checkpoint_path}")

        if self.debug_mode:
            print(f"[INFO] 从 {checkpoint_path} 加载模型")

        # 构建模型
        model = deeplabv3plus_mobilenet(
            num_classes=2,
            output_stride=16,
            pretrained_backbone=False
        )

        # 加载checkpoint
        try:
            # 尝试最安全的方式
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if self.debug_mode:
                print("[DEBUG] 使用 weights_only=True 加载检查点")
        except Exception as e1:
            if self.debug_mode:
                print(f"[DEBUG] weights_only=True 失败: {e1}")
            try:
                # 添加safe_globals后重试
                torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if self.debug_mode:
                    print("[DEBUG] 使用 safe_globals + weights_only=True 加载")
            except Exception as e2:
                if self.debug_mode:
                    print(f"[DEBUG] 安全加载失败: {e2}")
                # 最后兜底
                warnings.warn(
                    "使用 weights_only=False 加载检查点。"
                    "请确保信任检查点的来源！"
                )
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if self.debug_mode:
                    print("[DEBUG] 使用 weights_only=False 加载（不安全模式）")

        # 提取state_dict
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # 加载权重
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        if self.debug_mode:
            print(f"[INFO] 模型加载成功")
            print(f"[INFO] 模型在设备上: {next(model.parameters()).device}")

        return model

    def _preprocess_image(self, image_pil: Image.Image) -> torch.Tensor:
        """预处理图像"""
        # Resize
        img_resized = image_pil.resize((self.img_size, self.img_size), Image.BILINEAR)

        # ToTensor + Normalize
        img_tensor = self.transform(img_resized)  # [3, H, W]

        return img_tensor.unsqueeze(0)  # [1, 3, H, W]

    def _postprocess_mask(self, mask_tensor: torch.Tensor, orig_size: Tuple[int, int]) -> Image.Image:
        """后处理掩码"""
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255  # 0/1 -> 0/255
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.convert("L")  # 转为L模式
        mask_pil = mask_pil.resize(orig_size, Image.NEAREST)
        return mask_pil

    @torch.no_grad()
    def segment_single(
            self,
            image: Image.Image | str,
            return_mask: bool = True,
            min_area_ratio: float = None,
            **kwargs
    ) -> Tuple[int, Dict[str, Any]]:
        """
        对单张图像进行分割

        Args:
            image: PIL Image对象或图像路径
            return_mask: 是否返回掩码图像
            min_area_ratio: 最小有效区域比例阈值
            **kwargs: 其他参数（覆盖类初始化参数）

        Returns:
            tuple: (success, stats)
                - success: 0或1，表示是否检测到有效分割
                - stats: 包含统计信息的字典，可能包含'mask'字段
        """
        # 处理参数覆盖
        threshold = kwargs.get('threshold', self.threshold)
        min_area_ratio = min_area_ratio or self.config['min_area_ratio']

        # 加载图像
        if not isinstance(image, Image.Image):
            if self.debug_mode:
                print(f"[INFO] 处理图像: {image}")
            try:
                image_pil = Image.open(image).convert("RGB")
            except OSError as e:
                if "truncated" in str(e).lower():
                    print(f"[WARNING] 跳过损坏的图像: {image}")
                    return 0, {"error": "truncated_image", "message": str(e)}
                else:
                    raise
        else:
            image_pil = image

        orig_size = image_pil.size  # (W, H)
        orig_area = orig_size[0] * orig_size[1]

        if self.debug_mode:
            print(f"[INFO] 原始图像大小: {orig_size}")
            print(f"[INFO] 原始面积: {orig_area} 像素")

        # 预处理
        input_tensor = self._preprocess_image(image_pil)
        input_tensor = input_tensor.to(self.device)

        # 推理
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_mask = (probs[:, 1, :, :] > threshold).float()  # 前景通道 > 阈值
        pred_mask = pred_mask.squeeze(0)  # [H, W]

        # 后处理
        mask_pil = self._postprocess_mask(pred_mask, orig_size)

        # 计算统计信息
        mask_np = np.array(mask_pil)
        area = int((mask_np > 0).sum())
        area_ratio = area / float(orig_area)

        # 计算置信度统计
        prob_map = probs[0, 1, :, :].cpu().numpy()  # 前景概率图
        mean_prob = float(prob_map[pred_mask.cpu().numpy() > 0].mean()) if area > 0 else 0.0
        max_prob = float(prob_map.max())

        # 判断是否成功（有有效分割）
        success = int((area_ratio >= min_area_ratio) and (area > 0))

        # 构建统计信息
        stats = {
            "area": area,
            "area_ratio": area_ratio,
            "mean_prob": mean_prob,
            "max_prob": max_prob,
            "image_size": orig_size,
            "threshold_used": threshold,
            "model_input_size": self.img_size,
            "success": bool(success)
        }

        if return_mask:
            stats["mask"] = mask_pil

        if self.debug_mode:
            print(f"[RESULT] 面积: {area} 像素")
            print(f"[RESULT] 面积比例: {area_ratio:.4f} (最小: {min_area_ratio})")
            print(f"[RESULT] 平均概率: {mean_prob:.4f}")
            print(f"[RESULT] 最大概率: {max_prob:.4f}")
            print(f"[RESULT] 成功: {'是' if success else '否'}")

        return success, stats

    @torch.no_grad()
    def batch_segment(
            self,
            image_paths: list,
            output_dir: str = "outputs",
            min_area_ratio: float = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        批量处理图像

        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            min_area_ratio: 最小有效区域比例阈值
            **kwargs: 其他参数

        Returns:
            批量处理结果统计
        """
        import os
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "details": []
        }

        min_area_ratio = min_area_ratio or self.config['min_area_ratio']

        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                print(f"[WARNING] 图像未找到: {img_path}")
                results["failed"] += 1
                continue

            # 处理单张图像
            success, stats = self.segment_single(
                str(img_path),
                return_mask=True,
                min_area_ratio=min_area_ratio,
                **kwargs
            )

            # 保存掩码
            if success and "mask" in stats:
                out_name = img_path.stem + "_mask.png"
                out_path = output_dir / out_name
                stats["mask"].save(out_path)
                stats["mask_path"] = str(out_path)

            # 记录结果
            results["processed"] += 1
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1

            results["details"].append({
                "image": str(img_path),
                "success": bool(success),
                "area_ratio": stats.get("area_ratio", 0),
                "mean_prob": stats.get("mean_prob", 0)
            })

            if self.debug_mode:
                status = "✓" if success else "✗"
                print(f"[BATCH] {status} {img_path.name}: area_ratio={stats.get('area_ratio', 0):.4f}")

        return results


# 全局变量，用于存储 BinarySegmentor 实例
segmentor: Optional[BinarySegmentor] = None
_init_lock = threading.Lock()


def initialize_segmentor(
        checkpoint_path: str = None,
        config_path: str = None,
        device: str = None,
        img_size: int = None,
        threshold: float = None,
        min_area_ratio: float = None,
        debug_mode: bool = None
) -> BinarySegmentor:
    """初始化二值分割器（单例模式）"""
    global segmentor

    if segmentor is None:
        with _init_lock:
            if segmentor is None:
                segmentor = BinarySegmentor(
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    device=device,
                    img_size=img_size,
                    threshold=threshold,
                    min_area_ratio=min_area_ratio,
                    debug_mode=debug_mode
                )
                print("[INFO] 分割器初始化完成")

    return segmentor


def BinarySegment_API(
        image: Image.Image | str,
        checkpoint_path: str = None,
        config_path: str = None,
        device: str = None,
        img_size: int = None,
        threshold: float = None,
        return_mask: bool = True,
        min_area_ratio: float = None,
        debug_mode: bool = None,
        **kwargs
) -> Tuple[int, Dict[str, Any]]:
    """
    API接口函数 - 返回元组格式 (success, stats)

    Args:
        image: Image对象或图像文件路径
        checkpoint_path: 模型权重路径（可选）
        config_path: 配置文件路径（可选）
        device: 设备类型（可选）
        img_size: 输入图像大小（可选）
        threshold: 分割阈值（可选）
        return_mask: 是否返回掩码
        min_area_ratio: 最小有效区域比例（可选）
        debug_mode: 调试模式（可选）
        **kwargs: 其他参数

    Returns:
        tuple: (success, stats)
            - success: 0或1，表示是否检测到有效分割
            - stats: 包含统计信息的字典，可能包含'mask'字段
    """
    global segmentor

    # 初始化segmentor（如果尚未初始化或参数不同）
    if segmentor is None:
        segmentor = initialize_segmentor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            img_size=img_size,
            threshold=threshold,
            min_area_ratio=min_area_ratio,
            debug_mode=debug_mode
        )

    # 进行图像分割处理
    success, stats = segmentor.segment_single(
        image,
        return_mask=return_mask,
        min_area_ratio=min_area_ratio,
        **kwargs
    )
    print(success, stats)
    return success, stats


if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    print("加载的配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")