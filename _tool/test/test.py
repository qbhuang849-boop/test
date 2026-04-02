import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageFile
import cv2
import datetime
from torchvision import transforms
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from network.modeling import deeplabv3plus_mobilenet, deeplabv3plus_resnet101


# 设置中文字体
def setup_chinese_font():
    """设置中文字体，支持中文显示"""
    try:
        font_path = "simhei.ttf"
        if os.path.exists(font_path):
            matplotlib.font_manager.fontManager.addfont(font_path)
            font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"✓ 成功加载中文字体: {font_name}")
        else:
            font_names = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            matplotlib.rcParams['font.sans-serif'] = font_names
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"✓ 使用系统字体")
    except Exception as e:
        print(f"⚠ 字体设置失败: {e}")


setup_chinese_font()

# 设置PIL处理截断图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class BinarySegmentationTester:
    def __init__(self, model_path, device='cuda', img_size=960, threshold=0.3,
                 backbone='resnet101', output_stride=16):
        """
        初始化测试器
        Args:
            model_path: 训练好的模型路径
            device: 使用设备
            img_size: 输入图像大小
            threshold: 分割阈值
            backbone: 骨干网络 ('resnet101' 或 'mobilenet')
            output_stride: 输出步长 (8 或 16)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.threshold = threshold
        self.backbone = backbone
        self.output_stride = output_stride

        print(f'使用设备: {self.device}')
        print(f'骨干网络: {backbone}')
        print(f'输出步长: {output_stride}')
        print(f'图像尺寸: {img_size}x{img_size}')
        print(f'分割阈值: {threshold}')

        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """加载训练好的模型"""
        # 根据backbone选择模型
        if self.backbone == 'resnet101':
            model = deeplabv3plus_resnet101(num_classes=2, output_stride=self.output_stride,
                                            pretrained_backbone=False)
        elif self.backbone == 'mobilenet':
            model = deeplabv3plus_mobilenet(num_classes=2, output_stride=self.output_stride,
                                            pretrained_backbone=False)
        else:
            raise ValueError(f"不支持的骨干网络: {self.backbone}")

        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除可能的'module.'前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)

        print(f'模型加载自 {model_path}')
        if 'best_iou' in checkpoint:
            print(f'训练时最佳IoU: {checkpoint["best_iou"]:.4f}')

        return model

    def safe_open_image(self, image_path):
        """安全打开图像，处理损坏文件"""
        try:
            image = Image.open(image_path).convert('RGB')
            image.verify()
            image = Image.open(image_path).convert('RGB')
            return image, None
        except Exception as e:
            return None, f"图像文件损坏或无法读取: {str(e)}"

    def safe_open_mask(self, mask_path):
        """安全打开mask，处理损坏文件"""
        try:
            mask = Image.open(mask_path).convert('L')
            mask.verify()
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)
            mask_np = (mask_np > 128).astype(np.uint8)
            return mask_np, None
        except Exception as e:
            return None, f"Mask文件损坏或无法读取: {str(e)}"

    def preprocess_image(self, image_path):
        """预处理图像"""
        image, error = self.safe_open_image(image_path)
        if error:
            raise ValueError(error)

        original_size = image.size
        image_resized = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image_tensor = self.transform(image_resized).unsqueeze(0)

        return image_tensor.to(self.device), original_size

    def inference(self, image_path, save_dir=None, save_visualization=True):
        """
        单张图像推理
        Returns:
            pred_mask: 预测的mask
            confidence_map: 置信度图
            stats: 统计信息字典
        """
        try:
            image_tensor, original_size = self.preprocess_image(image_path)
        except ValueError as e:
            print(f"⚠ 跳过损坏图像: {image_path}")
            empty_mask = np.zeros((100, 100), dtype=np.uint8)
            empty_confidence = np.zeros((100, 100), dtype=np.float32)
            stats = {
                "area": 0, "area_ratio": 0.0, "mean_prob": 0.0, "max_prob": 0.0,
                "image_size": (100, 100), "threshold_used": self.threshold,
                "success": False, "error": str(e)
            }
            return empty_mask, empty_confidence, stats

        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence = probs[1]

        # 转为numpy
        confidence_map = confidence.cpu().numpy()
        binary_mask = (confidence_map > self.threshold).astype(np.uint8)

        # 恢复到原始尺寸
        if binary_mask.shape != original_size[::-1]:
            binary_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            confidence_map = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)

        # 计算统计信息
        area = int(binary_mask.sum())
        orig_area = original_size[0] * original_size[1]
        area_ratio = area / float(orig_area) if orig_area > 0 else 0.0
        mean_prob = float(confidence_map[binary_mask > 0].mean()) if area > 0 else 0.0
        max_prob = float(confidence_map.max())

        stats = {
            "area": area, "area_ratio": area_ratio, "mean_prob": mean_prob,
            "max_prob": max_prob, "image_size": original_size,
            "threshold_used": self.threshold, "success": bool(area > 0)
        }

        # 保存可视化结果
        if save_dir and save_visualization:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.visualize_result(image_path, binary_mask, confidence_map,
                                  save_dir / f'result_{Path(image_path).stem}.jpg')

        return binary_mask, confidence_map, stats

    def visualize_result(self, image_path, pred_mask, confidence_map, save_path):
        """可视化结果"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠ 无法读取图像: {image_path}")
                return

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay = image.copy()
            overlay[pred_mask == 1] = [255, 0, 0]
            blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('原始图像')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('预测Mask')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(blended)
            plt.title('叠加结果')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠ 保存可视化结果失败: {save_path}, 错误: {str(e)}")

    def calculate_metrics(self, pred, true):
        """计算各种评估指标"""
        pred_flat = pred.ravel()
        true_flat = true.ravel()
        tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat, labels=[0, 1]).ravel()

        # 安全计算各项指标
        iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
        mIoU = (iou_fg + iou_bg) / 2
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

        # 纯背景图片的指标
        pure_bg_acc = tn / (tp + tn + fp + fn) if np.sum(true_flat) == 0 and (tp + tn + fp + fn) > 0 else None

        return {
            'true_positive': int(tp), 'false_positive': int(fp),
            'true_negative': int(tn), 'false_negative': int(fn),
            'iou_foreground': iou_fg, 'iou_background': iou_bg, 'mIoU': mIoU,
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1_score': f1_score, 'dice_coefficient': dice,
            'false_positive_rate': fpr, 'false_discovery_rate': fdr,
            'pure_background_accuracy': pure_bg_acc
        }

    def evaluate_dataset(self, images_dir, masks_dir, save_dir=None):
        """评估整个测试集"""
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        image_files = sorted([f for f in os.listdir(images_dir)
                              if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))])

        print(f'找到 {len(image_files)} 张测试图像')

        all_metrics = []
        skipped_images = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建保存目录
        if save_dir:
            save_dir = Path(save_dir)
            timestamped_dir = save_dir / f'results_{timestamp}'
            vis_dir = timestamped_dir / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)

        # 逐张评估
        for img_idx, img_name in enumerate(tqdm(image_files, desc='评估进度')):
            img_path = images_dir / img_name
            base_name = os.path.splitext(img_name)[0]

            # 查找对应的mask文件
            mask_path = None
            for ext in ['.png', '.jpg']:
                test_path = masks_dir / (base_name + ext)
                if test_path.exists():
                    mask_path = test_path
                    break

            if mask_path is None:
                print(f'警告: 未找到 {img_name} 对应的mask文件')
                skipped_images.append((img_name, "mask文件不存在"))
                continue

            try:
                # 检查文件是否可读
                image_test, error = self.safe_open_image(str(img_path))
                if error:
                    skipped_images.append((img_name, error))
                    continue

                mask_test, error = self.safe_open_mask(str(mask_path))
                if error:
                    skipped_images.append((img_name, error))
                    continue

                # 推理
                pred_mask, confidence_map, stats = self.inference(
                    str(img_path), save_dir=None, save_visualization=False
                )

                # 加载真实mask并计算指标
                true_mask, _ = self.safe_open_mask(mask_path)
                if pred_mask.shape != true_mask.shape:
                    pred_mask = cv2.resize(pred_mask, (true_mask.shape[1], true_mask.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

                # 保存可视化
                if save_dir:
                    try:
                        self.visualize_result(str(img_path), pred_mask, confidence_map,
                                              vis_dir / f'vis_{base_name}.jpg')
                    except Exception as e:
                        print(f"⚠ 保存可视化失败: {img_name}")

                # 计算指标
                metrics = self.calculate_metrics(pred_mask, true_mask)
                metrics.update(stats)
                metrics['image_name'] = img_name
                all_metrics.append(metrics)

                # 清理内存
                del pred_mask, confidence_map, true_mask, mask_test, image_test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f'⚠ 处理图像 {img_name} 时发生错误: {str(e)}')
                skipped_images.append((img_name, str(e)))

        # 打印跳过的图像
        if skipped_images:
            print(f'\n⚠ 跳过了 {len(skipped_images)} 张图像')

        if not all_metrics:
            return "错误: 没有成功处理任何图像!"

        # 计算总体指标
        overall_cm = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        for m in all_metrics:
            overall_cm['tp'] += m['true_positive']
            overall_cm['fp'] += m['false_positive']
            overall_cm['tn'] += m['true_negative']
            overall_cm['fn'] += m['false_negative']

        overall_metrics = self.calculate_metrics(
            np.array([1] * overall_cm['tp'] + [0] * overall_cm['tn'] +
                     [1] * overall_cm['fp'] + [0] * overall_cm['fn']),
            np.array([1] * overall_cm['tp'] + [0] * overall_cm['tn'] +
                     [0] * overall_cm['fp'] + [1] * overall_cm['fn'])
        )
        overall_metrics['num_images'] = len(all_metrics)

        # 分类统计
        pure_bg = [m for m in all_metrics if m.get('pure_background_accuracy') is not None]
        foreground = [m for m in all_metrics if m.get('pure_background_accuracy') is None]

        # 计算总准确率
        total_correct = sum(1 for m in foreground if m['true_positive'] > 0)
        total_correct += sum(1 for m in pure_bg if m['false_positive'] == 0)
        overall_metrics['total_accuracy'] = total_correct / len(all_metrics)

        # 生成报告
        evaluation_text = self.generate_report(overall_metrics, pure_bg, foreground, skipped_images)
        print(evaluation_text)

        # 保存结果
        if save_dir:
            result_path = timestamped_dir / 'evaluation_results.txt'
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(evaluation_text)
            print(f'\n评估结果已保存到: {result_path}')

        return evaluation_text

    def generate_report(self, overall, pure_bg, foreground, skipped):
        """生成评估报告"""
        lines = ['=' * 60, '评估结果汇总', '=' * 60, '']

        lines.append(f'📊 整体指标 (基于 {overall["num_images"]} 张图像):')
        lines.append(f'   总和准确率:            {overall.get("total_accuracy", 0):.4f}')
        lines.append(f'   平均IoU (mIoU):        {overall["mIoU"]:.4f}')
        lines.append(f'   前景IoU:               {overall["iou_foreground"]:.4f}')
        lines.append(f'   精确率:                {overall["precision"]:.4f}')
        lines.append(f'   召回率:                {overall["recall"]:.4f}')
        lines.append(f'   F1分数:                {overall["f1_score"]:.4f}')
        lines.append('')

        lines.append(f'🎯 前景图像 ({len(foreground)} 张):')
        if foreground:
            fg_iou = np.mean([m['iou_foreground'] for m in foreground])
            fg_detected = sum(1 for m in foreground if m['true_positive'] > 0)
            lines.append(
                f'   检测准确率:            {fg_detected / len(foreground):.4f} ({fg_detected}/{len(foreground)})')
            lines.append(f'   平均IoU:               {fg_iou:.4f}')
        lines.append('')

        lines.append(f'🚫 纯背景图像 ({len(pure_bg)} 张):')
        if pure_bg:
            bg_perfect = sum(1 for m in pure_bg if m['false_positive'] == 0)
            lines.append(f'   完美预测率:            {bg_perfect / len(pure_bg):.4f} ({bg_perfect}/{len(pure_bg)})')
        lines.append('')

        if skipped:
            lines.append(f'⚠ 跳过了 {len(skipped)} 张图像')

        lines.append('=' * 60)
        return '\n'.join(lines)


def main():
    # ================ 配置参数 ================

    # 模型配置
    MODEL_PATH = "/home/ej/桌面/翼界项目总览/git/DeepLabV3Plus/models/20260118_test_ten101/best_model.pth"
    BACKBONE = 'mobilenet'  # 'resnet101' 或 'mobilenet'
    OUTPUT_STRIDE = 16  # 8 或 16

    # 测试数据路径
    TEST_IMAGES_DIR = "/media/ej/data/物种识别/空拍分割算法/_tool/20260128/val/images"
    TEST_MASKS_DIR = "/media/ej/data/物种识别/空拍分割算法/_tool/20260128/val/masks"
    OUTPUT_DIR = "/media/ej/data/物种识别/空拍分割算法/_tool/20260128/val/test"

    # 单张图像测试（可选）
    TEST_SINGLE_IMAGE = False
    SINGLE_IMAGE_PATH = "test_image.jpg"
    SINGLE_MASK_PATH = "test_mask.png"

    # 模型参数
    IMG_SIZE = 960
    THRESHOLD = 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PREDICTIONS = True

    # ================ 开始测试 ================

    print('=' * 60)
    print('DeepLabV3+ 二值分割模型测试')
    print('=' * 60)
    print(f'模型路径: {MODEL_PATH}')
    print(f'骨干网络: {BACKBONE}')
    print(f'输出步长: {OUTPUT_STRIDE}')
    print('=' * 60)

    if not os.path.exists(MODEL_PATH):
        print(f'错误: 模型文件不存在: {MODEL_PATH}')
        return

    # 创建测试器
    tester = BinarySegmentationTester(
        MODEL_PATH, device=DEVICE, img_size=IMG_SIZE, threshold=THRESHOLD,
        backbone=BACKBONE, output_stride=OUTPUT_STRIDE
    )

    if TEST_SINGLE_IMAGE:
        # 单张图像测试
        print(f'\n正在测试单张图像: {SINGLE_IMAGE_PATH}')

        if SAVE_PREDICTIONS:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            single_output_dir = Path(OUTPUT_DIR) / f"single_test_{timestamp}"
            pred_mask, confidence_map, stats = tester.inference(
                SINGLE_IMAGE_PATH, save_dir=single_output_dir, save_visualization=True
            )
            print(f'结果已保存到: {single_output_dir}')
            print(f'分割面积: {stats["area"]} 像素')
            print(f'面积占比: {stats["area_ratio"]:.4f}')
    else:
        # 整个测试集评估
        print(f'\n正在进行数据集评估...')

        if not os.path.exists(TEST_IMAGES_DIR):
            print(f'错误: 测试图像目录不存在: {TEST_IMAGES_DIR}')
            return
        if not os.path.exists(TEST_MASKS_DIR):
            print(f'错误: 测试mask目录不存在: {TEST_MASKS_DIR}')
            return

        tester.evaluate_dataset(
            TEST_IMAGES_DIR, TEST_MASKS_DIR,
            save_dir=OUTPUT_DIR if SAVE_PREDICTIONS else None
        )
        print('\n评估完成!')


if __name__ == '__main__':
    main()