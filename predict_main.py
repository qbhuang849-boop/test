import os
import time
from pathlib import Path
from typing import List, Dict, Any

# 从你的推理代码导入API
from predict_test import BinarySegment_API, initialize_segmentor


def is_image_file(file_path: str) -> bool:
    """判断文件是否为图像文件"""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return Path(file_path).suffix.lower() in img_exts


def collect_image_files(input_dir: str, recursive: bool = True) -> List[str]:
    """收集输入目录中的所有图像文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_dir}")

    image_files = []

    if input_path.is_file():
        # 单个文件
        if is_image_file(input_dir):
            image_files.append(input_dir)
        else:
            print(f"[WARNING] 不是支持的图像格式: {input_dir}")
    else:
        # 目录
        if recursive:
            # 递归搜索
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
                image_files.extend([str(p) for p in input_path.rglob(ext)])
        else:
            # 仅当前目录
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
                image_files.extend([str(p) for p in input_path.glob(ext)])

        # 按文件名排序
        image_files.sort()

    return image_files


def save_binary_mask(image_path: str, mask_pil, output_dir: str = None) -> str:
    """保存二值化黑白掩码图片"""
    from PIL import Image

    # 创建输出目录结构
    img_path = Path(image_path)
    stem = img_path.stem

    if output_dir is None:
        # 默认保存到输入目录的masks子文件夹
        output_dir = img_path.parent / "masks"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存纯掩码（黑白二值图）
    mask_path = output_dir / f"{stem}_mask.png"
    mask_pil.save(mask_path)

    return str(mask_path)


def process_single_image(
        image_path: str,
        segmentor_config: Dict[str, Any] = None,
        output_dir: str = None,
        save_results: bool = True
) -> Dict[str, Any]:
    """处理单张图像"""
    if segmentor_config is None:
        segmentor_config = {}

    # 记录开始时间
    t0 = time.perf_counter()

    # 调用分割API
    success, stats = BinarySegment_API(
        image=image_path,
        return_mask=True,
        **segmentor_config
    )

    # 记录处理时间
    dt = time.perf_counter() - t0

    # 准备结果
    result = {
        'filename': os.path.basename(image_path),
        'filepath': image_path,
        'success': success,
        'area': stats.get('area', 0),
        'area_ratio': stats.get('area_ratio', 0.0),
        'mean_prob': stats.get('mean_prob', 0.0),
        'max_prob': stats.get('max_prob', 0.0),
        'time_ms': dt * 1000,
        'stats': stats
    }

    # 保存结果
    if success and save_results and 'mask' in stats:
        mask_path = save_binary_mask(image_path, stats['mask'], output_dir)
        result['mask_path'] = mask_path

    return result


def main():
    # === 配置参数 ===
    # 输入路径（可以是文件或目录）
    INPUT_PATH = "/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/test_1212/测试数据集"

    # 模型配置
    SEGMENTOR_CONFIG = {
        'checkpoint_path': "/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/checkpoints/best_model.pth",  # 模型权重路径
        'device': "cuda",  # 或 "cpu"
        'img_size': 960,  # 模型输入大小
        'threshold': 0.5,  # 分割阈值
        'min_area_ratio': 0.001,  # 最小有效区域比例
        'debug_mode': False  # 是否显示调试信息
    }

    # 输出目录
    OUTPUT_DIR = "/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/test_1212/test"

    # 是否递归搜索子目录
    RECURSIVE_SEARCH = True

    # 是否保存结果
    SAVE_RESULTS = True

    # === 检查输入路径 ===
    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] 输入路径不存在: {INPUT_PATH}")
        return

    print(f"[INFO] 输入路径: {INPUT_PATH}")
    print(f"[INFO] 模型配置: {SEGMENTOR_CONFIG}")

    # === 收集图像文件 ===
    if os.path.isfile(INPUT_PATH):
        image_files = [INPUT_PATH]
        print(f"[INFO] 处理单个文件: {INPUT_PATH}")
    else:
        image_files = collect_image_files(INPUT_PATH, RECURSIVE_SEARCH)
        print(f"[INFO] 发现 {len(image_files)} 张图片")

    if not image_files:
        print("[WARNING] 没有找到支持的图像文件")
        return

    # === 初始化分割器 ===
    print("[INFO] 初始化分割器...")
    # 使用 initialize_segmentor 初始化分割器
    segmentor = initialize_segmentor(
        checkpoint_path=SEGMENTOR_CONFIG['checkpoint_path'],
        device=SEGMENTOR_CONFIG['device'],
        img_size=SEGMENTOR_CONFIG['img_size'],
        threshold=SEGMENTOR_CONFIG['threshold'],
        debug_mode=SEGMENTOR_CONFIG['debug_mode']
    )
    print("[INFO] 分割器初始化完成")

    # === 处理所有图片 ===
    all_results = []
    times = []
    success_count = 0

    print("\n" + "=" * 60)
    print("开始处理图片...")
    print("=" * 60)

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {os.path.basename(img_path)}")

        try:
            # 处理单张图像
            result = process_single_image(
                image_path=img_path,
                segmentor_config=SEGMENTOR_CONFIG,
                output_dir=OUTPUT_DIR,
                save_results=SAVE_RESULTS
            )

            all_results.append(result)
            times.append(result['time_ms'])

            if result['success']:
                success_count += 1
                status = "✓"
            else:
                status = "✗"

            # 显示处理结果
            print(f"  {status} 面积占比: {result['area_ratio']:.3%}")
            print(f"     置信度: {result['mean_prob']:.4f}")
            print(f"     耗时: {result['time_ms']:.1f} ms")

            if result['success'] and 'mask_path' in result:
                print(f"     掩码保存至: {result['mask_path']}")

        except Exception as e:
            print(f"[ERROR] 处理失败: {img_path}")
            print(f"        错误信息: {e}")

            # 记录失败结果
            failed_result = {
                'filename': os.path.basename(img_path),
                'filepath': img_path,
                'success': False,
                'error': str(e),
                'time_ms': 0
            }
            all_results.append(failed_result)

    # === 汇总统计 ===
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)

    if times:
        # 时间统计
        avg_time = sum(times) / len(times)
        sorted_times = sorted(times)
        median_time = sorted_times[len(times) // 2]
        p95_time = sorted_times[int(len(times) * 0.95) - 1] if len(times) > 1 else times[0]

        # 成功失败统计
        success_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]

        print(f"\n【总体统计】")
        print(f"  总计图片: {len(image_files)} 张")
        print(f"  成功分割: {success_count} 张 ({success_count / len(image_files) * 100:.1f}%)")
        print(f"  分割失败: {len(failed_results)} 张")

        print(f"\n【性能统计】")
        print(f"  平均耗时: {avg_time:.1f} ms/张")
        print(f"  中位数: {median_time:.1f} ms")
        print(f"  P95耗时: {p95_time:.1f} ms")
        print(f"  总耗时: {sum(times) / 1000:.2f} 秒")

        # 质量统计（成功案例）
        if success_results:
            avg_area_ratio = sum(r['area_ratio'] for r in success_results) / len(success_results)
            avg_mean_prob = sum(r['mean_prob'] for r in success_results) / len(success_results)
            areas = [r['area'] for r in success_results]
            max_area = max(areas)
            min_area = min(areas)

            print(f"\n【分割质量】")
            print(f"  平均面积占比: {avg_area_ratio:.3%}")
            print(f"  平均置信度: {avg_mean_prob:.4f}")
            print(f"  最大分割面积: {max_area} 像素")
            print(f"  最小分割面积: {min_area} 像素")

            # 面积分布统计
            area_ratios = [r['area_ratio'] for r in success_results]
            small_mask = sum(1 for r in area_ratios if r < 0.01)  # <1%
            medium_mask = sum(1 for r in area_ratios if 0.01 <= r < 0.1)  # 1%-10%
            large_mask = sum(1 for r in area_ratios if r >= 0.1)  # >=10%

            print(f"\n【面积分布】")
            print(f"  小目标 (<1%): {small_mask} 张 ({small_mask / len(success_results) * 100:.1f}%)")
            print(f"  中目标 (1-10%): {medium_mask} 张 ({medium_mask / len(success_results) * 100:.1f}%)")
            print(f"  大目标 (≥10%): {large_mask} 张 ({large_mask / len(success_results) * 100:.1f}%)")

        # 显示失败文件
        if failed_results:
            print(f"\n【失败文件】({len(failed_results)} 个):")
            for i, r in enumerate(failed_results[:10], 1):
                error_msg = f" - {r.get('error', '未知错误')}" if 'error' in r else ""
                print(f"  {i:2d}. {r['filename']}{error_msg}")

            if len(failed_results) > 10:
                print(f"  ... 还有 {len(failed_results) - 10} 个失败文件")

            # 保存失败列表
            fail_list_path = os.path.join(OUTPUT_DIR, "failed_files.txt")
            with open(fail_list_path, 'w', encoding='utf-8') as f:
                for r in failed_results:
                    f.write(f"{r['filepath']}\n")
            print(f"  失败列表已保存至: {fail_list_path}")

    print("\n处理结束！")


# === 辅助函数 ===
def get_results_summary(all_results: List[Dict]) -> Dict[str, Any]:
    """获取结果摘要"""
    success_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]

    summary = {
        'total': len(all_results),
        'success': len(success_results),
        'failed': len(failed_results),
        'success_rate': len(success_results) / len(all_results) if all_results else 0,
        'avg_time': sum(r.get('time_ms', 0) for r in all_results) / len(all_results) if all_results else 0,
    }

    if success_results:
        summary['avg_area_ratio'] = sum(r['area_ratio'] for r in success_results) / len(success_results)
        summary['avg_confidence'] = sum(r['mean_prob'] for r in success_results) / len(success_results)

    return summary


def export_results_to_csv(all_results: List[Dict], output_path: str = "results.csv"):
    """导出结果到CSV文件"""
    import csv

    if not all_results:
        print("[WARNING] 没有结果可导出")
        return

    fieldnames = ['filename', 'success', 'area', 'area_ratio', 'mean_prob', 'max_prob', 'time_ms', 'filepath']

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            row = {
                'filename': result['filename'],
                'success': result['success'],
                'area': result.get('area', 0),
                'area_ratio': result.get('area_ratio', 0),
                'mean_prob': result.get('mean_prob', 0),
                'max_prob': result.get('max_prob', 0),
                'time_ms': result.get('time_ms', 0),
                'filepath': result['filepath']
            }
            writer.writerow(row)

    print(f"[INFO] 结果已导出至: {output_path}")


if __name__ == "__main__":
    # 示例：直接运行主函数
    main()
