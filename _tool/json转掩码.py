import json
import numpy as np
from PIL import Image, ImageDraw
import os
from tqdm import tqdm


def create_masks_from_json_file(json_file_path, output_dir='masks'):
    """
    从JSON文件读取标注数据并生成掩码图片
    每张图片的所有标注合并到一张掩码图片中
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件: {json_file_path}")
    except Exception as e:
        print(f"读取JSON文件失败: {e}")
        return

    # 统计信息
    total_tasks = len(data)
    total_masks = 0

    # 处理每个任务（每张图片）
    for task_idx, task in enumerate(tqdm(data, desc="处理任务")):
        task_id = task.get('id', f'task_{task_idx}')
        image_path = task['data']['image']

        # 从图片路径提取文件名（不含扩展名）
        image_filename = os.path.basename(image_path)
        base_name = os.path.splitext(image_filename)[0]

        # 获取图片尺寸（使用第一个标注的尺寸）
        width = None
        height = None
        all_polygons = []

        # 收集该图片的所有多边形标注
        annotations = task.get('annotations', [])
        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                if result.get('type') == 'polygonlabels':
                    # 获取图片尺寸
                    if width is None:
                        width = result['original_width']
                        height = result['original_height']

                    # 获取多边形坐标（百分比）
                    points = result['value']['points']

                    # 转换为像素坐标
                    pixel_points = []
                    for point in points:
                        x = int(point[0] * width / 100)
                        y = int(point[1] * height / 100)
                        pixel_points.append((x, y))

                    all_polygons.append(pixel_points)

        # 如果有标注，创建合并的掩码
        if all_polygons and width is not None:
            # 创建空白掩码
            mask = np.zeros((height, width), dtype=np.uint8)

            # 使用PIL绘制所有多边形
            img = Image.fromarray(mask)
            draw = ImageDraw.Draw(img)

            # 将所有多边形绘制到同一个掩码上
            for polygon in all_polygons:
                draw.polygon(polygon, fill=255)

            # 生成输出文件名（每张图片只有一个掩码文件）
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)

            # 保存掩码图像
            img.save(output_path)
            total_masks += 1

    print(f"\n处理完成!")
    print(f"总任务数: {total_tasks}")
    print(f"总掩码数: {total_masks}")
    print(f"掩码图片已保存到: {output_dir}")


def create_masks_from_json_string(json_string, output_dir='masks'):
    """
    从JSON字符串读取标注数据并生成掩码图片
    每张图片的所有标注合并到一张掩码图片中
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 解析JSON字符串
    try:
        data = json.loads(json_string)
        print("成功解析JSON字符串")
    except Exception as e:
        print(f"解析JSON字符串失败: {e}")
        return

    # 统计信息
    total_tasks = len(data)
    total_masks = 0

    # 处理每个任务（每张图片）
    for task_idx, task in enumerate(tqdm(data, desc="处理任务")):
        task_id = task.get('id', f'task_{task_idx}')
        image_path = task['data']['image']

        # 从图片路径提取文件名（不含扩展名）
        image_filename = os.path.basename(image_path)
        base_name = os.path.splitext(image_filename)[0]

        # 获取图片尺寸（使用第一个标注的尺寸）
        width = None
        height = None
        all_polygons = []

        # 收集该图片的所有多边形标注
        annotations = task.get('annotations', [])
        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                if result.get('type') == 'polygonlabels':
                    # 获取图片尺寸
                    if width is None:
                        width = result['original_width']
                        height = result['original_height']

                    # 获取多边形坐标（百分比）
                    points = result['value']['points']

                    # 转换为像素坐标
                    pixel_points = []
                    for point in points:
                        x = int(point[0] * width / 100)
                        y = int(point[1] * height / 100)
                        pixel_points.append((x, y))

                    all_polygons.append(pixel_points)

        # 如果有标注，创建合并的掩码
        if all_polygons and width is not None:
            # 创建空白掩码
            mask = np.zeros((height, width), dtype=np.uint8)

            # 使用PIL绘制所有多边形
            img = Image.fromarray(mask)
            draw = ImageDraw.Draw(img)

            # 将所有多边形绘制到同一个掩码上
            for polygon in all_polygons:
                draw.polygon(polygon, fill=255)

            # 生成输出文件名（每张图片只有一个掩码文件）
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)

            # 保存掩码图像
            img.save(output_path)
            total_masks += 1

    print(f"\n处理完成!")
    print(f"总任务数: {total_tasks}")
    print(f"总掩码数: {total_masks}")
    print(f"掩码图片已保存到: {output_dir}")


# 使用示例1：从JSON文件读取（推荐）
if __name__ == "__main__":
    # 方法1：从文件读取
    json_file_path = "/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/20160105_data/original_dataset/project-66-at-2026-01-05-03-20-3e0d7024.json"  # 替换为你的JSON文件路径
    create_masks_from_json_file(json_file_path, output_dir="/home/ej/桌面/翼界项目总览/其他项目/DeepLabV3Plus-Pytorch-master/_tool/20160105_data/original_dataset/1")

# 使用示例2：直接使用JSON字符串
# json_data = """[{"id":15225,"annotations":[...]}]"""  # 你的JSON字符串
# create_masks_from_json_string(json_data, output_dir="merged_masks")
