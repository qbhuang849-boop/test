#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

# ==== 在这里写死目录路径 ====
INPUT_DIR = "/其他项目/DeepLabV3Plus-Pytorch-master/VOCdevkit/VOC2012/SegmentationClass"  # 输入掩码目录
OUTPUT_DIR = "/其他项目/DeepLabV3Plus-Pytorch-master/VOCdevkit/VOC2012/1"  # 输出目录（会自动创建）
THRESH = 0                               # 像素值 > THRESH 视为前景
# ===========================================


def to_binary_mask(input_path, output_path, thresh=0):
    """将任意掩码图转为单通道0/255二值PNG"""
    im = Image.open(input_path)
    arr = np.array(im)

    # 多通道 → 灰度
    if arr.ndim == 3:
        arr = np.array(im.convert('L'))

    # 统一二值化
    binary = (arr > thresh).astype(np.uint8) * 255

    out = Image.fromarray(binary, mode='L')
    out.save(output_path, format='PNG')


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 允许的图像扩展名
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for filename in os.listdir(INPUT_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in exts:
            continue

        in_path = os.path.join(INPUT_DIR, filename)
        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"Processing: {filename}")
        to_binary_mask(in_path, out_path, THRESH)

    print("Done.")


if __name__ == "__main__":
    main()
