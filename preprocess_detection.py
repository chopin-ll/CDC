"""
从 CT 和 annotations 提取结节中心切片，保存为单通道 PNG，并生成 YOLO 标签
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm

# ========== 配置 ==========
DATA_DIR = "data"
SUBSETS = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('subset')]
if not SUBSETS:
    raise RuntimeError("在 data 目录下未找到任何 subset 文件夹（如 subset0, subset1...）")
print(f"找到的 subset 文件夹: {SUBSETS}")

ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
OUTPUT_IMG_DIR = "detection_dataset/images"
OUTPUT_LABEL_DIR = "detection_dataset/labels"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

DEFAULT_DIAMETER_MM = 10

# ========== 读取 annotations ==========
print("读取 annotations.csv ...")
annotations = pd.read_csv(ANNOTATIONS_FILE)
print(f"annotations 中共有 {len(annotations)} 个结节")

required_cols = ['seriesuid', 'coordX', 'coordY', 'coordZ']
for col in required_cols:
    if col not in annotations.columns:
        raise ValueError(f"annotations.csv 缺少 {col} 列")

if 'diameter_mm' not in annotations.columns:
    print("警告: annotations.csv 中没有 diameter_mm 列，将使用默认直径 {:.1f} mm".format(DEFAULT_DIAMETER_MM))
    annotations['diameter_mm'] = DEFAULT_DIAMETER_MM

# ========== 建立 seriesuid 到文件路径的映射 ==========
print("建立 CT 文件索引...")
seriesuid_to_path = {}
for subset in SUBSETS:
    subset_path = os.path.join(DATA_DIR, subset)
    if not os.path.isdir(subset_path):
        continue
    for fname in os.listdir(subset_path):
        if fname.endswith('.mhd'):
            suid = fname[:-4]
            seriesuid_to_path[suid] = os.path.join(subset_path, fname)
print(f"索引完成，共找到 {len(seriesuid_to_path)} 个 CT 文件")

# ========== 辅助函数 ==========
def get_ct_volume(seriesuid):
    mhd_path = seriesuid_to_path.get(seriesuid)
    if mhd_path is None:
        raise FileNotFoundError(f"未找到 seriesuid={seriesuid} 的 CT 文件")
    itk_img = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(itk_img)
    return volume, itk_img

def world_to_voxel(coord_world, origin, spacing):
    x = int((coord_world[0] - origin[0]) / spacing[0])
    y = int((coord_world[1] - origin[1]) / spacing[1])
    z = int((coord_world[2] - origin[2]) / spacing[2])
    return (x, y, z)

def save_slice_and_label(volume, itk_img, center_voxel, diameter_mm, seriesuid, nodule_idx, output_img_dir, output_label_dir):
    z = center_voxel[2]
    if z < 0 or z >= volume.shape[0]:
        return False

    slice_2d = volume[z, :, :]
    # 归一化到 0-255
    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)
    if max_val > min_val:
        slice_norm = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        slice_norm = np.zeros_like(slice_2d, dtype=np.uint8)

    # 保存为单通道灰度图
    img_filename = f"{seriesuid}_nodule{nodule_idx}_z{z}.png"
    img_path = os.path.join(output_img_dir, img_filename)
    cv2.imwrite(img_path, slice_norm)

    # 生成 YOLO 标签
    spacing = itk_img.GetSpacing()
    origin = itk_img.GetOrigin()
    cx_pixel = center_voxel[0]
    cy_pixel = center_voxel[1]
    pixel_spacing_xy = (spacing[0] + spacing[1]) / 2.0
    diameter_pixel = diameter_mm / pixel_spacing_xy
    img_h, img_w = slice_2d.shape
    x_center = cx_pixel / img_w
    y_center = cy_pixel / img_h
    width = diameter_pixel / img_w
    height = diameter_pixel / img_h
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)

    label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    label_filename = img_filename.replace('.png', '.txt')
    label_path = os.path.join(output_label_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write(label_line)
    return True

# ========== 主循环 ==========
print("开始生成检测数据集（单通道）...")
success_count = 0
skip_count = 0

for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
    seriesuid = row['seriesuid']
    coord_world = (row['coordX'], row['coordY'], row['coordZ'])
    diameter_mm = row['diameter_mm']

    try:
        volume, itk_img = get_ct_volume(seriesuid)
        origin = itk_img.GetOrigin()
        spacing = itk_img.GetSpacing()
        center_voxel = world_to_voxel(coord_world, origin, spacing)

        ok = save_slice_and_label(volume, itk_img, center_voxel, diameter_mm,
                                  seriesuid, idx, OUTPUT_IMG_DIR, OUTPUT_LABEL_DIR)
        if ok:
            success_count += 1
        else:
            skip_count += 1
    except FileNotFoundError:
        skip_count += 1
        continue
    except Exception as e:
        print(f"处理 {seriesuid} 结节 {idx} 时出错: {e}")
        skip_count += 1

print(f"处理完成: 成功生成 {success_count} 个样本，跳过 {skip_count} 个（文件缺失或越界）")
print(f"图像保存在 {OUTPUT_IMG_DIR}, 标签保存在 {OUTPUT_LABEL_DIR}")