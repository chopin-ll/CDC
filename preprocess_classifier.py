"""
从 annotations 提取正样本，从 candidates_V2 提取负样本（排除靠近金标准的），保存为 64x64 灰度图
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
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates_V2.csv")
OUTPUT_POS_DIR = "classifier_dataset/positive"
OUTPUT_NEG_DIR = "classifier_dataset/negative"
os.makedirs(OUTPUT_POS_DIR, exist_ok=True)
os.makedirs(OUTPUT_NEG_DIR, exist_ok=True)

PATCH_SIZE = 64           # 分类器输入尺寸 64x64
DEFAULT_DIAMETER_MM = 10
MATCH_TOLERANCE_MM = 5.0  # 排除与金标准结节距离小于此值的负样本

# 建立 seriesuid -> CT 文件路径的映射
seriesuid_to_path = {}
for subset in SUBSETS:
    subset_path = os.path.join(DATA_DIR, subset)
    if not os.path.isdir(subset_path):
        continue
    for fname in os.listdir(subset_path):
        if fname.endswith('.mhd'):
            suid = fname[:-4]
            seriesuid_to_path[suid] = os.path.join(subset_path, fname)

def get_ct_volume(seriesuid):
    mhd_path = seriesuid_to_path.get(seriesuid)
    if mhd_path is None:
        raise FileNotFoundError(f"未找到 {seriesuid}")
    itk_img = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(itk_img)
    return volume, itk_img

def world_to_voxel(coord_world, origin, spacing):
    x = int((coord_world[0] - origin[0]) / spacing[0])
    y = int((coord_world[1] - origin[1]) / spacing[1])
    z = int((coord_world[2] - origin[2]) / spacing[2])
    return (x, y, z)

def extract_patch_2d(volume, center_voxel, patch_size=64):
    """提取轴向切片并缩放到 patch_size x patch_size，返回单通道 numpy 数组"""
    z = center_voxel[2]
    if z < 0 or z >= volume.shape[0]:
        return None
    slice_2d = volume[z, :, :]
    min_val, max_val = np.min(slice_2d), np.max(slice_2d)
    if max_val > min_val:
        slice_norm = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        slice_norm = np.zeros_like(slice_2d, dtype=np.uint8)
    resized = cv2.resize(slice_norm, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return resized  # 返回 (64,64) 单通道

# ========== 处理正样本（annotations） ==========
print("处理正样本（结节）...")
annotations = pd.read_csv(ANNOTATIONS_FILE)
pos_count = 0
for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
    seriesuid = row['seriesuid']
    coord = (row['coordX'], row['coordY'], row['coordZ'])
    try:
        volume, itk_img = get_ct_volume(seriesuid)
        origin = itk_img.GetOrigin()
        spacing = itk_img.GetSpacing()
        center_voxel = world_to_voxel(coord, origin, spacing)
        patch = extract_patch_2d(volume, center_voxel, PATCH_SIZE)
        if patch is not None:
            out_path = os.path.join(OUTPUT_POS_DIR, f"{seriesuid}_pos_{idx}.png")
            cv2.imwrite(out_path, patch)
            pos_count += 1
    except Exception as e:
        print(f"正样本跳过 {seriesuid}: {e}")
print(f"正样本提取完成: {pos_count} 张")

# ========== 处理负样本（candidates_V2.csv 中 class=0） ==========
print("处理负样本（非结节）...")
candidates = pd.read_csv(CANDIDATES_FILE)
# 只取 class==0 的候选
neg_candidates = candidates[candidates['class'] == 0].copy()
print(f"候选负样本总数: {len(neg_candidates)}")

# 为了排除与金标准结节重叠的负样本，构建金标准坐标列表
gold_std = []
for _, row in annotations.iterrows():
    gold_std.append((row['seriesuid'], (row['coordX'], row['coordY'], row['coordZ'])))

neg_count = 0
for idx, row in tqdm(neg_candidates.iterrows(), total=len(neg_candidates)):
    seriesuid = row['seriesuid']
    coord = (row['coordX'], row['coordY'], row['coordZ'])
    # 检查是否与任一金标准结节距离过近
    too_close = False
    for gs_suid, gs_coord in gold_std:
        if gs_suid == seriesuid:
            dist = np.sqrt(sum((coord[i]-gs_coord[i])**2 for i in range(3)))
            if dist < MATCH_TOLERANCE_MM:
                too_close = True
                break
    if too_close:
        continue
    try:
        volume, itk_img = get_ct_volume(seriesuid)
        origin = itk_img.GetOrigin()
        spacing = itk_img.GetSpacing()
        center_voxel = world_to_voxel(coord, origin, spacing)
        patch = extract_patch_2d(volume, center_voxel, PATCH_SIZE)
        if patch is not None:
            out_path = os.path.join(OUTPUT_NEG_DIR, f"{seriesuid}_neg_{idx}.png")
            cv2.imwrite(out_path, patch)
            neg_count += 1
    except Exception as e:
        print(f"负样本跳过 {seriesuid}: {e}")
print(f"负样本提取完成: {neg_count} 张")
print(f"数据集准备完成！正样本: {pos_count}, 负样本: {neg_count}")