"""
使用 yolov8m.pt 训练检测模型，支持多尺度、自定义增强，模型保存至 detection_model/best.pt
"""


from ultralytics import YOLO
import os
import torch
from sklearn.model_selection import train_test_split
import shutil
import yaml

# ========== 配置 ==========
DATASET_ROOT = "detection_dataset"
OUTPUT_MODEL_DIR = "detection_model"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
# DEVICE = 'cpu'
DEVICE = 'cuda'
MULTI_SCALE = False
HYP_FILE = "hyp_custom.yaml"

# 读取自定义超参数
with open(HYP_FILE, 'r', encoding='utf-8') as f:
    hyp = yaml.safe_load(f)

# ========== 数据集划分 ==========
images_dir = os.path.join(DATASET_ROOT, "images")
labels_dir = os.path.join(DATASET_ROOT, "labels")
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

if len(image_files) == 0:
    raise RuntimeError("未找到任何图像，请先运行 preprocess_detection.py")

train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
print(f"训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")

# ========== 创建临时 YOLO 数据集 ==========
temp_yolo_dir = "temp_yolo_dataset"
if os.path.exists(temp_yolo_dir):
    shutil.rmtree(temp_yolo_dir)
os.makedirs(os.path.join(temp_yolo_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(temp_yolo_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(temp_yolo_dir, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(temp_yolo_dir, "val", "labels"), exist_ok=True)

def copy_files(file_list, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    for f in file_list:
        shutil.copy(os.path.join(src_img_dir, f), os.path.join(dst_img_dir, f))
        label_file = f.replace('.png', '.txt')
        src_label = os.path.join(src_label_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(dst_label_dir, label_file))

copy_files(train_files, images_dir, labels_dir,
           os.path.join(temp_yolo_dir, "train", "images"),
           os.path.join(temp_yolo_dir, "train", "labels"))
copy_files(val_files, images_dir, labels_dir,
           os.path.join(temp_yolo_dir, "val", "images"),
           os.path.join(temp_yolo_dir, "val", "labels"))

# 创建 data.yaml（使用相对路径）
yaml_content = f"""
train: ./train
val: ./val
nc: 1
names: ['nodule']
"""
with open(os.path.join(temp_yolo_dir, "data.yaml"), 'w') as f:
    f.write(yaml_content)

# ========== 核心执行逻辑（必须加这个保护） ==========
if __name__ == '__main__':
    # 检查设备
    if DEVICE == 'cpu' and torch.cuda.is_available():
        print("检测到 GPU，但根据配置使用 CPU。如需 GPU 请修改 DEVICE='cuda'")
        
    #强制保存路径
    save_dir = os.path.join('detection_checkpoints', 'lung_nodule_det', 'weights')
    os.makedirs(save_dir, exist_ok=True) 

    # 加载模型
    model = YOLO('yolov8m.pt')

    # 开始训练
    results = model.train(
        data=os.path.join(temp_yolo_dir, "data.yaml"),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project='detection_checkpoints',
        name='lung_nodule_det',
        exist_ok=True,
        verbose=True,
        multi_scale=MULTI_SCALE,
        patience=20,
        save_period=5,
        **hyp
    )

    # 复制最佳模型
    best_src = os.path.join('detection_checkpoints', 'lung_nodule_det', 'weights', 'best.pt')
    if os.path.exists(best_src):
        shutil.copy(best_src, os.path.join(OUTPUT_MODEL_DIR, 'best.pt'))
        print(f"模型已保存至 {os.path.join(OUTPUT_MODEL_DIR, 'best.pt')}")
    else:
        print("未找到 best.pt，请检查训练日志")

    # 清理临时目录 (可选)
    # shutil.rmtree(temp_yolo_dir)
    print("训练完成！")