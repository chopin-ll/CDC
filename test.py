from ultralytics import YOLO
import os

# 1. 获取当前脚本所在的绝对路径目录
# 这样可以确保无论你在哪里运行，基准目录都是对的
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 定义图片文件名 (你的长文件名)
img_filename = "1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860_nodule1_z117.png"

# 3. 拼接完整的图片绝对路径
# 假设图片在 detection_dataset/images/ 文件夹下
image_path = os.path.join(base_dir, "detection_dataset", "images", img_filename)

# 4. 加载模型
model_path = os.path.join(base_dir, "detection_model", "best.pt")
model = YOLO(model_path)

# 5. 关键：打印路径并检查文件是否存在
print(f"正在查找图片: {image_path}")

if os.path.exists(image_path):
    print("✅ 图片找到了！开始检测...")
    # 开始预测
    results = model(image_path, conf=0.25)
    # 显示结果
    results[0].show()
else:
    print("❌ 错误：找不到图片")
    print(f"请检查该路径下是否有文件：{image_path}")
    # 列出当前目录下的所有文件夹，帮你排查是不是文件夹名字写错了
    print(f"当前目录下的文件夹有: {os.listdir(base_dir)}")