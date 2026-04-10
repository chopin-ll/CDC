from ultralytics import YOLO
import argparse
import cv2
import os

def detect_image(model_path, image_path, conf_threshold=0.25):
    """检测图像中的结节，打印边界框和置信度，并保存带框的图像"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)
    result = results[0]
    
    # 获取检测结果
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("未检测到任何结节。")
        return
    
    print(f"检测到 {len(boxes)} 个结节：")
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"  结节 {i+1}: 置信度 {conf:.2%}, 边界框 {xyxy}")
    
    # 可选：保存带检测框的图像
    annotated_img = result.plot()  # BGR 图像
    output_path = image_path.replace('.png', '_detected.png').replace('.jpg', '_detected.jpg')
    cv2.imwrite(output_path, annotated_img)
    print(f"带框图像已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="detection_model/best.pt", help="检测模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()
    
    try:
        detect_image(args.model, args.image, args.conf)
    except Exception as e:
        print(f"错误: {e}")