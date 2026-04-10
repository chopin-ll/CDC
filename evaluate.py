"""
评估模型效果,得到mAP50,mAP50-95,精准率,召回率和F1分数
"""


import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ========== 配置 ==========
MODEL_PATH = "detection_model/best.pt"
DATA_YAML = "temp_yolo_dataset/data.yaml"
DEVICE = 'cpu'
# DEVICE = 'cuda'
CONF_THRESH = 0.25
IOU_THRESH = 0.5

def evaluate():
    print("加载模型...")
    model = YOLO(MODEL_PATH)

    print("开始在验证集上评估...")
    results = model.val(
        data=DATA_YAML,
        device=DEVICE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        batch=16
    )

    metrics = results.box
    map50 = metrics.map50
    map50_95 = metrics.map
    # 对于单类别，取第一个元素
    precision = metrics.p[0] if isinstance(metrics.p, (list, tuple, np.ndarray)) else metrics.p
    recall = metrics.r[0] if isinstance(metrics.r, (list, tuple, np.ndarray)) else metrics.r
    f1 = metrics.f1[0] if isinstance(metrics.f1, (list, tuple, np.ndarray)) else metrics.f1

    print("\n========== 模型评估结果 ==========")
    print(f"mAP50 (IoU=0.5):      {map50:.4f}")
    print(f"mAP50-95:             {map50_95:.4f}")
    print(f"精确率 (Precision):   {precision:.4f}")
    print(f"召回率 (Recall):      {recall:.4f}")
    print(f"F1分数:               {f1:.4f}")

    # 保存到 CSV
    results_df = pd.DataFrame({
        'Metric': ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1'],
        'Value': [map50, map50_95, precision, recall, f1]
    })
    results_df.to_csv('yolo_evaluation_results.csv', index=False)
    print("评估结果已保存至 yolo_evaluation_results.csv")

if __name__ == "__main__":
    evaluate()