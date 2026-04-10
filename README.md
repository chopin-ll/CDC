# 肺结节检测系统（YOLOv8 检测 + CNN 二次过滤）

## 项目简介
本项目使用 **YOLOv8 检测模型** 对肺部 CT 轴向切片进行结节初步定位，再使用 **轻量级 CNN 分类器** 对每个候选框进行二次判断，有效过滤假阳性。  
- **输入**：2D CT 轴向切片（PNG/JPG，灰度图）  
- **输出**：结节边界框、检测置信度、分类器置信度、直径（mm）、中心位置、可下载的诊断报告

## 文件结构
肺结节检测/
├── data/ # 原始数据（subset0~4, annotations.csv, candidates_V2.csv）
├── detection_dataset/ # 检测模型数据集（自动生成）
├── detection_model/ # 训练好的检测模型 best.pt
├── classifier_dataset/ # 分类器数据集（正负样本，自动生成）
├── classifier_model/ # 训练好的分类器模型 best.pth
├── preprocess_detection.py # 预处理：生成检测数据集（单通道）
├── train_detection.py # 训练 YOLOv8 检测模型
├── preprocess_classifier.py # 预处理：生成分类数据集
├── train_classifier.py # 训练 CNN 分类器
├── classifier_filter.py # 分类器推理模块
├── app_detection.py # 增强版 Web 界面（集成检测+分类过滤）
├── evaluate.py # 评估原始检测模型
├── evaluate_with_classifier.py # 评估检测+分类器整体性能（可选）
├── hyp_custom.yaml # 数据增强配置
├── requirements.txt # 依赖列表
└── README.md # 本文件

text

## 环境要求
- Python 3.8+
- 推荐 GPU（也可 CPU 运行，但训练较慢）
- 安装依赖：
  ```bash
  pip install -r requirements.txt
快速使用（无需重新训练）
确保已训练好检测模型和分类器（模型文件已提供）。

启动 Web 界面：

bash
streamlit run app_detection.py
在浏览器中上传 CT 切片（PNG/JPG），侧边栏可调节：

像素间距 (mm/pixel)：用于计算结节真实直径（通常 0.5~1.0）。

分类器阈值：值越高过滤越严格（推荐 0.5）。

系统会显示：

检测框（绿色）

检测置信度 + 分类器置信度

结节直径、中心位置

点击 “生成诊断报告” 下载文本报告。

训练自己的模型（如需重新训练）
1. 准备原始数据
将 subset0~subset4 文件夹、annotations.csv、candidates_V2.csv 放入 data/ 目录。

2. 生成检测数据集
bash
python preprocess_detection.py
输出：detection_dataset/images/（单通道灰度图）和 detection_dataset/labels/（YOLO 格式标签）。

3. 训练检测模型（YOLOv8）
bash
python train_detection.py
训练参数可在脚本中修改（如模型大小、轮数、批量大小）。训练完成后模型保存至 detection_model/best.pt。

4. 生成分类数据集
bash
python preprocess_classifier.py
正样本：来自 annotations.csv（结节）

负样本：来自 candidates_V2.csv 中 class=0 且与金标准距离 ≥5mm 的候选
输出：classifier_dataset/positive/ 和 classifier_dataset/negative/（64x64 单通道）。

5. 训练分类器（CNN）
bash
python train_classifier.py
训练完成后模型保存至 classifier_model/best.pth。

6. 启动 Web 界面（集成过滤）
bash
streamlit run app_detection.py
评估指标
原始检测模型
bash
python evaluate.py
输出：mAP50, mAP50-95, Precision, Recall, F1，并保存至 yolo_evaluation_results.csv。

检测 + 分类器整体性能（可选）
bash
python evaluate_with_classifier.py
需要安装 torchmetrics，输出带分类器过滤后的 mAP50 和预测框数量对比。



注意事项
所有图像处理基于单通道灰度图，上传的彩色图像会自动转为灰度。

分类器阈值可在 Web 界面实时调节，无需重新训练。

若需使用 GPU 训练，请修改 train_detection.py 中的 DEVICE = 'cuda'，并确保已安装 CUDA 版 PyTorch。

本系统仅供科研辅助，不构成医疗诊断。