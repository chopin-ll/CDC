"""
定义 SimpleCNN/ResNet18 结构，加载分类器权重，提供 predict_patch 返回结节概率
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 注意：cv2 在 predict_patch 内部导入，避免模块加载时出错

def load_classifier(model_path="classifier_model/best_resnet18.pth", device='cpu'):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_patch(patch_img, model, device='cpu'):
    """
    返回结节类别的概率 (0~1)，不做阈值判断
    """
    import cv2  # 延迟导入，避免部署环境缺少图形库时模块加载失败

    # 转为灰度图
    if len(patch_img.shape) == 3 and patch_img.shape[2] == 3:
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2GRAY)
    if len(patch_img.shape) == 3:
        patch_img = patch_img[:, :, 0]
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    if isinstance(patch_img, np.ndarray):
        patch_img = Image.fromarray(patch_img.astype('uint8'), mode='L')
    input_tensor = transform(patch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob_nodule = probs[0, 1].item()
    return prob_nodule
