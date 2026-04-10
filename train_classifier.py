"""
训练 ResNet18 二分类器，使用正负样本，支持数据增强，保存最佳模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ========== 配置 ==========
DATA_ROOT = "classifier_dataset"
POS_DIR = os.path.join(DATA_ROOT, "positive")
NEG_DIR = os.path.join(DATA_ROOT, "negative")
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.0005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 128
MODEL_SAVE_PATH = "classifier_model/best_resnet18.pth"
os.makedirs("classifier_model", exist_ok=True)

# ========== 数据集类 ==========
class NoduleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ========== 主程序 ==========
if __name__ == '__main__':
    # 加载数据
    print("加载正样本...")
    pos_images = [os.path.join(POS_DIR, f) for f in os.listdir(POS_DIR) if f.endswith('.png')]
    print(f"正样本数量: {len(pos_images)}")
    print("加载负样本...")
    neg_images = [os.path.join(NEG_DIR, f) for f in os.listdir(NEG_DIR) if f.endswith('.png')]
    print(f"负样本数量: {len(neg_images)}")

    if len(pos_images) == 0 or len(neg_images) == 0:
        raise RuntimeError("正样本或负样本为空，请先运行 preprocess_classifier.py")

    images = pos_images + neg_images
    labels = [1] * len(pos_images) + [0] * len(neg_images)

    # 分层划分
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"训练集: {len(train_imgs)} (正:{sum(train_labels)}, 负:{len(train_labels)-sum(train_labels)})")
    print(f"验证集: {len(val_imgs)} (正:{sum(val_labels)}, 负:{len(val_labels)-sum(val_labels)})")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=10, scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = NoduleDataset(train_imgs, train_labels, transform=train_transform)
    val_dataset = NoduleDataset(val_imgs, val_labels, transform=val_transform)
    # Windows 下 num_workers 设为 0 避免多进程错误
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 构建模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += lbls.size(0)
            train_correct += (pred == lbls).sum().item()
            loop.set_postfix(loss=loss.item(), acc=train_correct/train_total)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val ]", leave=False):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += lbls.size(0)
                val_correct += (pred == lbls).sum().item()
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 保存最佳模型，验证准确率: {best_val_acc:.4f}")

    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存至: {MODEL_SAVE_PATH}")