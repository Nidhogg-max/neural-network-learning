"""
predict.py - 使用训练好的模型识别单张手写数字图片
支持命令行参数:python predict.py 图片路径
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 模型定义 -------------------
class CNNHandwriting(nn.Module):
    def __init__(self):
        super(CNNHandwriting, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def predict_image(image_path, model_path='cnn_mnist_model.pth'):
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件 {model_path} 不存在，请先运行 train.py")
        return None
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片文件 {image_path} 不存在")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNHandwriting()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        img = Image.open(image_path).convert('L')
        img_resized = img.resize((28, 28))
        tensor = transform(img_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
        print(f"\n[OK] 预测结果: {pred}")
        print("各类别概率:")
        for i, p in enumerate(prob.cpu().numpy()[0]):
            print(f"  {i}: {p*100:.2f}%")

        # 保存预测结果图
        plt.imshow(img, cmap='gray')
        plt.title(f'预测数字: {pred}')
        plt.axis('off')
        plt.savefig('custom_prediction.png')
        plt.close()
        print("预测结果图已保存为 custom_prediction.png")
        return pred
    except Exception as e:
        print(f"[ERROR] 识别失败: {e}")
        return None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("用法: python predict.py 图片路径")
        path = input("请输入图片路径: ").strip()
        if path:
            predict_image(path)