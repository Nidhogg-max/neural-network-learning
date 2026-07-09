"""
train.py - 训练CNN手写数字识别模型,输出结果报告
生成结果：
  1. cnn_mnist_model.pth         - 模型权重
  2. training_curves.png         - 损失和准确率曲线
  3. learning_rate.png           - 学习率变化曲线
  4. confusion_matrix.png        - 混淆矩阵（热力图）
  5. confusion_matrix_norm.png   - 归一化混淆矩阵
  6. class_accuracy.png          - 各类别准确率条形图
  7. error_samples.png           - 错误预测样本
  8. classification_report.txt   - 精确率/召回率/F1-score文本报告
  9. model_summary.txt           - 模型参数量、结构概要
 10. training_summary.txt        - 训练总体信息
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import RandomPerspective, ColorJitter, RandomAffine, RandomErasing
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# ------------------- 中文字体设置 -------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)
np.random.seed(42)

# ------------------- CNN模型（带BatchNorm） -------------------
class CNNHandwriting(nn.Module):
    def __init__(self):
        super(CNNHandwriting, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  #3*3卷积核
            nn.BatchNorm2d(32),  #标准化分布
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(2),  #池化层
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

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

# ------------------- 本地数据集支持 -------------------
class MNISTLocalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        for label in range(10):
            folder = os.path.join(root_dir, str(label))
            if not os.path.exists(folder):
                continue
            for ext in ('*.png', '*.jpg'):
                for img_path in glob.glob(os.path.join(folder, ext)):
                    self.images.append(img_path)
                    self.labels.append(label)
        if len(self.images) == 0:
            raise FileNotFoundError(f"在 {root_dir} 下未找到按数字分类的子文件夹")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ------------------- 数据加载器 -------------------
def get_data_loaders(batch_size=64, mode='camera'):
    """
    mode: 'standard' → 标准增强
          'camera'   → 强增强
    """
    data_root = "./MNIST_Dataset"

    if mode == 'standard':
        # 标准模式
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("使用 standard 模式（标准增强，测试集准确率优先）")
    else:  # mode == 'camera'
        # 摄像头模式
        train_transform = transforms.Compose([
            transforms.RandomRotation(12),
            transforms.RandomAffine(0, translate=(0.12, 0.12)),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.3),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("使用 camera 模式（强增强，摄像头场景优先，测试集准确率可能略降）")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if os.path.exists(data_root) and os.path.isdir(data_root):
        print("使用本地 MNIST_Dataset")
        train_dataset = MNISTLocalDataset(os.path.join(data_root, 'train_images'), train_transform)
        test_dataset = MNISTLocalDataset(os.path.join(data_root, 'test_images'), test_transform)
    else:
        print("未找到本地数据集，下载官方 MNIST")
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, test_dataset

# ------------------- 训练函数（记录学习率） -------------------
def train_model(model, train_loader, epochs=10, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_losses, train_accuracies, lrs = [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(acc)
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%  LR: {current_lr:.6f}')
    return train_losses, train_accuracies, lrs

# ------------------- 评估函数 -------------------
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    return all_preds, all_labels

# ------------------- 绘图函数 -------------------
def plot_training_curves(losses, accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, len(losses)+1), losses, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.grid(True)
    ax2.plot(range(1, len(accuracies)+1), accuracies, 'r-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练准确率曲线')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print("训练曲线已保存为 training_curves.png")

def plot_learning_rate(lrs):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(lrs)+1), lrs, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('学习率变化曲线')
    plt.grid(True)
    plt.savefig('learning_rate.png')
    plt.close()
    print("学习率曲线已保存为 learning_rate.png")

def plot_confusion_matrices(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("混淆矩阵已保存为 confusion_matrix.png")

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('归一化混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix_norm.png')
    plt.close()
    print("归一化混淆矩阵已保存为 confusion_matrix_norm.png")

def plot_class_accuracy(labels, preds):
    class_acc = []
    for cls in range(10):
        mask = (np.array(labels) == cls)
        if sum(mask) == 0:
            class_acc.append(0)
        else:
            class_acc.append(accuracy_score(np.array(labels)[mask], np.array(preds)[mask]))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(10), class_acc, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('数字类别')
    plt.ylabel('准确率')
    plt.title('各类别识别准确率')
    for bar, acc in zip(bars, class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.2%}', ha='center', va='bottom')
    plt.xticks(range(10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.close()
    print("各类别准确率图已保存为 class_accuracy.png")

def plot_error_samples(model, test_loader, device, num_samples=20):
    model.eval()
    error_imgs, error_labels, error_preds = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            mask = (pred != target)
            if mask.sum() > 0:
                error_imgs.extend(data[mask].cpu())
                error_labels.extend(target[mask].cpu().numpy())
                error_preds.extend(pred[mask].cpu().numpy())
            if len(error_imgs) >= num_samples:
                break
    if len(error_imgs) == 0:
        print("没有错误样本，跳过 error_samples.png 生成")
        return

    error_imgs = error_imgs[:num_samples]
    error_labels = error_labels[:num_samples]
    error_preds = error_preds[:num_samples]

    cols = 5
    rows = (len(error_imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i in range(len(error_imgs)):
        ax = axes[i]
        img = error_imgs[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'真:{error_labels[i]} 预:{error_preds[i]}', color='red')
        ax.axis('off')
    for i in range(len(error_imgs), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('error_samples.png')
    plt.close()
    print(f"错误样本已保存为 error_samples.png(共{len(error_imgs)}个)")

# ------------------- 文本报告 -------------------
def save_classification_report(labels, preds):
    report = classification_report(labels, preds, digits=4)
    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("分类报告(精确率/召回率/F1-score)\n")
        f.write("="*60 + "\n")
        f.write(report)
    print("分类报告已保存为 classification_report.txt")

def save_model_summary(model):
    total, trainable = model.get_model_size()
    with open('model_summary.txt', 'w', encoding='utf-8') as f:
        f.write("模型结构概要\n")
        f.write("="*60 + "\n")
        f.write(str(model) + "\n\n")
        f.write(f"总参数量: {total:,} (约 {total/1e6:.2f} M)\n")
        f.write(f"可训练参数量: {trainable:,} (约 {trainable/1e6:.2f} M)\n")
        f.write("\n各层参数量明细:\n")
        for name, param in model.named_parameters():
            f.write(f"  {name}: {param.numel():,}\n")
    print("模型概要已保存为 model_summary.txt")

# ------------------- 主程序 -------------------
def main():
    # ========== 在这里切换模式 ==========
    mode = 'standard'   # 可选 'standard' 或 'camera'
    # ===================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(batch_size=64, mode=mode)
    print(f"训练集: {len(train_dataset)} 张, 测试集: {len(test_dataset)} 张")

    model = CNNHandwriting()
    print("\n模型结构:\n", model)

    print("\n开始训练...")
    start_time = time.time()
    losses, accuracies, lrs = train_model(model, train_loader, epochs=10, device=device)
    train_time = time.time() - start_time
    print(f"训练耗时: {train_time:.2f} 秒")

    print("\n评估模型...")
    preds, labels = evaluate_model(model, test_loader, device=device)

    plot_training_curves(losses, accuracies)
    plot_learning_rate(lrs)
    plot_confusion_matrices(labels, preds)
    plot_class_accuracy(labels, preds)
    plot_error_samples(model, test_loader, device, num_samples=20)

    save_classification_report(labels, preds)
    save_model_summary(model)

    torch.save(model.state_dict(), 'cnn_mnist_model.pth')
    print("模型已保存为 cnn_mnist_model.pth")

    final_acc = accuracy_score(labels, preds) * 100
    print(f"测试集准确率: {final_acc:.2f}%")

    with open('training_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"训练模式: {mode}\n")
        f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练耗时: {train_time:.2f} 秒\n")
        f.write(f"最终测试准确率: {final_acc:.2f}%\n")
        f.write(f"模型总参数量: {model.get_model_size()[0]:,}\n")
    print("训练摘要已保存为 training_summary.txt")

    print("\n所有结果文件已生成!")

if __name__ == '__main__':
    main()