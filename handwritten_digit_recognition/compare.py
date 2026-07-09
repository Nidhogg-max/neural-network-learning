"""
compare.py - 不同训练样本数对识别性能的影响对比实验
支持命令行参数覆盖默认值：
  --epochs     训练轮次
  --samples    样本数列表,逗号分隔
示例：
  python compare.py --epochs 3 --samples 1000,2000,5000
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
import time
import argparse
from sklearn.metrics import accuracy_score

# 中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(42)
np.random.seed(42)

# ------------------- CNN模型 -------------------
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

# ------------------- 本地数据集 -------------------
class MNISTLocalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images, self.labels = [], []
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

def get_data_loaders(batch_size=64):
    data_root = "./MNIST_Dataset"
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
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

# ------------------- 训练单个子集 -------------------
def train_subset(model, loader, epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for data, target in loader:
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
        avg_loss = running_loss / len(loader)
        acc = 100 * correct / total
        losses.append(avg_loss)
        accs.append(acc)
        print(f'Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%')
    return losses, accs

# ------------------- 评估 -------------------
def evaluate(model, test_loader, device):
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
def plot_single_curves(losses, accs, size, save_dir='compare_training_curves'):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, len(losses)+1), losses, 'b-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'样本数={size} - 训练损失曲线')
    ax1.grid(True)
    ax2.plot(range(1, len(accs)+1), accs, 'r-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'样本数={size} - 训练准确率曲线')
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'train_curves_{size}.png'))
    plt.close()

def plot_comparison(results):
    sizes = sorted(results.keys())
    train_accs = [results[s]['train_acc'] for s in sizes]
    test_accs = [results[s]['test_acc'] for s in sizes]
    times = [results[s]['time'] for s in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(sizes, train_accs, 'bo-', label='训练准确率', linewidth=2, markersize=8)
    ax1.plot(sizes, test_accs, 'rs-', label='测试准确率', linewidth=2, markersize=8)
    ax1.set_xlabel('训练样本数')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('不同样本数对准确率的影响')
    ax1.legend()
    ax1.grid(True)
    for i, s in enumerate(sizes):
        ax1.annotate(f'{test_accs[i]:.1f}%', (s, test_accs[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

    ax2 = axes[1]
    bars = ax2.bar(sizes, times, color='skyblue')
    ax2.set_xlabel('训练样本数')
    ax2.set_ylabel('训练时间 (秒)')
    ax2.set_title('不同样本数的训练耗时对比')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{t:.1f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('sample_size_comparison.png')
    plt.close()
    print("汇总对比图已保存为 sample_size_comparison.png")

def save_report(results, sample_sizes, epochs):
    with open('comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("不同训练样本数对识别性能的影响对比报告\n")
        f.write("="*70 + "\n")
        f.write(f"对比样本数: {sample_sizes}\n")
        f.write(f"每项训练轮次: {epochs}\n")
        f.write(f"训练日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'样本数':>10} | {'训练准确率':>12} | {'测试准确率':>12} | {'训练耗时(秒)':>14} | {'测试集准确率提升':>16}\n")
        f.write("-"*70 + "\n")
        base_acc = None
        for size in sorted(results.keys()):
            r = results[size]
            if base_acc is None:
                base_acc = r['test_acc']
                improve = 0
            else:
                improve = r['test_acc'] - base_acc
            f.write(f"{size:>10} | {r['train_acc']:>11.2f}% | {r['test_acc']:>11.2f}% | {r['time']:>13.2f}s | {improve:>+15.2f}%\n")
        f.write("\n" + "="*70 + "\n")
        f.write("结论：随着训练样本数增加，测试准确率呈上升趋势，但训练耗时也相应增加。\n")
        f.write("建议在实际应用中根据可用数据量和计算资源合理选择训练样本数。\n")
    print("对比报告已保存为 comparison_report.txt")

# ------------------- 主程序 -------------------
def main():
    parser = argparse.ArgumentParser(description='不同训练样本数对比实验')
    parser.add_argument('--epochs', type=int, default=5,
                        help='每项训练轮次 (默认5)')
    parser.add_argument('--samples', type=str, default='1000,5000,10000,30000,60000',
                        help='样本数列表，逗号分隔 (默认1000,5000,10000,30000,60000)')
    args = parser.parse_args()

    # 解析样本数列表
    sample_sizes = [int(x.strip()) for x in args.samples.split(',') if x.strip()]
    epochs = args.epochs
    print(f"使用参数: 训练轮次={epochs}, 样本数={sample_sizes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders()
    print(f"完整训练集: {len(train_dataset)} 张, 测试集: {len(test_dataset)} 张\n")

    results = {}
    os.makedirs('compare_training_curves', exist_ok=True)

    for size in sample_sizes:
        if size > len(train_dataset):
            print(f"样本数 {size} 超过数据集大小，跳过")
            continue

        print(f"\n{'='*50}")
        print(f"开始训练: 样本数 = {size}")
        print('='*50)

        indices = np.random.choice(len(train_dataset), size, replace=False)
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=64, shuffle=True)

        model = CNNHandwriting()
        start_time = time.time()
        losses, accs = train_subset(model, loader, epochs, device)
        train_time = time.time() - start_time

        preds, labels = evaluate(model, test_loader, device)
        test_acc = accuracy_score(labels, preds) * 100
        train_acc = accs[-1]

        results[size] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'time': train_time,
            'losses': losses,
            'accs': accs
        }

        print(f"\n样本数 {size} 完成: 训练准确率 {train_acc:.2f}%, 测试准确率 {test_acc:.2f}%, 耗时 {train_time:.2f}s")
        plot_single_curves(losses, accs, size)

    plot_comparison(results)
    save_report(results, sample_sizes, epochs)

    print("\n" + "="*60)
    print("[OK] 对比实验全部完成！")
    print("生成文件:")
    print("  - sample_size_comparison.png  (汇总对比图)")
    print("  - comparison_report.txt       (详细对比报告)")
    print("  - compare_training_curves/   (各样本数训练曲线子文件夹)")
    print("="*60)

if __name__ == '__main__':
    main()