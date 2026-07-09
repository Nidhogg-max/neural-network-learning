# 📝 Handwritten Digit Recognition System

> 基于卷积神经网络（CNN）的手写数字识别系统 —— 从训练到部署完整解决方案

---

## 📌 项目简介

本项目是一个完整的**手写数字识别系统**，基于 **PyTorch** 深度学习框架构建。系统以 MNIST 数据集为基础，实现了从数据加载、模型训练、多维度评估到摄像头实时识别的全链路功能。

与仅完成基础分类的简易实现不同，本系统在模型结构上引入了 **BatchNorm**、**Dropout**、**Adam 优化器** 和 **ReduceLROnPlateau 学习率调度**；在数据层面实现了**六重强数据增强策略**，有效解决了训练数据（MNIST 纯黑背景）与真实摄像头场景之间的领域差异问题；在工程层面构建了**完整的摄像头预处理管道**（灰度转换、高斯滤波、自适应阈值二值化、轮廓检测、居中缩放）和 **Tkinter 三标签页图形界面**，实现了从学术基准到工程落地的跨越。

---

## ✨ 核心功能

| 模块 | 功能说明 |
|------|----------|
| **模型训练** | 支持本地/在线双模式数据加载，训练时自动输出 10 项评估报告（含图表与文本） |
| **图片识别** | 加载模型对用户手写数字图片进行识别，输出概率分布并保存预测结果 |
| **对比实验** | 分析不同训练样本数（1000/5000/10000/30000/60000）对性能的影响 |
| **摄像头识别** | 实时识别摄像头画面中的手写数字，含预处理管道、绿色识别框、置信度校准 |
| **图形界面** | Tkinter 三标签页 GUI，整合全部功能，支持实时日志、任务终止、参数自定义 |

---

## 🛠 技术栈

| 技术领域 | 使用工具 |
|----------|----------|
| 深度学习框架 | PyTorch |
| 图像处理 | OpenCV、PIL |
| 数据可视化 | Matplotlib、Seaborn |
| 评估指标 | scikit-learn |
| 图形界面 | Tkinter |

---

## 📁 项目结构

```
handwritten_digit_recognition/
│
├── train.py                 # 训练模块（含数据加载、模型定义、训练评估、结果可视化）
├── predict.py               # 预测模块（单张图片识别 + 概率输出）
├── compare.py               # 对比实验模块（不同样本规模性能分析）
├── camera.py                # 摄像头识别模块（预处理管道 + 实时推理）
├── gui.py                   # 图形界面模块（三标签页整合全部功能）
│
├── cnn_mnist_model.pth      # 训练好的模型权重文件
│
├── MNIST_Dataset/           # 本地数据集（按 0-9 子文件夹存放，可选）
│
├── training_curves.png      # 训练损失与准确率曲线
├── confusion_matrix.png     # 混淆矩阵（计数版）
├── confusion_matrix_norm.png# 归一化混淆矩阵
├── class_accuracy.png       # 各类别准确率条形图
├── error_samples.png        # 错误样本展示
├── learning_rate.png        # 学习率变化曲线
├── sample_size_comparison.png # 对比实验汇总图
├── custom_prediction.png    # 用户图片预测结果
│
├── classification_report.txt # 精确率/召回率/F1-score 报告
├── model_summary.txt        # 模型参数量统计
├── training_summary.txt     # 训练摘要（准确率/耗时/参数量）
├── comparison_report.txt    # 对比实验详细报告
│
└── compare_training_curves/ # 各样本数的单独训练曲线
    ├── train_curves_1000.png
    ├── train_curves_5000.png
    ├── train_curves_10000.png
    ├── train_curves_30000.png
    └── train_curves_60000.png
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（conda）
conda create -n digit_env python=3.8
conda activate digit_env

# 安装依赖
pip install torch torchvision matplotlib numpy pillow scikit-learn seaborn opencv-python
```

### 2. 训练模型

```bash
python train.py
```

训练完成后会自动生成模型权重及全部评估报告。

### 3. 启动图形界面

```bash
python gui.py
```

### 4. 其他功能

```bash
# 单张图片识别
python predict.py test.png

# 对比实验（自定义参数）
python compare.py --epochs 5 --samples 1000,5000,10000,30000,60000

# 摄像头识别（独立运行）
python camera.py
```

---

## 🧠 模型结构

| 层级 | 类型 | 输入尺寸 | 输出尺寸 | 参数量 |
|------|------|----------|----------|--------|
| 0 | Conv2d (1→32) | (1,28,28) | (32,28,28) | 320 |
| 1 | BatchNorm2d | (32,28,28) | (32,28,28) | 64 |
| 2 | ReLU | - | - | 0 |
| 3 | MaxPool2d | (32,28,28) | (32,14,14) | 0 |
| 4 | Conv2d (32→64) | (32,14,14) | (64,14,14) | 18,496 |
| 5 | BatchNorm2d | (64,14,14) | (64,14,14) | 128 |
| 6 | ReLU | - | - | 0 |
| 7 | MaxPool2d | (64,14,14) | (64,7,7) | 0 |
| 8 | Flatten | (64,7,7) | 3136 | 0 |
| 9 | Dropout(0.25) | 3136 | 3136 | 0 |
| 10 | Linear | 3136 | 128 | 401,536 |
| 11 | ReLU | - | - | 0 |
| 12 | Dropout(0.5) | 128 | 128 | 0 |
| 13 | Linear | 128 | 10 | 1,290 |

**总参数量：约 42 万**

---

## 📊 核心结果

| 指标 | 结果 |
|------|------|
| 测试集准确率 | **99.16%** |
| 模型参数量 | 421,834 |
| 训练耗时（GPU） | 约 30 秒 |

> 详细结果见 `training_summary.txt`、`classification_report.txt` 及各类 PNG 图表。

---

## 🏆 个人提高

相较于仅用基础 CNN 完成 MNIST 分类的实现，本设计在以下维度进行了大幅扩展：

| 维度 | 基础实现 | 本设计 |
|------|----------|--------|
| 模型结构 | Conv + Pool + FC | +BatchNorm + Dropout |
| 优化器 | SGD 固定学习率 | Adam + ReduceLROnPlateau |
| 数据增强 | 无 | 六重强增强（旋转/平移/透视/亮度/噪声/擦除） |
| 评估维度 | 仅准确率 | 6 类图表 + 3 类文本报告 |
| 摄像头识别 | 不支持 | 完整预处理管道 + 置信度校准 |
| 用户交互 | 命令行 | Tkinter 三标签页 GUI |

---

## 🙋 作者

- **Nidhogg-max**
- 完成日期：2026 年 7 月 7 日
- 课程设计：Python 与 Matlab 程序设计综合课程设计

---

## 📄 许可证

本项目仅供学习交流使用。

---

**⭐ 如果这个项目对你有帮助，欢迎点个 Star！**
```

---

这个 README.md 文件可以直接复制粘贴到你的项目根目录下，然后提交到 GitHub。内容包含了项目介绍、文件结构、使用说明、模型结果和你的作者信息。
