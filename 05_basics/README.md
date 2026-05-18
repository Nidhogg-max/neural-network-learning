## 线性回归的简洁实现 (PyTorch)

> 作者：出自 Nidhogg-max

本 Jupyter Notebook 使用 PyTorch 的高级 API，以极简代码完成线性回归模型的完整实现。通过 `nn.Sequential` 搭建线性层、`nn.MSELoss` 计算均方误差、`torch.optim.SGD` 自动执行随机梯度下降，展示数据加载、模型定义、参数初始化、损失计算、优化器设置和训练循环的标准化流程。与前一章手动实现形成对比，帮助理解 PyTorch 的模块化设计如何简化深度学习模型开发。

---

## 📄 文件说明

- **文件名**：`chapt5.ipynb`
- **内容**：利用 PyTorch 的 `DataLoader` 和 `TensorDataset` 加载人造数据集，通过 `nn.Linear` 构建单层网络，使用 `MSELoss` 和 `SGD` 优化器进行训练，仅需少量代码即可实现与手动实现同样效果的线性回归。

---

## 📖 章节概览

| 节号 | 标题 | 主要内容 |
|------|------|----------|
| 1.1 | 生成数据集 | 复用 `synthetic_data` 函数生成含噪声的线性数据集，提供 1000 个样本。 |
| 1.2 | 读取数据集 | 使用 `TensorDataset` 和 `DataLoader` 构造小批量数据迭代器，支持随机打乱和批量加载。 |
| 1.3 | 定义模型 | 通过 `nn.Sequential(nn.Linear(2, 1))` 定义单层全连接网络。 |
| 1.4 | 初始化模型参数 | 使用 `normal_` 和 `fill_` 方法自定义权重和偏置的初始值。 |
| 1.5 | 定义损失函数 | 直接调用 `nn.MSELoss()` 作为均方误差损失。 |
| 1.6 | 定义优化算法 | 创建 `torch.optim.SGD` 优化器，指定学习率 0.03。 |
| 1.7 | 训练 | 标准化训练循环：清零梯度、前向传播、反向传播、参数更新，每 epoch 评估整体损失。 |

---

## ⚙️ 运行环境

- **Python 3.7+**
- **PyTorch** ≥ 1.0
- **Jupyter Notebook** 或 **JupyterLab**

安装依赖（如果尚未安装）：
```bash
pip install torch notebook
