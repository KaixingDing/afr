# 快速开始指南 (Quick Start Guide)

## 项目概述

本项目实现了物理引导扩散模型（Physics-Guided Diffusion Model）用于稀疏声场重建。

## 环境设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装:
```bash
pip install torch numpy scipy matplotlib tqdm pyyaml pillow
```

### 2. 验证安装

运行快速演示:
```bash
python demo.py
```

## 使用流程

### 步骤1: 生成训练数据

生成模拟声场数据:

```bash
cd data
python generate_data.py
```

这将生成100个样本，每个包含:
- 完整声压场（真值）
- 稀疏观测点
- 折射率场
- 声源信息

**自定义数据生成:**

```python
from data.generate_data import generate_dataset

generate_dataset(
    num_samples=1000,      # 样本数量
    grid_size=128,         # 网格大小
    output_dir='data/samples'
)
```

### 步骤2: 训练模型

运行训练脚本:

```bash
cd src
python train.py
```

**配置训练参数:**

编辑 `experiments/config.yaml` 修改:
- `batch_size`: 批次大小 (默认: 8)
- `num_epochs`: 训练轮数 (默认: 100)
- `learning_rate`: 学习率 (默认: 1e-4)
- `lambda_helmholtz`: Helmholtz损失权重 (默认: 1.0)

### 步骤3: 评估模型

评估训练好的模型:

```bash
cd src
python evaluate.py
```

这将生成:
- 重建结果可视化
- Helmholtz残差图
- 评估指标报告
- 训练曲线图

### 步骤4: 使用模型进行推理

```python
import torch
from src.model_diffusion import PhysicsGuidedDiffusionModel
from src.evaluate import load_model

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('experiments/checkpoints/best_model.pt', device)

# 准备输入 (稀疏观测)
sparse_obs = ...  # [B, 2, H, W] 张量

# 重建声场
model.eval()
with torch.no_grad():
    reconstructed = model.sample(sparse_obs, device)
```

## 项目结构说明

```
afr/
│
├── data/
│   ├── generate_data.py       # 数据生成脚本
│   └── samples/               # 生成的数据样本
│
├── src/
│   ├── model_diffusion.py     # 扩散模型定义
│   ├── physics_loss.py        # 物理约束损失
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数
│
├── experiments/
│   ├── config.yaml            # 配置文件
│   ├── checkpoints/           # 模型保存
│   ├── logs/                  # 训练日志
│   └── results/               # 评估结果
│
├── demo.py                    # 快速演示脚本
├── README.md                  # 项目说明
├── paper_draft.md             # 论文初稿
└── requirements.txt           # 依赖列表
```

## 核心功能

### 1. Helmholtz求解器

生成满足波动方程的声场:

```python
from data.generate_data import HelmholtzSolver

solver = HelmholtzSolver(grid_size=128)
n = solver.create_refractive_index('circle')
s = solver.create_source('point')
p = solver.solve_helmholtz(k=20.0, n=n, s=s)
```

### 2. 物理约束损失

计算多种物理约束:

```python
from src.physics_loss import PhysicsLoss

physics_loss = PhysicsLoss(
    lambda_helmholtz=1.0,
    lambda_energy=0.5,
    lambda_boundary=0.3
)

loss, loss_dict = physics_loss(p_pred, p_true, k, n, s)
```

### 3. 扩散采样

从稀疏观测重建完整场:

```python
from src.model_diffusion import PhysicsGuidedDiffusionModel

model = PhysicsGuidedDiffusionModel()
reconstructed = model.sample(sparse_obs, device)
```

## 性能指标

预期结果（训练100 epochs后）:

| 指标 | 目标值 |
|------|--------|
| MAE | < 0.05 |
| PSNR | > 25 dB |
| SSIM | > 0.90 |
| PDE残差 | < 0.01 |

## 常见问题

**Q: GPU内存不足？**
- 减小 `batch_size`
- 减小 `grid_size`
- 减少模型 `base_channels`

**Q: 训练很慢？**
- 使用GPU加速
- 减少 `timesteps`
- 使用更小的数据集测试

**Q: 重建质量不好？**
- 增加训练epochs
- 调整物理损失权重
- 增加训练数据量
- 检查数据归一化

## 进阶使用

### 多频训练

在不同频率下联合训练以提高泛化:

```python
frequencies = [500, 1000, 1500, 2000]  # Hz
# 数据生成时随机采样频率
```

### 不确定性估计

多次采样生成置信度图:

```python
samples = []
for _ in range(10):
    sample = model.sample(sparse_obs, device)
    samples.append(sample)

# 计算均值和标准差
mean = torch.stack(samples).mean(dim=0)
std = torch.stack(samples).std(dim=0)
```

### 自定义物理约束

修改 `src/physics_loss.py` 添加新的约束项。

## 可视化

所有可视化结果保存在 `experiments/results/`:
- `reconstruction_*.png`: 重建对比图
- `helmholtz_residual_*.png`: PDE残差图
- `training_curves.png`: 训练曲线
- `demo_reconstruction.png`: 演示结果

## 引用

如果使用本代码，请引用:

```bibtex
@misc{pgdm2024,
  title={Physics-Guided Diffusion Model for Sparse Acoustic Field Reconstruction},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/KaixingDing/afr}}
}
```

## 技术支持

遇到问题? 
1. 查看 `README.md` 详细文档
2. 阅读 `paper_draft.md` 了解方法
3. 运行 `demo.py` 测试环境
4. 在GitHub提issue

---

**祝使用愉快！**
