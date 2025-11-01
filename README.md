# Physics-Guided Diffusion Model for Sparse Acoustic Field Reconstruction

## 项目简介

本项目实现了一个**物理引导的扩散模型（Physics-Guided Diffusion Model, PGDM）**，用于从稀疏观测点重建二维声场。该模型结合了深度学习的强大表示能力和物理约束（Helmholtz方程），能够在有限测量点的条件下高精度重建完整的声压场。

### 核心特性

- ✅ **物理一致性约束**：通过Helmholtz残差、能量守恒和边界条件三重物理约束，确保重建结果符合声学波动方程
- ✅ **扩散模型架构**：基于DDPM的去噪扩散概率模型，支持高质量生成
- ✅ **稀疏观测重建**：可从少量麦克风阵列观测点重建完整声场
- ✅ **多频声场支持**：支持多种频率条件下的声场重建
- ✅ **不确定性评估**：通过扩散过程提供重建不确定性估计

---

## 项目结构

```
PhysicsGuidedDiffusion/
│
├── data/
│   ├── generate_data.py      # 声场模拟脚本 (Helmholtz求解器)
│   └── samples/              # 存放npz格式的模拟数据
│
├── src/
│   ├── model_diffusion.py    # 扩散模型定义
│   ├── physics_loss.py       # 物理约束损失函数
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估与可视化
│   └── utils.py              # 工具函数
│
├── experiments/
│   ├── config.yaml           # 配置文件
│   ├── checkpoints/          # 模型checkpoint
│   ├── logs/                 # 训练日志
│   └── results/              # 评估结果和可视化
│
├── agent_task.md             # 项目任务书
└── README.md                 # 本文件
```

---

## 理论基础

### Helmholtz方程

声学波动方程的Helmholtz形式：

```
(∇² + k²n²)p = s
```

其中：
- `p(x,y)`: 声压复场
- `n(x,y)`: 折射率场
- `k = 2πf/c₀`: 波数
- `s(x,y)`: 声源项

### 物理约束损失

```
L = L_recon + λ₁·L_helmholtz + λ₂·L_energy + λ₃·L_boundary
```

- **L_recon**: 重建损失（与真值对比）
- **L_helmholtz**: Helmholtz残差 `||∇²p + k²n²p - s||²`
- **L_energy**: 能量守恒 `|E(p) - E(p_true)|`
- **L_boundary**: 边界条件约束

---

## 安装依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision numpy scipy matplotlib tqdm pyyaml
```

**主要依赖：**
- Python >= 3.8
- PyTorch >= 1.10
- NumPy, SciPy, Matplotlib

---

## 快速开始

### 1. 生成模拟数据

```bash
cd data
python generate_data.py
```

这将生成100个模拟声场样本，保存在 `data/samples/` 目录。每个样本包含：
- 完整声压场（真值）
- 稀疏观测点
- 折射率场
- 声源和波数信息

### 2. 训练模型

```bash
cd src
python train.py
```

训练过程将：
- 自动划分训练/验证集
- 定期保存checkpoint
- 记录训练日志和指标

训练配置可在 `experiments/config.yaml` 中修改。

### 3. 评估模型

```bash
cd src
python evaluate.py
```

评估将生成：
- 重建结果可视化
- Helmholtz残差分布图
- 评估指标报告（MAE, PSNR, SSIM）
- 扩散采样动画

结果保存在 `experiments/results/` 目录。

---

## 使用示例

### 生成自定义数据

```python
from data.generate_data import HelmholtzSolver, generate_dataset

# 生成1000个样本
generate_dataset(
    num_samples=1000,
    grid_size=128,
    output_dir='data/samples'
)
```

### 加载和推理

```python
import torch
from src.model_diffusion import PhysicsGuidedDiffusionModel
from src.evaluate import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = load_model('experiments/checkpoints/best_model.pt', device)

# 准备稀疏观测 (shape: [1, 2, 128, 128])
sparse_obs = ...  # 您的稀疏观测数据

# 重建完整声场
with torch.no_grad():
    reconstructed = model.sample(sparse_obs, device)
```

---

## 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| **MAE** | 平均绝对误差 | 越小越好 |
| **PSNR** | 峰值信噪比 (dB) | 越大越好 |
| **SSIM** | 结构相似性 | 接近1.0 |
| **PDE残差** | Helmholtz方程残差 | 越小越好 |

---

## 可视化示例

训练和评估过程将生成以下可视化：

1. **声场重建对比图**：真值 vs 稀疏观测 vs 重建结果
2. **Helmholtz残差分布**：展示物理一致性
3. **训练曲线**：损失和指标随epoch变化
4. **扩散采样动画**：从噪声到重建的完整过程

---

## 配置说明

主要配置参数在 `experiments/config.yaml`：

```yaml
# 数据
data:
  grid_size: 128           # 网格大小
  num_samples: 1000        # 样本数量

# 模型
model:
  timesteps: 1000          # 扩散步数
  base_channels: 64        # 基础通道数

# 物理约束权重
physics:
  lambda_helmholtz: 1.0    # Helmholtz残差权重
  lambda_energy: 0.5       # 能量守恒权重
  lambda_boundary: 0.3     # 边界条件权重

# 训练
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.0001
```

---

## 实验结果

预期结果（在100个epoch训练后）：

- **MAE**: < 0.05
- **PSNR**: > 25 dB
- **SSIM**: > 0.90
- **PDE残差**: < 0.01

---

## 扩展功能

### 1. 神经算符预训练

可选的二阶段训练策略：
1. 使用Neural Operator预测粗略初场
2. 用Diffusion Model精细化重建

### 2. 多频融合

支持在多个频率条件下联合训练，提升泛化能力。

### 3. 不确定性估计

扩散模型自然支持不确定性量化，可通过多次采样生成置信度图。

---

## 参考文献

本项目基于以下研究领域：

1. **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
2. **Physics-Informed Neural Networks**: Raissi et al., "Physics-Informed Neural Networks", JCP 2019
3. **Acoustic Field Reconstruction**: Various works on inverse scattering and field reconstruction

---

## 故障排除

### 常见问题

**Q: 训练时内存不足？**
- 减小 `batch_size`（如改为4或2）
- 减小 `grid_size`（如改为64）
- 减少模型通道数

**Q: 数据生成很慢？**
- Helmholtz求解器使用稀疏矩阵求解，计算密集
- 可以减少 `num_samples` 或使用更小的 `grid_size`
- 考虑并行生成

**Q: 重建结果不理想？**
- 增加训练epochs
- 调整物理损失权重
- 检查数据归一化
- 增加训练数据量

---

## 贡献指南

欢迎贡献！可以：
- 报告bug或提出功能建议
- 提交代码改进
- 补充文档和示例
- 分享实验结果

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

## 致谢

感谢所有为声学反演和扩散模型研究做出贡献的研究者们。
