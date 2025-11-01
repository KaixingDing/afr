# 论文图表索引 / Figure and Table Index

本文档列出了所有生成的图表及其对应的代码文件。

## 图表列表 / List of Figures and Tables

### Figure 1: Helmholtz求解示例 / Helmholtz Solver Examples
- **文件**: `experiments/results/figures/fig1_helmholtz_examples.png`
- **代码**: `data/generate_data.py` - `HelmholtzSolver` class
- **描述**: 展示不同类型散射体（圆形、方形、随机）及其对应的折射率场和声压场
- **用途**: 论文第3节方法论，展示数据生成过程

### Figure 2: 模型架构 / Model Architecture
- **文件**: `experiments/results/figures/fig2_architecture.png`
- **代码**: `src/model_diffusion.py` - `UNet`, `PhysicsGuidedDiffusionModel`
- **描述**: 物理引导扩散模型的整体架构，包括U-Net结构和物理损失
- **用途**: 论文第3.2节，说明模型设计

### Figure 3: 稀疏采样模式 / Sparse Sampling Patterns
- **文件**: `experiments/results/figures/fig3_sampling_patterns.png`
- **代码**: `data/generate_data.py` - `generate_sparse_observations()`
- **描述**: 展示不同采样模式（随机、网格、线性）下的稀疏观测
- **用途**: 论文第4节实验设置，说明数据采集方式

### Figure 4: 重建结果对比 / Reconstruction Comparison
- **文件**: `experiments/results/figures/fig4_reconstruction_comparison.png`
- **代码**: `src/model_diffusion.py` - `sample()` method
- **描述**: 对比真实场、稀疏观测、重建场和误差图
- **用途**: 论文第5节实验结果，展示重建质量

### Figure 5: 物理约束可视化 / Physics Constraints Visualization
- **文件**: `experiments/results/figures/fig5_physics_constraints.png`
- **代码**: `src/physics_loss.py` - `helmholtz_residual()`, `energy_conservation()`
- **描述**: 展示Helmholtz残差、能量分布和残差统计分布
- **用途**: 论文第5.2节，验证物理一致性

### Figure 6: 训练曲线 / Training Curves
- **文件**: `experiments/results/figures/fig6_training_curves.png`
- **代码**: `src/train.py` - `train_epoch()`, `validate()`
- **描述**: 训练损失、验证MAE和PSNR随epoch变化
- **用途**: 论文第5节，展示训练过程和收敛性

### Figure 7: 消融实验 / Ablation Study
- **文件**: `experiments/results/figures/fig7_ablation_study.png`
- **代码**: `src/physics_loss.py` - 各物理约束项
- **描述**: 不同物理约束组合对PSNR和PDE残差的影响
- **用途**: 论文第5.2节，验证各物理约束的有效性

### Table 1: 性能对比 / Performance Comparison
- **文件**: `experiments/results/figures/table1_performance_comparison.md` (Markdown)
- **文件**: `experiments/results/figures/table1_performance_comparison.tex` (LaTeX)
- **代码**: `src/evaluate.py` - `compute_metrics()`
- **描述**: 与基线方法的定量性能对比（MAE, PSNR, SSIM, PDE残差）
- **用途**: 论文第5.1节，量化评估

## 代码文件对应关系 / Code File Mapping

| 代码文件 | 主要功能 | 相关图表 |
|---------|---------|---------|
| `data/generate_data.py` | Helmholtz求解、数据生成 | Figure 1, 3 |
| `src/model_diffusion.py` | 扩散模型架构 | Figure 2, 4 |
| `src/physics_loss.py` | 物理约束损失 | Figure 5, 7; Table 1 |
| `src/train.py` | 训练流程 | Figure 6 |
| `src/evaluate.py` | 评估和可视化 | Figure 4, 5, 6; Table 1 |
| `demo.py` | 端到端演示 | - |

## 生成方法 / How to Generate

所有图表可通过以下脚本重新生成：
```bash
python generate_paper_figures.py
```

单独生成某个图表：
```python
# Figure 1
from data.generate_data import HelmholtzSolver
solver = HelmholtzSolver(grid_size=128)
# ... (详见generate_paper_figures.py)

# Figure 4
from src.model_diffusion import PhysicsGuidedDiffusionModel
model = PhysicsGuidedDiffusionModel()
# ... (详见generate_paper_figures.py)
```

## 图表引用格式 / Citation Format

在论文中引用图表时使用以下格式：
- 英文: "As shown in Figure 1, ..."
- 中文: "如图1所示，..."
- LaTeX: "\ref{fig:helmholtz_examples}"

