# 🎓 项目任务书（Draft）

请使用中文进行代码注释和文档撰写。但是如果绘图，图中应当为全英文。

## 1️⃣ 课题名称

**Physics-Guided Diffusion Model for Sparse Acoustic Field Reconstruction**

---

## 2️⃣ 研究目标

构建一个**物理一致性约束的扩散模型（Physically Consistent Diffusion Model, PCDM）**，在稀疏观测条件下从有限声压测量点重建二维声场（pressure field）或介质折射率分布。

* **核心问题**：传统深度学习反演声场易忽略波动方程约束，导致结果物理不一致。
* **研究目标**：

  1. 提出包含多物理约束的扩散模型损失；
  2. 在多频声场条件下实现稳健重建；
  3. 给出反演结果的不确定性评估。

---

## 3️⃣ 理论基础

### 3.1 声学波动方程（Helmholtz Form）

[
(\nabla^2 + k^2 n(x,y)^2) p(x,y) = s(x,y)
]
其中：

* ( p(x,y) )：声压复场；
* ( n(x,y) )：折射率场；
* ( k = \frac{2\pi f}{c_0} )：波数；
* ( s(x,y) )：声源项。

### 3.2 反问题定义

> 已知若干观测点处的 ( p(x_i, y_i) )，重建全场 ( p(x,y) ) 或介质 ( n(x,y) )。

### 3.3 数值数据生成

* 使用有限差分（FDTD）或有限元（FEM）方法生成1000组模拟场；
* 随机生成不同 ( n(x,y) )（含障碍/散射体）；
* 在每个场上采样少量观测点（稀疏麦克风阵列）；
* 保存：

  * 真值场（Ground Truth）
  * 稀疏观测（Input）
  * 折射率场（可选）

---

## 4️⃣ 模型总体结构

### 4.1 生成主干：Diffusion Model (DDPM)

* 以稀疏声场观测为条件输入；
* 生成完整声压场 ( \hat{p}(x,y) )。

### 4.2 Physics-Guided Loss 设计

[
\mathcal{L} = \mathcal{L}*{recon} +
\lambda_1 \underbrace{||\nabla^2 \hat{p} + k^2 n^2 \hat{p} - s||^2}*{\text{Helmholtz残差}} +
\lambda_2 \underbrace{|E(\hat{p}) - E(p)|}*{\text{能量守恒}} +
\lambda_3 \underbrace{||\partial_n \hat{p}|*{\partial \Omega}||^2}_{\text{边界条件}}
]

### 4.3 模型创新点

| 模块     | 创新内容                                    | 理论意义     |
| ------ | --------------------------------------- | -------- |
| 物理约束损失 | PDE + 能量 + 边界三重约束                       | 强化物理一致性  |
| 多频融合   | 共享U-Net特征跨频训练                           | 提升泛化     |
| 不确定性估计 | 在DDPM输出方差头中生成置信度图                       | 增加解释性    |
| 二阶段结构  | Neural Operator 预测初场 → Diffusion refine | 提高效率与收敛性 |

---

## 5️⃣ 实施路线（Copilot执行计划）

| 阶段              | 子任务                   | 可由 Copilot 自动生成            | 输出产物                |
| --------------- | --------------------- | -------------------------- | ------------------- |
| **阶段 1：数据准备**   | 生成模拟声场（Helmholtz求解器）  | ✅ 完整 Python 模块             | `.npz` 格式波场数据       |
| **阶段 2：模型搭建**   | 搭建扩散网络 / 神经算符模块       | ✅ PyTorch 实现               | `model.py`          |
| **阶段 3：物理约束嵌入** | 定义 PDE/能量残差           | ✅ 自动微分实现                   | `physics_loss.py`   |
| **阶段 4：训练与记录**  | 日志、可视化、checkpoint     | ✅ TensorBoard & Matplotlib | `logs/`, `results/` |
| **阶段 5：评估与分析**  | 重建误差 / PDE残差 / 不确定性热图 | ✅                          | 图像报告、CSV结果          |
| **阶段 6：可视化展示**  | 动画演化 + 对比可视化          | ✅                          | `.mp4` 动画与论文图表      |

---

## 6️⃣ 评价指标

| 类别    | 指标              | 含义        |
| ----- | --------------- | --------- |
| 重建性能  | PSNR, MAE, SSIM | 重建质量      |
| 物理一致性 | PDE残差均值         | 满足波动方程程度  |
| 泛化能力  | 频率域误差           | 多频鲁棒性     |
| 解释性   | 置信度热图相关性        | 不确定性估计可靠性 |

---

## 7️⃣ 可视化计划

* 声压真值 vs 重建（二维热图）
* Helmholtz残差分布图
* 置信度图（Variance Map）
* 训练过程损失曲线
* 扩散生成演化动画（从噪声→重建）

---

## 8️⃣ 项目目录结构（建议）

```
PhysicsGuidedDiffusion/
│
├── data/
│   ├── generate_data.py      # 声场模拟脚本
│   └── samples/              # 存放 npz 数据
│
├── src/
│   ├── model_diffusion.py    # 主模型定义
│   ├── model_operator.py     # 神经算符(可选)
│   ├── physics_loss.py       # PDE & 物理约束
│   ├── train.py              # 训练流程
│   ├── evaluate.py           # 评估与可视化
│   └── utils.py
│
├── experiments/
│   ├── config.yaml           # 超参与路径
│   ├── logs/                 # 训练日志
│   └── results/              # 输出结果
│
└── README.md
```

---

## 9️⃣ 预期成果

* ✅ 可运行的物理一致性扩散反演模型；
* ✅ 多频声场数据集与可复现实验；
* ✅ 可视化动画与分析报告；
* ✅ 完成论文初稿，放在md里。

---
