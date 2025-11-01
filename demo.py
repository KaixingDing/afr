"""
快速演示脚本
展示如何使用物理引导扩散模型进行声场重建
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_diffusion import PhysicsGuidedDiffusionModel
from physics_loss import PhysicsLoss, compute_metrics
from utils import set_seed


def demo_quick_test():
    """快速测试示例"""
    print("=" * 60)
    print("物理引导扩散模型 - 快速演示")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n1. 初始化模型...")
    model = PhysicsGuidedDiffusionModel(
        grid_size=64,  # 使用较小的网格以加快演示速度
        timesteps=50   # 使用较少的时间步
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 模型参数量: {total_params:,}")
    
    # 创建模拟数据
    print("\n2. 创建模拟数据...")
    batch_size = 2
    
    # 真实声场 (模拟)
    p_true = torch.randn(batch_size, 2, 64, 64).to(device)
    
    # 稀疏观测 (随机掩码)
    mask = torch.zeros_like(p_true)
    for b in range(batch_size):
        # 随机选择20个观测点
        num_obs = 20
        obs_indices = torch.randperm(64 * 64)[:num_obs]
        for idx in obs_indices:
            y, x = divmod(idx.item(), 64)
            mask[b, :, y, x] = 1.0
    
    p_sparse = p_true * mask
    print(f"   ✓ 真实场形状: {p_true.shape}")
    print(f"   ✓ 稀疏观测点数: {num_obs}")
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    model.eval()
    with torch.no_grad():
        # 随机时间步
        t = torch.randint(0, 50, (batch_size,)).to(device)
        # 前向传播
        output = model(p_true, t, p_sparse)
        print(f"   ✓ 输出形状: {output.shape}")
    
    # 测试采样 (重建)
    print("\n4. 测试采样 (重建声场)...")
    with torch.no_grad():
        p_reconstructed = model.sample(p_sparse, device)
        print(f"   ✓ 重建完成: {p_reconstructed.shape}")
    
    # 计算评估指标
    print("\n5. 计算评估指标...")
    metrics = compute_metrics(p_reconstructed, p_true)
    for key, val in metrics.items():
        print(f"   {key}: {val:.4f}")
    
    # 计算物理损失
    physics_loss = PhysicsLoss().to(device)
    k = torch.tensor([10.0, 15.0]).to(device)
    n = torch.ones(batch_size, 1, 64, 64).to(device)
    
    total_loss, loss_dict = physics_loss(p_reconstructed, p_true, k, n)
    print("\n6. 物理损失:")
    for key, val in loss_dict.items():
        if val < 1e6:  # 只显示合理范围的值
            print(f"   {key}: {val:.6f}")
    
    # 可视化
    print("\n7. 生成可视化...")
    visualize_demo(p_true[0], p_reconstructed[0], p_sparse[0], mask[0])
    
    print("\n" + "=" * 60)
    print("✅ 演示完成!")
    print("=" * 60)


def visualize_demo(p_true, p_pred, p_sparse, mask):
    """可视化演示结果"""
    # 转换为numpy
    p_true = p_true.cpu().numpy()
    p_pred = p_pred.cpu().numpy()
    p_sparse = p_sparse.cpu().numpy()
    mask = mask[0].cpu().numpy()
    
    # 计算幅度
    amp_true = np.sqrt(p_true[0]**2 + p_true[1]**2)
    amp_pred = np.sqrt(p_pred[0]**2 + p_pred[1]**2)
    amp_sparse = np.sqrt(p_sparse[0]**2 + p_sparse[1]**2)
    error = np.abs(amp_pred - amp_true)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 真实场
    im0 = axes[0].imshow(amp_true, cmap='jet', origin='lower')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    # 稀疏观测
    im1 = axes[1].imshow(amp_sparse, cmap='jet', origin='lower')
    axes[1].scatter(*np.where(mask > 0)[::-1], c='white', s=10, alpha=0.5)
    axes[1].set_title(f'Sparse Obs ({int(mask.sum())} points)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])
    
    # 重建场
    im2 = axes[2].imshow(amp_pred, cmap='jet', origin='lower')
    axes[2].set_title('Reconstructed')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])
    
    # 误差
    im3 = axes[3].imshow(error, cmap='hot', origin='lower')
    axes[3].set_title('Error')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    
    # 保存
    os.makedirs('experiments/results', exist_ok=True)
    save_path = 'experiments/results/demo_reconstruction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ 可视化保存至: {save_path}")


if __name__ == '__main__':
    demo_quick_test()
