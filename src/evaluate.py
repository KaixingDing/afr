"""
评估与可视化模块
用于模型评估、结果可视化和生成报告
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from matplotlib.animation import FuncAnimation
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_diffusion import PhysicsGuidedDiffusionModel
from physics_loss import PhysicsLoss, compute_metrics


def load_model(checkpoint_path, device):
    """
    加载训练好的模型
    
    参数:
        checkpoint_path: checkpoint文件路径
        device: 设备
    
    返回:
        model: 加载的模型
    """
    model = PhysicsGuidedDiffusionModel(grid_size=128, timesteps=1000).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载完成 (epoch {checkpoint.get('epoch', 'N/A')})")
    
    return model


def visualize_reconstruction(p_true, p_pred, p_sparse, n, save_path):
    """
    可视化重建结果
    
    参数:
        p_true: 真实声场 [2, H, W]
        p_pred: 预测声场 [2, H, W]
        p_sparse: 稀疏观测 [2, H, W]
        n: 折射率场 [1, H, W]
        save_path: 保存路径
    """
    # 转换为numpy
    p_true = p_true.cpu().numpy()
    p_pred = p_pred.cpu().numpy()
    p_sparse = p_sparse.cpu().numpy()
    n = n.cpu().numpy()[0]
    
    # 计算幅度
    amp_true = np.sqrt(p_true[0]**2 + p_true[1]**2)
    amp_pred = np.sqrt(p_pred[0]**2 + p_pred[1]**2)
    amp_sparse = np.sqrt(p_sparse[0]**2 + p_sparse[1]**2)
    
    # 计算相位
    phase_true = np.arctan2(p_true[1], p_true[0])
    phase_pred = np.arctan2(p_pred[1], p_pred[0])
    
    # 计算误差
    error = np.abs(amp_pred - amp_true)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 真实声场幅度
    im0 = axes[0, 0].imshow(amp_true, cmap='jet', origin='lower')
    axes[0, 0].set_title('Ground Truth |p|', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 稀疏观测
    im1 = axes[0, 1].imshow(amp_sparse, cmap='jet', origin='lower')
    axes[0, 1].set_title('Sparse Observations', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 预测声场幅度
    im2 = axes[0, 2].imshow(amp_pred, cmap='jet', origin='lower')
    axes[0, 2].set_title('Reconstructed |p|', fontsize=12)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 折射率场
    im3 = axes[1, 0].imshow(n, cmap='viridis', origin='lower')
    axes[1, 0].set_title('Refractive Index n(x,y)', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 误差图
    im4 = axes[1, 1].imshow(error, cmap='hot', origin='lower')
    axes[1, 1].set_title('Reconstruction Error', fontsize=12)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 相位对比
    im5 = axes[1, 2].imshow(phase_pred - phase_true, cmap='hsv', origin='lower', 
                            vmin=-np.pi, vmax=np.pi)
    axes[1, 2].set_title('Phase Difference', fontsize=12)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_helmholtz_residual(p_pred, k, n, save_path):
    """
    可视化Helmholtz残差分布
    
    参数:
        p_pred: 预测声场 [2, H, W]
        k: 波数 (标量)
        n: 折射率场 [1, H, W]
        save_path: 保存路径
    """
    # 计算物理损失
    device = p_pred.device
    physics_loss = PhysicsLoss().to(device)
    
    # 扩展维度用于计算
    p_pred_batch = p_pred.unsqueeze(0)
    k_batch = torch.tensor([k]).to(device)
    n_batch = n.unsqueeze(0)
    
    # 计算残差
    residual = physics_loss.helmholtz_residual(p_pred_batch, k_batch, n_batch)
    residual = residual.squeeze(0).cpu().numpy()
    
    # 计算残差幅度
    residual_amp = np.sqrt(residual[0]**2 + residual[1]**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 残差幅度
    im0 = axes[0].imshow(residual_amp, cmap='hot', origin='lower')
    axes[0].set_title('Helmholtz Residual |∇²p + k²n²p|', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    # 残差统计直方图
    axes[1].hist(residual_amp.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Residual Magnitude')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_diffusion_animation(model, condition, save_path, device, num_frames=50):
    """
    创建扩散生成过程动画
    
    参数:
        model: 扩散模型
        condition: 条件输入 [1, 2, H, W]
        save_path: 保存路径
        device: 设备
        num_frames: 动画帧数
    """
    model.eval()
    
    # 生成过程中的中间帧
    frames = []
    
    with torch.no_grad():
        # 从纯噪声开始
        x = torch.randn_like(condition)
        
        # 采样间隔
        step = model.diffusion.timesteps // num_frames
        
        for i in reversed(range(0, model.diffusion.timesteps, step)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            x = model.diffusion.p_sample(model.unet, x, t, condition)
            
            # 保存当前帧
            amp = torch.sqrt(x[0, 0]**2 + x[0, 1]**2).cpu().numpy()
            frames.append(amp)
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(frames[0], cmap='jet', origin='lower', animated=True)
    ax.set_title('Diffusion Sampling Process', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Diffusion Step {(num_frames - frame_idx - 1) * step}/{model.diffusion.timesteps}')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    anim.save(save_path, writer='pillow', fps=10)
    plt.close()
    
    print(f"动画保存至: {save_path}")


def evaluate_model(model, data_dir, device, num_samples=20, output_dir='experiments/results'):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        data_dir: 数据目录
        device: 设备
        num_samples: 评估样本数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    test_files = all_files[-num_samples:]  # 使用最后的样本作为测试集
    
    all_metrics = []
    physics_loss = PhysicsLoss().to(device)
    
    print(f"评估 {len(test_files)} 个样本...")
    
    for i, filename in enumerate(tqdm(test_files)):
        filepath = os.path.join(data_dir, filename)
        data = np.load(filepath)
        
        # 加载数据
        p_true = data['pressure_field']
        p_sparse = data['sparse_observations']
        n = data['refractive_index']
        k = float(data['wavenumber'])
        
        # 转换为tensor
        p_true_tensor = torch.from_numpy(
            np.stack([p_true.real, p_true.imag], axis=0).astype(np.float32)
        ).unsqueeze(0).to(device)
        
        p_sparse_tensor = torch.from_numpy(
            np.stack([p_sparse.real, p_sparse.imag], axis=0).astype(np.float32)
        ).unsqueeze(0).to(device)
        
        n_tensor = torch.from_numpy(n[np.newaxis, :, :].astype(np.float32)).unsqueeze(0).to(device)
        k_tensor = torch.tensor([k], dtype=torch.float32).to(device)
        
        # 归一化
        p_max = torch.max(torch.abs(p_true_tensor)) + 1e-8
        p_true_norm = p_true_tensor / p_max
        p_sparse_norm = p_sparse_tensor / p_max
        
        # 生成重建
        with torch.no_grad():
            p_pred_norm = model.sample(p_sparse_norm, device)
        
        # 计算指标
        metrics = compute_metrics(p_pred_norm, p_true_norm)
        
        # 计算物理损失
        phys_loss, phys_dict = physics_loss(p_pred_norm, p_true_norm, k_tensor, n_tensor)
        metrics.update(phys_dict)
        
        all_metrics.append(metrics)
        
        # 可视化前几个样本
        if i < 5:
            vis_path = os.path.join(output_dir, f'reconstruction_{i:03d}.png')
            visualize_reconstruction(
                p_true_norm[0], p_pred_norm[0], p_sparse_norm[0], n_tensor[0], vis_path
            )
            
            # Helmholtz残差
            res_path = os.path.join(output_dir, f'helmholtz_residual_{i:03d}.png')
            visualize_helmholtz_residual(p_pred_norm[0], k, n_tensor[0], res_path)
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    # 保存结果
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    print("\n评估结果:")
    for key, stats in avg_metrics.items():
        print(f"{key:15s}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    
    print(f"\n结果保存至: {output_dir}")
    
    return avg_metrics


def plot_training_curves(history_path, save_path):
    """
    绘制训练曲线
    
    参数:
        history_path: 训练历史JSON文件路径
        save_path: 保存路径
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练损失
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 验证指标
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        val_metrics = history['val_metrics']
        epochs = np.arange(len(val_metrics))
        
        # 提取指标
        if 'MAE' in val_metrics[0]:
            mae = [m['MAE'] for m in val_metrics]
            axes[1].plot(epochs, mae, marker='o', label='MAE')
        
        if 'PSNR' in val_metrics[0]:
            psnr = [m['PSNR'] for m in val_metrics]
            ax2 = axes[1].twinx()
            ax2.plot(epochs, psnr, marker='s', color='orange', label='PSNR')
            ax2.set_ylabel('PSNR (dB)')
            ax2.legend(loc='upper right')
        
        axes[1].set_xlabel('Validation Step')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Validation Metrics')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线保存至: {save_path}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型 (如果存在)
    checkpoint_path = 'experiments/checkpoints/best_model.pt'
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path, device)
        
        # 评估模型
        evaluate_model(
            model, 
            data_dir='data/samples',
            device=device,
            num_samples=20,
            output_dir='experiments/results'
        )
        
        # 绘制训练曲线
        history_path = 'experiments/logs/history.json'
        if os.path.exists(history_path):
            plot_training_curves(history_path, 'experiments/results/training_curves.png')
    else:
        print(f"Checkpoint不存在: {checkpoint_path}")
        print("请先训练模型")
