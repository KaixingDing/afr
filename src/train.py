"""
训练脚本
训练物理引导扩散模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# 添加src到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_diffusion import PhysicsGuidedDiffusionModel
from physics_loss import PhysicsLoss, DiffusionLoss, compute_metrics


class AcousticFieldDataset(Dataset):
    """声场数据集"""
    
    def __init__(self, data_dir, split='train', train_ratio=0.8):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录
            split: 'train' 或 'val'
            train_ratio: 训练集比例
        """
        self.data_dir = data_dir
        self.split = split
        
        # 获取所有样本文件
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
        # 划分训练/验证集
        num_train = int(len(all_files) * train_ratio)
        if split == 'train':
            self.files = all_files[:num_train]
        else:
            self.files = all_files[num_train:]
        
        print(f"{split}集: {len(self.files)} 个样本")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 加载数据
        filepath = os.path.join(self.data_dir, self.files[idx])
        data = np.load(filepath)
        
        # 提取字段
        p_true = data['pressure_field']  # 复数
        p_sparse = data['sparse_observations']  # 复数
        n = data['refractive_index']  # 实数
        k = float(data['wavenumber'])
        
        # 转换为实部/虚部表示
        p_true_tensor = np.stack([p_true.real, p_true.imag], axis=0).astype(np.float32)
        p_sparse_tensor = np.stack([p_sparse.real, p_sparse.imag], axis=0).astype(np.float32)
        n_tensor = n[np.newaxis, :, :].astype(np.float32)
        
        # 归一化
        # 声场归一化
        p_max = np.max(np.abs(p_true)) + 1e-8
        p_true_tensor = p_true_tensor / p_max
        p_sparse_tensor = p_sparse_tensor / p_max
        
        return {
            'pressure_true': torch.from_numpy(p_true_tensor),
            'pressure_sparse': torch.from_numpy(p_sparse_tensor),
            'refractive_index': torch.from_numpy(n_tensor),
            'wavenumber': torch.tensor(k, dtype=torch.float32),
            'normalization': torch.tensor(p_max, dtype=torch.float32),
        }


def train_epoch(model, dataloader, optimizer, diffusion_loss, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_dict_sum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # 移动数据到设备
        p_true = batch['pressure_true'].to(device)
        p_sparse = batch['pressure_sparse'].to(device)
        n = batch['refractive_index'].to(device)
        k = batch['wavenumber'].to(device)
        
        batch_size = p_true.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, model.diffusion.timesteps, (batch_size,), device=device)
        
        # 前向扩散: 添加噪声
        noise = torch.randn_like(p_true)
        p_noisy, _ = model.diffusion.q_sample(p_true, t, noise)
        
        # 预测噪声
        noise_pred = model(p_noisy, t, p_sparse)
        
        # 计算损失
        loss, loss_dict = diffusion_loss(noise_pred, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item()
        for key, val in loss_dict.items():
            loss_dict_sum[key] = loss_dict_sum.get(key, 0) + val
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {key: val / len(dataloader) for key, val in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict


def validate(model, dataloader, physics_loss, device):
    """验证"""
    model.eval()
    total_metrics = {}
    total_physics_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            p_true = batch['pressure_true'].to(device)
            p_sparse = batch['pressure_sparse'].to(device)
            n = batch['refractive_index'].to(device)
            k = batch['wavenumber'].to(device)
            
            # 生成重建场
            p_pred = model.sample(p_sparse, device)
            
            # 计算物理损失
            phys_loss, _ = physics_loss(p_pred, p_true, k, n)
            total_physics_loss += phys_loss.item()
            
            # 计算评估指标
            metrics = compute_metrics(p_pred, p_true)
            for key, val in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0) + val
    
    # 平均
    num_batches = len(dataloader)
    avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
    avg_metrics['physics_loss'] = total_physics_loss / num_batches
    
    return avg_metrics


def train(config):
    """
    主训练函数
    
    参数:
        config: 配置字典
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = AcousticFieldDataset(
        config['data_dir'], 
        split='train', 
        train_ratio=config['train_ratio']
    )
    val_dataset = AcousticFieldDataset(
        config['data_dir'], 
        split='val', 
        train_ratio=config['train_ratio']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 创建模型
    model = PhysicsGuidedDiffusionModel(
        grid_size=config['grid_size'],
        timesteps=config['timesteps']
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    physics_loss = PhysicsLoss(
        lambda_helmholtz=config['lambda_helmholtz'],
        lambda_energy=config['lambda_energy'],
        lambda_boundary=config['lambda_boundary'],
    ).to(device)
    
    diffusion_loss = DiffusionLoss(physics_loss=None)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.01
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_metrics': [],
    }
    
    # 创建输出目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, optimizer, diffusion_loss, device, epoch
        )
        
        print(f"训练损失: {train_loss:.6f}")
        for key, val in train_loss_dict.items():
            print(f"  {key}: {val:.6f}")
        
        # 验证
        if epoch % config['val_interval'] == 0:
            val_metrics = validate(model, val_loader, physics_loss, device)
            
            print(f"验证指标:")
            for key, val in val_metrics.items():
                print(f"  {key}: {val:.6f}")
            
            # 保存历史
            history['val_metrics'].append(val_metrics)
            
            # 保存最佳模型
            if val_metrics['physics_loss'] < best_val_loss:
                best_val_loss = val_metrics['physics_loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, os.path.join(config['checkpoint_dir'], 'best_model.pt'))
                print(f"保存最佳模型 (physics_loss={best_val_loss:.6f})")
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史
        history['train_loss'].append(train_loss)
        
        # 定期保存checkpoint
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt'))
    
    # 保存最终模型
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'history': history,
    }, os.path.join(config['checkpoint_dir'], 'final_model.pt'))
    
    # 保存训练历史
    with open(os.path.join(config['log_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n训练完成!")


if __name__ == '__main__':
    # 配置
    config = {
        # 数据
        'data_dir': 'data/samples',
        'train_ratio': 0.8,
        
        # 模型
        'grid_size': 128,
        'timesteps': 1000,
        
        # 物理损失权重
        'lambda_helmholtz': 1.0,
        'lambda_energy': 0.5,
        'lambda_boundary': 0.3,
        
        # 训练
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        
        # 验证和保存
        'val_interval': 5,
        'save_interval': 10,
        'checkpoint_dir': 'experiments/checkpoints',
        'log_dir': 'experiments/logs',
    }
    
    # 创建目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # 保存配置
    with open(os.path.join(config['log_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 开始训练
    train(config)
