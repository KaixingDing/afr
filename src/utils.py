"""
工具函数模块
"""

import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    设置随机种子以保证可重复性
    
    参数:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    计算模型参数量
    
    参数:
        model: PyTorch模型
    
    返回:
        total: 总参数量
        trainable: 可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    保存训练checkpoint
    
    参数:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        metrics: 评估指标
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer=None, scheduler=None, filepath=None, device='cpu'):
    """
    加载checkpoint
    
    参数:
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        filepath: checkpoint路径
        device: 设备
    
    返回:
        epoch: 当前epoch
        metrics: 评估指标
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


def complex_to_real(field):
    """
    将复数场转换为实部/虚部表示
    
    参数:
        field: 复数数组 [..., H, W]
    
    返回:
        real_field: [..., 2, H, W] (最后一维前为实部，后为虚部)
    """
    real = np.real(field)
    imag = np.imag(field)
    return np.stack([real, imag], axis=-3)


def real_to_complex(field):
    """
    将实部/虚部表示转换为复数场
    
    参数:
        field: [..., 2, H, W]
    
    返回:
        complex_field: [..., H, W] (复数)
    """
    real = field[..., 0, :, :]
    imag = field[..., 1, :, :]
    return real + 1j * imag


def normalize_field(field, method='max'):
    """
    归一化声场
    
    参数:
        field: 输入场
        method: 归一化方法 ('max', 'std', 'minmax')
    
    返回:
        normalized_field: 归一化后的场
        norm_factor: 归一化因子
    """
    if method == 'max':
        norm_factor = np.max(np.abs(field)) + 1e-8
        normalized_field = field / norm_factor
    elif method == 'std':
        norm_factor = np.std(field) + 1e-8
        normalized_field = (field - np.mean(field)) / norm_factor
    elif method == 'minmax':
        min_val = np.min(field)
        max_val = np.max(field)
        norm_factor = (min_val, max_val)
        normalized_field = (field - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_field, norm_factor


def denormalize_field(field, norm_factor, method='max'):
    """
    反归一化声场
    
    参数:
        field: 归一化的场
        norm_factor: 归一化因子
        method: 归一化方法
    
    返回:
        denormalized_field: 恢复的场
    """
    if method == 'max':
        return field * norm_factor
    elif method == 'std':
        return field * norm_factor['std'] + norm_factor['mean']
    elif method == 'minmax':
        min_val, max_val = norm_factor
        return field * (max_val - min_val) + min_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_device(gpu_id=None):
    """
    获取计算设备
    
    参数:
        gpu_id: GPU ID (None表示自动选择)
    
    返回:
        device: torch设备
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device


def create_grid(size, domain_size=1.0):
    """
    创建计算网格
    
    参数:
        size: 网格点数
        domain_size: 物理域大小
    
    返回:
        X, Y: 网格坐标
    """
    x = np.linspace(0, domain_size, size)
    y = np.linspace(0, domain_size, size)
    X, Y = np.meshgrid(x, y)
    return X, Y


if __name__ == '__main__':
    # 测试工具函数
    print("测试工具函数...")
    
    # 设置随机种子
    set_seed(42)
    print("✓ 随机种子设置完成")
    
    # 测试复数转换
    field_complex = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)
    field_real = complex_to_real(field_complex)
    field_back = real_to_complex(field_real)
    
    assert np.allclose(field_complex, field_back), "复数转换失败"
    print("✓ 复数转换测试通过")
    
    # 测试归一化
    field = np.random.randn(64, 64)
    normalized, norm_factor = normalize_field(field, method='max')
    denormalized = denormalize_field(normalized, norm_factor, method='max')
    
    assert np.allclose(field, denormalized), "归一化/反归一化失败"
    print("✓ 归一化测试通过")
    
    # 测试设备获取
    device = get_device()
    print(f"✓ 使用设备: {device}")
    
    print("\n所有测试通过!")
