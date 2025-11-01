"""
物理约束损失函数
包含Helmholtz残差、能量守恒、边界条件等物理约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    物理约束损失
    L = L_recon + λ1*L_helmholtz + λ2*L_energy + λ3*L_boundary
    """
    
    def __init__(self, 
                 lambda_helmholtz=1.0,
                 lambda_energy=0.5,
                 lambda_boundary=0.3,
                 grid_spacing=1.0/128):
        super().__init__()
        
        self.lambda_helmholtz = lambda_helmholtz
        self.lambda_energy = lambda_energy
        self.lambda_boundary = lambda_boundary
        self.dx = grid_spacing
        self.dy = grid_spacing
    
    def compute_laplacian(self, field):
        """
        计算拉普拉斯算子 ∇²p
        使用五点差分格式
        
        参数:
            field: 输入场 [B, 2, H, W] (实部和虚部)
        
        返回:
            laplacian: ∇²field [B, 2, H, W]
        """
        # 分离实部和虚部
        field_real = field[:, 0:1, :, :]
        field_imag = field[:, 1:2, :, :]
        
        # 计算二阶导数
        # ∂²/∂x²
        d2_dx2_real = (
            F.pad(field_real[:, :, :, 2:], (0, 2, 0, 0)) +
            F.pad(field_real[:, :, :, :-2], (2, 0, 0, 0)) -
            2 * field_real
        ) / (self.dx ** 2)
        
        d2_dx2_imag = (
            F.pad(field_imag[:, :, :, 2:], (0, 2, 0, 0)) +
            F.pad(field_imag[:, :, :, :-2], (2, 0, 0, 0)) -
            2 * field_imag
        ) / (self.dx ** 2)
        
        # ∂²/∂y²
        d2_dy2_real = (
            F.pad(field_real[:, :, 2:, :], (0, 0, 0, 2)) +
            F.pad(field_real[:, :, :-2, :], (0, 0, 2, 0)) -
            2 * field_real
        ) / (self.dy ** 2)
        
        d2_dy2_imag = (
            F.pad(field_imag[:, :, 2:, :], (0, 0, 0, 2)) +
            F.pad(field_imag[:, :, :-2, :], (0, 0, 2, 0)) -
            2 * field_imag
        ) / (self.dy ** 2)
        
        # ∇² = ∂²/∂x² + ∂²/∂y²
        laplacian_real = d2_dx2_real + d2_dy2_real
        laplacian_imag = d2_dx2_imag + d2_dy2_imag
        
        laplacian = torch.cat([laplacian_real, laplacian_imag], dim=1)
        
        return laplacian
    
    def helmholtz_residual(self, p_pred, k, n, s=None):
        """
        计算Helmholtz方程残差
        残差 = ∇²p + k²n²p - s
        
        参数:
            p_pred: 预测声场 [B, 2, H, W]
            k: 波数 [B]
            n: 折射率场 [B, 1, H, W]
            s: 声源项 [B, 2, H, W] (可选)
        
        返回:
            residual: Helmholtz残差 [B, 2, H, W]
        """
        # 计算∇²p
        laplacian_p = self.compute_laplacian(p_pred)
        
        # 计算k²n²p
        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        p_real = p_pred[:, 0:1, :, :]
        p_imag = p_pred[:, 1:2, :, :]
        
        k2_n2 = (k[:, None, None, None] ** 2) * (n ** 2)
        
        k2n2p_real = k2_n2 * p_real
        k2n2p_imag = k2_n2 * p_imag
        
        k2n2p = torch.cat([k2n2p_real, k2n2p_imag], dim=1)
        
        # 计算残差
        residual = laplacian_p + k2n2p
        
        if s is not None:
            residual = residual - s
        
        return residual
    
    def energy_conservation(self, p_pred, p_true):
        """
        计算能量守恒损失
        E(p) = ∫|p|² dx dy
        
        参数:
            p_pred: 预测声场 [B, 2, H, W]
            p_true: 真实声场 [B, 2, H, W]
        
        返回:
            energy_loss: 能量差异
        """
        # 计算幅度平方 |p|² = real² + imag²
        energy_pred = p_pred[:, 0, :, :] ** 2 + p_pred[:, 1, :, :] ** 2
        energy_true = p_true[:, 0, :, :] ** 2 + p_true[:, 1, :, :] ** 2
        
        # 积分 (求和近似)
        E_pred = torch.sum(energy_pred, dim=[1, 2])
        E_true = torch.sum(energy_true, dim=[1, 2])
        
        # 相对误差
        energy_loss = torch.abs(E_pred - E_true) / (E_true + 1e-8)
        
        return energy_loss.mean()
    
    def boundary_loss(self, p_pred):
        """
        计算边界条件损失
        假设边界为吸收边界: ∂p/∂n ≈ 0
        
        参数:
            p_pred: 预测声场 [B, 2, H, W]
        
        返回:
            boundary_loss: 边界法向导数的L2范数
        """
        # 提取边界
        # 上边界
        top_grad = torch.abs(p_pred[:, :, 0, :] - p_pred[:, :, 1, :])
        # 下边界
        bottom_grad = torch.abs(p_pred[:, :, -1, :] - p_pred[:, :, -2, :])
        # 左边界
        left_grad = torch.abs(p_pred[:, :, :, 0] - p_pred[:, :, :, 1])
        # 右边界
        right_grad = torch.abs(p_pred[:, :, :, -1] - p_pred[:, :, :, -2])
        
        # 总边界损失
        boundary_loss = (
            top_grad.mean() + bottom_grad.mean() +
            left_grad.mean() + right_grad.mean()
        ) / 4.0
        
        return boundary_loss
    
    def reconstruction_loss(self, p_pred, p_true, mask=None):
        """
        重建损失 (L2)
        
        参数:
            p_pred: 预测声场 [B, 2, H, W]
            p_true: 真实声场 [B, 2, H, W]
            mask: 可选的掩码 [B, 1, H, W]
        
        返回:
            recon_loss: 重建误差
        """
        if mask is not None:
            diff = (p_pred - p_true) * mask
        else:
            diff = p_pred - p_true
        
        recon_loss = torch.mean(diff ** 2)
        
        return recon_loss
    
    def forward(self, p_pred, p_true, k, n, s=None, mask=None):
        """
        计算总物理损失
        
        参数:
            p_pred: 预测声场 [B, 2, H, W]
            p_true: 真实声场 [B, 2, H, W]
            k: 波数 [B]
            n: 折射率场 [B, 1, H, W]
            s: 声源项 [B, 2, H, W] (可选)
            mask: 观测掩码 [B, 1, H, W] (可选)
        
        返回:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 重建损失
        L_recon = self.reconstruction_loss(p_pred, p_true, mask)
        
        # Helmholtz残差
        helmholtz_res = self.helmholtz_residual(p_pred, k, n, s)
        L_helmholtz = torch.mean(helmholtz_res ** 2)
        
        # 能量守恒
        L_energy = self.energy_conservation(p_pred, p_true)
        
        # 边界条件
        L_boundary = self.boundary_loss(p_pred)
        
        # 总损失
        total_loss = (
            L_recon +
            self.lambda_helmholtz * L_helmholtz +
            self.lambda_energy * L_energy +
            self.lambda_boundary * L_boundary
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': L_recon.item(),
            'helmholtz': L_helmholtz.item(),
            'energy': L_energy.item(),
            'boundary': L_boundary.item(),
        }
        
        return total_loss, loss_dict


class DiffusionLoss(nn.Module):
    """
    扩散模型训练损失
    """
    
    def __init__(self, physics_loss=None):
        super().__init__()
        self.physics_loss = physics_loss
        self.mse = nn.MSELoss()
    
    def forward(self, noise_pred, noise_true, x0_pred=None, x0_true=None, 
                k=None, n=None, s=None):
        """
        计算扩散损失
        
        参数:
            noise_pred: 预测的噪声 [B, 2, H, W]
            noise_true: 真实噪声 [B, 2, H, W]
            x0_pred: 预测的原始场 [B, 2, H, W] (可选)
            x0_true: 真实原始场 [B, 2, H, W] (可选)
            k, n, s: 物理参数 (可选)
        
        返回:
            total_loss: 总损失
            loss_dict: 损失字典
        """
        # 噪声预测损失
        L_noise = self.mse(noise_pred, noise_true)
        
        loss_dict = {
            'noise': L_noise.item(),
        }
        
        total_loss = L_noise
        
        # 如果提供了x0预测，添加物理损失
        if x0_pred is not None and x0_true is not None and self.physics_loss is not None:
            physics_loss, physics_dict = self.physics_loss(
                x0_pred, x0_true, k, n, s
            )
            total_loss = total_loss + 0.1 * physics_loss
            loss_dict.update({f'physics_{key}': val for key, val in physics_dict.items()})
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def compute_metrics(p_pred, p_true):
    """
    计算评估指标
    
    参数:
        p_pred: 预测声场 [B, 2, H, W]
        p_true: 真实声场 [B, 2, H, W]
    
    返回:
        metrics: 指标字典
    """
    # 转换为numpy用于计算
    p_pred_np = p_pred.detach().cpu().numpy()
    p_true_np = p_true.detach().cpu().numpy()
    
    # 计算幅度
    amp_pred = np.sqrt(p_pred_np[:, 0, :, :] ** 2 + p_pred_np[:, 1, :, :] ** 2)
    amp_true = np.sqrt(p_true_np[:, 0, :, :] ** 2 + p_true_np[:, 1, :, :] ** 2)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(amp_pred - amp_true))
    
    # PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((amp_pred - amp_true) ** 2)
    if mse > 0:
        psnr = 20 * np.log10(np.max(amp_true) / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    # SSIM (简化版本 - 使用相关系数近似)
    def correlation(x, y):
        x_flat = x.flatten()
        y_flat = y.flatten()
        return np.corrcoef(x_flat, y_flat)[0, 1]
    
    ssim_values = [correlation(amp_pred[i], amp_true[i]) 
                   for i in range(amp_pred.shape[0])]
    ssim = np.mean(ssim_values)
    
    metrics = {
        'MAE': mae,
        'PSNR': psnr,
        'SSIM': ssim,
    }
    
    return metrics


if __name__ == '__main__':
    # 测试物理损失
    import numpy as np
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    B, H, W = 4, 128, 128
    p_pred = torch.randn(B, 2, H, W).to(device)
    p_true = torch.randn(B, 2, H, W).to(device)
    k = torch.tensor([10.0, 15.0, 20.0, 25.0]).to(device)
    n = torch.ones(B, 1, H, W).to(device)
    s = torch.zeros(B, 2, H, W).to(device)
    
    # 计算物理损失
    physics_loss = PhysicsLoss().to(device)
    total_loss, loss_dict = physics_loss(p_pred, p_true, k, n, s)
    
    print("物理损失测试:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.6f}")
    
    # 计算评估指标
    metrics = compute_metrics(p_pred, p_true)
    print("\n评估指标:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")
