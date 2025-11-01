"""
声场模拟数据生成模块
使用有限差分法（FDTD）求解Helmholtz方程生成声场数据
"""

import numpy as np
import os
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class HelmholtzSolver:
    """
    二维Helmholtz方程求解器
    方程形式: (∇² + k²n²)p = s
    """
    
    def __init__(self, grid_size=128, domain_size=1.0):
        """
        初始化求解器
        
        参数:
            grid_size: 网格点数
            domain_size: 物理域大小 (米)
        """
        self.N = grid_size
        self.L = domain_size
        self.dx = domain_size / grid_size
        self.dy = self.dx
        
        # 生成网格
        x = np.linspace(0, domain_size, grid_size)
        y = np.linspace(0, domain_size, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        
    def create_refractive_index(self, obstacle_type='circle'):
        """
        创建折射率场 n(x,y)
        
        参数:
            obstacle_type: 障碍物类型 ('circle', 'square', 'random')
        
        返回:
            n: 折射率场 (N x N)
        """
        n = np.ones((self.N, self.N))
        
        if obstacle_type == 'circle':
            # 圆形散射体
            cx, cy = np.random.uniform(0.3, 0.7, 2) * self.L
            radius = np.random.uniform(0.05, 0.15) * self.L
            mask = (self.X - cx)**2 + (self.Y - cy)**2 < radius**2
            n[mask] = np.random.uniform(1.2, 1.8)
            
        elif obstacle_type == 'square':
            # 方形散射体
            cx, cy = np.random.uniform(0.3, 0.7, 2) * self.L
            size = np.random.uniform(0.1, 0.2) * self.L
            mask = (np.abs(self.X - cx) < size/2) & (np.abs(self.Y - cy) < size/2)
            n[mask] = np.random.uniform(1.2, 1.8)
            
        elif obstacle_type == 'random':
            # 随机多个散射体
            num_obstacles = np.random.randint(2, 5)
            for _ in range(num_obstacles):
                cx, cy = np.random.uniform(0.2, 0.8, 2) * self.L
                radius = np.random.uniform(0.03, 0.1) * self.L
                mask = (self.X - cx)**2 + (self.Y - cy)**2 < radius**2
                n[mask] = np.random.uniform(1.1, 1.6)
        
        return n
    
    def create_source(self, source_type='point'):
        """
        创建声源项 s(x,y)
        
        参数:
            source_type: 声源类型 ('point', 'line', 'gaussian')
        
        返回:
            s: 声源场 (N x N)
        """
        s = np.zeros((self.N, self.N), dtype=complex)
        
        if source_type == 'point':
            # 点源
            sx, sy = 0.1 * self.L, 0.5 * self.L
            ix = int(sx / self.dx)
            iy = int(sy / self.dy)
            s[iy, ix] = 1.0
            
        elif source_type == 'line':
            # 线源
            ix = int(0.1 * self.N)
            s[:, ix] = 1.0
            
        elif source_type == 'gaussian':
            # 高斯源
            sx, sy = 0.1 * self.L, 0.5 * self.L
            sigma = 0.05 * self.L
            s = np.exp(-((self.X - sx)**2 + (self.Y - sy)**2) / (2 * sigma**2))
        
        return s
    
    def solve_helmholtz(self, k, n, s):
        """
        求解Helmholtz方程: (∇² + k²n²)p = s
        使用有限差分法
        
        参数:
            k: 波数 (2πf/c0)
            n: 折射率场 (N x N)
            s: 声源场 (N x N)
        
        返回:
            p: 声压场 (N x N, 复数)
        """
        N = self.N
        dx2 = self.dx ** 2
        
        # 构建Laplacian矩阵 (五点差分)
        # ∇²p ≈ (p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j}) / dx²
        
        # 将二维问题展平为一维
        n_flat = n.flatten()
        s_flat = s.flatten()
        
        # 构建稀疏矩阵 A, 使得 A*p_flat = s_flat
        # A = ∇²/dx² + k²n²
        
        # 主对角线: -4/dx² + k²n²
        diag_main = -4.0 / dx2 + k**2 * n_flat**2
        
        # 上下左右对角线: 1/dx²
        diag_off = np.ones(N * N) / dx2
        
        # 构建矩阵 (考虑边界条件)
        diagonals = [diag_main]
        offsets = [0]
        
        # 上下邻居 (±1)
        diag_lr = diag_off.copy()
        for i in range(N):
            diag_lr[i * N] = 0  # 左边界
            if i > 0:
                diag_lr[i * N - 1] = 0  # 右边界
        
        diagonals.extend([diag_lr[1:], diag_lr[:-1]])
        offsets.extend([1, -1])
        
        # 前后邻居 (±N)
        diagonals.extend([diag_off[N:], diag_off[:-N]])
        offsets.extend([N, -N])
        
        A = diags(diagonals, offsets, shape=(N*N, N*N), format='csr')
        
        # 求解线性系统
        try:
            p_flat = spsolve(A, s_flat)
        except:
            # 如果求解失败，使用迭代法
            from scipy.sparse.linalg import gmres
            p_flat, info = gmres(A, s_flat, maxiter=1000, tol=1e-6)
            if info != 0:
                print(f"Warning: GMRES did not converge, info={info}")
        
        p = p_flat.reshape((N, N))
        return p
    
    def generate_sparse_observations(self, p, num_points=50, pattern='random'):
        """
        从完整声场中采样稀疏观测点
        
        参数:
            p: 完整声压场 (N x N)
            num_points: 观测点数量
            pattern: 采样模式 ('random', 'grid', 'line')
        
        返回:
            sparse_field: 稀疏观测场 (N x N, 其余位置为0)
            indices: 观测点位置 (num_points x 2)
        """
        N = self.N
        sparse_field = np.zeros_like(p)
        
        if pattern == 'random':
            # 随机采样
            indices_x = np.random.randint(5, N-5, num_points)
            indices_y = np.random.randint(5, N-5, num_points)
            indices = np.stack([indices_y, indices_x], axis=1)
            
        elif pattern == 'grid':
            # 规则网格采样
            grid_size = int(np.sqrt(num_points))
            x_idx = np.linspace(5, N-5, grid_size, dtype=int)
            y_idx = np.linspace(5, N-5, grid_size, dtype=int)
            xx, yy = np.meshgrid(x_idx, y_idx)
            indices = np.stack([yy.flatten(), xx.flatten()], axis=1)
            
        elif pattern == 'line':
            # 线性阵列
            x_idx = np.full(num_points, N // 4)
            y_idx = np.linspace(10, N-10, num_points, dtype=int)
            indices = np.stack([y_idx, x_idx], axis=1)
        
        # 填充稀疏场
        for idx_y, idx_x in indices:
            sparse_field[idx_y, idx_x] = p[idx_y, idx_x]
        
        return sparse_field, indices


def generate_dataset(num_samples=1000, grid_size=128, output_dir='data/samples'):
    """
    生成声场数据集
    
    参数:
        num_samples: 生成样本数量
        grid_size: 网格大小
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    solver = HelmholtzSolver(grid_size=grid_size)
    
    # 声速和频率范围
    c0 = 343.0  # 空气中声速 (m/s)
    frequencies = [500, 1000, 1500, 2000]  # Hz
    
    obstacle_types = ['circle', 'square', 'random']
    source_types = ['point', 'gaussian']
    
    print(f"开始生成 {num_samples} 个声场样本...")
    
    for i in range(num_samples):
        # 随机选择参数
        freq = np.random.choice(frequencies)
        k = 2 * np.pi * freq / c0
        
        obstacle_type = np.random.choice(obstacle_types)
        source_type = np.random.choice(source_types)
        
        # 生成场
        n = solver.create_refractive_index(obstacle_type)
        s = solver.create_source(source_type)
        p = solver.solve_helmholtz(k, n, s)
        
        # 生成稀疏观测
        num_obs = np.random.randint(30, 80)
        sparse_p, obs_indices = solver.generate_sparse_observations(
            p, num_points=num_obs, pattern='random'
        )
        
        # 保存数据
        data = {
            'pressure_field': p,  # 真值场
            'sparse_observations': sparse_p,  # 稀疏观测
            'refractive_index': n,  # 折射率场
            'source': s,  # 声源
            'wavenumber': k,  # 波数
            'frequency': freq,  # 频率
            'observation_indices': obs_indices,  # 观测点位置
        }
        
        filename = os.path.join(output_dir, f'sample_{i:05d}.npz')
        np.savez_compressed(filename, **data)
        
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本")
    
    print(f"数据生成完成！保存在 {output_dir}")


def visualize_sample(sample_path):
    """
    可视化单个样本
    
    参数:
        sample_path: 样本文件路径
    """
    data = np.load(sample_path)
    
    p = data['pressure_field']
    sparse_p = data['sparse_observations']
    n = data['refractive_index']
    freq = data['frequency']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 真值声压场 (幅度)
    im0 = axes[0, 0].imshow(np.abs(p), cmap='jet', origin='lower')
    axes[0, 0].set_title(f'Ground Truth |p| (f={freq}Hz)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 稀疏观测
    im1 = axes[0, 1].imshow(np.abs(sparse_p), cmap='jet', origin='lower')
    axes[0, 1].set_title('Sparse Observations |p|')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 折射率场
    im2 = axes[1, 0].imshow(n, cmap='viridis', origin='lower')
    axes[1, 0].set_title('Refractive Index n(x,y)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 声压场 (相位)
    im3 = axes[1, 1].imshow(np.angle(p), cmap='hsv', origin='lower')
    axes[1, 1].set_title('Phase of p')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("可视化保存至 sample_visualization.png")


if __name__ == '__main__':
    # 生成小规模数据集用于测试
    generate_dataset(num_samples=100, grid_size=128, output_dir='data/samples')
    
    # 可视化第一个样本
    if os.path.exists('data/samples/sample_00000.npz'):
        visualize_sample('data/samples/sample_00000.npz')
