"""
物理引导扩散模型 (Physics-Guided Diffusion Model)
基于DDPM架构实现声场重建
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码用于时间步嵌入"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """残差块，用于U-Net"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim else None
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 计算注意力
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class UNet(nn.Module):
    """
    U-Net架构，用于扩散模型
    输入: 噪声场 + 稀疏观测条件
    输出: 预测的噪声
    """
    
    def __init__(self, 
                 in_channels=2,  # 实部和虚部
                 out_channels=2,
                 base_channels=64,
                 channel_mults=(1, 2, 4, 8),
                 num_res_blocks=2,
                 attention_resolutions=(16, 8),
                 dropout=0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 时间嵌入
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 条件嵌入 (稀疏观测)
        self.cond_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 输入投影
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [base_channels]
        now_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, out_ch, time_emb_dim)
                )
                now_channels = out_ch
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.down_samples.append(nn.Conv2d(now_channels, now_channels, 3, 2, 1))
                channels.append(now_channels)
        
        # 中间层
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_emb_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(now_channels + channels.pop(), out_ch, time_emb_dim)
                )
                now_channels = out_ch
            
            if i != len(channel_mults) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(now_channels, now_channels, 4, 2, 1)
                )
        
        # 输出投影
        self.out_norm = nn.GroupNorm(8, now_channels)
        self.out_conv = nn.Conv2d(now_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t, condition):
        """
        前向传播
        
        参数:
            x: 噪声场 [B, 2, H, W]
            t: 时间步 [B]
            condition: 稀疏观测条件 [B, 2, H, W]
        
        返回:
            预测的噪声 [B, 2, H, W]
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 条件嵌入
        cond_emb = self.cond_conv(condition)
        
        # 输入 + 条件
        h = self.input_conv(x) + cond_emb
        
        # 下采样
        hs = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            hs.append(h)
        
        for downsample in self.down_samples:
            h = downsample(h)
            hs.append(h)
        
        # 中间层
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 上采样
        for i, block in enumerate(self.up_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, t_emb)
            
            if i % (len(self.up_blocks) // len(self.up_samples)) == 0 and i > 0:
                idx = i // (len(self.up_blocks) // len(self.up_samples)) - 1
                if idx < len(self.up_samples):
                    h = self.up_samples[idx](h)
        
        # 输出
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


class GaussianDiffusion:
    """
    高斯扩散过程 (DDPM)
    """
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # 定义beta schedule (线性)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算用于扩散的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 计算用于反向过程的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程: q(x_t | x_0)
        
        参数:
            x_start: 初始数据 [B, C, H, W]
            t: 时间步 [B]
            noise: 可选的噪声
        
        返回:
            x_t: 加噪后的数据
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_t = sqrt_alpha.to(x_start.device) * x_start + \
              sqrt_one_minus_alpha.to(x_start.device) * noise
        
        return x_t, noise
    
    def p_sample(self, model, x_t, t, condition):
        """
        反向去噪过程: p(x_{t-1} | x_t)
        
        参数:
            model: 去噪模型
            x_t: 当前噪声数据 [B, C, H, W]
            t: 当前时间步 [B]
            condition: 条件输入
        
        返回:
            x_{t-1}: 去噪后的数据
        """
        # 预测噪声
        noise_pred = model(x_t, t, condition)
        
        # 计算均值
        alpha = self.alphas[t][:, None, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        mean = sqrt_recip_alpha.to(x_t.device) * (
            x_t - (1 - alpha).to(x_t.device) / sqrt_one_minus_alpha_cumprod.to(x_t.device) * noise_pred
        )
        
        # 添加噪声 (t > 0时)
        variance = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x_t)
        
        # t == 0 时不添加噪声
        nonzero_mask = (t != 0).float()[:, None, None, None]
        
        x_prev = mean + nonzero_mask.to(x_t.device) * torch.sqrt(variance.to(x_t.device)) * noise
        
        return x_prev
    
    def p_sample_loop(self, model, shape, condition, device):
        """
        完整的反向采样过程
        
        参数:
            model: 去噪模型
            shape: 输出形状
            condition: 条件输入
            device: 设备
        
        返回:
            x_0: 生成的数据
        """
        b = shape[0]
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition)
        
        return x


class PhysicsGuidedDiffusionModel(nn.Module):
    """
    物理引导扩散模型 - 主模型
    结合扩散模型和物理约束
    """
    
    def __init__(self, 
                 grid_size=128,
                 in_channels=2,
                 timesteps=1000):
        super().__init__()
        
        self.grid_size = grid_size
        self.unet = UNet(in_channels=in_channels, out_channels=in_channels)
        self.diffusion = GaussianDiffusion(timesteps=timesteps)
    
    def forward(self, x, t, condition):
        """前向传播"""
        return self.unet(x, t, condition)
    
    def sample(self, condition, device):
        """
        从条件输入生成完整声场
        
        参数:
            condition: 稀疏观测 [B, 2, H, W]
            device: 设备
        
        返回:
            生成的完整声场
        """
        shape = condition.shape
        return self.diffusion.p_sample_loop(self.unet, shape, condition, device)


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PhysicsGuidedDiffusionModel(grid_size=128).to(device)
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 2, 128, 128).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, 2, 128, 128).to(device)
    
    # 前向传播
    output = model(x, t, condition)
    print(f"模型输出形状: {output.shape}")
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
