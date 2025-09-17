
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # 注意：这里 GroupNorm 的 num_groups 和 num_channels 都是 channels // groups，
        # 等价于每组一个通道。如果你本意是按 groups 分组，可改为：
        # nn.GroupNorm(self.groups, channels // self.groups)
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SimplifiedAFF(nn.Module):
    """Modified version of the AFF module to include frequency domain processing"""
    def __init__(self, in_channels, out_channels):
        super(SimplifiedAFF, self).__init__()

        # MBConv module
        self.mbconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution for channel mixing（未使用，可按需移除）
        self.group_linear = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Processing in the frequency domain
        self.freq_process = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention module
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 频域处理（保持数值稳定：FFT 用 float32，再回到原始 dtype）
        orig_dtype = x.dtype
        x_fp32 = x.float()
        x_freq = torch.fft.fft2(x_fp32)
        x_freq_magnitude = torch.abs(x_freq).to(orig_dtype)
        x_freq = self.freq_process(x_freq_magnitude)

        # 空间域处理
        x_spatial = self.mbconv(x)

        # 注意力融合
        attention_map = self.attention(torch.cat([x_freq, x_spatial], dim=1))
        out = x_freq * attention_map + x_spatial * (1 - attention_map)
        return out


class EMAAFFCombined(nn.Module):
    """Combined EMA and SimplifiedAFF Module - Dual Domain Attention Module
    在该模块的每一层引入残差：
      - 预处理（仅当需要通道投影时）
      - EMA 分支
      - AFF 分支
      - 融合输出到模块输入
    """
    def __init__(self, in_channels, ema_factor=32):
        super(EMAAFFCombined, self).__init__()
        out_channels = in_channels  # 保持进出通道一致，便于各处残差

        # 检查 EMA 分组可整除
        assert out_channels % ema_factor == 0, f"in_channels ({in_channels}) must be divisible by ema_factor ({ema_factor})"

        # EMA 模块
        self.ema = EMA(out_channels, factor=ema_factor)

        # Simplified AFF 模块
        self.aff = SimplifiedAFF(in_channels, out_channels)

        # 预处理：仅当通道不一致时才做 1x1 投影，并加残差
        if in_channels != out_channels:
            self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.pre_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.use_pre_residual = True
        else:
            self.pre_conv = nn.Identity()
            self.pre_skip = nn.Identity()
            self.use_pre_residual = False  # 避免 Identity + Identity = 2x

        # 融合调制
        self.output_modulation = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        # 最终输出残差（与模块输入对齐）
        self.out_skip = nn.Identity()  # 此处 in==out，若以后变动通道，可改为 1x1 Conv

    def forward(self, x):
        x_in = x  # 保存模块输入供最终残差

        # 预处理层（条件残差）
        x_processed = self.pre_conv(x_in)
        if self.use_pre_residual:
            x_processed = x_processed + self.pre_skip(x_in)

        # 并行分支：EMA 分支 + 残差
        ema_out = self.ema(x_processed)
        ema_out = ema_out + x_processed

        # 并行分支：AFF 分支 + 残差
        aff_out = self.aff(x_in)
        aff_out = aff_out + x_in

        # 调制与融合
        combined = torch.cat([ema_out, aff_out], dim=1)
        modulation = self.output_modulation(combined)
        fused = ema_out * modulation + aff_out * (1 - modulation)

        # 最终与模块输入残差
        out = fused + self.out_skip(x_in)
        return out


# Test case
if __name__ == '__main__':
    # Create test tensor [batch_size, channels, height, width]
    x = torch.randn(2, 512, 64, 64)

    # Initialize model (ensure input channels are divisible by the ema factor)
    model = EMAAFFCombined(in_channels=512, ema_factor=32)

    # Forward pass
    output = model(x)

    # Verify output shape (should match input shape)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == x.shape, f"期望输出形状 {x.shape}, 得到 {output.shape}"

    # Verify model parameters
    print("模型参数数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("测试通过! ")
