# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom CSPResNet-based blocks optimized for timber defect detection."""

import torch
import torch.nn as nn

# Import YOLOv8 standard modules
from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck
from ultralytics.nn.modules.conv import Conv

__all__ = (
    "ECAAttention",
    "CSPResNetBackbone",
    "YOLONeckP2Enhanced",
)


class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA) - lightweight and effective."""
    
    def __init__(self, channels, kernel_size=3):
        """Initialize ECA attention.
        
        Args:
            channels (int): Number of input channels
            kernel_size (int): Adaptive kernel size for 1D conv
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass with efficient channel attention."""
        # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        # Squeeze: [B, C, 1, 1] -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D convolution across channels
        y = self.conv(y)
        # Transpose back: [B, 1, C] -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Apply attention
        return x * self.sigmoid(y)


class CSPBottleneck(nn.Module):
    """CSP Bottleneck with residual connection and optional attention."""
    
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        """Initialize CSP Bottleneck.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            n (int): Number of bottleneck layers
            shortcut (bool): Use residual connection
            e (float): Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))
    
    def forward(self, x):
        """Forward pass through CSP bottleneck."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class CSPResNetBackbone(nn.Module):
    """
    CSPResNet backbone optimized for timber defect detection.
    Outputs 4 feature pyramid levels: P2, P3, P4, P5
    
    Architecture:
    - Stem: Conv(3->32) + Conv(32->64)
    - P2 (stride 4):  64 channels, 160x160 (for small defects like cracks)
    - P3 (stride 8):  128 channels, 80x80
    - P4 (stride 16): 256 channels, 40x40
    - P5 (stride 32): 512 channels, 20x20 (with SPPF for context)
    """
    
    def __init__(self, pretrained=False):
        """Initialize CSPResNet backbone.
        
        Args:
            pretrained (bool): Load pretrained weights (not used, kept for compatibility)
        """
        super().__init__()
        
        # Stem - initial feature extraction
        self.stem = nn.Sequential(
            Conv(3, 32, k=3, s=2, p=1),  # 320x320
            Conv(32, 64, k=3, s=2, p=1),  # 160x160 (P2 level)
        )
        
        # P2 level (stride 4, 160x160) - High resolution for small defects
        self.stage1 = nn.Sequential(
            CSPBottleneck(64, 64, n=1, shortcut=True, e=0.5),
            ECAAttention(64, kernel_size=3)
        )
        
        # P3 level (stride 8, 80x80)
        self.down1 = Conv(64, 128, k=3, s=2, p=1)
        self.stage2 = nn.Sequential(
            CSPBottleneck(128, 128, n=1, shortcut=True, e=0.5),  # Reduced from n=2
            ECAAttention(128, kernel_size=3)
        )
        
        # P4 level (stride 16, 40x40)
        self.down2 = Conv(128, 256, k=3, s=2, p=1)
        self.stage3 = nn.Sequential(
            CSPBottleneck(256, 256, n=1, shortcut=True, e=0.5),  # Reduced from n=2
            ECAAttention(256, kernel_size=5)
        )
        
        # P5 level (stride 32, 20x20) - Deep context
        self.down3 = Conv(256, 384, k=3, s=2, p=1)  # Reduced from 512 to 384
        self.stage4 = nn.Sequential(
            CSPBottleneck(384, 384, n=1, shortcut=True, e=0.5),
            SPPF(384, 384, k=5),  # Spatial pyramid pooling for multi-scale context
            ECAAttention(384, kernel_size=5)
        )
        
        # Output channels for each pyramid level
        self.out_channels = [64, 128, 256, 384]  # [P2, P3, P4, P5] - reduced P5
    
    def forward(self, x):
        """Forward pass through backbone.
        
        Args:
            x (torch.Tensor): Input image [B, 3, 640, 640]
            
        Returns:
            list: Feature maps [P2, P3, P4, P5]
        """
        # Stem
        x = self.stem(x)  # [B, 64, 160, 160]
        
        # P2 - High resolution
        p2 = self.stage1(x)  # [B, 64, 160, 160]
        
        # P3
        x = self.down1(p2)
        p3 = self.stage2(x)  # [B, 128, 80, 80]
        
        # P4
        x = self.down2(p3)
        p4 = self.stage3(x)  # [B, 256, 40, 40]
        
        # P5
        x = self.down3(p4)
        p5 = self.stage4(x)  # [B, 512, 20, 20]
        
        return [p2, p3, p4, p5]


class YOLONeckP2Enhanced(nn.Module):
    """
    YOLOv8-style neck with P2 support for 4-scale detection.
    FPN + PAN architecture with C2f fusion modules.
    
    Input: [P2, P3, P4, P5] from backbone
    Output: [P2_out, P3_out, P4_out, P5_out] for detection heads
    """
    
    def __init__(self, in_channels=[64, 128, 256, 384]):
        """Initialize P2-enhanced neck.
        
        Args:
            in_channels (list): Input channels for [P2, P3, P4, P5]
        """
        super().__init__()
        c2, c3, c4, c5 = in_channels
        
        # Output channels for each level
        # Keep them moderate to stay under 6M params
        out_c2, out_c3, out_c4, out_c5 = 64, 96, 128, 160  # Reduced P5 from 192 to 160
        
        # Top-down pathway (FPN) - start from P5
        # P5 -> P4
        self.reduce_p5 = Conv(c5, out_c4, k=1, s=1)
        self.c2f_p4_td = C2f(c4 + out_c4, out_c4, n=1, shortcut=False)  # Reduced from n=2
        
        # P4 -> P3
        self.reduce_p4 = Conv(out_c4, out_c3, k=1, s=1)
        self.c2f_p3_td = C2f(c3 + out_c3, out_c3, n=1, shortcut=False)  # Reduced from n=2
        
        # P3 -> P2
        self.reduce_p3 = Conv(out_c3, out_c2, k=1, s=1)
        self.c2f_p2_td = C2f(c2 + out_c2, out_c2, n=1, shortcut=False)  # Reduced from n=2
        
        # Bottom-up pathway (PAN) - start from P2
        # P2 -> P3
        self.downsample_p2 = Conv(out_c2, out_c2, k=3, s=2)
        self.c2f_p3_bu = C2f(out_c2 + out_c3, out_c3, n=1, shortcut=False)  # Reduced from n=2
        
        # P3 -> P4
        self.downsample_p3 = Conv(out_c3, out_c3, k=3, s=2)
        self.c2f_p4_bu = C2f(out_c3 + out_c4, out_c4, n=1, shortcut=False)  # Reduced from n=2
        
        # P4 -> P5
        self.downsample_p4 = Conv(out_c4, out_c4, k=3, s=2)
        self.c2f_p5_bu = C2f(out_c4 + c5, out_c5, n=1, shortcut=False)  # Reduced from n=2
        
        self.out_channels = [out_c2, out_c3, out_c4, out_c5]  # [64, 96, 128, 160]
    
    def forward(self, feats):
        """Forward pass through neck.
        
        Args:
            feats (list): Multi-scale features [P2, P3, P4, P5]
            
        Returns:
            list: Enhanced features [P2_out, P3_out, P4_out, P5_out]
        """
        p2, p3, p4, p5 = feats
        
        # Top-down pathway (FPN)
        # P5 -> P4
        p5_reduce = self.reduce_p5(p5)
        p5_up = nn.functional.interpolate(p5_reduce, size=p4.shape[-2:], mode='nearest')
        p4_td = self.c2f_p4_td(torch.cat([p4, p5_up], dim=1))
        
        # P4 -> P3
        p4_reduce = self.reduce_p4(p4_td)
        p4_up = nn.functional.interpolate(p4_reduce, size=p3.shape[-2:], mode='nearest')
        p3_td = self.c2f_p3_td(torch.cat([p3, p4_up], dim=1))
        
        # P3 -> P2
        p3_reduce = self.reduce_p3(p3_td)
        p3_up = nn.functional.interpolate(p3_reduce, size=p2.shape[-2:], mode='nearest')
        p2_out = self.c2f_p2_td(torch.cat([p2, p3_up], dim=1))
        
        # Bottom-up pathway (PAN)
        # P2 -> P3
        p2_down = self.downsample_p2(p2_out)
        p3_out = self.c2f_p3_bu(torch.cat([p2_down, p3_td], dim=1))
        
        # P3 -> P4
        p3_down = self.downsample_p3(p3_out)
        p4_out = self.c2f_p4_bu(torch.cat([p3_down, p4_td], dim=1))
        
        # P4 -> P5
        p4_down = self.downsample_p4(p4_out)
        p5_out = self.c2f_p5_bu(torch.cat([p4_down, p5], dim=1))
        
        return [p2_out, p3_out, p4_out, p5_out]
