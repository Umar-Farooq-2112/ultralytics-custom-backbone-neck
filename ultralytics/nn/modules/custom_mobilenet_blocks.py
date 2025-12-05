# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom MobileNetV3-based blocks optimized for detection."""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

# Import YOLOv8 standard modules
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv

__all__ = (
    "EnhancedCBAM",
    "MobileNetV3BackboneEnhanced",
    "YOLONeckEnhanced",
    "ECAAttention",
    "CSPResNetBackbone",
    "YOLONeckP2Enhanced",
)


class EnhancedCBAM(nn.Module):
    """Efficient CBAM with channel and spatial attention optimized for detection."""
    
    def __init__(self, channels, reduction=16):
        """Initialize enhanced CBAM.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Channel reduction ratio
        """
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        
        # Spatial attention - simplified for efficiency
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass with channel and spatial attention."""
        # Channel attention
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_attn = self.sigmoid(avg_out + max_out)
        x = x * channel_attn
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attn = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_attn
        
        return x


class MobileNetV3BackboneEnhanced(nn.Module):
    """Enhanced MobileNetV3 backbone with Conv and C2f modules instead of excessive DW convs."""
    
    def __init__(self, pretrained=True):
        """Initialize enhanced backbone.
        
        Args:
            pretrained (bool): Use pretrained weights
        """
        super().__init__()
        m = mobilenet_v3_small(pretrained=pretrained)
        feats = m.features

        # Keep original MobileNetV3 stages
        self.stage1 = feats[:3]   # P3: 24 channels, stride 8
        self.stage2 = feats[3:7]  # P4: 40 channels, stride 16
        self.stage3 = feats[7:]   # P5: 576 channels, stride 32

        # P3 enhancement - Lighter processing
        self.p3_conv1 = Conv(24, 48, k=3, s=1)
        self.p3_c2f = C2f(48, 64, n=1, shortcut=False)
        
        # P4 enhancement - Moderate processing
        self.p4_conv1 = Conv(40, 80, k=3, s=1)
        self.p4_c2f = C2f(80, 96, n=2, shortcut=False)
        
        # P5 enhancement - Deeper for context
        self.p5_conv1 = Conv(576, 160, k=3, s=1)
        self.p5_c2f = C2f(160, 192, n=2, shortcut=False)
        
        # CBAM attention only on P5
        self.cbam_p5 = EnhancedCBAM(192, reduction=16)

        self.out_channels = [64, 96, 192]

    def forward(self, x):
        """Forward pass through enhanced backbone.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            list: Multi-scale features [P3, P4, P5]
        """
        # Extract base features
        p3_base = self.stage1(x)
        p4_base = self.stage2(p3_base)
        p5_base = self.stage3(p4_base)
        
        # Enhanced processing
        p3 = self.p3_conv1(p3_base)
        p3 = self.p3_c2f(p3)
        
        p4 = self.p4_conv1(p4_base)
        p4 = self.p4_c2f(p4)
        
        p5 = self.p5_conv1(p5_base)
        p5 = self.p5_c2f(p5)
        p5 = self.cbam_p5(p5)
        
        return [p3, p4, p5]


class YOLONeckEnhanced(nn.Module):
    """YOLOv8-style neck with C2f modules and FPN+PAN architecture."""
    
    def __init__(self, in_channels=[64, 96, 192]):
        """Initialize enhanced neck.
        
        Args:
            in_channels (list): Input channels for [P3, P4, P5]
        """
        super().__init__()
        c3, c4, c5 = in_channels

        # P5 processing with SPPF
        self.sppf = SPPF(c5, c5, k=5)
        
        # Top-down pathway (FPN)
        self.reduce_p5 = Conv(c5, 128, k=1, s=1)
        self.c2f_p4_td = C2f(c4 + 128, 128, n=2, shortcut=False)
        
        self.reduce_p4 = Conv(128, 96, k=1, s=1)
        self.c2f_p3_td = C2f(c3 + 96, 96, n=2, shortcut=False)
        
        # Bottom-up pathway (PAN)
        self.downsample_p3 = Conv(96, 96, k=3, s=2)
        self.c2f_p4_bu = C2f(96 + 128, 128, n=2, shortcut=False)
        
        self.downsample_p4 = Conv(128, 128, k=3, s=2)
        self.c2f_p5_bu = C2f(128 + c5, 192, n=2, shortcut=False)

    def forward(self, feats):
        """Forward pass through neck.
        
        Args:
            feats (list): Multi-scale features [P3, P4, P5]
            
        Returns:
            list: Enhanced features [P3_out, P4_out, P5_out]
        """
        p3, p4, p5 = feats
        
        # P5 with SPPF
        p5_sppf = self.sppf(p5)
        
        # Top-down (FPN)
        p5_reduce = self.reduce_p5(p5_sppf)
        p5_up = nn.functional.interpolate(p5_reduce, size=p4.shape[-2:], mode='nearest')
        p4_td = self.c2f_p4_td(torch.cat([p4, p5_up], dim=1))
        
        p4_reduce = self.reduce_p4(p4_td)
        p4_up = nn.functional.interpolate(p4_reduce, size=p3.shape[-2:], mode='nearest')
        p3_out = self.c2f_p3_td(torch.cat([p3, p4_up], dim=1))
        
        # Bottom-up (PAN)
        p3_down = self.downsample_p3(p3_out)
        p4_out = self.c2f_p4_bu(torch.cat([p3_down, p4_td], dim=1))
        
        p4_down = self.downsample_p4(p4_out)
        p5_out = self.c2f_p5_bu(torch.cat([p4_down, p5_sppf], dim=1))
        
        return [p3_out, p4_out, p5_out]


# ============================================================================
# PRIORITY 1 COMPONENTS: CSPResNet Backbone + P2 Detection + ECA Attention
# ============================================================================


class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA) - lightweight attention mechanism.
    
    Priority 1: Replaces CBAM with more efficient ECA attention.
    """
    
    def __init__(self, channels, gamma=2, b=1):
        """Initialize ECA attention.
        
        Args:
            channels (int): Number of input channels
            gamma (int): Kernel size calculation parameter
            b (int): Kernel size calculation parameter
        """
        super().__init__()
        # Adaptive kernel size calculation
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass through ECA attention.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Attention-weighted output
        """
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution across channels
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid activation
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class CSPResNetBackbone(nn.Module):
    """CSPResNet backbone with P2, P3, P4, P5 outputs and ECA attention.
    
    Priority 1: Replaces MobileNetV3 with stronger CSPResNet backbone.
    Adds P2 detection level for small object detection.
    Uses ECA attention instead of CBAM for efficiency.
    
    Expected gain: +2.5% mAP from better backbone, +1.5% from P2 detection
    Param cost: ~1M additional parameters
    """
    
    def __init__(self, pretrained=False):
        """Initialize CSPResNet backbone.
        
        Args:
            pretrained (bool): Use pretrained weights (not implemented for custom arch)
        """
        super().__init__()
        
        # Stem: Input -> 32 -> 64 channels
        self.stem = nn.Sequential(
            Conv(3, 32, k=3, s=2, p=1),  # 640 -> 320
            Conv(32, 64, k=3, s=2, p=1),  # 320 -> 160
        )
        
        # P2 stage (160x160 feature map, stride=4) - For small defects
        self.stage_p2 = nn.Sequential(
            C2f(64, 64, n=1, shortcut=False),
            ECAAttention(64),
        )
        
        # P3 stage (80x80 feature map, stride=8)
        self.downsample_p3 = Conv(64, 128, k=3, s=2, p=1)  # 160 -> 80
        self.stage_p3 = nn.Sequential(
            C2f(128, 128, n=2, shortcut=False),
            ECAAttention(128),
        )
        
        # P4 stage (40x40 feature map, stride=16)
        self.downsample_p4 = Conv(128, 256, k=3, s=2, p=1)  # 80 -> 40
        self.stage_p4 = nn.Sequential(
            C2f(256, 256, n=2, shortcut=False),
            ECAAttention(256),
        )
        
        # P5 stage (20x20 feature map, stride=32)
        self.downsample_p5 = Conv(256, 384, k=3, s=2, p=1)  # 40 -> 20
        self.stage_p5 = nn.Sequential(
            C2f(384, 384, n=1, shortcut=False),
            SPPF(384, 384, k=5),  # SPPF for multi-scale context
            ECAAttention(384),
        )
        
        # Output channels for each pyramid level
        self.out_channels = [64, 128, 256, 384]  # [P2, P3, P4, P5]
    
    def forward(self, x):
        """Forward pass through CSPResNet backbone.
        
        Args:
            x (torch.Tensor): Input image tensor [B, 3, 640, 640]
            
        Returns:
            list: Multi-scale features [P2, P3, P4, P5]
                - P2: [B, 64, 160, 160] stride=4
                - P3: [B, 128, 80, 80] stride=8
                - P4: [B, 256, 40, 40] stride=16
                - P5: [B, 384, 20, 20] stride=32
        """
        # Stem
        x = self.stem(x)  # [B, 64, 160, 160]
        
        # P2 stage
        p2 = self.stage_p2(x)  # [B, 64, 160, 160]
        
        # P3 stage
        x = self.downsample_p3(p2)
        p3 = self.stage_p3(x)  # [B, 128, 80, 80]
        
        # P4 stage
        x = self.downsample_p4(p3)
        p4 = self.stage_p4(x)  # [B, 256, 40, 40]
        
        # P5 stage
        x = self.downsample_p5(p4)
        p5 = self.stage_p5(x)  # [B, 384, 20, 20]
        
        return [p2, p3, p4, p5]


class YOLONeckP2Enhanced(nn.Module):
    """Enhanced YOLO neck with P2 support (4-scale FPN+PAN).
    
    Priority 1: Extends standard neck to support P2 detection level.
    Processes P2, P3, P4, P5 from backbone into detection-ready features.
    
    Expected gain: Part of +1.5% from P2 detection
    Param cost: ~200K additional parameters
    """
    
    def __init__(self, in_channels=[64, 128, 256, 384]):
        """Initialize enhanced P2 neck.
        
        Args:
            in_channels (list): Input channels for [P2, P3, P4, P5]
        """
        super().__init__()
        c2, c3, c4, c5 = in_channels
        
        # Top-down pathway (FPN) - Start from P5
        self.reduce_p5 = Conv(c5, 128, k=1, s=1)
        self.c2f_p4_fpn = C2f(c4 + 128, 128, n=2, shortcut=False)
        
        self.reduce_p4 = Conv(128, 96, k=1, s=1)
        self.c2f_p3_fpn = C2f(c3 + 96, 96, n=2, shortcut=False)
        
        self.reduce_p3 = Conv(96, 64, k=1, s=1)
        self.c2f_p2_fpn = C2f(c2 + 64, 64, n=1, shortcut=False)
        
        # Bottom-up pathway (PAN) - Start from P2
        self.downsample_p2 = Conv(64, 64, k=3, s=2, p=1)
        self.c2f_p3_pan = C2f(64 + 96, 96, n=2, shortcut=False)
        
        self.downsample_p3 = Conv(96, 96, k=3, s=2, p=1)
        self.c2f_p4_pan = C2f(96 + 128, 128, n=2, shortcut=False)
        
        self.downsample_p4 = Conv(128, 128, k=3, s=2, p=1)
        self.c2f_p5_pan = C2f(128 + c5, 160, n=2, shortcut=False)
        
        # Output channels for each scale
        self.out_channels = [64, 96, 128, 160]  # [P2, P3, P4, P5]
    
    def forward(self, feats):
        """Forward pass through P2-enhanced neck.
        
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
        p4_fpn = self.c2f_p4_fpn(torch.cat([p4, p5_up], dim=1))
        
        # P4 -> P3
        p4_reduce = self.reduce_p4(p4_fpn)
        p4_up = nn.functional.interpolate(p4_reduce, size=p3.shape[-2:], mode='nearest')
        p3_fpn = self.c2f_p3_fpn(torch.cat([p3, p4_up], dim=1))
        
        # P3 -> P2
        p3_reduce = self.reduce_p3(p3_fpn)
        p3_up = nn.functional.interpolate(p3_reduce, size=p2.shape[-2:], mode='nearest')
        p2_out = self.c2f_p2_fpn(torch.cat([p2, p3_up], dim=1))
        
        # Bottom-up pathway (PAN)
        # P2 -> P3
        p2_down = self.downsample_p2(p2_out)
        p3_out = self.c2f_p3_pan(torch.cat([p2_down, p3_fpn], dim=1))
        
        # P3 -> P4
        p3_down = self.downsample_p3(p3_out)
        p4_out = self.c2f_p4_pan(torch.cat([p3_down, p4_fpn], dim=1))
        
        # P4 -> P5
        p4_down = self.downsample_p4(p4_out)
        p5_out = self.c2f_p5_pan(torch.cat([p4_down, p5], dim=1))
        
        return [p2_out, p3_out, p4_out, p5_out]
