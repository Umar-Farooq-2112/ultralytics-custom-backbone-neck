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
