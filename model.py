import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class DualAttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se = SqueezeExcitation(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.se(x)
        x = self.sa(x)
        return x

class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, stride=1):
        super().__init__()
        use_groups = groups if in_ch >= groups and out_ch >= groups else 1
        mid_ch = out_ch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride, 1, groups=use_groups, bias=False),
            nn.BatchNorm2d(mid_ch), nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 5, stride, 2, groups=use_groups, bias=False),
            nn.BatchNorm2d(mid_ch), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU()
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        out = self.fuse(torch.cat([b1, b2], dim=1))
        return self.act(out + self.skip(x))

class SSMTokenMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru_h = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.gru_w = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_norm = self.norm(x_flat)

        x_h = rearrange(x_norm, 'b (h w) c -> (b h) w c', h=H, w=W)
        out_h, _ = self.gru_h(x_h)
        out_h = rearrange(out_h, '(b h) w c -> b (h w) c', b=B, h=H)

        x_v = rearrange(x_norm, 'b (h w) c -> (b w) h c', h=H, w=W)
        out_v, _ = self.gru_w(x_v)
        out_v = rearrange(out_v, '(b w) h c -> b (h w) c', b=B, w=W)

        out = torch.cat([out_h, out_v], dim=-1)
        out = self.drop(self.proj(out))
        out = out + x_flat
        return rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(1),
                          nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.GELU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.GELU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6), nn.BatchNorm2d(out_ch), nn.GELU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12), nn.BatchNorm2d(out_ch), nn.GELU()),
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18), nn.BatchNorm2d(out_ch), nn.GELU()),
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch*5, out_ch, 1), nn.BatchNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x):
        H, W = x.shape[2:]
        outs = []
        for i, branch in enumerate(self.branches):
            out = branch(x)
            if i == 0:
                out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
            outs.append(out)
        return self.fuse(torch.cat(outs, dim=1))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.gate = DualAttentionGate(skip_ch)
        self.conv = nn.Sequential(
            ResNeXtBlock(in_ch + skip_ch, out_ch),
            ResNeXtBlock(out_ch, out_ch)
        )
        self.deep_sup = nn.Conv2d(out_ch, 2, 1)

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.gate(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        ds = self.deep_sup(x)
        return x, ds

class HybridFormerUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.enc1 = nn.Sequential(ResNeXtBlock(in_ch, 64),  ResNeXtBlock(64, 64))
        self.enc2 = nn.Sequential(ResNeXtBlock(64, 128, stride=2), ResNeXtBlock(128, 128))
        self.enc3 = nn.Sequential(ResNeXtBlock(128, 256, stride=2), ResNeXtBlock(256, 256))
        self.enc4 = nn.Sequential(ResNeXtBlock(256, 512, stride=2), ResNeXtBlock(512, 512))

        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(512, 256)
        self.ssm = SSMTokenMixer(256)
        self.bottleneck_conv = ResNeXtBlock(256, 512)

        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.final = nn.Conv2d(32, num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.pool(s4)
        b = self.aspp(b)
        b = self.ssm(b)
        b = self.bottleneck_conv(b)

        d4, ds4 = self.dec4(b, s4)
        d3, ds3 = self.dec3(d4, s3)
        d2, ds2 = self.dec2(d3, s2)
        d1, ds1 = self.dec1(d2, s1)

        out = self.final(d1)
        return out, [ds4, ds3, ds2, ds1]