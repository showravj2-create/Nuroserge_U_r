import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),nn.GELU(),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )
    def forward(self, x): return self.block(x)

class AttentionGate(nn.Module):
    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        self.x_proj = nn.Conv2d(x_ch, inter_ch,1, bias= False)
        self.g_proj = nn.Conv2d(g_ch, inter_ch, 1, bias= False)
        self.psi = nn.Sequential(nn.GELU(), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())
    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
             g = nn.functional.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = self.psi(self.x_proj(x) + self.g_proj(g))
        return x * a
class UNet(nn.Module):

    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        base_channels=32,
        dropout=0.1,
        attention=True
    ):
        super().__init__()

        b = base_channels

        self.attention = attention

        self.enc1 = ConvBlock(
            in_channels,
            b,
            dropout
        )

        self.enc2 = ConvBlock(
            b,
            b * 2,
            dropout
        )

        self.enc3 = ConvBlock(
            b * 2,
            b * 4,
            dropout
        )

        self.enc4 = ConvBlock(
            b * 4,
            b * 8,
            dropout
        )

        self.pool = nn.MaxPool2d(2)

        self.bridge = ConvBlock(
            b * 8,
            b * 16,
            dropout
        )

        self.up4 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            2,
            2
        )

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            2,
            2
        )

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            2,
            2
        )

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            2,
            2
        )

        self.ag4 = (
            AttentionGate(
                b * 8,
                b * 8,
                b * 4
            )
            if attention
            else nn.Identity()
        )

        self.ag3 = (
            AttentionGate(
                b * 4,
                b * 4,
                b * 2
            )
            if attention
            else nn.Identity()
        )

        self.ag2 = (
            AttentionGate(
                b * 2,
                b * 2,
                b
            )
            if attention
            else nn.Identity()
        )

        self.dec4 = ConvBlock(
            b * 16,
            b * 8,
            dropout
        )

        self.dec3 = ConvBlock(
            b * 8,
            b * 4,
            dropout
        )

        self.dec2 = ConvBlock(
            b * 4,
            b * 2,
            dropout
        )

        self.dec1 = ConvBlock(
            b * 2,
            b,
            dropout
        )

        self.head = nn.Conv2d(
            b,
            out_channels,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        z = self.bridge(
            self.pool(e4)
        )

        d4 = self.up4(z)

        s4 = (
            self.ag4(e4, d4)
            if self.attention
            else e4
        )

        d4 = self.dec4(
            torch.cat([d4, s4], 1)
        )

        d3 = self.up3(d4)

        s3 = (
            self.ag3(e3, d3)
            if self.attention
            else e3
        )

        d3 = self.dec3(
            torch.cat([d3, s3], 1)
        )

        d2 = self.up2(d3)

        s2 = (
            self.ag2(e2, d2)
            if self.attention
            else e2
        )

        d2 = self.dec2(
            torch.cat([d2, s2], 1)
        )

        d1 = self.up1(d2)

        d1 = self.dec1(
            torch.cat([d1, e1], 1)
        )

        return self.head(d1)

    