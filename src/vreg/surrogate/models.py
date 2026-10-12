"""Small residual U-Nets for the platform emulator E and the pre-compensation generator G."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
    )


class ResUNet(nn.Module):
    """y = x + scale * tanh(unet(x)); the last conv is zero-initialised so the net starts as identity.

    Inputs/outputs are RGB in [0, 1], NCHW, H and W divisible by 2**levels.
    """

    def __init__(self, width: int = 32, levels: int = 3, max_residual: float = 1.0):
        super().__init__()
        chans = [width * 2**i for i in range(levels + 1)]
        self.inp = _block(3, chans[0])
        self.down = nn.ModuleList(_block(chans[i], chans[i + 1]) for i in range(levels))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(levels))
        )
        self.dec = nn.ModuleList(_block(2 * chans[i], chans[i]) for i in reversed(range(levels)))
        self.out = nn.Conv2d(chans[0], 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.max_residual = max_residual
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inp(x)]
        h = skips[0]
        for d in self.down:
            h = d(F.avg_pool2d(h, 2))
            skips.append(h)
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], dim=1))
        return x + self.max_residual * torch.tanh(self.out(h))


def emulator(width: int = 32) -> ResUNet:
    """E: uploaded frame -> playback frame (models denoise, blur, blocking)."""
    return ResUNet(width=width, levels=3, max_residual=1.0)


def generator(width: int = 24, max_residual: float = 0.15) -> ResUNet:
    """G: source frame -> pre-compensated frame; residual bounded to keep it a 'nudge'."""
    return ResUNet(width=width, levels=2, max_residual=max_residual)


@torch.no_grad()
def apply_full_frames(net: nn.Module, frames, device: str, batch: int = 2):
    """Run a fully convolutional net on (N,H,W,3) uint8 frames; pads to a multiple of 16."""
    import numpy as np

    net.eval()
    n, h, w, _ = frames.shape
    ph, pw = (-h) % 16, (-w) % 16
    out = np.empty_like(frames)
    for i in range(0, n, batch):
        x = torch.from_numpy(frames[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        y = net(x)[:, :, :h, :w].clamp(0, 1)
        out[i:i + batch] = (y * 255 + 0.5).byte().permute(0, 2, 3, 1).cpu().numpy()
    return out
