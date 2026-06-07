from __future__ import annotations

import torch
import torch.nn as nn

from .modules import A2C2f, A2C2fBCRS, C3k2, Conv, Detect, make_divisible


class _BaseYOLOv12Scales(nn.Module):
    SCALES = {
        "n": (0.50, 0.25, 1024),
        "s": (0.50, 0.50, 1024),
        "m": (0.50, 1.00, 512),
        "l": (1.00, 1.00, 512),
        "x": (1.00, 1.50, 512),
    }

    def __init__(self, nc: int = 1, scale: str = "n"):
        super().__init__()
        if scale not in self.SCALES:
            raise ValueError(f"Unsupported scale '{scale}', expected one of {tuple(self.SCALES)}")
        self.nc = nc
        self.scale = scale
        self.depth, self.width, self.max_channels = self.SCALES[scale]

    def _c(self, c2: int) -> int:
        return make_divisible(min(c2, self.max_channels) * self.width, 8)

    def _r(self, n: int) -> int:
        return max(round(n * self.depth), 1) if n > 1 else n


class YOLOv12Baseline(_BaseYOLOv12Scales):
    """Explicit original YOLOv12 P3-P5 detector for comparison."""

    def __init__(self, nc: int = 1, scale: str = "n", ch: int = 3):
        super().__init__(nc=nc, scale=scale)
        mlp_ratio = 1.5 if scale in {"l", "x"} else 2.0
        c3k_for_mlx = scale in {"m", "l", "x"}

        c0 = self._c(64)
        c1 = self._c(128)
        c2 = self._c(256)
        c4 = self._c(512)
        c7 = self._c(1024)
        p3c = self._c(256)
        p4c = self._c(512)
        p5c = self._c(1024)

        n2 = self._r(2)
        n4 = self._r(4)

        self.l0 = Conv(ch, c0, 3, 2)
        self.l1 = Conv(c0, c1, 3, 2, 1, 2)
        self.l2 = C3k2(c1, c2, n2, False, 0.25)
        self.l3 = Conv(c2, c2, 3, 2, 1, 4)
        self.l4 = C3k2(c2, c4, n2, False, 0.25)
        self.l5 = Conv(c4, c4, 3, 2)
        self.l6 = A2C2f(c4, p4c, n4, use_attention=True, area=4, residual=True, mlp_ratio=mlp_ratio)
        self.l7 = Conv(p4c, p4c * 2, 3, 2)
        self.l8 = A2C2f(p4c * 2, p5c, n4, use_attention=True, area=1, residual=True, mlp_ratio=mlp_ratio)

        self.up9 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l11 = A2C2f(p5c + p4c, p4c, n2, use_attention=False, area=-1)

        self.up12 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l14 = A2C2f(p4c + c4, p3c, n2, use_attention=False, area=-1)

        self.l15 = Conv(p3c, p3c, 3, 2)
        self.l17 = A2C2f(p3c + p4c, p4c, n2, use_attention=False, area=-1)

        self.l18 = Conv(p4c, p4c, 3, 2)
        self.l20 = C3k2(p4c + p5c, p5c, n2, c3k_for_mlx)

        self.detect = Detect(nc=nc, ch=(p3c, p4c, p5c))
        self.detect.stride = torch.tensor((8.0, 16.0, 32.0))

    def forward(self, x: torch.Tensor):
        y0 = self.l0(x)
        y1 = self.l1(y0)
        y2 = self.l2(y1)
        y3 = self.l3(y2)
        y4 = self.l4(y3)
        y5 = self.l5(y4)
        y6 = self.l6(y5)
        y7 = self.l7(y6)
        y8 = self.l8(y7)

        y9 = self.up9(y8)
        y11 = self.l11(torch.cat((y9, y6), 1))

        y12 = self.up12(y11)
        y14 = self.l14(torch.cat((y12, y4), 1))

        y15 = self.l15(y14)
        y17 = self.l17(torch.cat((y15, y11), 1))

        y18 = self.l18(y17)
        y20 = self.l20(torch.cat((y18, y8), 1))

        if self.detect.stride.device != y14.device:
            self.detect.stride = self.detect.stride.to(y14.device)
        return self.detect([y14, y17, y20])


class BCRSYOLOv12P2(_BaseYOLOv12Scales):
    """Explicit YOLOv12 + BCRS attention + P2 detector."""

    def __init__(self, nc: int = 1, scale: str = "n", ch: int = 3):
        super().__init__()
        _BaseYOLOv12Scales.__init__(self, nc=nc, scale=scale)
        mlp_ratio = 1.5 if scale in {"l", "x"} else 2.0
        c3k_for_mlx = scale in {"m", "l", "x"}

        c0 = self._c(64)
        c1 = self._c(128)
        c2 = self._c(256)
        c4 = self._c(512)
        c7 = self._c(1024)
        p2c = self._c(128)
        p3c = self._c(256)
        p4c = self._c(512)
        p5c = self._c(1024)

        n2 = self._r(2)
        n4 = self._r(4)

        self.l0 = Conv(ch, c0, 3, 2)
        self.l1 = Conv(c0, c1, 3, 2, 1, 2)
        self.l2 = C3k2(c1, c2, n2, False, 0.25)
        self.l3 = Conv(c2, c2, 3, 2, 1, 4)
        self.l4 = C3k2(c2, c4, n2, False, 0.25)
        self.l5 = Conv(c4, c4, 3, 2)
        self.l6 = A2C2fBCRS(c4, p4c, n4, area=4, residual=True, mlp_ratio=mlp_ratio)
        self.l7 = Conv(p4c, p4c * 2, 3, 2)
        self.l8 = A2C2fBCRS(p4c * 2, p5c, n4, area=1, residual=True, mlp_ratio=mlp_ratio)

        self.up9 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l11 = A2C2f(p5c + p4c, p4c, n2, use_attention=False, area=-1)

        self.up12 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l14 = A2C2f(p4c + c4, p3c, n2, use_attention=False, area=-1)

        self.up15 = nn.Upsample(scale_factor=2, mode="nearest")
        self.l17 = C3k2(p3c + c2, p2c, n2, False)

        self.l18 = Conv(p2c, p2c, 3, 2)
        self.l20 = C3k2(p2c + p3c, p3c, n2, False)

        self.l21 = Conv(p3c, p3c, 3, 2)
        self.l23 = A2C2f(p3c + p4c, p4c, n2, use_attention=False, area=-1)

        self.l24 = Conv(p4c, p4c, 3, 2)
        self.l26 = C3k2(p4c + p5c, p5c, n2, c3k_for_mlx)

        self.detect = Detect(nc=nc, ch=(p2c, p3c, p4c, p5c))
        self.detect.stride = torch.tensor((4.0, 8.0, 16.0, 32.0))

    def forward(self, x: torch.Tensor):
        y0 = self.l0(x)
        y1 = self.l1(y0)
        y2 = self.l2(y1)
        y3 = self.l3(y2)
        y4 = self.l4(y3)
        y5 = self.l5(y4)
        y6 = self.l6(y5)
        y7 = self.l7(y6)
        y8 = self.l8(y7)

        y9 = self.up9(y8)
        y11 = self.l11(torch.cat((y9, y6), 1))

        y12 = self.up12(y11)
        y14 = self.l14(torch.cat((y12, y4), 1))

        y15 = self.up15(y14)
        y17 = self.l17(torch.cat((y15, y2), 1))

        y18 = self.l18(y17)
        y20 = self.l20(torch.cat((y18, y14), 1))

        y21 = self.l21(y20)
        y23 = self.l23(torch.cat((y21, y11), 1))

        y24 = self.l24(y23)
        y26 = self.l26(torch.cat((y24, y8), 1))

        if self.detect.stride.device != y17.device:
            self.detect.stride = self.detect.stride.to(y17.device)
        return self.detect([y17, y20, y23, y26])


def build_model(nc: int = 1, scale: str = "n") -> BCRSYOLOv12P2:
    return BCRSYOLOv12P2(nc=nc, scale=scale)


def build_base_model(nc: int = 1, scale: str = "n") -> YOLOv12Baseline:
    return YOLOv12Baseline(nc=nc, scale=scale)
