from __future__ import annotations

import math

import torch
import torch.nn as nn


def make_divisible(x: float, divisor: int = 8) -> int:
    return math.ceil(x / divisor) * divisor


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class BCRSAttn(nn.Module):
    """Boundary-Contrast Repulsion Separation attention for dense object separation.

    Internal four branches:
    center, boundary, repulsion, and context.
    """

    def __init__(
        self,
        dim,
        num_heads=1,
        area=1,
        reduction=16,
        enable_center=True,
        enable_boundary=True,
        enable_repulsion=True,
        enable_context=True,
    ):
        super().__init__()
        if not any((enable_center, enable_boundary, enable_repulsion, enable_context)):
            raise ValueError("At least one BCRS branch must remain enabled.")

        hidden = max(dim // reduction, 8)
        self.enable_center = enable_center
        self.enable_boundary = enable_boundary
        self.enable_repulsion = enable_repulsion
        self.enable_context = enable_context
        self.norm = nn.LayerNorm(dim)
        self.value = Conv(dim, dim, 1, 1, act=False)
        self.local = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.context = Conv(dim, dim, 3, 1, g=dim, d=2, act=False)
        self.contrast_proj = Conv(dim, dim, 1, 1, act=False)
        self.boundary_proj = Conv(dim, dim, 1, 1, act=False)
        self.repulsion_proj = Conv(dim, dim, 1, 1, act=False)

        self.center_gate = nn.Sequential(
            nn.Conv2d(4, hidden, 3, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.repulsion_gate = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.branch_score = nn.Conv2d(dim * 4, 4, 1, 1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=False),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.register_buffer(
            "branch_enabled",
            torch.tensor(
                [enable_center, enable_boundary, enable_repulsion, enable_context], dtype=torch.bool
            ).view(1, 4, 1, 1),
            persistent=False,
        )

        self.sobel_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.sobel_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.laplace = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        with torch.no_grad():
            wgx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
            wgy = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
            wgl = torch.tensor([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]], dtype=torch.float32).view(1, 1, 3, 3)
            self.sobel_x.weight.copy_(wgx.expand(dim, 1, 3, 3))
            self.sobel_y.weight.copy_(wgy.expand(dim, 1, 3, 3))
            self.laplace.weight.copy_(wgl.expand(dim, 1, 3, 3))
        for m in (self.sobel_x, self.sobel_y, self.laplace):
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        branch_enabled = self.branch_enabled.to(device=x.device, dtype=torch.bool)
        enable_center, enable_boundary, enable_repulsion, enable_context = branch_enabled.view(-1).tolist()

        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        value = self.value(xn)
        local_feat = self.local(value)
        context_feat = self.context(value)
        if not enable_context:
            context_feat = torch.zeros_like(context_feat)

        contrast = value - torch.nn.functional.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        contrast_mag = contrast.abs().mean(1, keepdim=True)
        center_residual = torch.nn.functional.relu(
            value - torch.nn.functional.avg_pool2d(value, kernel_size=5, stride=1, padding=2)
        )
        center_score = self.center_gate(
            torch.cat(
                (
                    value.mean(1, keepdim=True),
                    value.amax(1, keepdim=True),
                    contrast_mag,
                    center_residual.mean(1, keepdim=True),
                ),
                1,
            )
        )
        if not enable_center:
            center_score = torch.zeros_like(center_score)

        edge_mag = self.sobel_x(value).abs() + self.sobel_y(value).abs()
        laplace_mag = self.laplace(value).abs()
        boundary_score = self.boundary_gate(
            torch.cat((edge_mag.mean(1, keepdim=True), laplace_mag.mean(1, keepdim=True), contrast_mag), 1)
        )
        if not enable_boundary:
            boundary_score = torch.zeros_like(boundary_score)

        crowd_score = torch.nn.functional.avg_pool2d(center_score, kernel_size=5, stride=1, padding=2)
        peak_score = torch.nn.functional.max_pool2d(center_score, kernel_size=3, stride=1, padding=1)
        repulsion_score = self.repulsion_gate(
            torch.cat((torch.nn.functional.relu(peak_score - center_score), crowd_score, boundary_score), 1)
        )
        if not enable_repulsion:
            repulsion_score = torch.zeros_like(repulsion_score)

        boundary_feat = self.boundary_proj(edge_mag + laplace_mag)
        contrast_feat = self.contrast_proj(contrast)
        repulsion_feat = self.repulsion_proj(contrast_feat * (1.0 + repulsion_score) + boundary_feat * repulsion_score)
        if not enable_boundary:
            boundary_feat = torch.zeros_like(boundary_feat)
        if not enable_repulsion:
            repulsion_feat = torch.zeros_like(repulsion_feat)

        center_branch = local_feat * center_score
        boundary_branch = (local_feat + context_feat + boundary_feat) * boundary_score
        repulsion_branch = repulsion_feat * repulsion_score
        context_branch = context_feat * (1.0 + 0.5 * center_score)
        if not enable_center:
            center_branch = torch.zeros_like(center_branch)
        if not enable_boundary:
            boundary_branch = torch.zeros_like(boundary_branch)
        if not enable_repulsion:
            repulsion_branch = torch.zeros_like(repulsion_branch)
        if not enable_context:
            context_branch = torch.zeros_like(context_branch)

        branch_logits = self.branch_score(torch.cat((center_branch, boundary_branch, repulsion_branch, context_branch), 1))
        branch_logits = branch_logits.masked_fill(~branch_enabled, torch.finfo(branch_logits.dtype).min)
        branch_gate = branch_logits.softmax(dim=1)
        fused = (
            center_branch * branch_gate[:, 0:1]
            + boundary_branch * branch_gate[:, 1:2]
            + repulsion_branch * branch_gate[:, 2:3]
            + context_branch * branch_gate[:, 3:4]
        )
        fused = fused * (1.0 + self.gamma.tanh() * repulsion_score) * self.channel_gate(fused)
        spatial = self.spatial_gate(torch.cat((fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)), 1))
        return self.proj(fused * spatial)


class ABlockBCRS(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=1.2,
        area=1,
        enable_center=True,
        enable_boundary=True,
        enable_repulsion=True,
        enable_context=True,
    ):
        super().__init__()
        self.attn = BCRSAttn(
            dim,
            num_heads=num_heads,
            area=area,
            enable_center=enable_center,
            enable_boundary=enable_boundary,
            enable_repulsion=enable_repulsion,
            enable_context=enable_context,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, use_attention=False, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Hidden channels must be a multiple of 32."
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones((c2)), requires_grad=True) if use_attention and residual else None
        if use_attention:
            num_heads = c_ // 32
            self.m = nn.ModuleList(nn.Sequential(*(ABlockBCRS(c_, num_heads, mlp_ratio, area) for _ in range(2))) for _ in range(n))
        else:
            self.m = nn.ModuleList(C3k(c_, c_, 2, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * out
        return out


class A2C2fBCRS(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        area=1,
        residual=False,
        enable_center=True,
        enable_boundary=True,
        enable_repulsion=True,
        enable_context=True,
        mlp_ratio=2.0,
        e=0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Hidden channels must be a multiple of 32."
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones((c2)), requires_grad=True) if residual else None
        self.m = nn.ModuleList(
            nn.Sequential(
                *(
                    ABlockBCRS(
                        c_,
                        num_heads,
                        mlp_ratio,
                        area,
                        enable_center=enable_center,
                        enable_boundary=enable_boundary,
                        enable_repulsion=enable_repulsion,
                        enable_context=enable_context,
                    )
                    for _ in range(2)
                )
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * out
        return out


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max)
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        return self._inference(x)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)
