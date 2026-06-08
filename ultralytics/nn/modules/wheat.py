# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Wheat-specific modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = (
    "WheatEnhance",
    "WheatCountRefine",
    "SCAMCountAttn",
    "ASFConcat2",
    "SMSSAAttn",
    "VLCCountAttn",
    "LSKCountAttn",
    "DBRACountAttn",
    "SARACountAttn",
    "BCRSAttn",
    "SpikeSepAttn",
    "DCLCAttn",
    "CoordCountAttn",
    "EMACountAttn",
    "SparseCountAttn",
)


def zero_shift2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Zero-padded spatial shift."""
    b, c, h, w = x.shape

    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    start_y = max(-dy, 0)
    start_x = max(-dx, 0)
    return x_pad[:, :, start_y : start_y + h, start_x : start_x + w]


class BlockTopKMask(nn.Module):
    """Generate a hard top-k block mask from a score map."""

    def __init__(self, block_size: int = 8, keep_ratio: float = 0.25):
        super().__init__()
        self.block_size = block_size
        self.keep_ratio = keep_ratio

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0 or self.block_size <= 1:
            return torch.ones_like(score)

        pooled = F.avg_pool2d(score, kernel_size=self.block_size, stride=self.block_size, ceil_mode=True)
        b, _, hb, wb = pooled.shape
        n = hb * wb
        k = max(1, int(round(n * self.keep_ratio)))

        flat = pooled.view(b, n)
        topk_idx = torch.topk(flat, k=k, dim=1, largest=True, sorted=False).indices
        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(1, topk_idx, 1.0)
        mask = mask_flat.view(b, 1, hb, wb)
        return F.interpolate(mask, size=score.shape[-2:], mode="nearest")


class DirectionalLineAggregator(nn.Module):
    """Sparse line-style directional aggregation across four canonical orientations."""

    def __init__(self, channels: int, radius: int = 5, tau: float = 3.0):
        super().__init__()
        self.channels = channels
        self.radius = radius
        self.tau = tau
        self.directions = [(0, 1), (1, 1), (1, 0), (1, -1)]

        decays = [math.exp(-float(s) / float(tau)) for s in range(1, radius + 1)]
        self.register_buffer("decays", torch.tensor(decays, dtype=torch.float32), persistent=False)
        self.scale = nn.Parameter(torch.ones(4, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(4, channels, 1, 1))

    def _aggregate_one(self, x: torch.Tensor, dy: int, dx: int, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = x
        for s in range(1, self.radius + 1):
            w = self.decays[s - 1]
            out = out + w * (zero_shift2d(x, s * dy, s * dx) + zero_shift2d(x, -s * dy, -s * dx))
        return out * scale + bias

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for i, (dy, dx) in enumerate(self.directions):
            outs.append(self._aggregate_one(x, dy, dx, self.scale[i : i + 1], self.bias[i : i + 1]))
        return outs


class ConvBNAct(nn.Module):
    """Simple conv-bn-activation block for custom attention internals."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SCAMCountAttn(nn.Module):
    """SCAM-style spatial-context aware attention adapted for dense small-object counting."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.norm = nn.GroupNorm(1, dim)
        self.local = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.ctx_reduce = nn.Conv2d(dim, hidden, 1, 1, 0, bias=False)
        self.ctx_bn = nn.BatchNorm2d(hidden)
        self.ctx_act = nn.SiLU(inplace=True)
        self.ctx_h = nn.Conv2d(hidden, dim, 1, 1, 0, bias=True)
        self.ctx_w = nn.Conv2d(hidden, dim, 1, 1, 0, bias=True)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, 1, 1, 0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=False),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        local = self.local(xn)

        h_ctx = xn.mean(dim=3, keepdim=True)
        w_ctx = xn.mean(dim=2, keepdim=True).transpose(2, 3)
        hw_ctx = torch.cat([h_ctx, w_ctx], dim=2)
        hw_ctx = self.ctx_act(self.ctx_bn(self.ctx_reduce(hw_ctx)))
        h_ctx, w_ctx = torch.split(hw_ctx, [x.shape[2], x.shape[3]], dim=2)
        w_ctx = w_ctx.transpose(2, 3)
        spatial_context = torch.sigmoid(self.ctx_h(h_ctx) + self.ctx_w(w_ctx))

        gap = F.adaptive_avg_pool2d(xn, 1)
        gmp = F.adaptive_max_pool2d(xn, 1)
        channel = self.channel_gate(torch.cat([gap, gmp], dim=1))

        spatial_stats = torch.cat([local.mean(dim=1, keepdim=True), local.amax(dim=1, keepdim=True)], dim=1)
        spatial = self.spatial_gate(spatial_stats)
        return self.proj(local * spatial_context * channel * spatial)


class ASFConcat2(nn.Module):
    """AFPN-inspired adaptive spatial fusion that outputs weighted concatenation."""

    def __init__(self, in_channels: list[int], hidden: int = 8):
        super().__init__()
        self.weight_proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            for c in in_channels
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(hidden * len(in_channels), hidden * len(in_channels), 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden * len(in_channels)),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden * len(in_channels), len(in_channels), 1, 1, 0, bias=True),
        )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        target_size = xs[-1].shape[-2:]
        aligned = []
        for x in xs:
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")
            aligned.append(x)

        weight_feats = [proj(x) for proj, x in zip(self.weight_proj, aligned)]
        weights = F.softmax(self.weight_head(torch.cat(weight_feats, dim=1)), dim=1)
        weighted = [x * weights[:, i : i + 1] for i, x in enumerate(aligned)]
        return torch.cat(weighted, dim=1)


class SMSSAAttn(nn.Module):
    """Shareable multi-semantic spatial attention adapted for dense wheat detection/counting."""

    def __init__(self, dim, num_heads=1, area=1, group_kernel_sizes=(3, 5, 7, 9), gate_layer="sigmoid"):
        super().__init__()
        assert dim % 4 == 0, "Input dimension of SMSSAAttn must be divisible by 4."
        self.dim = dim
        self.group_chans = dim // 4

        self.norm = nn.GroupNorm(4, dim)
        self.value = Conv(dim, dim, 1, 1, act=False)
        self.local_residual = Conv(dim, dim, 3, 1, g=dim, act=False)

        self.local_dwc = nn.Conv1d(
            self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], padding=group_kernel_sizes[0] // 2, groups=self.group_chans, bias=False
        )
        self.global_dwc_s = nn.Conv1d(
            self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1], padding=group_kernel_sizes[1] // 2, groups=self.group_chans, bias=False
        )
        self.global_dwc_m = nn.Conv1d(
            self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], padding=group_kernel_sizes[2] // 2, groups=self.group_chans, bias=False
        )
        self.global_dwc_l = nn.Conv1d(
            self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], padding=group_kernel_sizes[3] // 2, groups=self.group_chans, bias=False
        )

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 16, 8), 1, 1, 0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(dim // 16, 8), dim, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)

    def _split_semantics(self, x: torch.Tensor):
        return torch.split(x, self.group_chans, dim=1)

    def _semantic_mix(self, parts):
        l_x, g_x_s, g_x_m, g_x_l = parts
        return torch.cat(
            (
                self.local_dwc(l_x),
                self.global_dwc_s(g_x_s),
                self.global_dwc_m(g_x_m),
                self.global_dwc_l(g_x_l),
            ),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xn = self.norm(x)

        x_h = xn.mean(dim=3)
        x_w = xn.mean(dim=2)

        x_h_attn = self.sa_gate(self.norm_h(self._semantic_mix(self._split_semantics(x_h)))).view(b, c, h, 1)
        x_w_attn = self.sa_gate(self.norm_w(self._semantic_mix(self._split_semantics(x_w)))).view(b, c, 1, w)

        value = self.value(xn)
        fused = value * x_h_attn * x_w_attn
        fused = fused + self.local_residual(xn)
        fused = fused * self.channel_gate(fused)
        return self.proj(fused)


class WheatEnhance(nn.Module):
    """Dense small-object enhancement block for shallow wheat features."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        hidden = max(int(c2 * e), 16)
        self.reduce = Conv(c1, hidden, 1, 1)
        self.local_3 = Conv(hidden, hidden, 3, 1, g=hidden)
        self.local_5 = Conv(hidden, hidden, 5, 1, g=hidden)
        self.project = Conv(hidden * 2, c2, 1, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, 1), nn.Sigmoid())
        self.shortcut = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        y = self.reduce(x)
        y = torch.cat((self.local_3(y), self.local_5(y)), 1)
        y = self.project(y)
        return self.shortcut(x) + y * self.gate(y)


class WheatCountRefine(nn.Module):
    """Dense-count refinement block that emphasizes local separation cues."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        hidden = max(int(c2 * e), 16)
        self.reduce = Conv(c1, hidden, 1, 1)
        self.local_3 = Conv(hidden, hidden, 3, 1, g=hidden)
        self.context_d2 = Conv(hidden, hidden, 3, 1, g=hidden, d=2)
        self.sep = Conv(hidden, hidden, 3, 1, g=hidden, act=False)
        self.project = Conv(hidden * 3, c2, 1, 1)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, 1), nn.Sigmoid())
        self.spatial_gate = nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3, bias=False), nn.Sigmoid())
        self.shortcut = Conv(c1, c2, 1, 1, act=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        y = self.reduce(x)
        local_feat = self.local_3(y)
        context_feat = self.context_d2(y)

        # Highlight local contrast to separate adjacent crowded targets.
        contrast = y - torch.nn.functional.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        sep_feat = self.sep(contrast)

        fused = self.project(torch.cat((local_feat, context_feat, sep_feat), 1))
        channel_weight = self.channel_gate(fused)
        spatial_input = torch.cat(
            (fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)),
            1,
        )
        spatial_weight = self.spatial_gate(spatial_input)
        return self.shortcut(x) + fused * channel_weight * spatial_weight


class VLCCountAttn(nn.Module):
    """VLCA-inspired lightweight attention for dense agricultural small-object detection."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16, eps=1e-6):
        super().__init__()
        self.area = area
        self.eps = eps
        hidden = max(dim // reduction, 8)
        self.norm = nn.LayerNorm(dim)
        self.local = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.context = Conv(dim, dim, 3, 1, g=dim, d=2, act=False)
        self.contour_proj = Conv(dim, dim, 1, 1, act=False)
        self.branch_score = nn.Conv2d(dim * 3, 3, 1, 1, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        self.sobel_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.sobel_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        with torch.no_grad():
            wgx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
            wgy = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).view(1, 1, 3, 3)
            self.sobel_x.weight.copy_(wgx.expand(dim, 1, 3, 3))
            self.sobel_y.weight.copy_(wgy.expand(dim, 1, 3, 3))
        for p in self.sobel_x.parameters():
            p.requires_grad = False
        for p in self.sobel_y.parameters():
            p.requires_grad = False

    def forward(self, x):
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        local_feat = self.local(xn)
        context_feat = self.context(xn)
        contour_feat = self.contour_proj(self.sobel_x(xn).abs() + self.sobel_y(xn).abs())

        score = self.branch_score(torch.cat((local_feat, context_feat, contour_feat), 1))
        gate = score.softmax(dim=1)

        hard = -(gate.clamp_min(self.eps) * gate.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
        hard = hard / math.log(3.0)

        fused = (
            local_feat * gate[:, 0:1]
            + context_feat * gate[:, 1:2]
            + contour_feat * gate[:, 2:3]
        )
        fused = fused * (1.0 + self.gamma * hard) * self.channel_gate(fused)
        return self.proj(fused)


class LSKCountAttn(nn.Module):
    """Large selective kernel attention adapted for dense wheat spike counting."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 8)
        branch_dim = max(dim // 2, 16)
        self.norm = nn.LayerNorm(dim)
        self.dw5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=False)
        self.dw7_d3 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, dilation=3, groups=dim, bias=False)
        self.reduce1 = nn.Conv2d(dim, branch_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.reduce2 = nn.Conv2d(dim, branch_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch_gate = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, stride=1, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.value = Conv(dim, dim, 1, 1, act=False)
        self.spatial_proj = nn.Conv2d(branch_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)

    def forward(self, x):
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        local_feat = self.dw5(xn)
        context_feat = self.dw7_d3(local_feat)
        attn1 = self.reduce1(local_feat)
        attn2 = self.reduce2(context_feat)

        attn_cat = torch.cat((attn1, attn2), 1)
        squeeze = torch.cat((attn_cat.mean(1, keepdim=True), attn_cat.amax(1, keepdim=True)), 1)
        weights = self.branch_gate(squeeze)
        spatial = self.spatial_proj(attn1 * weights[:, 0:1] + attn2 * weights[:, 1:2]).sigmoid()

        value = self.value(xn)
        fused = value * spatial
        fused = fused * self.channel_gate(fused)
        return self.proj(fused)


class DBRACountAttn(nn.Module):
    """Deformable bi-level routing attention adapted for dense agricultural counting."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16, agent_grid=4, topk=2, sample_points=4):
        super().__init__()
        self.dim = dim
        self.area = max(int(area), 1)
        self.agent_grid = int(agent_grid)
        self.topk = int(topk)
        self.sample_points = int(sample_points)
        hidden = max(dim // reduction, 8)
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.offset = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, self.topk * self.sample_points * 2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.local = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)
        self.scale = dim**-0.5

    def forward(self, x):
        b, c, h, w = x.shape
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        q_map = self.q_proj(xn)
        k_map = self.k_proj(xn)
        v_map = self.v_proj(xn)

        agent_q_map = F.adaptive_avg_pool2d(q_map, (self.agent_grid, self.agent_grid))
        agent_count = self.agent_grid * self.agent_grid
        agents = agent_q_map.flatten(2).transpose(1, 2)  # B, A, C

        region_k = F.adaptive_avg_pool2d(k_map, (self.area, self.area)).flatten(2).transpose(1, 2)  # B, R, C
        region_v = F.adaptive_avg_pool2d(v_map, (self.area, self.area)).flatten(2).transpose(1, 2)  # B, R, C
        region_count = region_k.shape[1]
        topk = min(self.topk, region_count)

        routing_logits = torch.matmul(agents, region_k.transpose(1, 2)) * self.scale
        route_idx = routing_logits.topk(topk, dim=-1).indices  # B, A, K

        gather_idx = route_idx.unsqueeze(-1).expand(-1, -1, -1, c)
        region_k_sel = torch.gather(region_k.unsqueeze(1).expand(-1, agent_count, -1, -1), 2, gather_idx)
        region_v_sel = torch.gather(region_v.unsqueeze(1).expand(-1, agent_count, -1, -1), 2, gather_idx)

        offsets = self.offset(agent_q_map)
        offsets = offsets.view(b, self.topk, self.sample_points, 2, self.agent_grid, self.agent_grid)
        offsets = offsets.permute(0, 4, 5, 1, 2, 3).reshape(b, agent_count, self.topk, self.sample_points, 2)
        offsets = offsets[:, :, :topk]

        route_y = torch.div(route_idx, self.area, rounding_mode="floor").float()
        route_x = (route_idx % self.area).float()
        cell = 2.0 / self.area
        base_x = (route_x + 0.5) * cell - 1.0
        base_y = (route_y + 0.5) * cell - 1.0
        base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(3)  # B, A, K, 1, 2
        sample_grid = (base_grid + torch.tanh(offsets) * (0.5 * cell)).clamp(-1.0, 1.0)

        sample_grid_flat = sample_grid.reshape(b, agent_count * topk * self.sample_points, 1, 2)
        sample_grid_flat = sample_grid_flat.to(device=k_map.device, dtype=k_map.dtype)
        sampled_k = F.grid_sample(k_map, sample_grid_flat, mode="bilinear", padding_mode="border", align_corners=False)
        sampled_v = F.grid_sample(v_map, sample_grid_flat, mode="bilinear", padding_mode="border", align_corners=False)
        sampled_k = sampled_k.reshape(b, c, agent_count, topk, self.sample_points).permute(0, 2, 3, 4, 1)
        sampled_v = sampled_v.reshape(b, c, agent_count, topk, self.sample_points).permute(0, 2, 3, 4, 1)
        sampled_k = sampled_k.reshape(b, agent_count, topk * self.sample_points, c)
        sampled_v = sampled_v.reshape(b, agent_count, topk * self.sample_points, c)

        tokens_k = torch.cat((region_k_sel, sampled_k), dim=2)
        tokens_v = torch.cat((region_v_sel, sampled_v), dim=2)
        attn = (agents.unsqueeze(2) * tokens_k).sum(dim=-1) * self.scale
        attn = attn.softmax(dim=-1)
        agent_ctx = (attn.unsqueeze(-1) * tokens_v).sum(dim=2)
        agent_ctx = agent_ctx.transpose(1, 2).reshape(b, c, self.agent_grid, self.agent_grid)
        routed_context = F.interpolate(agent_ctx, size=(h, w), mode="bilinear", align_corners=False)

        fused = routed_context + self.local(xn)
        fused = fused * self.channel_gate(fused)
        return self.proj(fused)


class SARACountAttn(nn.Module):
    """Saliency-aware rotated aggregation attention for dense wheat spike counting."""

    def __init__(
        self,
        dim,
        num_heads=1,
        area=1,
        dir_radius=5,
        tau=3.0,
        block_size=8,
        keep_ratio=0.25,
        bg_kernel=7,
        lambda_bg=0.5,
        gate_hidden=16,
        residual_scale_init=0.1,
    ):
        super().__init__()
        self.channels = dim
        self.lambda_bg = lambda_bg
        self.norm = nn.LayerNorm(dim)

        self.saliency_conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.dir_agg = DirectionalLineAggregator(channels=dim, radius=dir_radius, tau=tau)
        self.dir_gate = nn.Sequential(
            ConvBNAct(4, gate_hidden, k=3, s=1, p=1),
            nn.Conv2d(gate_hidden, 4, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.block_mask = BlockTopKMask(block_size=block_size, keep_ratio=keep_ratio)
        self.bg_pool = nn.AvgPool2d(kernel_size=bg_kernel, stride=1, padding=bg_kernel // 2)
        self.fuse = nn.Sequential(
            ConvBNAct(dim * 3, dim, k=1, s=1, p=0),
            ConvBNAct(dim, dim, k=3, s=1, p=1, g=1, act=False),
        )
        self.out_act = nn.SiLU(inplace=True)
        self.res_scale = nn.Parameter(torch.tensor(residual_scale_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        avg_map = xn.mean(dim=1, keepdim=True)
        max_map = xn.amax(dim=1, keepdim=True)
        std_map = xn.std(dim=1, keepdim=True, unbiased=False)
        saliency = torch.sigmoid(self.saliency_conv(torch.cat([avg_map, max_map, std_map], dim=1)))

        dir_feats = self.dir_agg(xn)
        dir_stats = torch.cat([f.mean(dim=1, keepdim=True) for f in dir_feats], dim=1)
        alpha = F.softmax(self.dir_gate(dir_stats), dim=1)

        weighted_dir = (
            alpha[:, 0:1] * dir_feats[0]
            + alpha[:, 1:2] * dir_feats[1]
            + alpha[:, 2:3] * dir_feats[2]
            + alpha[:, 3:4] * dir_feats[3]
        )

        route_score = saliency + 0.3 * alpha.max(dim=1, keepdim=True).values
        hard_mask = self.block_mask(route_score)
        routed_x = xn * saliency * hard_mask
        routed_dir_feats = self.dir_agg(routed_x)
        context = (
            alpha[:, 0:1] * routed_dir_feats[0]
            + alpha[:, 1:2] * routed_dir_feats[1]
            + alpha[:, 2:3] * routed_dir_feats[2]
            + alpha[:, 3:4] * routed_dir_feats[3]
        )

        bg = self.bg_pool(xn)
        enhance = xn - self.lambda_bg * bg
        foreground = saliency * enhance

        y = self.fuse(torch.cat([weighted_dir, context, foreground], dim=1))
        return self.out_act(self.res_scale * y)


class BCRSAttn(nn.Module):
    """Boundary-Contrast Repulsion Separation attention for crowded instance separation.

    The implementation separates three concepts to avoid ambiguity:
    low-level cues, response maps, and the final guided fusion branches.
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
        branch_enabled_tensor = getattr(self, "branch_enabled", None)
        if branch_enabled_tensor is not None:
            branch_flags = branch_enabled_tensor.detach().view(-1).to(dtype=torch.bool).cpu().tolist()
            while len(branch_flags) < 4:
                branch_flags.append(True)
        else:
            branch_flags = [True, True, True, True]
        enable_center = bool(getattr(self, "enable_center", branch_flags[0]))
        enable_boundary = bool(getattr(self, "enable_boundary", branch_flags[1]))
        enable_repulsion = bool(getattr(self, "enable_repulsion", branch_flags[2]))
        enable_context = bool(getattr(self, "enable_context", branch_flags[3]))

        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        value = self.value(xn)
        local_texture_cue = self.local(value)
        context_receptive_cue = self.context(value)
        if not enable_context:
            context_receptive_cue = torch.zeros_like(context_receptive_cue)

        local_contrast_cue = value - F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        contrast_response = local_contrast_cue.abs().mean(1, keepdim=True)
        center_residual_cue = F.relu(value - F.avg_pool2d(value, kernel_size=5, stride=1, padding=2))
        center_response = self.center_gate(
            torch.cat(
                (
                    value.mean(1, keepdim=True),
                    value.amax(1, keepdim=True),
                    contrast_response,
                    center_residual_cue.mean(1, keepdim=True),
                ),
                1,
            )
        )
        if not enable_center:
            center_response = torch.zeros_like(center_response)

        sobel_edge_cue = self.sobel_x(value).abs() + self.sobel_y(value).abs()
        laplace_edge_cue = self.laplace(value).abs()
        boundary_response = self.boundary_gate(
            torch.cat(
                (
                    sobel_edge_cue.mean(1, keepdim=True),
                    laplace_edge_cue.mean(1, keepdim=True),
                    contrast_response,
                ),
                1,
            )
        )
        if not enable_boundary:
            boundary_response = torch.zeros_like(boundary_response)

        crowding_response = F.avg_pool2d(center_response, kernel_size=5, stride=1, padding=2)
        peak_response = F.max_pool2d(center_response, kernel_size=3, stride=1, padding=1)
        repulsion_response = self.repulsion_gate(
            torch.cat(
                (
                    F.relu(peak_response - center_response),
                    crowding_response,
                    boundary_response,
                ),
                1,
            )
        )
        if not enable_repulsion:
            repulsion_response = torch.zeros_like(repulsion_response)

        edge_boundary_cue = self.boundary_proj(sobel_edge_cue + laplace_edge_cue)
        projected_contrast_cue = self.contrast_proj(local_contrast_cue)
        repulsion_cue = self.repulsion_proj(
            local_contrast_cue * (1.0 + repulsion_response) + edge_boundary_cue * repulsion_response
        )
        if not enable_boundary:
            edge_boundary_cue = torch.zeros_like(edge_boundary_cue)
        if not enable_repulsion:
            repulsion_cue = torch.zeros_like(repulsion_cue)

        center_guided_branch = local_texture_cue * center_response
        boundary_guided_branch = (local_texture_cue + context_receptive_cue + edge_boundary_cue) * boundary_response
        repulsion_guided_branch = repulsion_cue * repulsion_response
        context_guided_branch = context_receptive_cue * (1.0 + 0.5 * center_response)
        if not enable_center:
            center_guided_branch = torch.zeros_like(center_guided_branch)
        if not enable_boundary:
            boundary_guided_branch = torch.zeros_like(boundary_guided_branch)
        if not enable_repulsion:
            repulsion_guided_branch = torch.zeros_like(repulsion_guided_branch)
        if not enable_context:
            context_guided_branch = torch.zeros_like(context_guided_branch)

        branch_logits = self.branch_score(
            torch.cat(
                (center_guided_branch, boundary_guided_branch, repulsion_guided_branch, context_guided_branch), 1
            )
        )
        if branch_enabled_tensor is None:
            branch_enabled_tensor = torch.tensor(branch_flags, dtype=torch.bool, device=branch_logits.device).view(1, 4, 1, 1)
        else:
            branch_enabled_tensor = branch_enabled_tensor.to(device=branch_logits.device, dtype=torch.bool)
        branch_logits = branch_logits.masked_fill(~branch_enabled_tensor, torch.finfo(branch_logits.dtype).min)
        branch_gate = branch_logits.softmax(dim=1)
        fused = (
            center_guided_branch * branch_gate[:, 0:1]
            + boundary_guided_branch * branch_gate[:, 1:2]
            + repulsion_guided_branch * branch_gate[:, 2:3]
            + context_guided_branch * branch_gate[:, 3:4]
        )
        fused = fused * (1.0 + self.gamma.tanh() * repulsion_response) * self.channel_gate(fused)
        spatial = self.spatial_gate(torch.cat((fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)), 1))
        out = self.proj(fused * spatial)
        if getattr(self, "collect_branch_maps", False):
            self.last_branch_maps = {
                "input_feature": x.detach().float().cpu(),
                "local_texture_cue": local_texture_cue.detach().float().cpu(),
                "context_receptive_cue": context_receptive_cue.detach().float().cpu(),
                "projected_contrast_cue": projected_contrast_cue.detach().float().cpu(),
                "edge_boundary_cue": edge_boundary_cue.detach().float().cpu(),
                "center_response": center_response.detach().float().cpu(),
                "boundary_response": boundary_response.detach().float().cpu(),
                "repulsion_response": repulsion_response.detach().float().cpu(),
                "center_guided_branch": center_guided_branch.detach().float().cpu(),
                "boundary_guided_branch": boundary_guided_branch.detach().float().cpu(),
                "repulsion_guided_branch": repulsion_guided_branch.detach().float().cpu(),
                "context_guided_branch": context_guided_branch.detach().float().cpu(),
                "center_branch": center_guided_branch.detach().float().cpu(),
                "boundary_branch": boundary_guided_branch.detach().float().cpu(),
                "repulsion_branch": repulsion_guided_branch.detach().float().cpu(),
                "context_branch": context_guided_branch.detach().float().cpu(),
                "branch_gate": branch_gate.detach().float().cpu(),
                "output_feature": out.detach().float().cpu(),
            }
        return out


class SpikeSepAttn(nn.Module):
    """Spike-oriented attention for dense, elongated, and overlapping agricultural targets."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.norm = nn.LayerNorm(dim)
        self.local = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.axial_h = nn.Conv2d(dim, dim, (1, 7), 1, (0, 3), groups=dim, bias=False)
        self.axial_v = nn.Conv2d(dim, dim, (7, 1), 1, (3, 0), groups=dim, bias=False)
        self.axial_proj = Conv(dim, dim, 1, 1, act=False)
        self.sep = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.branch_score = nn.Conv2d(dim * 3, 3, 1, 1, bias=True)
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
        self.density_gate = nn.Sequential(
            nn.Conv2d(1, 1, 5, 1, 2, bias=True),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)

    def forward(self, x):
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        local_feat = self.local(xn)
        axial_feat = self.axial_proj(self.axial_h(xn) + self.axial_v(xn))

        # Local contrast highlights adjacent spike boundaries in crowded regions.
        contrast = xn - torch.nn.functional.avg_pool2d(xn, kernel_size=3, stride=1, padding=1)
        sep_feat = self.sep(contrast)

        branch_gate = self.branch_score(torch.cat((local_feat, axial_feat, sep_feat), 1)).softmax(dim=1)
        fused = (
            local_feat * branch_gate[:, 0:1]
            + axial_feat * branch_gate[:, 1:2]
            + sep_feat * branch_gate[:, 2:3]
        )

        density = self.density_gate(contrast.abs().mean(1, keepdim=True))
        spatial = self.spatial_gate(torch.cat((fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)), 1))
        fused = fused * (1.0 + density) * self.channel_gate(fused) * spatial
        return self.proj(fused)


class DCLCAttn(nn.Module):
    """Density-Calibrated Local-Context Attention for count-stable dense object detection."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.norm = nn.LayerNorm(dim)
        self.identity = Conv(dim, dim, 1, 1, act=False)
        self.local = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.context = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(dim, dim, 5, 1, g=dim, act=False),
        )
        self.branch_score = nn.Conv2d(dim * 3, 3, 1, 1, bias=True)
        self.density_gate = nn.Sequential(
            nn.Conv2d(2, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, dim, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 5, 1, 2, bias=False),
            nn.Sigmoid(),
        )
        self.proj = Conv(dim, dim, 1, 1, act=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        xn = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

        identity_feat = self.identity(xn)
        local_feat = self.local(xn)
        context_feat = self.context(xn)

        branch_gate = self.branch_score(torch.cat((identity_feat, local_feat, context_feat), 1)).softmax(dim=1)
        fused = (
            identity_feat * branch_gate[:, 0:1]
            + local_feat * branch_gate[:, 1:2]
            + context_feat * branch_gate[:, 2:3]
        )

        density_input = torch.cat(
            (
                local_feat.mean(1, keepdim=True),
                (context_feat - identity_feat).abs().mean(1, keepdim=True),
            ),
            1,
        )
        density = self.density_gate(density_input)
        spatial = self.spatial_gate(torch.cat((fused.mean(1, keepdim=True), fused.amax(1, keepdim=True)), 1))
        fused = fused * (1.0 + self.alpha.tanh() * density) * self.channel_gate(fused) * spatial
        return self.proj(fused)


class CoordCountAttn(nn.Module):
    """Coordinate Attention adapted for count-oriented dense object detection."""

    def __init__(self, dim, num_heads=1, area=1, reduction=16):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.conv1 = nn.Conv2d(dim, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_w = nn.Conv2d(hidden, dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


class EMACountAttn(nn.Module):
    """EMA attention adapted for dense small-object counting in YOLOv12 backbone blocks."""

    def __init__(self, dim, num_heads=1, area=1, factor=8):
        super().__init__()
        groups = min(factor, dim)
        while dim % groups != 0 and groups > 1:
            groups -= 1
        self.groups = max(groups, 1)
        gc = dim // self.groups

        self.gn = nn.GroupNorm(1, gc)
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, stride=1, padding=1, bias=True)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, -1, h, w)

        x_h = gx.mean(dim=3, keepdim=True)
        x_w = gx.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x1 = self.gn(gx * x_h.sigmoid() * x_w.sigmoid())
        x2 = self.conv3x3(gx)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, 1, -1))
        x12 = x2.reshape(b * self.groups, -1, h * w)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, 1, -1))
        x22 = x1.reshape(b * self.groups, -1, h * w)

        weights = (torch.bmm(x11, x12) + torch.bmm(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (gx * weights.sigmoid()).reshape(b, c, h, w)
        return out


class SparseCountAttn(nn.Module):
    """Sparse block attention adapted from the user's cs3 SparseAttention for A2C2f integration."""

    def __init__(
        self,
        dim,
        num_heads=8,
        area=1,
        block_size=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.in_channels = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        assert dim % num_heads == 0, f"in_channels {dim} must be divisible by num_heads {num_heads}"

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def block_partition(self, x):
        """Partition feature maps into non-overlapping sparse-attention blocks."""
        b, c, h, w = x.shape
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        hp, wp = h + pad_h, w + pad_w
        nh, nw = hp // self.block_size, wp // self.block_size
        x = x.view(b, c, nh, self.block_size, nw, self.block_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b, nh * nw, c, self.block_size, self.block_size)
        return x, (h, w, hp, wp, nh, nw)

    def block_reverse(self, x, shape_info):
        """Restore sparse-attention blocks to the original feature map layout."""
        h, w, hp, wp, nh, nw = shape_info
        b, _, c, bh, bw = x.shape
        x = x.view(b, nh, nw, c, bh, bw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, hp, wp)
        if hp > h or wp > w:
            x = x[:, :, :h, :w]
        return x

    def forward_attention(self, x):
        """Compute block-sparse attention over local windows."""
        b, c, _, _ = x.shape
        x_blocks, shape_info = self.block_partition(x)
        b_blocks, n_blocks, c_blocks, bh, bw = x_blocks.shape
        x_blocks = x_blocks.view(b_blocks * n_blocks, c_blocks, bh, bw)

        qkv = self.qkv(x_blocks)
        qkv = qkv.reshape(b_blocks * n_blocks, 3, self.num_heads, self.head_dim, bh * bw)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_attn = (attn @ v)
        x_attn = x_attn.transpose(1, 2).reshape(b_blocks * n_blocks, c_blocks, bh, bw)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        x_attn = x_attn.view(b_blocks, n_blocks, c_blocks, bh, bw)
        return self.block_reverse(x_attn, shape_info)

    def forward(self, x):
        """Forward pass without the internal residual to avoid double-residual inside ABlock."""
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x_norm = x_norm + self.pos_embed(x_norm)
        return self.forward_attention(x_norm)
