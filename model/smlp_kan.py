import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from typing import *


# ════════════════════════════════════════════════════════════
#  FasterKAN components — used for lightweight timestep conditioning
# ════════════════════════════════════════════════════════════

class SplineLinear(nn.Linear):
    """Linear layer with Xavier uniform initialization for spline weights."""
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)


class ReflectionalSwitchFunction(nn.Module):
    """
    Reflectional switch (basis) function used in PSAB:
        phi(x) = 1 - tanh^2((x - g) / d)

    Grid points g are fixed and uniformly distributed over [grid_min, grid_max],
    introducing no additional learnable parameters (as required by the paper).
    """
    def __init__(self, grid_min=-2., grid_max=2., num_grids=8, exponent=2, denominator=0.33):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.inv_denominator = 1 / denominator

    def forward(self, x):
        diff = (x[..., None] - self.grid).mul(self.inv_denominator)
        t = torch.tanh(diff)
        return 1 - t.mul(t)


class KANCondLayer(nn.Module):
    """
    Single KAN layer used for timestep scale/shift conditioning.
    Renamed from FasterKANLayer to reflect its role in SMLP-KAN.
    """
    def __init__(self, input_dim, output_dim, grid_min=-2., grid_max=2.,
                 num_grids=8, exponent=2, denominator=0.33,
                 use_base_update=True, base_activation=F.silu,
                 spline_weight_init_scale=0.1):
        super().__init__()
        self.layernorm    = nn.LayerNorm(input_dim)
        self.rbf          = ReflectionalSwitchFunction(grid_min, grid_max, num_grids, exponent, denominator)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)

    def forward(self, x, time_benchmark=False):
        inp          = self.layernorm(x) if not time_benchmark else x
        spline_basis = self.rbf(inp).view(x.shape[0], -1)
        return self.spline_linear(spline_basis)


class KANCond(nn.Module):
    """
    Multi-layer KAN conditioning network (stacked KANCondLayers).
    Renamed from FasterKAN to reflect its role in SMLP-KAN.
    """
    def __init__(self, layers_hidden, grid_min=-2., grid_max=2., num_grids=8,
                 exponent=2, denominator=0.33, use_base_update=True,
                 base_activation=F.silu, spline_weight_init_scale=0.667):
        super().__init__()
        self.layers = nn.ModuleList([
            KANCondLayer(in_dim, out_dim, grid_min=grid_min, grid_max=grid_max,
                         num_grids=num_grids, exponent=exponent, denominator=denominator,
                         use_base_update=use_base_update, base_activation=base_activation,
                         spline_weight_init_scale=spline_weight_init_scale)
            for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ════════════════════════════════════════════════════════════
#  Sinusoidal timestep embedding
# ════════════════════════════════════════════════════════════

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps:  1-D Tensor of N timestep indices (one per batch element).
    :param dim:        Output embedding dimension.
    :param max_period: Controls the minimum frequency of the embeddings.
    :return: [N x dim] Tensor of positional embeddings.
    """
    half  = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args  = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Activation(Enum):
    none  = 'none'
    relu  = 'relu'
    lrelu = 'lrelu'
    silu  = 'silu'
    tanh  = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


# ════════════════════════════════════════════════════════════
#  PSAB — Probabilistic Spline Attention Block
#
#  Paper formulation:
#    phi(F_n) = 1 - tanh^2((F_n - g) / d)   [ReflectionalSwitchFunction]
#    F_y      = W_spline * phi(F_n)           [spline_linear]
#    F_p      = softmax(Linear(F_y)) * F_y    [attn_linear + softmax]
#
#  Grid g is fixed and uniformly distributed over [-2, 2],
#  introducing no extra learnable parameters (as stated in the paper).
#
#  Implementation note: a learnable temperature parameter scales the softmax
#  logits to prevent attention collapse when the channel dimension C is large
#  (e.g. C=256 would otherwise yield uniform weights ~1/C).
# ════════════════════════════════════════════════════════════

class PSAB(nn.Module):
    def __init__(self, in_channels: int, num_grids: int = 4, denominator: float = 0.33):
        super().__init__()
        self.rsf          = ReflectionalSwitchFunction(
            grid_min=-2., grid_max=2., num_grids=num_grids, denominator=denominator)
        # W_spline: (in_channels * num_grids) -> in_channels
        self.spline_linear = SplineLinear(in_channels * num_grids, in_channels)
        self.attn_linear   = nn.Linear(in_channels, in_channels)
        # Learnable temperature for numerically stable softmax
        self.temperature   = nn.Parameter(torch.ones(1) * math.sqrt(in_channels))

    def forward(self, x):
        # x = F_n: [B, C]
        basis = self.rsf(x).reshape(x.shape[0], -1)                     # [B, C*G]
        F_y   = self.spline_linear(basis)                                # [B, C]
        # Paper: F_p = softmax(Linear(F_y)) * F_y
        attn  = torch.softmax(self.attn_linear(F_y) / self.temperature, dim=-1)
        return attn * F_y                                                # F_p: [B, C]


# ════════════════════════════════════════════════════════════
#  CAAB — Clustered Approximate Attention Block
#
#  Paper formulation:
#    F_c = k-means(LayerNorm(F_n))
#    - Fixed k=2, independent of the number of spectral bands.
#    - Only the cluster with the higher mean response is retained,
#      forming a binary mask that emphasises dominant spectral patterns.
#
#  CAAB holds its own LayerNorm, directly matching the paper formula
#  "k-means(LayerNorm(F_n))".
#
#  Implementation note: the mean-threshold rule is the closed-form optimal
#  2-cluster partition for 1-D features, replacing iterative K-means at O(1)
#  cost. The mask is computed inside torch.no_grad() because K-means is
#  non-differentiable by definition.
# ════════════════════════════════════════════════════════════

class CAAB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Paper: input to k-means is LayerNorm(F_n); CAAB owns this LN.
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # x = F_n: [B, C]
        x_norm = self.norm(x)                                    # LayerNorm(F_n): [B, C]
        # Mean threshold = closed-form optimal 2-cluster split.
        # Computed without gradient (K-means is non-differentiable).
        with torch.no_grad():
            threshold = x_norm.mean(dim=-1, keepdim=True)        # [B, 1]
            mask      = (x_norm > threshold).float()             # [B, C]  binary mask
        # F_c: retain the dominant cluster from the normalised features
        return mask * x_norm                                     # F_c: [B, C]


# ════════════════════════════════════════════════════════════
#  RHAG — Residual Hybrid Attention Group
#
#  Paper formulation:
#    F_n = LayerNorm(F_s)
#    F_p = PSAB(F_n)
#    F_c = CAAB(F_n)   [CAAB applies its own LN internally, per the paper]
#    F_d = F_p + F_c
# ════════════════════════════════════════════════════════════

class RHAG(nn.Module):
    def __init__(self, channels: int, num_grids: int = 4, denominator: float = 0.33):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)
        self.psab      = PSAB(channels, num_grids=num_grids, denominator=denominator)
        self.caab      = CAAB(channels)

    def forward(self, F_s):
        F_n = self.layernorm(F_s)   # F_n = LayerNorm(F_s)
        F_p = self.psab(F_n)        # PSAB branch
        F_c = self.caab(F_n)        # CAAB branch (applies LayerNorm(F_n) internally)
        return F_p + F_c            # F_d


# ════════════════════════════════════════════════════════════
#  SMLPKANConfig — configuration dataclass for SMLP-KAN
#  (previously MLPSkipNetConfig)
# ════════════════════════════════════════════════════════════

@dataclass
class SMLPKANConfig:
    """Configuration for the SMLP-KAN denoising network."""
    num_channels:          int
    skip_layers:           Tuple[int]
    num_hid_channels:      int
    num_layers:            int
    num_time_emb_channels: int        = 64
    activation:            Activation = Activation.silu
    use_norm:              bool       = True
    condition_bias:        float      = 1
    dropout:               float      = 0
    last_act:              Activation = Activation.none
    num_time_layers:       int        = 2
    time_last_act:         bool       = False
    # Number of grid points for the KAN conditioning layers
    kan_num_grids:         int        = 4
    # Number of grid points for the PSAB spline basis
    psab_num_grids:        int        = 4
    psab_denominator:      float      = 0.33

    def make_model(self):
        return SMLPKANNet(self)


# Alias for backward compatibility with external code that imports MLPSkipNetConfig
MLPSkipNetConfig = SMLPKANConfig


# ════════════════════════════════════════════════════════════
#  SMLPKANLayer — one network layer implementing the three-stage pipeline
#  (previously MLPLNAct)
#
#  Paper stages (when use_cond=True):
#    Stage 1  MLP Feature Extraction : F_s = Linear(SiLU(Linear(x_s)))
#    Conditioning                    : F_s = F_s * (bias + KAN_scale(cond))
#                                            + KAN_shift(cond)
#    Stage 2  Deep Feature (RHAG)    : F_d = PSAB(LN(F_s)) + CAAB(LN(F_s))
#    Stage 3  Feature Reconstruction : out = Dropout(LayerNorm(F_d))
#
#  When use_cond=False (final layer), a plain Linear + norm + activation is
#  used without conditioning or RHAG.
# ════════════════════════════════════════════════════════════

class SMLPKANLayer(nn.Module):
    def __init__(
            self,
            in_channels:      int,
            out_channels:     int,
            norm:             bool,
            use_cond:         bool,
            activation:       Activation,
            cond_channels:    int,
            condition_bias:   float = 0,
            dropout:          float = 0,
            kan_num_grids:    int   = 4,
            psab_num_grids:   int   = 4,
            psab_denominator: float = 0.33,
    ):
        super().__init__()
        self.activation     = activation
        self.condition_bias = condition_bias
        self.use_cond       = use_cond

        # Stage 1 — paper: F_s = Linear(SiLU(Linear(x_s)))
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            activation.get_act(),
            nn.Linear(out_channels, out_channels),
        )

        if self.use_cond:
            # Lightweight single-layer KAN for scale and shift conditioning
            self.cond_scale = KANCondLayer(
                input_dim=cond_channels, output_dim=out_channels, num_grids=kan_num_grids)
            self.cond_shift = KANCondLayer(
                input_dim=cond_channels, output_dim=out_channels, num_grids=kan_num_grids)
            # Stage 2 — RHAG
            self.rhag = RHAG(channels=out_channels,
                             num_grids=psab_num_grids, denominator=psab_denominator)
            # Stage 3 — FRB: LayerNorm + Dropout
            self.norm = nn.LayerNorm(out_channels)
        else:
            # Final layer: no conditioning, no RHAG
            self.norm = nn.LayerNorm(out_channels) if norm else nn.Identity()
            self.act  = activation.get_act()

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight, a=0,   nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight, a=0.2, nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight, a=0,   nonlinearity='relu')

    def forward(self, x, cond=None):
        # Stage 1: shallow feature extraction
        F_s = self.linear(x)

        if self.use_cond:
            # Inject timestep conditioning via KAN scale-shift modulation
            scale = self.cond_scale(cond)
            shift = self.cond_shift(cond)
            F_s   = F_s * (self.condition_bias + scale) + shift
            # Stage 2: deep feature extraction via RHAG
            F_d   = self.rhag(F_s)
            # Stage 3: feature reconstruction (FRB)
            return self.dropout(self.norm(F_d))
        else:
            out = self.norm(F_s)
            out = self.act(out)
            return self.dropout(out)


# ════════════════════════════════════════════════════════════
#  SMLPKANNet — DDPM backbone integrating SMLP-KAN layers
#  (previously MLPSkipNet)
#
#  Optionally concatenates the original input x into hidden layers
#  at positions specified by skip_layers (skip connections).
# ════════════════════════════════════════════════════════════

class SMLPKANNet(nn.Module):
    """
    SMLP-KAN denoising network for DDPM.
    Each conditioned layer runs the three-stage SMLP-KAN pipeline:
    MLP feature extraction → KAN conditioning → RHAG → FRB.
    Supports optional skip connections that concatenate x into hidden layers.
    """
    def __init__(self, conf: SMLPKANConfig):
        super().__init__()
        self.conf = conf

        # Timestep embedding MLP
        layers = []
        for i in range(conf.num_time_layers):
            a = conf.num_time_emb_channels if i == 0 else conf.num_channels
            b = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())
        self.time_embed = nn.Sequential(*layers)

        # Main SMLP-KAN layers
        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act, norm, cond = conf.activation, conf.use_norm, True
                a, b    = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act, norm, cond = Activation.none, False, False
                a, b    = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act, norm, cond = conf.activation, conf.use_norm, True
                a, b    = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

            # Widen input for skip-connected layers
            if i in conf.skip_layers:
                a += conf.num_channels

            self.layers.append(
                SMLPKANLayer(
                    a, b,
                    norm=norm,
                    activation=act,
                    cond_channels=conf.num_channels,
                    use_cond=cond,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                    kan_num_grids=conf.kan_num_grids,
                    psab_num_grids=conf.psab_num_grids,
                    psab_denominator=conf.psab_denominator,
                ))

        self.last_act = conf.last_act.get_act()

    def forward(self, x, t, **kwargs):
        t    = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        h    = x
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                h = torch.cat([h, x], dim=1)   # inject original input (skip connection)
            h = self.layers[i].forward(h, cond=cond)
        return self.last_act(h)


# ════════════════════════════════════════════════════════════
#  Quick sanity check
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, S = 512, 128
    x = torch.randn(B, S, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    def benchmark(model, name, n=50):
        model.eval().to(device)
        with torch.no_grad():
            for _ in range(5):
                model(x, t)
        start  = time.perf_counter()
        with torch.no_grad():
            for _ in range(n):
                model(x, t)
        ms     = (time.perf_counter() - start) / n * 1000
        params = sum(p.numel() for p in model.parameters()) / 1e3
        print(f"[{name}]  avg={ms:.2f}ms  params={params:.1f}K")

    def train_test(model, name):
        model.train().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        try:
            for _ in range(3):
                opt.zero_grad()
                model(x, t).mean().backward()
                opt.step()
            print(f"[{name}] training: OK")
        except RuntimeError as e:
            print(f"[{name}] training failed: {e}")

    cfg = SMLPKANConfig(
        num_channels=128, skip_layers=(), num_hid_channels=256,
        num_layers=6, kan_num_grids=4, psab_num_grids=4,
    )
    model = cfg.make_model()
    benchmark(model, "SMLP-KAN")
    train_test(model, "SMLP-KAN")

    model.eval()
    with torch.no_grad():
        out = model(x, t)
    print(f"Output shape: {out.shape}  {'ok' if out.shape == (B, S) else 'WRONG'}")