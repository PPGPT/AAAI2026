import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContextualizer(nn.Module):

    def __init__(self, dim, hidden=None, dropout=0.1):
        super().__init__()
        hidden = hidden or dim * 2
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.mlp_task = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.mlp_sample = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, u):
        g = self.gate(u)
        u_task = g * u
        u_sample = (1.0 - g) * u
        return u + self.mlp_task(u_task) + self.mlp_sample(u_sample)

class SpatialRelationAggregator(nn.Module):

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, v):
        B, P, d = v.shape
        h = self.proj_in(v).view(B, P, self.heads, self.dh).transpose(1, 2)
        hn = F.normalize(h, dim=-1)
        aff = torch.matmul(hn, hn.transpose(-1, -2))
        adj = F.softmax(aff / self.dh ** 0.5, dim=-1)
        prop = torch.matmul(self.drop(adj), h)
        prop = prop.transpose(1, 2).contiguous().view(B, P, d)
        out = self.norm(v + self.proj_out(prop))
        return out

class BidirectionalAttentionFusion(nn.Module):

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn_uv = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_vu = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_u = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.mlp_u = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.mlp_v = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, u_vec, v_patches):
        u = u_vec.unsqueeze(1)
        a_uv, _ = self.attn_uv(u, v_patches, v_patches)
        u_tilde = self.norm_u(u + a_uv)
        a_vu, _ = self.attn_vu(v_patches, u, u)
        v_tilde = self.norm_v(v_patches + a_vu)
        u_out = self.mlp_u(u_tilde).squeeze(1)
        v_out = self.mlp_v(v_tilde)
        return (u_out, v_out)

class HSTFLayer(nn.Module):

    def __init__(self, dim, heads=4, dropout=0.1, ablate=None):
        super().__init__()
        self.ablate = ablate
        self.tc = TemporalContextualizer(dim, dropout=dropout)
        self.sra = SpatialRelationAggregator(dim, heads=heads, dropout=dropout)
        self.baf = BidirectionalAttentionFusion(dim, heads=heads, dropout=dropout)
        self.token_norm = nn.LayerNorm(2 * dim)

    def forward(self, u, v):
        a = self.ablate
        if a == 'temporal':
            u = self.tc(u)
            return (u, v, self.token_norm(torch.cat([u, u], dim=-1)))
        if a == 'spatial':
            v = self.sra(v)
            v_vec = v.mean(dim=1)
            return (u, v, self.token_norm(torch.cat([v_vec, v_vec], dim=-1)))
        if a != 'no_tc':
            u = self.tc(u)
        if a != 'no_sra':
            v = self.sra(v)
        if a == 'no_baf':
            u_out, v_out = (u, v)
        else:
            u_out, v_out = self.baf(u, v)
        v_vec = v_out.mean(dim=1)
        z = self.token_norm(torch.cat([u_out, v_vec], dim=-1))
        return (u_out, v_out, z)

class SignalStem(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.GELU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(), nn.Conv1d(64, dim, 3, stride=2, padding=1), nn.BatchNorm1d(dim), nn.GELU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)

class VisionStem(nn.Module):

    def __init__(self, dim, patches=16):
        super().__init__()
        self.patches = patches
        self.grid = int(round(patches ** 0.5))
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.Conv2d(64, dim, 3, stride=1, padding=1), nn.BatchNorm2d(dim), nn.GELU(), nn.AdaptiveAvgPool2d((self.grid, self.grid)))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        f = self.net(x)
        B, d, g, _ = f.shape
        return f.flatten(2).transpose(1, 2)

class HSTF(nn.Module):

    def __init__(self, dim=64, layers=6, patches=16, heads=4, dropout=0.1, ablate=None):
        super().__init__()
        self.dim = dim
        self.layers_n = layers
        self.ablate = ablate
        self.signal_stem = SignalStem(dim)
        self.vision_stem = VisionStem(dim, patches=patches)
        self.blocks = nn.ModuleList([HSTFLayer(dim, heads=heads, dropout=dropout, ablate=ablate) for _ in range(layers)])
        self.urg = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, signal, vision, return_final=False):
        u = self.signal_stem(signal)
        v = self.vision_stem(vision)
        tokens = []
        for blk in self.blocks:
            u, v, z = blk(u, v)
            tokens.append(z)
        Z = torch.stack(tokens, dim=1)
        if return_final:
            v_vec = v.mean(dim=1)
            p = self.urg(torch.cat([u, v_vec], dim=-1))
            return (Z, p)
        return Z

    def token_dim(self):
        return 2 * self.dim

def build_hstf(cfg=None):
    cfg = cfg or {}
    return HSTF(dim=cfg.get('dim', 64), layers=cfg.get('layers', 6), patches=cfg.get('patches', 16), heads=cfg.get('heads', 4), dropout=cfg.get('dropout', 0.1), ablate=cfg.get('ablate', None))
