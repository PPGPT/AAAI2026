import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class DeepSeekMoE(nn.Module):

    def __init__(self, dim, hidden, n_routed=4, top_k=2, n_shared=1, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.n_routed = n_routed

        def ffn():
            return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.shared = nn.ModuleList([ffn() for _ in range(n_shared)])
        self.routed = nn.ModuleList([ffn() for _ in range(n_routed)])
        self.router = nn.Linear(dim, n_routed)

    def forward(self, x):
        B, T, d = x.shape
        out = sum((e(x) for e in self.shared))
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-09)
        routed_out = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = topi[..., slot]
            weight = topv[..., slot].unsqueeze(-1)
            for e in range(self.n_routed):
                mask = (idx == e).unsqueeze(-1).float()
                if mask.any():
                    routed_out = routed_out + mask * weight * self.routed[e](x)
        importance = probs.mean(dim=(0, 1))
        aux = (importance * importance).sum() * self.n_routed
        return (out + routed_out, aux)

class DecoderBlock(nn.Module):

    def __init__(self, dim, heads=4, n_routed=4, top_k=2, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.moe = DeepSeekMoE(dim, dim * 4, n_routed=n_routed, top_k=top_k, dropout=dropout)

    def forward(self, x, attn_mask):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m, aux = self.moe(self.norm2(x))
        x = x + m
        return (x, aux)

class NFTPDecoder(nn.Module):

    def __init__(self, token_dim, d_model=128, depth=4, heads=4, n_routed=4, top_k=2, max_len=32, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([DecoderBlock(d_model, heads, n_routed, top_k, dropout) for _ in range(depth)])
        self.norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, token_dim)

    def forward(self, Z):
        B, L, _ = Z.shape
        x = self.in_proj(Z) + self.pos[:, :L]
        mask = torch.triu(torch.full((L, L), float('-inf'), device=Z.device), diagonal=1)
        aux_total = 0.0
        for blk in self.blocks:
            x, aux = blk(x, mask)
            aux_total = aux_total + aux
        x = self.norm(x)
        pred = self.out_proj(x)
        return (pred, aux_total)

    def nftp_loss(self, Z, aux_weight=0.01):
        pred, aux = self.forward(Z)
        target = Z[:, 1:, :]
        loss = F.mse_loss(pred[:, :-1, :], target)
        return (loss + aux_weight * aux, {'nftp_mse': loss.item(), 'moe_aux': float(aux.detach())})

def build_nftp(token_dim, cfg=None):
    cfg = cfg or {}
    return NFTPDecoder(token_dim, d_model=cfg.get('d_model', 128), depth=cfg.get('depth', 4), heads=cfg.get('heads', 4), n_routed=cfg.get('n_routed', 4), top_k=cfg.get('top_k', 2), max_len=cfg.get('max_len', 32), dropout=cfg.get('dropout', 0.1))
