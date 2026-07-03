import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce(p1, p2, temperature=0.07):
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    logits = p1 @ p2.t() / temperature
    targets = torch.arange(p1.size(0), device=p1.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

class JumpDecoder(nn.Module):

    def __init__(self, dim, out_len):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, out_len))

    def forward(self, p):
        return self.net(p)

def bidirectional_jump_loss(decoder, p1, p2, sig1, sig2):
    rec1 = decoder(p1)
    rec2 = decoder(p2)
    return F.mse_loss(rec1, sig2) + F.mse_loss(rec2, sig1)

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, f1, f2):
        return info_nce(f1, f2, self.temperature)
