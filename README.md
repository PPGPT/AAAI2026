# PPGPT

Reference implementation of the model described in **PPGPT: Transferring
Next-Token Modeling from Language to PPG Signals**.

PPGPT reformulates next-token prediction into Next-Feature Token Prediction
(NFTP) for photoplethysmography. This repository provides the core model and
components:

- `models/hstf.py` — Hierarchical Spatio-Temporal Fusion (HSTF) dual-stream
  encoder: Temporal Contextualizer, Spatial Relation Aggregator, Bidirectional
  Attention Fusion, producing hierarchical feature tokens.
- `models/nftp.py` — DeepSeekMoE decoder with the NFTP objective.
- `data/vision_ppg.py` — VisionPPG signal-to-image transform.
- `training/losses.py` — contrastive and bidirectional jump-prediction losses.
- `utils/metrics.py` — regression / classification / representation metrics.

## Install

```bash
pip install -r requirements.txt
```

## Model

```python
import torch
from models.hstf import build_hstf
from models.nftp import build_nftp

encoder = build_hstf({"dim": 96, "layers": 6})
decoder = build_nftp(encoder.token_dim(), {"d_model": 160, "depth": 6})

signal = torch.randn(8, 512)
vision = torch.rand(8, 32, 32, 3)
Z = encoder(signal, vision)          # [B, L, 2d] hierarchical feature tokens
loss, _ = decoder.nftp_loss(Z)       # next-feature-token prediction loss
```
