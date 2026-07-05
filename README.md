# PPGPT

Official repository for **PPGPT: Transferring Next-Token Modeling from Language
to PPG Signals** (AAAI 2026), providing the model and the BioMTL benchmark.

PPGPT reformulates next-token prediction into Next-Feature Token Prediction
(NFTP) for photoplethysmography, pairing a dual-stream Hierarchical
Spatio-Temporal Fusion (HSTF) encoder with a DeepSeekMoE decoder.

## Model

- `models/hstf.py` — HSTF dual-stream encoder: Temporal Contextualizer, Spatial
  Relation Aggregator, Bidirectional Attention Fusion, producing hierarchical
  feature tokens.
- `models/nftp.py` — DeepSeekMoE decoder with the NFTP objective.
- `data/vision_ppg.py` — VisionPPG signal-to-image transform.
- `training/losses.py` — contrastive and bidirectional jump-prediction losses.
- `utils/metrics.py` — regression / classification / representation metrics.

```bash
pip install -r requirements.txt
```

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

## BioMTL benchmark

BioMTL is a PPG-centric multi-task benchmark: 1,808 four-second fingertip-PPG
windows (1000 Hz) from 172 subjects, with labels for seven tasks — SBP, DBP,
blood glucose, heart rate, BMI (regression) and mental stress, hypertension,
type-2 diabetes (classification).

The dataset is released for **non-commercial academic research under the Data
Use Agreement**. To request access, complete `dataset/documents/en/DATA_USE_AGREEMENT.pdf`
and email the signed PDF to **biomtl@163.com**.

- `dataset/documents/en/`, `dataset/documents/zh/` — dataset documentation and
  data use agreement (English / Chinese).
- `dataset/SHA256SUMS.txt` — checksums for the released archive.
- `dataset/example_window.csv` — one example PPG window.

## Citation

```bibtex
@inproceedings{zhang2026ppgpt,
  title     = {PPGPT: Transferring Next-Token Modeling from Language to PPG Signals},
  author    = {Zhang, Zexing and Lu, Huimin and Zhao, Qingxin},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {34},
  pages     = {28573--28581},
  year      = {2026},
  doi       = {10.1609/aaai.v40i34.40088}
}
```
