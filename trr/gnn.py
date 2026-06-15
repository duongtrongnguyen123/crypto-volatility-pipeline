"""Graph Attention Network over the asset-relational graph.

The TRR pipeline hand-codes the *relational* step (PageRank over the impact
graph). Here we instead LEARN it: a GAT message-passes across a 6-node
asset graph (BTC, ETH, SOL, BNB, AVAX, DOGE) each day, so stress on one asset
can propagate to correlated neighbours (contagion) before predicting each
asset's crash.

  nodes   : the 6 portfolio assets
  edges   : asset-asset return correlation (static, |corr| > 0.3) + self-loops
  features: [per-asset LLM crash_prob, daily return, 10d volatility, F&G fear]
  target  : per-asset forward-3d crash (12% threshold)

Trained walk-forward (train on the past, predict the future). Compared against
the raw per-asset LLM signal to test whether learned relational propagation
adds value. Local, CPU. Run:  python -m trr.gnn
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from trr.labels import asset_crash_labels, build_portfolio
from trr.schema import PORTFOLIO

torch.manual_seed(0)
OUT_DIR = "reports"


def build_tensors():
    pa = pd.read_csv("kaggle/out_perasset_32b/trr_predictions.csv", index_col=0)
    pa.index = pd.to_datetime(pa.index).date
    port = build_portfolio(); port.index = pd.to_datetime(port.index).date
    lab = asset_crash_labels(threshold=0.12); lab.index = pd.to_datetime(lab.index).date
    fng = json.load(open("data/fng.json"))["data"]
    fser = pd.Series({datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date(): int(x["value"])
                      for x in fng})

    idx = [d for d in pa.index if d in lab.index]
    rets = port[[f"{t}_ret" for t in PORTFOLIO]]
    vol = rets.rolling(10).std().fillna(0.0)
    fear = (100 - fser.reindex(idx).ffill()).fillna(50).values / 100.0

    # Node features per day: [pa_prob, ret, vol, fng] -> X [D, 6, 4]
    X, Y = [], []
    for d in idx:
        feats = []
        for t in PORTFOLIO:
            feats.append([
                float(pa.loc[d, f"crash_prob_{t}"]),
                float(port.loc[d, f"{t}_ret"]) if d in port.index else 0.0,
                float(vol.loc[d, f"{t}_ret"]) if d in vol.index else 0.0,
            ])
        X.append(feats)
        Y.append([int(lab.loc[d, f"{t}_crash"]) for t in PORTFOLIO])
    X = np.array(X, dtype=np.float32)                       # [D, 6, 3]
    # broadcast global fear as a 4th feature
    X = np.concatenate([X, np.repeat(fear[:, None, None], 6, axis=1).astype(np.float32)], axis=2)
    Y = np.array(Y, dtype=np.float32)                       # [D, 6]

    # Static adjacency = |return correlation| > 0.3, plus self-loops.
    corr = rets.reindex(idx).corr().to_numpy()
    A = (np.abs(corr) > 0.3).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    A = A / A.sum(1, keepdims=True)                         # row-normalize
    return X, Y, A, idx


class GAT(nn.Module):
    """Tiny 2-layer graph-attention net over the 6-asset graph."""

    def __init__(self, in_dim: int, hid: int = 16):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid)
        self.a1 = nn.Linear(2 * hid, 1)
        self.w2 = nn.Linear(hid, hid)
        self.a2 = nn.Linear(2 * hid, 1)
        self.out = nn.Linear(hid, 1)
        self.act = nn.ELU()

    def _attn(self, h, a_lin, mask):
        # h: [B, N, H]; produce attention-weighted aggregation over neighbours.
        B, N, H = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, H)
        hj = h.unsqueeze(1).expand(B, N, N, H)
        e = a_lin(torch.cat([hi, hj], -1)).squeeze(-1)       # [B, N, N]
        e = torch.where(mask.bool(), e, torch.full_like(e, -1e9))
        alpha = torch.softmax(e, dim=-1)
        return torch.bmm(alpha, h)                           # [B, N, H]

    def forward(self, x, A):
        B = x.shape[0]
        mask = A.unsqueeze(0).expand(B, -1, -1)
        h = self.act(self.w1(x))
        h = self.act(self._attn(h, self.a1, mask))
        h = self.act(self.w2(h))
        h = self.act(self._attn(h, self.a2, mask))
        return self.out(h).squeeze(-1)                       # [B, N] logits


def walk_forward_train(X, Y, A, n_splits=4, epochs=300):
    D = len(X)
    A_t = torch.tensor(A)
    start = D // (n_splits + 1)
    oof = np.full(Y.shape, np.nan)
    for k in range(1, n_splits + 1):
        tr_end = start * k
        te_end = start * (k + 1) if k < n_splits else D
        Xtr = torch.tensor(X[:tr_end]); Ytr = torch.tensor(Y[:tr_end])
        Xte = torch.tensor(X[tr_end:te_end])
        # standardize features on train
        mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        model = GAT(X.shape[2])
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        pos_w = torch.tensor((Ytr.numel() - Ytr.sum()) / (Ytr.sum() + 1))
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(model(Xtr, A_t), Ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            oof[tr_end:te_end] = torch.sigmoid(model(Xte, A_t)).numpy()
    return oof


def main():
    X, Y, A, idx = build_tensors()
    print(f"=== GAT on asset-relational graph | {len(X)} days, 6 assets ===")
    print(f"    node features: [pa_prob, ret, vol, fng] | edges: |corr|>0.3")
    oof = walk_forward_train(X, Y, A)
    mask = ~np.isnan(oof[:, 0])
    pa = pd.read_csv("kaggle/out_perasset_32b/trr_predictions.csv", index_col=0)
    pa.index = pd.to_datetime(pa.index).date
    raw = np.array([[float(pa.loc[d, f"crash_prob_{t}"]) for t in PORTFOLIO] for d in idx])

    print(f"\n    {'asset':6s} {'crashes':>8s} {'rawLLM':>7s} {'GAT':>7s}")
    gat_aus, raw_aus = [], []
    for j, t in enumerate(PORTFOLIO):
        y = Y[mask, j]
        if y.sum() < 3 or y.sum() == len(y):
            print(f"    {t:6s} {int(y.sum()):>8d}  (too few)")
            continue
        a_raw = roc_auc_score(y, raw[mask, j]); a_gat = roc_auc_score(y, oof[mask, j])
        raw_aus.append(a_raw); gat_aus.append(a_gat)
        print(f"    {t:6s} {int(y.sum()):>8d} {a_raw:>7.3f} {a_gat:>7.3f}")
    print(f"    {'MACRO':6s} {'':>8s} {np.mean(raw_aus):>7.3f} {np.mean(gat_aus):>7.3f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"macro_raw_llm": float(np.mean(raw_aus)), "macro_gat": float(np.mean(gat_aus)),
               "n_days": int(mask.sum())}, open(f"{OUT_DIR}/analysis_gnn.json", "w"), indent=2)
    print(f"\n[saved] {OUT_DIR}/analysis_gnn.json")


if __name__ == "__main__":
    main()
