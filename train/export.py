"""Fit the TRR meta-learner on ALL available data and persist it for serving.

Produces models/trr_meta.pkl = {model, features, metadata}. The serving layer
loads this to turn a day's (LLM crash_prob + news counts + price technicals)
into a calibrated crash probability — the training -> serving handoff.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np

from train.features import FEATURES_FULL, build_dataset
from train.run import _gbm

OUT = "models/trr_meta.pkl"


def main():
    df = build_dataset()
    X, y = df[FEATURES_FULL], df["label_true"].to_numpy()
    model = _gbm().fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES_FULL,
                 "n_train": int(len(df)), "n_crash": int(y.sum()),
                 "trained_on": f"{df.index.min()}..{df.index.max()}"}, OUT)
    # reload + sanity check
    art = joblib.load(OUT)
    p = art["model"].predict_proba(X[:5])[:, 1]
    print(f"[export] saved {OUT}  features={len(art['features'])} "
          f"trained_on={art['trained_on']} n={art['n_train']} crashes={art['n_crash']}")
    print(f"[export] reload OK, sample probs: {np.round(p,3).tolist()}")


if __name__ == "__main__":
    main()
