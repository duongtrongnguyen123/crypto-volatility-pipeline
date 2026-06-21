"""Daily advisory via the 32B on Kaggle (the validated quality model).

Kaggle kernels have NO internet, so this orchestrates the round-trip:
  1. LOCAL: fetch recent prices (yfinance) + current headlines -> a staging dir
  2. version a private Kaggle dataset with them
  3. push a kernel that runs the 32B TRR pipeline over the recent window..today
  4. poll; download trr_predictions.csv; take the LAST row (= today) and write
     data/live/daily_report.json (same shape the webapp reads) tagged 32B-Kaggle.

Daily cadence makes the ~20-min 32B run feasible (it never was for minute-level).
Run:  .venv/bin/python -m scripts.daily_kaggle        # uses ~/.kaggle (acct1)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

KG = ".venv/bin/kaggle"
TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
STAGE = "kaggle/daily_stage"
PUSH = "kaggle/daily_push"
USER = os.environ.get("KAGGLE_USER", "nguyenduongtrong")
DATASET = f"{USER}/stock-live-bundle"
SLUG = f"{USER}/daily-advisory-32b"
WINDOW_DAYS = 45  # recent calendar window staged + processed


def _sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def stage_data():
    import pandas as pd
    import yfinance as yf
    from webapp.live import fetch_live_headlines
    os.makedirs(STAGE, exist_ok=True)
    start = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS * 3)).strftime("%Y-%m-%d")
    for t in TICKERS:
        df = yf.download(t, start=start, progress=False, auto_adjust=True)
        close = df["Close"]; close = close[t] if hasattr(close, "columns") else close
        # top-level files (the kernel globs **/AAPL.csv) — avoids subdir/zip upload issues
        pd.DataFrame({"date": pd.to_datetime(df.index).strftime("%Y-%m-%d"),
                      "close": close.to_numpy().ravel()}).to_csv(
            f"{STAGE}/{t}.csv", index=False)
    heads = fetch_live_headlines(max_per=12)
    rows = [{"date": h.timestamp.strftime("%Y-%m-%d"), "title": h.title,
             "assets": h.assets[0], "source": "yfinance"} for h in heads]
    pd.DataFrame(rows).to_csv(f"{STAGE}/stocknews.csv", index=False)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    wstart = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    print(f"[daily-kaggle] staged {len(rows)} headlines + prices; window {wstart}..{end}")
    return wstart, end


def push_dataset():
    meta = {"title": "Stock live bundle", "id": DATASET,
            "licenses": [{"name": "other"}], "isPrivate": True}
    json.dump(meta, open(f"{STAGE}/dataset-metadata.json", "w"))
    # upload top-level files individually (no zip/subdir); version, else create
    r = _sh(KG, "datasets", "version", "-p", STAGE, "-m", "daily")
    if any(s in (r.stderr + r.stdout).lower() for s in ("not found", "404", "does not exist")):
        _sh(KG, "datasets", "create", "-p", STAGE)
    for _ in range(20):  # wait until the price files are actually queryable
        time.sleep(10)
        files = _sh(KG, "datasets", "files", DATASET).stdout
        if "AAPL.csv" in files and "stocknews" in files:
            print("[daily-kaggle] dataset ready (AAPL.csv + stocknews present)")
            return True
    print("[daily-kaggle] WARN: dataset files not confirmed; proceeding")
    return True


def push_kernel(wstart, end):
    os.makedirs(PUSH, exist_ok=True)
    src = open("kaggle/stock_standalone.py").read()
    import re
    src = re.sub(r'DEFAULT_START, DEFAULT_END = .*',
                 f'DEFAULT_START, DEFAULT_END = "{wstart}", "{end}"', src)
    src = src.replace('"crash,direction"', '"crash"')
    open(f"{PUSH}/daily_advisory.py", "w").write(src)
    meta = {"id": SLUG, "title": "daily-advisory-32b", "code_file": "daily_advisory.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": False,
            "machine_shape": "NvidiaRtxPro6000",
            "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"],
            "dataset_sources": [DATASET],
            "model_sources": ["qwen-lm/qwen2.5/transformers/32b-instruct/1"]}
    json.dump(meta, open(f"{PUSH}/kernel-metadata.json", "w"))
    print("[daily-kaggle]", _sh(KG, "kernels", "push", "-p", PUSH).stdout.strip()[-80:])


def poll_and_fetch():
    out = "kaggle/out_daily"
    for i in range(80):
        time.sleep(60)
        st = _sh(KG, "kernels", "status", SLUG).stdout
        if "COMPLETE" in st:
            os.makedirs(out, exist_ok=True)
            _sh(KG, "kernels", "output", SLUG, "-p", out)
            return _to_report(out)
        if "ERROR" in st or "CANCEL" in st:
            print("[daily-kaggle] kernel failed:", st[:120]); return None
        print(f"[daily-kaggle] poll {i}: running…")
    return None


def _to_report(out):
    import pandas as pd
    from webapp.live import compose_advisory
    p = f"{out}/crash/trr_predictions.csv"
    if not os.path.exists(p):
        print("[daily-kaggle] no predictions"); return None
    d = pd.read_csv(p, index_col=0)
    last = d.iloc[-1]
    sig = {"crash_prob": float(last["crash_prob"]), "edges": [],
           "rationale": str(last.get("rationale", "")),
           "backend": "Qwen2.5-32B",
           "asof": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    adv = compose_advisory(sig)
    adv["n_headlines"] = int(last.get("n_news", 0))
    adv["as_of_day"] = str(d.index[-1])
    json.dump(adv, open("data/live/daily_report.json", "w"), indent=2)
    print(f"[daily-kaggle] WROTE daily_report.json — risk {adv['risk_level']} "
          f"crash_prob {adv['crash_prob']:.0%} (day {adv['as_of_day']})")
    return adv


def main():
    import sys
    wstart, end = stage_data()
    push_dataset()
    push_kernel(wstart, end)
    adv = poll_and_fetch()
    sys.exit(0 if adv else 1)  # non-zero lets the cron fall back to local 7B


if __name__ == "__main__":
    main()
