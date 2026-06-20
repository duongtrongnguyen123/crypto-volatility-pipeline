"""Assemble a SELF-CONTAINED Kaggle kernel that runs TRR on EQUITIES.

Same inlined trr/ package as build_standalone.py, plus trr/prices.py (the
asset-agnostic daily loader), with a stock MAIN that runs either the CRASH or
the DIRECTION target (TARGET_MODE env) over the 6 large-cap portfolio, and
evaluates against daily stock labels.

Attaches:
  - a stock-data dataset: prices/{TICKER}.csv + stocknews.csv
  - a HuggingFace model (Qwen2.5-32B-Instruct) -> the reasoner
Output: kaggle/stock_standalone.py
"""
import os

from build_standalone import HEADER, MODULES, strip_module  # reuse the inliner

# prices.py is appended after labels.py so crash_labels_daily/direction_labels_daily
# are inlined too.
STOCK_MODULES = MODULES + ["prices"]

STOCK_MAIN = '''

# =========================================================================== #
# STOCK kernel orchestration (equities; CRASH or DIRECTION target).
# =========================================================================== #
KAGGLE_WORKING = "/kaggle/working"
SMOKE_OUT_DIR = "/tmp/trr_stock_smoke_out"
STOCK_TICKERS = ["AAPL", "AMZN", "GOOGL", "NVDA", "TSLA", "NFLX"]
# COVID-crash window: the news corpus (2019-06..2020-06) and the price labels.
DEFAULT_START, DEFAULT_END = "2019-06-03", "2020-06-10"
SMOKE_START, SMOKE_END = "2020-02-20", "2020-03-20"
# Both targets run in one GPU session (model load is the cost) unless overridden.
TARGET_MODES = os.environ.get("TARGET_MODES", "crash,direction").split(",")
MAX_ITEMS_PER_DAY = 20
GEN_BATCH_SIZE = 8
MAX_INPUT_TOKENS = 2048
LAM = 0.6
TOP_K = 30
REASON_MAXTOK = 256
BRAINSTORM_MAXTOK = 768


def _is_smoke():
    return os.environ.get("SMOKE", "0") == "1"


def _glob1(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    return None


def _gpu_gate():
    import torch
    if not torch.cuda.is_available():
        print("[gpu] CUDA not available (CPU/SMOKE).", flush=True)
        return "float32"
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[gpu] {torch.cuda.get_device_name(0)}  sm_{major}{minor}  torch={torch.__version__}", flush=True)
    if (major, minor) == (6, 0):
        print("[gpu] FATAL: P100/sm_60 fallback — the RTX 6000 Pro gate failed.", flush=True)
        sys.exit(1)
    return "bfloat16" if major >= 8 else "float16"


def _find_model_dir():
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(cfg)
        if any(os.path.exists(os.path.join(d, t)) for t in
               ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
            return d
    return None


def _save_outputs(pred_df, metrics, out_dir, score_col):
    os.makedirs(out_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(out_dir, "trr_predictions.csv"))
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(pred_df)), pred_df[score_col], label=score_col)
        for i, c in enumerate(pred_df["label_true"].to_numpy()):
            if c == 1:
                ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15)
        ax.set_title(f"TRR {score_col} vs actual positive days (shaded) — stocks")
        ax.legend()
        fig.savefig(os.path.join(out_dir, "trr_timeline.png"), dpi=120, bbox_inches="tight")
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)


def _evaluate(pred_df, price_dir, out_dir, target_mode):
    from sklearn.metrics import average_precision_score, roc_auc_score

    if target_mode == "direction":
        lab = direction_labels_daily(price_dir, STOCK_TICKERS)
        truth, score_col = "up", "up_prob"
    else:
        lab = crash_labels_daily(price_dir, STOCK_TICKERS)
        truth, score_col = "crash", "crash_prob"

    s = lab[truth].copy()
    s.index = pd.to_datetime(s.index).date
    df = pred_df.copy()
    df["label_true"] = [int(s.get(d, 0)) for d in df.index]

    y = df["label_true"].to_numpy()
    metrics = {"summary": {"target": target_mode, "n_days": int(len(df)),
                           "n_pos_days": int(y.sum()), "base_rate": float(y.mean()),
                           "date_start": str(df.index.min()),
                           "date_end": str(df.index.max())},
               "metrics": {}}
    if 0 < y.sum() < len(y):
        metrics["metrics"]["TRR"] = {
            "auroc": float(roc_auc_score(y, df[score_col])),
            "pr_auc": float(average_precision_score(y, df[score_col])),
        }
        if "n_news" in df:
            metrics["metrics"]["news_volume"] = {
                "auroc": float(roc_auc_score(y, df["n_news"])),
            }
    else:
        metrics["summary"]["single_class_window"] = True
    _save_outputs(df, metrics, out_dir, score_col)
    return metrics


def main():
    smoke = _is_smoke()
    print(f"[kernel] STOCK BUILD={BUILD_TAG} targets={TARGET_MODES} "
          f"mode={'SMOKE' if smoke else 'KAGGLE'}", flush=True)

    price1 = _glob1("/kaggle/input/**/prices/AAPL.csv", "/kaggle/input/**/AAPL.csv",
                    "data/stockdata/prices/AAPL.csv")
    if not price1:
        print("[kernel] FATAL: AAPL.csv not found", flush=True); sys.exit(1)
    price_dir = os.path.dirname(price1)
    print(f"[kernel] price_dir={price_dir}", flush=True)

    news_csv = _glob1("/kaggle/input/**/stocknews.csv", "data/stockdata/stocknews.csv")
    print(f"[kernel] news={news_csv}", flush=True)
    news = load_news(news_csv)

    start = os.environ.get("TRR_START", SMOKE_START if smoke else DEFAULT_START)
    end = os.environ.get("TRR_END", SMOKE_END if smoke else DEFAULT_END)
    out_dir = SMOKE_OUT_DIR if smoke else (KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else "/tmp")

    dtype = _gpu_gate()
    if smoke:
        llm = MockLLM()
    else:
        model_dir = _find_model_dir()
        print(f"[kernel] model dir: {model_dir}", flush=True)
        llm = HFReasoningLLM(model_path=model_dir, dtype=dtype,
                             batch_size=GEN_BATCH_SIZE, max_input_tokens=MAX_INPUT_TOKENS)

    print(f"[kernel] window {start}..{end}  news_items={len(news)}", flush=True)
    by_day = group_by_day(news)
    all_metrics = {}
    for tmode in TARGET_MODES:
        tmode = tmode.strip()
        if not tmode:
            continue
        print(f"[kernel] === running target={tmode} ===", flush=True)
        pipe = TRRPipeline(llm=llm, portfolio=STOCK_TICKERS, target_mode=tmode,
                           batch=True, cross_batch=True, max_items_per_day=MAX_ITEMS_PER_DAY,
                           lam=LAM, top_k=TOP_K, reason_max_new_tokens=REASON_MAXTOK,
                           brainstorm_max_new_tokens=BRAINSTORM_MAXTOK)
        pred = pipe.run(by_day, start=start, end=end)
        print(f"[kernel] target={tmode} predicted {len(pred)} days", flush=True)
        mode_dir = os.path.join(out_dir, tmode)
        metrics = _evaluate(pred, price_dir, mode_dir, tmode)
        all_metrics[tmode] = metrics
        print(f"[kernel] target={tmode} metrics: {json.dumps(metrics.get('metrics', {}))}", flush=True)
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"[kernel] wrote outputs -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def build() -> str:
    parts = [HEADER.replace("standalone-v4-32b-fewshot", "stock-v1-32b")]
    for m in STOCK_MODULES:
        src = open(os.path.join("trr", f"{m}.py")).read()
        parts.append(f"\n# ===================== trr/{m}.py =====================\n")
        parts.append(strip_module(src))
    parts.append(STOCK_MAIN)
    return "\n".join(parts)


if __name__ == "__main__":
    code = build()
    out = os.path.join("kaggle", "stock_standalone.py")
    with open(out, "w") as f:
        f.write(code)
    print(f"wrote {out} ({len(code)} bytes, {code.count(chr(10))} lines)")
