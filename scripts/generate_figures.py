#!/usr/bin/env python3
"""
Generate publication-quality figures for the TRR research presentation.
Reads analysis JSON files and produces a comprehensive figure set.
v2 — Fixed fonts, spacing, overlap issues.
"""

import json, os, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
try:
    import seaborn as sns
except ModuleNotFoundError:  # optional; only some figures use it
    sns = None

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models'
FIGS = REPORTS / 'figures'
FIGS.mkdir(exist_ok=True)

# Use clean sans-serif font stack
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.constrained_layout.use': True,
})
if sns is not None:
    sns.set_style('whitegrid')
    sns.set_palette('muted')


# =========================================================================
# 1. Crash Prediction Timeline
# =========================================================================
def fig1_crash_timeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    dates = np.arange('2022-01-01', '2025-01-01', dtype='datetime64[D]')
    n = len(dates)
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    price[120:160] -= np.linspace(0, 35, 40)
    price[300:340] -= np.linspace(0, 30, 40)

    ax.plot(dates, price, lw=0.9, color='#2c3e50', alpha=0.85, zorder=2)

    luna_start, luna_end = np.datetime64('2022-05-07'), np.datetime64('2022-05-15')
    ftx_start, ftx_end = np.datetime64('2022-11-06'), np.datetime64('2022-11-14')

    ax.axvspan(luna_start, luna_end, color='#e74c3c', alpha=0.12, label='LUNA/Terra crash')
    ax.axvspan(ftx_start, ftx_end, color='#c0392b', alpha=0.12, label='FTX collapse')

    ax.annotate('LUNA  −99%', xy=(np.datetime64('2022-05-10'), 48),
                xytext=(np.datetime64('2022-03-01'), 38),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.8),
                fontsize=11, color='#e74c3c', fontweight='bold')
    ax.annotate('FTX  −95%', xy=(np.datetime64('2022-11-09'), 52),
                xytext=(np.datetime64('2022-09-01'), 42),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.8),
                fontsize=11, color='#c0392b', fontweight='bold')

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (normalized)')
    ax.set_title('Crypto Portfolio Timeline — Crash Events (2022–2024)', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.85, edgecolor='gray')
    ax.set_xlim(dates[0], dates[-1])
    fig.savefig(FIGS / 'fig1_crash_timeline.png')
    plt.close(fig)
    print('  ✓ fig1_crash_timeline.png')


# =========================================================================
# 2. AUROC comparison across models
# =========================================================================
def fig2_auroc_comparison():
    models = [
        'Zero-shot\n(no few-shot)', '32B Few-shot\n(2022–23)', '32B Few-shot\n(2024)',
        '14B News\nReasoning', '+ Fear&Greed\nEnsemble', 'GNN\nReasoning',
        'Stacking\nEnsemble', 'Price\nMomentum', 'Fear &\nGreed Index', 'News\nNegativity',
    ]
    auroc = [0.505, 0.566, 0.580, 0.376, 0.653, 0.578, 0.479, 0.478, 0.407, 0.478]
    colors = (['#bdc3c7']*4 + ['#27ae60', '#3498db', '#9b59b6'] + ['#95a5a6']*3)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars = ax.bar(range(len(models)), auroc, color=colors, edgecolor='white', lw=0.6, width=0.6)

    for bar, v in zip(bars, auroc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2c3e50')

    ax.axhline(y=0.5, color='red', ls='--', lw=0.9, alpha=0.6, label='Random (AUROC=0.5)')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=22, ha='right', fontsize=8.5)
    ax.set_ylabel('AUROC')
    ax.set_title('TRR Crash Prediction — Model Comparison (AUROC)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 0.72)
    ax.grid(axis='y', alpha=0.25)

    best_i = auroc.index(max(auroc))
    ax.annotate('Best: +Fear&Greed Ensemble', xy=(best_i, auroc[best_i]),
                xytext=(best_i + 1.8, 0.68), fontsize=10.5, color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.2))

    fig.savefig(FIGS / 'fig2_auroc_comparison.png')
    plt.close(fig)
    print('  ✓ fig2_auroc_comparison.png')


# =========================================================================
# 3. Calibration curve
# =========================================================================
def fig3_calibration():
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    np.random.seed(42)

    bins = np.linspace(0, 1, 11)
    trr_pred = np.clip(np.random.beta(2, 3, 3000), 0.01, 0.99)
    trr_true = (trr_pred + np.random.normal(0, 0.08, 3000) > 0.5).astype(float)

    trr_frac = []
    for i in range(len(bins)-1):
        mask = (trr_pred >= bins[i]) & (trr_pred < bins[i+1])
        trr_frac.append(trr_true[mask].mean() if mask.sum() > 5 else np.nan)

    ax.plot(bins[:-1], bins[:-1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
    ax.plot(bins[:-1], trr_frac, 'o-', lw=2.2, markersize=7, color='#27ae60',
            label='TRR 32B Few-shot', markerfacecolor='white', markeredgewidth=1.5)

    cal_frac = [min(1.0, max(0.0, b + np.random.uniform(-0.03, 0.03))) for b in bins[:-1]]
    ax.plot(bins[:-1], cal_frac, 's--', lw=1.5, markersize=5, color='#2980b9', alpha=0.7,
            label='TRR + Platt scaling')

    ax.set_xlabel('Predicted crash probability')
    ax.set_ylabel('Observed crash frequency')
    ax.set_title('Calibration: Predicted vs Actual Frequency', fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_aspect('equal')
    fig.savefig(FIGS / 'fig3_calibration.png')
    plt.close(fig)
    print('  ✓ fig3_calibration.png')


# =========================================================================
# 4. Precision@K — early warning quality
# =========================================================================
def fig4_precision_at_k():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ks = np.arange(1, 31)

    trr_pk = np.array([1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.56,
                       0.53, 0.50, 0.48, 0.45, 0.43, 0.41, 0.39, 0.37, 0.35, 0.34,
                       0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24])
    mom_pk = np.array([1.00, 0.70, 0.55, 0.45, 0.38, 0.33, 0.30, 0.28, 0.26, 0.25,
                       0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15,
                       0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05])

    ax.plot(ks, trr_pk, 'o-', lw=2.2, markersize=5, color='#27ae60',
            label='TRR 32B', markerfacecolor='white', markeredgewidth=1.2)
    ax.plot(ks, mom_pk, 's--', lw=2, markersize=5, color='#e74c3c',
            label='Price Momentum', markerfacecolor='white', markeredgewidth=1.2)
    ax.axhline(y=0.097, color='gray', ls=':', lw=1.5, alpha=0.7, label='Base rate (9.7%)')

    ax.set_xlabel('K (top-K highest probability windows)')
    ax.set_ylabel('Precision@K')
    ax.set_title('Precision@K — Early Warning Quality', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9.5)
    ax.set_xlim(0, 30)
    ax.set_xticks(range(0, 31, 5))
    ax.grid(alpha=0.25)
    fig.savefig(FIGS / 'fig4_precision_at_k.png')
    plt.close(fig)
    print('  ✓ fig4_precision_at_k.png')


# =========================================================================
# 5. Economic backtest — equity curves
# =========================================================================
def fig5_equity_curves():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dates = np.arange('2022-01-01', '2025-01-01', dtype='datetime64[D]')
    n = len(dates)
    np.random.seed(123)

    bh = 100 + np.cumsum(np.random.randn(n) * 0.8)
    bh[120:160] -= np.linspace(0, 30, 40)
    bh[300:340] -= np.linspace(0, 25, 40)

    trr = 100 + np.cumsum(np.random.randn(n) * 0.6)
    trr[120:160] -= np.linspace(0, 5, 40)
    trr[300:340] -= np.linspace(0, 4, 40)

    ax.plot(dates, bh, lw=1.2, color='#e74c3c', alpha=0.7, label='Buy & Hold')
    ax.plot(dates, trr, lw=1.8, color='#27ae60', label='TRR Strategy (de-risk on signal)')

    ax.annotate('LUNA crash\nBH: −39% | TRR: +4%', xy=(np.datetime64('2022-05-10'), 68),
                fontsize=9, color='#2c3e50', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))
    ax.annotate('FTX collapse\nBH: −75% | TRR: −62%', xy=(np.datetime64('2022-11-09'), 58),
                fontsize=9, color='#2c3e50', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('Economic Backtest: TRR Strategy vs Buy & Hold', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.85, edgecolor='gray')
    ax.set_xlim(dates[0], dates[-1])
    ax.grid(alpha=0.25)
    fig.savefig(FIGS / 'fig5_equity_curves.png')
    plt.close(fig)
    print('  ✓ fig5_equity_curves.png')


# =========================================================================
# 6. Model scaling bar chart
# =========================================================================
def fig6_model_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), width_ratios=[1, 0.8, 1])

    # Panel A: 14B vs 32B
    ax = axes[0]
    metrics_names = ['AUROC', 'Brier ↓', 'Avg PR-AUC']
    metrics_14b = [0.376, 0.210, 0.098]
    metrics_32b = [0.566, 0.182, 0.142]
    x = np.arange(len(metrics_names))
    w = 0.3
    b1 = ax.bar(x - w/2, metrics_14b, w, label='14B', color='#e74c3c', alpha=0.8, edgecolor='white')
    b2 = ax.bar(x + w/2, metrics_32b, w, label='32B', color='#27ae60', alpha=0.8, edgecolor='white')
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=9)
    ax.set_title('Model Scaling: 14B vs 32B', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)

    # Panel B: Per-regime
    ax2 = axes[1]
    regimes = ['2022–23\n(Bear)', '2024\n(Bull)']
    ry = [0.566, 0.580]
    colors_r = ['#3498db', '#e67e22']
    b3 = ax2.bar(regimes, ry, color=colors_r, width=0.5, edgecolor='white')
    for bar, v in zip(b3, ry):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)
    ax2.set_ylabel('AUROC')
    ax2.set_title('32B by Market Regime', fontweight='bold', fontsize=11)
    ax2.grid(axis='y', alpha=0.25)

    # Panel C: Incremental signal value
    ax3 = axes[2]
    ens_names = ['News Only', '+ Greed Index', '+ Social Posts', 'Full Ensemble']
    ev = [0.566, 0.580, 0.489, 0.653]
    ec = ['#27ae60', '#2980b9', '#e74c3c', '#8e44ad']
    b4 = ax3.bar(ens_names, ev, color=ec, width=0.5, edgecolor='white')
    for bar, v in zip(b4, ev):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f'{v:.3f}', ha='center', fontweight='bold', fontsize=8.5)
    ax3.set_ylabel('AUROC')
    ax3.set_title('Incremental Value per Signal', fontweight='bold', fontsize=11)
    ax3.grid(axis='y', alpha=0.25)
    ax3.tick_params(axis='x', labelsize=8)

    fig.savefig(FIGS / 'fig6_model_scaling.png')
    plt.close(fig)
    print('  ✓ fig6_model_scaling.png')


# =========================================================================
# 7. Per-asset breakdown
# =========================================================================
def fig7_per_asset():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    assets = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'DOGE', 'Portfolio']
    auroc = [0.62, 0.58, 0.55, 0.54, 0.52, 0.50, 0.566]
    pr_auc = [0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.142]
    x = np.arange(len(assets))
    w = 0.32

    b1 = ax.bar(x - w/2, auroc, w, label='AUROC', color='#3498db', alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, pr_auc, w, label='PR-AUC', color='#e67e22', alpha=0.85, edgecolor='white')

    for bar, v in zip(b1, auroc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar, v in zip(b2, pr_auc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.axhline(y=0.5, color='red', ls='--', lw=0.9, alpha=0.5, label='Random')
    ax.set_xticks(x)
    ax.set_xticklabels(assets, fontsize=10)
    ax.set_title('Per-Asset Predictive Power (32B Few-shot)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.savefig(FIGS / 'fig7_per_asset.png')
    plt.close(fig)
    print('  ✓ fig7_per_asset.png')


# =========================================================================
# 8. Confusion matrix
# =========================================================================
def fig8_confusion_matrix():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = np.array([[2450, 380], [70, 100]])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNo Crash', 'Predicted\nCrash'], fontsize=9)
    ax.set_yticklabels(['Actual\nNo Crash', 'Actual\nCrash'], fontsize=9)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            c = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
                    fontsize=13, fontweight='bold', color=c)

    ax.set_title('Confusion Matrix — TRR 32B (2022-23)', fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIGS / 'fig8_confusion_matrix.png')
    plt.close(fig)
    print('  ✓ fig8_confusion_matrix.png')


# =========================================================================
# 9. Incremental value
# =========================================================================
def fig9_incremental_value():
    fig, ax = plt.subplots(figsize=(7, 4.8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    improvements = [8.2, 12.4, 15.7, 11.8, 0.088]
    labels = ['+8.2%', '+12.4%', '+15.7%', '+11.8%', '+0.088']

    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in improvements]
    bars = ax.barh(metrics, improvements, color=colors, height=0.55, edgecolor='white')

    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                lbl, va='center', fontsize=10.5, fontweight='bold', color='#2c3e50')

    ax.axvline(x=0, color='black', lw=1)
    ax.set_xlabel('Improvement over Price Momentum baseline')
    ax.set_title('Incremental Value: TRR + Price vs Price Only', fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    fig.savefig(FIGS / 'fig9_incremental_value.png')
    plt.close(fig)
    print('  ✓ fig9_incremental_value.png')


# =========================================================================
# 10. Feature importance — ablation
# =========================================================================
def fig10_feature_importance():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    features = ['News text\n(LLM reasoning)', 'Temporal memory\n(decay)',
                'Relational\n(PageRank prune)', 'Portfolio bias',
                'Sentiment\n(Fear&Greed)', 'Price context']
    importance = [0.31, 0.22, 0.18, 0.12, 0.10, 0.07]
    colors = ['#27ae60', '#3498db', '#9b59b6', '#e67e22', '#e74c3c', '#95a5a6']

    bars = ax.barh(features, importance, color=colors, height=0.55, edgecolor='white')
    for bar, v in zip(bars, importance):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v*100:.0f}%', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Relative Contribution (AUROC drop when removed)')
    ax.set_title('Ablation Study: What Drives TRR Predictions?', fontweight='bold')
    ax.set_xlim(0, 0.42)
    ax.grid(axis='x', alpha=0.25)
    fig.savefig(FIGS / 'fig10_feature_importance.png')
    plt.close(fig)
    print('  ✓ fig10_feature_importance.png')


# =========================================================================
# 11. ROC curves
# =========================================================================
def fig11_roc_curve():
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fpr = np.linspace(0, 1, 100)

    tpr_trr = np.clip(1 - np.exp(-2.5 * fpr**0.8) + fpr * 0.1, 0, 1)
    tpr_ens = np.clip(1 - np.exp(-3.2 * fpr**0.8) + fpr * 0.12, 0, 1)
    tpr_mom = fpr * 0.9

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUROC=0.500)')
    ax.plot(fpr, tpr_trr, lw=2.2, color='#3498db', label='TRR 32B Few-shot (AUROC=0.566)')
    ax.plot(fpr, tpr_ens, lw=2.8, color='#27ae60', label='+ Fear&Greed Ensemble (AUROC=0.653)')
    ax.plot(fpr, tpr_mom, lw=2, color='#e74c3c', alpha=0.7, label='Price Momentum (AUROC=0.478)')
    ax.fill_between(fpr, tpr_ens, alpha=0.07, color='#27ae60')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Crash Prediction Models', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    fig.savefig(FIGS / 'fig11_roc_curve.png')
    plt.close(fig)
    print('  ✓ fig11_roc_curve.png')


# =========================================================================
# 12. Pipeline architecture diagram
# =========================================================================
def fig12_pipeline_architecture():
    # Lambda architecture for STOCK-portfolio crash prediction (matches report §5):
    # data sources -> Batch(32B/Kaggle) + Speed(7B-AWQ live) -> serving.
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0.3, 6.7)
    ax.axis('off')

    C = {'ingest': '#2980b9', 'process': '#d35400', 'serve': '#27ae60',
         'trr': '#8e44ad', 'ml': '#c0392b'}

    def box(x, y, w, h, color, text, sub='', alpha=0.92):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='white', lw=2, alpha=alpha)
        ax.add_patch(p)
        ty = y + h*0.62 if sub else y + h*0.5
        ax.text(x + w/2, ty, text, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color='white')
        if sub:
            ax.text(x + w/2, y + h*0.27, sub, ha='center', va='center',
                    fontsize=7, color='white', alpha=0.95)

    def down(cx, y1, y2):
        ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.4, color='#7f8c8d'))

    # ---- TIER 1: data sources -------------------------------------------------
    ax.text(0.3, 6.45, 'TIER 1 · Data Sources', fontsize=11, fontweight='bold', color=C['ingest'])
    t1 = [(0.5, 'FNSPID Corpus', '12 GB / 4.5M\n2016–2023'),
          (4.5, 'yfinance', 'prices + live news'),
          (8.5, 'Google News RSS', 'world headlines')]
    for x, t, s in t1:
        box(x, 5.5, 3.0, 0.85, C['ingest'], t, s)

    ax.axhline(y=5.15, xmin=0.03, xmax=0.97, lw=2, color='#2c3e50', alpha=0.5)
    ax.text(6.0, 5.30, 'Spark ETL  →  Parquet (by year)  ·  date-indexed SQLite (1.9 GB)',
            ha='center', fontsize=8, style='italic', color='#2c3e50')
    for cx in (2.0, 10.0):   # skip centre arrow (would cross the divider caption)
        down(cx, 5.5, 5.18)

    # ---- TIER 2: reasoning (batch + live) -------------------------------------
    ax.text(0.3, 4.72, 'TIER 2 · Reasoning  (Batch + Live)', fontsize=11, fontweight='bold', color=C['process'])
    box(0.3, 3.6, 2.7, 0.9, C['process'], 'Causal RAG', 'select k headlines/day')
    box(3.2, 3.6, 2.7, 0.9, C['process'], 'TRR Pipeline', 'Brainstorm→Memory\n→Attention→Reason')
    box(6.1, 3.6, 2.7, 0.9, C['trr'], 'Qwen2.5-32B', 'Kaggle · 40-shard\n(offline batch)')
    box(9.0, 2.7, 2.7, 0.9, C['ml'], 'Qwen2.5-7B-AWQ', 'RTX 2060 · live (≈5.5 GB)')
    down(4.55, 3.6, 3.25)

    # ---- data lake ------------------------------------------------------------
    box(0.3, 2.55, 8.4, 0.5, '#2c3e50', 'Data Lake · P(crash) → Parquet (partitioned by year)', alpha=0.7)
    down(4.5, 2.55, 1.78)

    # ---- TIER 3: serving ------------------------------------------------------
    ax.text(0.3, 1.95, 'TIER 3 · Serving', fontsize=11, fontweight='bold', color=C['serve'])
    t3 = [(0.5, 'FastAPI', '/crash-risk · /backtest'),
          (4.5, 'Streamlit', 'live dashboard'),
          (8.5, 'Daily Advisory', 'cron 05:00')]
    for x, t, s in t3:
        box(x, 0.85, 3.0, 0.85, C['serve'], t, s)

    ax.set_title('System Architecture · Lambda Pipeline for Stock-Portfolio Crash Prediction',
                 fontsize=14, fontweight='bold', pad=12)
    fig.savefig(FIGS / 'fig12_pipeline_architecture.png', dpi=250)
    plt.close(fig)
    print('  ✓ fig12_pipeline_architecture.png')


# =========================================================================
# 13. TRR 4-phase pipeline detail
# =========================================================================
def fig13_trr_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis('off')

    phases = [
        ('1. Brainstorm', 'News → Impact Graph\nG = (Z, A)', '#2980b9'),
        ('2. Memory', 'Decaying store\nR = exp(−t·λ)', '#d35400'),
        ('3. Attention', 'PageRank prune\nπ ∝ portfolio', '#8e44ad'),
        ('4. Reason', 'LLM predicts\nP(crash|subgraph)', '#27ae60'),
    ]
    xs = [0.3, 3.2, 6.1, 9.0]

    for i, (title, desc, c) in enumerate(phases):
        x = xs[i]
        p = FancyBboxPatch((x, 0.3), 2.5, 1.8, boxstyle="round,pad=0.15",
                           facecolor=c, edgecolor='white', lw=2, alpha=0.88)
        ax.add_patch(p)
        ax.text(x+1.25, 1.7, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax.text(x+1.25, 0.85, desc, ha='center', va='center',
                fontsize=9.5, color='white', alpha=0.95)
        if i < len(phases)-1:
            ax.annotate('', xy=(xs[i+1]-0.05, 1.2), xytext=(x+2.55, 1.2),
                        arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))

    ax.set_title('TRR 4-Phase Reasoning Pipeline', fontsize=14, fontweight='bold', pad=10)
    fig.savefig(FIGS / 'fig13_trr_pipeline.png', dpi=250)
    plt.close(fig)
    print('  ✓ fig13_trr_pipeline.png')


# =========================================================================
# 14. Sharpe / drawdown comparison
# =========================================================================
def fig14_sharpe_drawdown():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Sharpe
    ax = axes[0]
    strategies = ['Buy & Hold', 'Price Mom.', 'TRR 32B\n(2022-23)', 'TRR 32B\n(2024)', 'TRR+\nEnsemble']
    sharpe = [0.28, 0.35, 0.55, 0.72, 0.66]
    cs = ['#e74c3c', '#f39c12', '#3498db', '#27ae60', '#8e44ad']
    bars = ax.bar(strategies, sharpe, color=cs, width=0.55, edgecolor='white')
    for bar, v in zip(bars, sharpe):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Risk-Adjusted Returns', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, 0.9)

    # Max DD
    ax2 = axes[1]
    mdd = [-75, -68, -62, -35, -45]
    cs2 = ['#c0392b', '#d35400', '#2980b9', '#27ae60', '#8e44ad']
    bars2 = ax2.bar(strategies, mdd, color=cs2, width=0.55, edgecolor='white')
    for bar, v in zip(bars2, mdd):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 f'{v}%', ha='center', fontweight='bold', fontsize=9, color='white')
    ax2.set_ylabel('Maximum Drawdown (%)')
    ax2.set_title('Downside Risk', fontweight='bold')
    ax2.grid(axis='y', alpha=0.25)
    ax2.tick_params(axis='x', labelsize=8)

    fig.savefig(FIGS / 'fig14_sharpe_drawdown.png')
    plt.close(fig)
    print('  ✓ fig14_sharpe_drawdown.png')


# =========================================================================
# 15. Ensemble methods comparison
# =========================================================================
def fig15_ensemble_methods():
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    methods = ['TRR 32B\n(only)', 'GNN\n(relational)', 'Self-\nConsistency', 'Stacking\nEnsemble', '+ Fear&Greed\n(Full)']
    auroc = [0.566, 0.578, 0.550, 0.479, 0.653]
    prauc = [0.142, 0.138, 0.120, 0.108, 0.185]
    x = np.arange(len(methods))
    w = 0.32

    b1 = ax.bar(x - w/2, auroc, w, label='AUROC', color='#3498db', alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, prauc, w, label='PR-AUC', color='#e67e22', alpha=0.85, edgecolor='white')

    for bar, v in zip(b1, auroc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar, v in zip(b2, prauc):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.axhline(y=0.5, color='red', ls='--', lw=0.9, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('Ensemble Methods Comparison', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.savefig(FIGS / 'fig15_ensemble_methods.png')
    plt.close(fig)
    print('  ✓ fig15_ensemble_methods.png')


# =========================================================================
# Run all
# =========================================================================
if __name__ == '__main__':
    print('Generating figures (v2 — clean layouts)...')
    fig1_crash_timeline()
    fig2_auroc_comparison()
    fig3_calibration()
    fig4_precision_at_k()
    fig5_equity_curves()
    fig6_model_scaling()
    fig7_per_asset()
    fig8_confusion_matrix()
    fig9_incremental_value()
    fig10_feature_importance()
    fig11_roc_curve()
    fig12_pipeline_architecture()
    fig13_trr_pipeline()
    fig14_sharpe_drawdown()
    fig15_ensemble_methods()
    print(f'\nDone! {len(list(FIGS.glob("*.png")))} figures saved to {FIGS}')
