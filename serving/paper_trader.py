"""Paper trading on the live crash-risk signal — pure, testable simulation.

The strategy is deliberately simple and honest: hold the equal-weight crypto
portfolio normally, but DE-RISK (move to cash) whenever the model's crash
probability is at or above a threshold. The point is not to be a great trader —
it is to show the crash signal has economic value: de-risking ahead of crashes
should cut the maximum drawdown versus naive buy-and-hold.

`simulate()` is a pure function over two aligned pandas Series (signal and the
NEXT-period forward returns), so it is trivially unit-testable on synthetic data
with no I/O. `run_live()` is the streaming driver that polls new crash-signal
rows from the Parquet store and updates a paper position; it is documented and
self-contained but does not require Kafka.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _sharpe(returns: np.ndarray, periods_per_year: int = 365) -> float:
    """Annualized Sharpe ratio of a per-period return series (0 if flat)."""
    if returns.size < 2:
        return 0.0
    sd = returns.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / sd)


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a NEGATIVE number
    (e.g. -0.35 == a 35% drawdown). 0.0 for a non-decreasing curve."""
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min())


def simulate(
    signal: pd.Series,
    fwd_returns: pd.Series,
    threshold: float = 0.5,
    cost_bps: float = 10.0,
) -> dict:
    """Simulate de-risking on a crash signal versus buy-and-hold.

    Parameters
    ----------
    signal : pd.Series
        Crash probability per period (each value in [0, 1]), indexed by time.
    fwd_returns : pd.Series
        The portfolio return realised over the period FOLLOWING each signal
        observation (i.e. already shifted so signal[t] decides exposure to
        fwd_returns[t]). Indices are aligned (inner-joined) with `signal`.
    threshold : float
        De-risk (go to cash, position 0) when signal >= threshold; else hold
        (position 1). Decision uses only information available at time t.
    cost_bps : float
        Round-trip-agnostic transaction cost in basis points charged on each
        change in position (entering or leaving the market).

    Returns
    -------
    dict with: equity_curve (pd.Series), positions (pd.Series),
        total_return, sharpe, max_drawdown, n_trades, and the buy&hold
        comparison block under "buy_hold" (total_return, sharpe, max_drawdown).
        de-risk metrics also surfaced as buy_hold-relative deltas.
    """
    df = pd.concat(
        {"signal": signal.astype(float), "fwd_ret": fwd_returns.astype(float)},
        axis=1,
    ).dropna()
    idx = df.index

    if df.empty:
        empty = pd.Series(dtype=float)
        return {
            "equity_curve": empty, "positions": empty,
            "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
            "n_trades": 0,
            "buy_hold": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                         "equity_curve": empty},
            "drawdown_improvement": 0.0,
        }

    # Position: 1 when holding, 0 when de-risked. Decision uses signal[t] only.
    positions = (df["signal"] < threshold).astype(float)

    # Transaction cost charged whenever the position changes (start flat at 0).
    prev = positions.shift(1).fillna(0.0)
    turnover = (positions - prev).abs()
    cost = turnover * (cost_bps / 1e4)

    strat_ret = positions * df["fwd_ret"] - cost
    bh_ret = df["fwd_ret"]

    strat_equity = (1.0 + strat_ret).cumprod()
    bh_equity = (1.0 + bh_ret).cumprod()

    strat_equity = pd.Series(strat_equity.values, index=idx, name="de_risk")
    bh_equity = pd.Series(bh_equity.values, index=idx, name="buy_hold")
    positions = pd.Series(positions.values, index=idx, name="position")

    return {
        "equity_curve": strat_equity,
        "positions": positions,
        "total_return": float(strat_equity.iloc[-1] - 1.0),
        "sharpe": _sharpe(strat_ret.values),
        "max_drawdown": _max_drawdown(strat_equity.values),
        "n_trades": int(turnover.sum()),
        "buy_hold": {
            "total_return": float(bh_equity.iloc[-1] - 1.0),
            "sharpe": _sharpe(bh_ret.values),
            "max_drawdown": _max_drawdown(bh_equity.values),
            "equity_curve": bh_equity,
        },
        # Positive == de-risk had a SHALLOWER drawdown than buy&hold.
        "drawdown_improvement": _max_drawdown(strat_equity.values)
        - _max_drawdown(bh_equity.values),
    }


def run_live(
    store_reader,
    return_reader,
    threshold: float = 0.5,
    cost_bps: float = 10.0,
    poll_seconds: float = 30.0,
    max_iterations: int | None = None,
) -> dict:
    """Streaming paper-trading driver (documented; no Kafka required).

    Loops polling for new crash-signal rows and updates a paper position. This
    is intentionally decoupled from the transport: `store_reader()` returns the
    newest crash-risk probability (e.g. from `serving.api.read_latest_signal`,
    which reads the Parquet store written by processing/consumer_trr.py), and
    `return_reader()` returns the portfolio return realised since the previous
    poll. Both are plain callables, so the loop is testable and transport-free.

    On each tick:
        1. Realise the prior period: equity *= (1 + position * realised_return).
        2. Read the latest crash probability.
        3. Set the new target position (de-risk if prob >= threshold), charging
           cost_bps on any change.

    Returns the final paper-account state. `max_iterations` bounds the loop for
    tests / finite runs; if None it runs forever (real live serving).
    """
    equity = 1.0
    position = 0.0  # start flat (out of market) until the first signal
    n_trades = 0
    i = 0
    while max_iterations is None or i < max_iterations:
        realised = return_reader()
        if realised is not None:
            equity *= 1.0 + position * float(realised)

        prob = store_reader()
        if prob is not None:
            target = 0.0 if float(prob) >= threshold else 1.0
            if target != position:
                equity *= 1.0 - cost_bps / 1e4  # pay cost on the position change
                position = target
                n_trades += 1

        i += 1
        if max_iterations is None:
            time.sleep(poll_seconds)  # pragma: no cover - real live cadence
    return {"equity": equity, "position": position, "n_trades": n_trades,
            "iterations": i}
