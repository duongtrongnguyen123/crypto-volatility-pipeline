"""Alternative prediction targets derived from the price portfolio.

The TRR pipeline was built around a single ground truth — the 3-day portfolio
*crash* (`trr.labels.crash_labels`). The assignment, however, is titled "...for
Stock Price Prediction", so this module adds the more on-point price targets and
keeps them all in one place:

    direction_labels(horizon)  -> price UP/DOWN over the next `horizon` days
    return_target(horizon)     -> forward `horizon`-day portfolio return (regr.)
    vol_regime_labels(horizon) -> next-window volatility tercile {0,1,2}
    severity_target(horizon)   -> forward max-drawdown magnitude (regression)

All targets are built from `trr.labels.build_portfolio`, are indexed by day, and
are strictly FORWARD-looking: each value at day *t* describes the window AFTER
*t* (a `shift(-h)`), so there is no lookahead. The undefined tail (where the
forward window runs past the data) is dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trr.labels import build_portfolio


def _portfolio(hist_dir: str = None) -> pd.DataFrame:
    """Equal-weight portfolio frame (cached-free thin wrapper for typing)."""
    return build_portfolio(hist_dir)


def _forward_return(level: pd.Series, horizon: int) -> pd.Series:
    """Forward `horizon`-day simple return: level[t+h] / level[t] - 1.

    Defined for days that have a full `horizon`-day window ahead; the tail is
    NaN and dropped by the public builders.
    """
    return level.shift(-horizon) / level - 1.0


def direction_labels(horizon: int = 1, hist_dir: str = None) -> pd.DataFrame:
    """Price up/down target.

    Returns a frame indexed by day with:
        fwd_ret : forward `horizon`-day portfolio return.
        up      : 1 if `fwd_ret` > 0 (price rises), else 0.

    This is the literal "will the price go up?" target. The tail without a full
    forward window is dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    level = _portfolio(hist_dir)["portfolio_level"]
    fwd = _forward_return(level, horizon)
    out = pd.DataFrame({"fwd_ret": fwd})
    out["up"] = (out["fwd_ret"] > 0).astype(int)
    return out.dropna(subset=["fwd_ret"]).copy()


def return_target(horizon: int = 1, hist_dir: str = None) -> pd.Series:
    """Forward `horizon`-day portfolio return as a regression target.

    Series indexed by day; the undefined tail is dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    level = _portfolio(hist_dir)["portfolio_level"]
    fwd = _forward_return(level, horizon)
    return fwd.dropna().rename("fwd_ret")


def vol_regime_labels(
    horizon: int = 1, window: int = 5, hist_dir: str = None
) -> pd.DataFrame:
    """Next-window volatility regime as a 3-class target {0,1,2}.

    Volatility of the upcoming window is the rolling standard deviation of the
    daily portfolio return over the next `window` days. Days are bucketed into
    low/med/high terciles (0/1/2) by the empirical 33rd/67th percentiles of that
    forward volatility.

    Returns a frame indexed by day with:
        fwd_vol : forward `window`-day return volatility (std), shifted by
                  `horizon` so day *t* sees the window starting `horizon` days
                  ahead.
        regime  : tercile class in {0,1,2} (0=low, 1=med, 2=high).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if window < 2:
        raise ValueError("window must be >= 2 for a std estimate")
    ret = _portfolio(hist_dir)["portfolio_ret"]
    # Forward rolling std of the NEXT `window` returns: reverse-roll then re-flip,
    # then shift so the window begins `horizon` days after today.
    fwd_vol = (
        ret.iloc[::-1].rolling(window, min_periods=window).std().iloc[::-1]
    ).shift(-horizon)
    fwd_vol = fwd_vol.dropna()
    if fwd_vol.empty:
        return pd.DataFrame({"fwd_vol": fwd_vol, "regime": fwd_vol.astype(int)})
    lo, hi = np.quantile(fwd_vol, [1 / 3, 2 / 3])
    regime = np.where(fwd_vol <= lo, 0, np.where(fwd_vol <= hi, 1, 2))
    out = pd.DataFrame({"fwd_vol": fwd_vol})
    out["regime"] = regime.astype(int)
    return out


def severity_target(horizon: int = 3, hist_dir: str = None) -> pd.Series:
    """Forward max-drawdown MAGNITUDE over the next `horizon` days (regression).

    For each day *t*, the deepest trough of the portfolio level reached within
    the next `horizon` days relative to today's level, returned as a positive
    magnitude (0 = no drawdown, 0.10 = a 10% trough). This is the continuous
    analogue of the binary crash label and the natural "how bad" severity score.

    Series indexed by day; the undefined tail is dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    level = _portfolio(hist_dir)["portfolio_level"]
    # Lowest close within the NEXT `horizon` days (excluding today).
    fwd_low = (
        level.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    )
    drawdown = fwd_low / level - 1.0          # <= 0
    severity = (-drawdown).clip(lower=0.0)    # positive magnitude
    severity = severity.where(fwd_low.notna())
    return severity.dropna().rename("fwd_drawdown")


if __name__ == "__main__":
    for h in (1, 3):
        d = direction_labels(h)
        print(f"[targets] direction h={h}: {len(d)} days, "
              f"up-rate {d['up'].mean():.1%}")
    r = return_target(1)
    print(f"[targets] return h=1: mean {r.mean():+.4f}, std {r.std():.4f}")
    v = vol_regime_labels(1)
    print(f"[targets] vol-regime h=1: {len(v)} days, "
          f"class counts {v['regime'].value_counts().sort_index().to_dict()}")
    s = severity_target(3)
    print(f"[targets] severity h=3: mean {s.mean():.3f}, "
          f"max {s.max():.3f}, p90 {s.quantile(0.9):.3f}")
