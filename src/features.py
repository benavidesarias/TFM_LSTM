"""Shared feature-engineering for the LSTM trend-classification pipeline.

These helpers are the single source of truth used by both:
- ``notebooks/03_preprocess_data.ipynb`` (builds the train/val/test datasets), and
- ``notebooks/05_inference.ipynb`` (builds live per-ticker recommendation windows),

so the features fed to the model at inference time are derived exactly the same
way as the features it was trained on.

Design notes
------------
- Each sector ticker produces the same ordered feature vector: a few generic
  per-ticker trend features followed by the shared macro features. This lets the
  per-ticker windows stack into one dataset and lets a single model serve every
  ticker.
- The ``Date`` index is NEVER turned into a feature; it is only used for
  alignment/windowing and for labelling the outputs. The trained model is
  therefore time-agnostic and can be reused at any later date.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's exponential smoothing.

    Returns a 0-100 oscillator. The first ``window`` rows are ``NaN`` (warmup)
    and are dropped downstream together with the other feature warmups.
    """
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # A flat/zero-loss window yields RS = inf -> RSI = 100.
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    return rsi


def resolve_tickers(
    metadata: Mapping, columns: Iterable[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Resolve which sector/macro tickers from the metadata are present in ``columns``.

    Returns ``(sector_tickers, macro_tickers, meta_sector_tickers, meta_macro_tickers)``
    where the first two are filtered to tickers that actually have an
    ``<TICKER>_Adj Close`` column, and the last two are the full lists declared in
    the metadata (useful for reporting what is missing).
    """
    columns = set(columns)
    meta_sector = [t for tk in metadata["sectors"].values() for t in tk.keys()]
    meta_macro = list(metadata.get("macro", {}).keys())
    sector = [t for t in meta_sector if f"{t}_Adj Close" in columns]
    macro = [t for t in meta_macro if f"{t}_Adj Close" in columns]
    return sector, macro, meta_sector, meta_macro


def _normalize_macro_reps(reps: str | Sequence[str]) -> list[str]:
    """Coerce a single representation or a list into a normalized list."""
    if isinstance(reps, str):
        return [reps]
    return list(reps)


def _macro_column_name(macro_ticker: str, rep: str) -> str:
    """Stable feature name for a macro ticker + representation token."""
    if rep == "level":
        suffix = "level"
    elif rep == "return":
        suffix = "ret"
    elif rep == "logret_5d":
        suffix = "logret5"
    elif rep == "diff_1d":
        suffix = "diff1"
    elif rep == "diff_5d":
        suffix = "diff5"
    elif rep == "zscore":
        suffix = "zscore"
    elif rep == "log_ma_ratio":
        suffix = "log_ma_ratio"
    elif rep.startswith("logret_") and rep.endswith("d"):
        suffix = f"logret{rep[7:-1]}"
    else:
        raise ValueError(
            f"Unknown macro representation '{rep}' for {macro_ticker} "
            "(use 'level', 'return', 'logret_{n}d', 'diff_5d', 'zscore', "
            "or 'log_ma_ratio')"
        )
    return f"macro::{macro_ticker}::{suffix}"


def build_macro_features(
    df: pd.DataFrame,
    macro_features_active: Mapping[str, str | Sequence[str]],
    macro_windows: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Build the shared macro feature columns (identical for every ticker).

    ``macro_features_active`` maps each macro ticker to one representation or a
    list of representations. Supported tokens:

    - ``"level"``        -> raw Adj Close (non-stationary),
    - ``"return"``       -> 1-day pct change,
    - ``"logret_{n}d"``  -> n-day log return ``log(series).diff(n)``,
    - ``"diff_5d"``      -> 5-day level change (yield points for ^TNX),
    - ``"zscore"``       -> trailing z-score ``(x - MA) / std`` (VIX regime),
    - ``"log_ma_ratio"`` -> ``log(x / MA)`` (parallel to equity MA features).

    ``macro_windows`` optionally supplies ``zscore_window`` and ``ma_window``
    (default 20) used by ``zscore`` and ``log_ma_ratio``.
    """
    windows = macro_windows or {}
    zscore_window = int(windows.get("zscore_window", 20))
    ma_window = int(windows.get("ma_window", 20))

    cols = {}
    for mt, reps in macro_features_active.items():
        series = df[f"{mt}_Adj Close"]
        for rep in _normalize_macro_reps(reps):
            col_name = _macro_column_name(mt, rep)
            if rep == "return":
                cols[col_name] = series.pct_change(1)
            elif rep == "level":
                cols[col_name] = series
            elif rep == "logret_5d":
                cols[col_name] = np.log(series).diff(5)
            elif rep.startswith("logret_") and rep.endswith("d"):
                n = int(rep[7:-1])
                cols[col_name] = np.log(series).diff(n)
            elif rep == "diff_1d":
                cols[col_name] = series.diff(1)
            elif rep == "diff_5d":
                cols[col_name] = series.diff(5)
            elif rep == "zscore":
                ma = series.rolling(zscore_window).mean()
                std = series.rolling(zscore_window).std()
                cols[col_name] = (series - ma) / std.replace(0, np.nan)
            elif rep == "log_ma_ratio":
                ma = series.rolling(ma_window).mean()
                cols[col_name] = np.log(series / ma)
            else:
                raise ValueError(
                    f"Unknown macro representation '{rep}' for {mt} "
                    "(use 'level', 'return', 'logret_{n}d', 'diff_5d', "
                    "'zscore', or 'log_ma_ratio')"
                )
    return pd.DataFrame(cols, index=df.index)


def build_ticker_features(
    df: pd.DataFrame,
    ticker: str,
    feature_flags: Mapping[str, bool],
    return_windows: Sequence[int],
    ma_windows: Sequence[int],
    rsi_window: int,
    vol_ma_window: int,
    has_volume: bool,
    vol_window: int = 20,
    horizon: int = 10,
) -> pd.DataFrame:
    """Build the per-ticker engineered (trend) features for one sector ticker.

    Mirrors the reference pipeline's per-ticker features:
    - log returns over each window in ``return_windows`` (``logret_{n}d``),
    - log price/MA deviation for each window in ``ma_windows``
      (``log_price_ma{n}`` = ``log(price / MA_n)``),
    - ``rsi_{rsi_window}`` (Wilder RSI),
    - ``log_vol_ratio`` (``log1p(volume) - log1p(MA_{vol_ma_window}(volume))``)
      when volume is available, and
    - ``realized_vol`` (``std(1d log returns, vol_window) * sqrt(horizon)``):
      trailing realized volatility scaled to the prediction horizon, the same
      quantity used to volatility-scale the labels (no look-ahead).
    """
    price = df[f"{ticker}_Adj Close"]
    log_price = np.log(price)
    cols = {}
    if feature_flags.get("use_log_returns", True):
        for n in return_windows:
            cols[f"logret_{n}d"] = log_price - log_price.shift(n)
    if feature_flags.get("use_ma_ratios", True):
        for n in ma_windows:
            ma = price.rolling(n).mean()
            cols[f"log_price_ma{n}"] = np.log(price / ma)
    if feature_flags.get("use_rsi", True):
        cols[f"rsi_{rsi_window}"] = compute_rsi(price, rsi_window)
    if feature_flags.get("use_volume_ratio", True) and has_volume:
        vol_col = f"{ticker}_Volume"
        if vol_col in df.columns:
            vol = df[vol_col]
            vol_ma = vol.rolling(vol_ma_window).mean()
            cols["log_vol_ratio"] = np.log1p(vol) - np.log1p(vol_ma)
    if feature_flags.get("use_realized_vol", True):
        daily_logret = log_price.diff()
        cols["realized_vol"] = daily_logret.rolling(vol_window).std() * np.sqrt(horizon)
    return pd.DataFrame(cols, index=df.index)


def build_ticker_frame(
    df: pd.DataFrame,
    ticker: str,
    macro_df: pd.DataFrame,
    feature_flags: Mapping[str, bool],
    return_windows: Sequence[int],
    ma_windows: Sequence[int],
    rsi_window: int,
    vol_ma_window: int,
    has_volume: bool,
    vol_window: int = 20,
    horizon: int = 10,
) -> pd.DataFrame:
    """Per-ticker features concatenated with the shared macro features (ordered)."""
    feats = build_ticker_features(
        df,
        ticker,
        feature_flags,
        return_windows,
        ma_windows,
        rsi_window,
        vol_ma_window,
        has_volume,
        vol_window=vol_window,
        horizon=horizon,
    )
    return pd.concat([feats, macro_df], axis=1)


def build_labels(
    df: pd.DataFrame,
    ticker: str,
    horizon: int,
    ordered_classes: Sequence[str],
    edges: Sequence[float],
    class_map: Mapping[str, int],
    log_return: bool = True,
    mode: str = "absolute",
    vol_window: int = 20,
    cost: float = 0.0,
) -> pd.Series:
    """Next-``horizon``-day forward return binned into ordinal class codes.

    The label for day ``t`` uses the forward return
    ``r = AdjClose[t+horizon] / AdjClose[t] - 1`` (log return when
    ``log_return`` is True). The value placed into the bins depends on ``mode``:

    - ``"absolute"`` bins the raw forward return ``r`` against ``edges`` given in
      return space (the original, fixed per-stock thresholds).
    - ``"vol_scaled"`` bins the volatility-normalized return ``z = r / sigma_H``
      where ``sigma_H = std(daily_log_ret, vol_window) * sqrt(horizon)`` is the
      stock's own trailing volatility scaled to the horizon. ``edges`` are then
      expressed in units of sigma, so a class means the same risk-adjusted move
      for every ticker. The trailing volatility uses only returns up to and
      including day ``t`` (no look-ahead).

    ``cost`` is an optional transaction-cost buffer (in return units) that
    shrinks the forward return toward zero before binning
    (``r -> sign(r) * max(|r| - cost, 0)``). This is equivalent to widening
    every threshold by ``cost`` on each side, so a move must beat both
    volatility and round-trip cost to count as a signal (matching the reference
    pipeline's ``... +/- COST_BPS`` thresholds).

    ``r`` (or ``z``) is placed into one of ``len(edges) + 1`` bins defined by the
    ascending ``edges`` and mapped through ``ordered_classes`` (ordered
    most-bearish -> most-bullish) and ``class_map`` to an integer code. This
    handles any class count (e.g. 2 edges -> 3 classes, 4 edges -> 5 classes).
    Days without a defined forward return (the final ``horizon`` rows) - and,
    under ``"vol_scaled"``, days whose trailing volatility is undefined (warmup)
    or zero (flat) - are left as ``NaN``.
    """
    if len(ordered_classes) != len(edges) + 1:
        raise ValueError(
            f"ordered_classes ({len(ordered_classes)}) must have exactly "
            f"len(edges)+1 ({len(edges) + 1}) entries"
        )
    price = df[f"{ticker}_Adj Close"]
    if log_return:
        fwd_ret = np.log(price.shift(-horizon)) - np.log(price)
    else:
        fwd_ret = price.shift(-horizon) / price - 1.0

    if cost:
        # Shrink the forward return toward zero by the round-trip cost; moves
        # smaller than `cost` collapse to 0 (-> the neutral/HOLD bin).
        fwd_ret = np.sign(fwd_ret) * (fwd_ret.abs() - cost).clip(lower=0.0)

    if mode == "vol_scaled":
        # Trailing daily-return volatility, scaled to the horizon. Uses only
        # past returns (ending at t), so it does not leak future information.
        daily_logret = np.log(price).diff()
        sigma_daily = daily_logret.rolling(vol_window).std()
        sigma_h = sigma_daily * np.sqrt(horizon)
        # Zero/NaN sigma -> undefined z -> dropped as warmup downstream.
        sigma_h = sigma_h.replace(0.0, np.nan)
        binned = fwd_ret / sigma_h
    elif mode == "absolute":
        binned = fwd_ret
    else:
        raise ValueError(f"Unknown label mode '{mode}' (use 'absolute' or 'vol_scaled')")

    label = pd.Series(np.full(len(price), np.nan), index=price.index, dtype="float64")
    valid = binned.notna()
    # np.digitize returns the bin index (0..len(edges)) for each value.
    bins = np.digitize(binned[valid].to_numpy(), np.asarray(edges, dtype="float64"))
    codes = np.array([class_map[ordered_classes[b]] for b in bins], dtype="float64")
    label[valid] = codes
    return label


def make_windows(
    scaled: np.ndarray,
    labels: np.ndarray,
    dates: Sequence,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a ``window``-length window over consecutive rows.

    For each window the label and ``as_of`` date are taken from the LAST row in
    the window (the most recent input day, i.e. the day being classified). The
    date array is returned purely for alignment/reporting and is never part of X.
    """
    n = len(scaled)
    n_features = scaled.shape[1] if scaled.ndim == 2 else 0
    if n < window:
        return (
            np.empty((0, window, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
        )
    xs, ys, ds = [], [], []
    for i in range(n - window + 1):
        xs.append(scaled[i : i + window])
        ys.append(labels[i + window - 1])
        ds.append(dates[i + window - 1])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ds, dtype=object),
    )
