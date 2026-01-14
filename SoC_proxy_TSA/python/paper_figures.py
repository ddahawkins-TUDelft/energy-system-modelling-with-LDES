#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot LDES capacity error for a chosen (W_proxy, number_reps) across horizons,
with NL points on the left and BE points on the right.

Key design choices (fixes your “only 10y shows” issue):
- Treat the CEM cache as the source of truth for horizon / W_proxy / number_reps.
- Merge cache fields into log rows and always filter/colour using *_use columns.
- Print a small coverage debug summary so you can immediately see whether other
  horizons exist for the selected (W, k).

Two plotters included:
  1) strip plot (jittered points)
  2) boxplot (optionally with point overlay)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import matplotlib as mpl

# --- Optional: only needed if you want the script to auto-build/refresh cache ---
import json
import yaml
from netCDF4 import Dataset
import calliope

from helper_signals import build_signal_metrics

# ------------------------- Paths & constants ---------------------------------

SAVE_FIGURES = True
OUT_DIR = Path(".")

COUNTRIES = ["NL", "BE"]  # plotting order: left -> right

LOGS_F123_BY_CC = {
    cc: [Path(f"SoC_proxy_TSA/data/notes/log_{cc}_rerun.csv")]
    for cc in COUNTRIES
}

MODELS_DIR = Path("SoC_proxy_TSA/data/calliope_models")
PARAM_DIR  = Path("SoC_proxy_TSA/data/parameters")

CACHE_CEM = Path("SoC_proxy_TSA/data/notes/cem_cache.csv")

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig_width = 5.0
fig_height = 2.8


# ------------------------- I/O helpers ---------------------------------------

def savefig(fig, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.show()
    plt.close(fig)


# ------------------------- Log parsing (lightweight) --------------------------

def _parse_single_log(log_path: Path) -> pd.DataFrame:
    """
    Parse the rerun log. IMPORTANT: horizon in the logs may be unreliable, so we
    will *not* depend on it. Still parse it for fallback/debug.
    """
    df = pd.read_csv(log_path)

    required_cols = {"id", "dates", "date_range", "model_name", "tvp"}
    missing = required_cols - set(df.columns)
    for col in missing:
        df[col] = np.nan

    df = df.copy()

    # Often unreliable: keep it but don't trust it
    df["horizon"] = pd.to_numeric(df.get("date_range", np.nan), errors="coerce")

    reps = df["model_name"].astype(str).str.extract(r"reps\s*=\s*(\d+)", expand=False)
    reps_k = df["model_name"].astype(str).str.extract(r"k\s*=\s*(\d+)", expand=False)
    df["number_reps"] = pd.to_numeric(reps.fillna(reps_k), errors="coerce")

    w_direct = df["model_name"].astype(str).str.extract(
        r"W[_ ]?proxy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False
    )
    L_ab = df["model_name"].astype(str).str.extract(
        r"L\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)/([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        expand=True
    )
    w_from_L = pd.Series(np.nan, index=df.index, dtype=float)
    if L_ab.notna().any().any():
        a = pd.to_numeric(L_ab[0], errors="coerce")
        b = pd.to_numeric(L_ab[1], errors="coerce")
        equal_mask = a.notna() & b.notna() & (np.isclose(a, b))
        w_from_L.loc[equal_mask] = a.loc[equal_mask]

    df["W_proxy"] = pd.to_numeric(w_direct, errors="coerce").fillna(w_from_L)

    keep = ["id", "dates", "horizon", "number_reps", "W_proxy", "model_name", "runtime", "tvp"]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    return df[keep]


def parse_logs(log_paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in log_paths:
        if p.exists():
            frames.append(_parse_single_log(p))
        else:
            print(f"[parse_logs] Missing log: {p}")
    if not frames:
        return pd.DataFrame(columns=["id", "dates", "horizon", "number_reps", "W_proxy", "model_name", "runtime", "tvp"])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df


# ------------------------- Cache building (copied from your script) ----------

def resolve_reference_nc(dates: str, tvp: Optional[str], models_dir: Path) -> Optional[Path]:
    years = re.findall(r"\d{4}", str(dates))
    if len(years) < 2:
        return None
    y0, y1 = years[0], years[1]

    candidates = []
    tvp_str = str(tvp).strip() if pd.notna(tvp) else ""
    if tvp_str and tvp_str.lower() not in {"", "nan", "none"}:
        stem = Path(tvp_str).stem
        parts = stem.split("_")
        cc = None
        if len(parts) > 1 and len(parts[-1]) == 2:
            cc = parts[-1]

        candidates += [
            models_dir / f"standard_{y0}_{y1}_reference_{cc}.nc" if cc else models_dir / f"standard_{y0}_{y1}_reference.nc",
            models_dir / f"{tvp_str}.nc",
        ]

    candidates.append(models_dir / f"standard_{y0}_{y1}_reference.nc")

    for c in candidates:
        if c.exists():
            return c
    return None


def read_clustered_netcdf(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise Exception(f"File does not exist at: {path}")
    with Dataset(p, "a") as nc:
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)
        g.setncattr("config", yaml.safe_dump(cfg))
    return calliope.read_netcdf(path)


def get_capacities(m: calliope.Model):
    df_power_caps = (
        m.results["flow_cap"].fillna(0).to_series().dropna().to_frame("capacity").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )

    mask_drop = (
        df_power_caps["techs"].isin(["battery", "h2_salt_cavern", "demand"])
        | df_power_caps["techs"].str.startswith("demand")
    )
    df_power_caps = df_power_caps[
        ~mask_drop
        & (
            ((df_power_caps["techs"] == "electrolyser") & (df_power_caps["carriers"] == "hydrogen"))
            | ((df_power_caps["techs"] != "electrolyser") & (df_power_caps["carriers"] == "power"))
        )
    ]
    df_power_caps.set_index("techs", inplace=True)

    df_energy_caps = (
        m.results["storage_cap"].fillna(0).to_series().dropna().to_frame("capacity").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    df_energy_caps = df_energy_caps[df_energy_caps["capacity"] > 0]
    df_energy_caps.set_index("techs", inplace=True)

    return df_power_caps["capacity"], df_energy_caps["capacity"]


def relative_error(df_ref, df_test):
    e = (df_ref - df_test) / df_ref
    e_mean_abs = np.mean(np.abs(e))
    return e, e_mean_abs


def build_or_load_cem_cache(df_needed: pd.DataFrame, models_dir: Path, cache_path: Path) -> pd.DataFrame:
    """
    Builds/extends cache if needed. IMPORTANT:
    It *writes* row.get("horizon") / row.get("W_proxy") / row.get("number_reps") into cache.
    So: if your df_needed doesn't contain correct horizons, cache horizon can be wrong.
    """
    if cache_path.exists():
        df_cache = pd.read_csv(cache_path)
    else:
        df_cache = pd.DataFrame(columns=[
            "id", "horizon", "number_reps", "W_proxy",
            "ldes_error", "macme", "abs_ldes_error",
            "proxy_cem_pearson", "proxy_cem_nrmse"
        ])

    cached_ids = set(df_cache["id"]) if not df_cache.empty else set()
    ref_cache: Dict[Tuple[str, str], calliope.Model] = {}
    rows = []

    num_reads = len(df_needed)
    counter = 0

    for _, row in df_needed.iterrows():
        mid = row["id"]
        counter += 1
        print(f"Reading {mid} for capacities, {counter}/{num_reads}")

        if pd.isna(mid) or (mid in cached_ids):
            continue

        test_nc = models_dir / f"{mid}.nc"
        if not test_nc.exists():
            continue

        dates = str(row.get("dates", ""))
        tvp   = row.get("tvp", np.nan)

        ref_key = (dates, str(tvp))
        if ref_key not in ref_cache:
            ref_nc = resolve_reference_nc(dates, tvp, models_dir)
            if ref_nc is None:
                continue
            try:
                ref_cache[ref_key] = calliope.read_netcdf(str(ref_nc))
            except Exception:
                continue

        model_reference = ref_cache[ref_key]

        try:
            model_test = read_clustered_netcdf(test_nc)
        except Exception:
            continue

        try:
            power_caps_reference, energy_caps_reference = get_capacities(model_reference)
            power_caps_test,      energy_caps_test      = get_capacities(model_test)
            pearson_r, nrmse = build_signal_metrics(tvp, mid)
        except Exception:
            continue

        _, e_mean_power = relative_error(power_caps_reference, power_caps_test)

        e_storage, _ = relative_error(energy_caps_reference, energy_caps_test)
        ldes_err = e_storage.get("h2_salt_cavern", np.nan)

        rows.append({
            "id": mid,
            "horizon": row.get("horizon", np.nan),
            "number_reps": row.get("number_reps", np.nan),
            "W_proxy": row.get("W_proxy", np.nan),
            "ldes_error": ldes_err,
            "macme": e_mean_power,
            "proxy_cem_pearson": pearson_r,
            "proxy_cem_nrmse": nrmse,
            "abs_ldes_error": abs(ldes_err) if pd.notna(ldes_err) else np.nan
        })

    if rows:
        df_cache = pd.concat([df_cache, pd.DataFrame(rows)], ignore_index=True)

    df_cache.drop_duplicates(subset=["id"], keep="first", inplace=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_cache.to_csv(cache_path, index=False)
    return df_cache


# ------------------------- Data prep for plotting ----------------------------

def load_logs_with_cache(debug: bool = True) -> pd.DataFrame:
    """
    1) parse NL+BE logs
    2) build/refresh cache for those IDs
    3) merge cache fields into logs
    4) create *_use columns (cache-first)
    """
    df_logs_by_cc = {cc: parse_logs(LOGS_F123_BY_CC[cc]).assign(country=cc) for cc in COUNTRIES}
    df_logs = pd.concat(list(df_logs_by_cc.values()), ignore_index=True)
    if df_logs.empty:
        raise RuntimeError("No log rows found.")

    df_cache = build_or_load_cem_cache(df_logs, MODELS_DIR, CACHE_CEM)

    # Merge cache fields
    df = df_logs.merge(
        df_cache[["id", "ldes_error", "horizon", "number_reps", "W_proxy"]],
        on="id",
        how="left",
        suffixes=("_log", "_cache"),
    )

    # cache-first usable columns
    df["horizon_use"] = df["horizon_cache"].fillna(df["horizon_log"])
    df["number_reps_use"] = df["number_reps_cache"].fillna(df["number_reps_log"])
    df["W_proxy_use"] = df["W_proxy_cache"].fillna(df["W_proxy_log"])

    # Horizon int
    df["horizon_int"] = pd.to_numeric(df["horizon_use"], errors="coerce").round().astype("Int64")

    if debug:
        tmp = df.dropna(subset=["W_proxy_use", "number_reps_use", "horizon_int"])
        if not tmp.empty:
            print("\n=== DEBUG: overall coverage (cache-first) ===")
            print("Unique horizons:", sorted(tmp["horizon_int"].dropna().unique().tolist()))
            print("Counts by country x horizon:")
            print(tmp.groupby(["country", "horizon_int"]).size())
        else:
            print("\n=== DEBUG: coverage is empty after basic dropna. Check cache/log fields. ===")

    return df


def filter_for_params(df: pd.DataFrame, weight: float, number_reps: int, horizons: Sequence[int], debug: bool = True) -> pd.DataFrame:
    horizons = [int(h) for h in horizons]

    d = df.copy()
    d = d.dropna(subset=["W_proxy_use", "number_reps_use", "horizon_int", "ldes_error"])

    d = d[np.isclose(d["W_proxy_use"].astype(float), float(weight), atol=1e-9)]
    d = d[d["number_reps_use"].astype(int) == int(number_reps)]

    if debug:
        print("\n=== DEBUG: coverage for chosen (W,k) BEFORE horizon filter ===")
        if d.empty:
            print("No rows for this (W,k) at all.")
        else:
            print("Unique horizons for (W,k):", sorted(d["horizon_int"].unique().tolist()))
            print(d.groupby(["country", "horizon_int"]).size())

    d = d[d["horizon_int"].isin(horizons)]

    if debug:
        missing = sorted(set(horizons) - set(d["horizon_int"].unique().tolist()))
        if missing:
            print(f"[warn] Missing horizons for this (W,k): {missing}")

    return d


# ------------------------- Plot 1: strip (jittered) --------------------------

def plot_ldes_error_strip(
    df_all: pd.DataFrame,
    weight: float,
    number_reps: int,
    horizons: Sequence[int],
    out_path: Path,
    jitter: float = 0.12,
    seed: int = 42,
    debug: bool = True,
) -> None:
    df = filter_for_params(df_all, weight, number_reps, horizons, debug=debug)

    if df.empty:
        print("[plot_strip] No data after filtering.")
        return

    unique_h = sorted(df["horizon_int"].dropna().unique().tolist())
    cmap = plt.get_cmap("plasma")
    col_lut = {h: cmap(i / max(len(unique_h) - 1, 1)) for i, h in enumerate(unique_h)}

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_base = {"NL": 0.0, "BE": 1.0}
    rng = np.random.default_rng(seed)

    for cc in COUNTRIES:
        sub_cc = df[df["country"] == cc]
        if sub_cc.empty:
            continue

        for h in unique_h:
            sub = sub_cc[sub_cc["horizon_int"] == h]
            if sub.empty:
                continue

            x0 = x_base[cc]
            xs = x0 + rng.normal(0.0, jitter, size=len(sub))
            ys = sub["ldes_error"].astype(float).values

            ax.scatter(
                xs, ys,
                s=30,
                marker="o",
                linewidths=0.3,
                edgecolors="black",
                alpha=0.9,
                c=[col_lut[h]] * len(sub),
                zorder=3,
            )

    ax.axhline(0, color="black", linewidth=1, zorder=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NL", "BE"])
    ax.set_ylabel("LDES capacity error")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title_h = ", ".join(str(h) for h in unique_h)
    ax.set_title(f"LDES capacity error | W={weight:g}, reps={number_reps}, horizon={title_h}y")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=col_lut[h], markeredgecolor="black",
               label=f"{h}y", markersize=6)
        for h in unique_h
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best", title="Horizon")

    savefig(fig, out_path, tight=True)


# ------------------------- Plot 2: box (with optional points) ----------------

def plot_ldes_error_box(
    df_all: pd.DataFrame,
    weight: float,
    number_reps: int,
    horizons: Sequence[int],
    out_path: Path,
    show_points: bool = True,
    jitter: float = 0.06,
    seed: int = 42,
    debug: bool = True,
    whiskers: str = "iqr",  # "iqr" or "minmax"
) -> None:
    df = filter_for_params(df_all, weight, number_reps, horizons, debug=debug)

    if df.empty:
        print("[plot_box] No data after filtering.")
        return

    unique_h = sorted(df["horizon_int"].dropna().unique().tolist())
    cmap = plt.get_cmap("plasma")
    col_lut = {h: cmap(i / max(len(unique_h) - 1, 1)) for i, h in enumerate(unique_h)}

    x_base = {"NL": 0.0, "BE": 1.0}

    if len(unique_h) == 1:
        h_offsets = {unique_h[0]: 0.0}
        box_width = 0.45
    else:
        spread = 0.55
        offsets = np.linspace(-spread / 2, spread / 2, len(unique_h))
        h_offsets = {h: offsets[i] for i, h in enumerate(unique_h)}
        box_width = min(0.18, 0.8 / len(unique_h))

    boxes = []
    positions = []
    facecols = []

    for cc in COUNTRIES:
        for h in unique_h:
            vals = df[(df["country"] == cc) & (df["horizon_int"] == h)]["ldes_error"].astype(float).values
            if len(vals) == 0:
                continue
            boxes.append(vals)
            positions.append(x_base[cc] + h_offsets[h])
            facecols.append(col_lut[h])

    if not boxes:
        print("[plot_box] Nothing to plot (no non-empty groups).")
        return

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    whis = 1.5
    if whiskers == "minmax":
        whis = (0, 100)

    bp = ax.boxplot(
        boxes,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        whis=whis,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        boxprops=dict(color="black", linewidth=0.8),
    )

    for patch, fc in zip(bp["boxes"], facecols):
        patch.set_facecolor(fc)
        patch.set_alpha(0.45)

    if show_points:
        rng = np.random.default_rng(seed)
        for cc in COUNTRIES:
            for h in unique_h:
                sub = df[(df["country"] == cc) & (df["horizon_int"] == h)]
                if sub.empty:
                    continue
                x0 = x_base[cc] + h_offsets[h]
                xs = x0 + rng.normal(0.0, jitter, size=len(sub))
                ys = sub["ldes_error"].astype(float).values
                ax.scatter(
                    xs, ys,
                    s=18,
                    marker="o",
                    linewidths=0.25,
                    edgecolors="black",
                    alpha=0.75,
                    c=[col_lut[h]] * len(sub),
                    zorder=3,
                )

    ax.axhline(0, color="black", linewidth=1, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NL", "BE"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("LDES capacity error")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title_h = ", ".join(str(h) for h in unique_h)
    ax.set_title(f"LDES capacity error | W={weight:g}, reps={number_reps}, horizon={title_h}y")

    if len(unique_h) > 1:
        legend_handles = [
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=col_lut[h], markeredgecolor="black",
                   label=f"{h}y", markersize=8, alpha=0.6)
            for h in unique_h
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="best", title="Horizon")

    savefig(fig, out_path, tight=True)


# ------------------------- Main ----------------------------------------------

def main():
    # Choose parameters
    W_PROXY = 0.5
    NUMBER_REPS = 45
    HORIZONS = [2, 5, 10]

    # Output
    OUT_STRIP = OUT_DIR / f"ldes_error_strip_W{W_PROXY:g}_k{NUMBER_REPS}_H{'-'.join(map(str,HORIZONS))}.pdf"
    OUT_BOX   = OUT_DIR / f"ldes_error_box_W{W_PROXY:g}_k{NUMBER_REPS}_H{'-'.join(map(str,HORIZONS))}.pdf"

    # Load once
    df_all = load_logs_with_cache(debug=True)

    # Plot
    plot_ldes_error_strip(df_all, W_PROXY, NUMBER_REPS, HORIZONS, OUT_STRIP, debug=True)
    # plot_ldes_error_box(df_all, W_PROXY, NUMBER_REPS, HORIZONS, OUT_BOX, show_points=True, whiskers="iqr", debug=True)

if __name__ == "__main__":
    main()
