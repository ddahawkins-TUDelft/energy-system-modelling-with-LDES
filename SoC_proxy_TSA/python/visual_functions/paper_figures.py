#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot figure generator for CEM results and runtimes.

Now supports multiple log files:
- FIGS 1–3 use LOGS_F123 (e.g., per-date logs like 'log_2006,2010,2015.csv')
- FIGS 4–5 use LOGS_F45 (e.g., 'log_horizon.csv')

Usage:
    python plot_cem_results.py

This script:
  - Loads all specified logs, builds a union of model IDs for caching
  - Builds/updates a lightweight cache of CEM metrics (absolute LDES error, MACME)
    so model files are read only when needed
  - Builds/updates a lightweight cache of runtimes
  - Generates all requested figures in sequence without manual tweaking

Notes:
  * Colour/marker choices are preserved from the previous code:
        MACME = '#0D0887'
        LDES  = '#CC4778'
        W=0   = grey outline '#666666' for hollow markers
  * A solid black horizontal line (linewidth=1) at y=0 is added to ALL error
    box-and-whisker charts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import yaml
from netCDF4 import Dataset
import calliope

import matplotlib as mpl



# ------------------------- Paths & constants ---------------------------------

# Adjust these if your repo layout differs
# Per your note: separate logs for figs 1–3 and 4–5:
LOGS_F123     = [Path("SoC_proxy_TSA/data/notes/log_14-180_NL_only.csv")] 
LOGS_F45      = [Path("SoC_proxy_TSA/data/notes/log_45-60-90.csv")]

MODELS_DIR      = Path("SoC_proxy_TSA/data/calliope_models")
PARAM_DIR       = Path("SoC_proxy_TSA/data/parameters")
OUT_DIR         = Path(".")

CACHE_CEM       = Path("SoC_proxy_TSA/data/notes/cem_cache.csv")
CACHE_RUNTIME   = Path("SoC_proxy_TSA/data/notes/runtime_cache.csv")

# Output filenames
FIG1_HEATMAP_10Y      = OUT_DIR / "fig1_heatmap_abs_ldes_error_10y.pdf"
FIG2_ERR_VS_REPS      = OUT_DIR / "fig2_error_box_vs_reps.pdf"
FIG3_ERR_VS_PROXY     = OUT_DIR / "fig3_error_box_vs_proxy_excl14.pdf"
FIG4_ERR_VS_HORIZON   = OUT_DIR / "fig4_error_vs_horizon_W01_reps60to90.pdf"
FIG5_RUNTIME_VS_HOR   = OUT_DIR / "fig5_runtime_vs_horizon.pdf"

# Colours (preserve existing)
COLOUR_MACME = "#0D0887"
COLOUR_LDES  = "#CC4778"
COLOUR_W0_EDGE = "#666666"   # hollow marker edge for W=0
COLOUR_GREY_MEAN = "#666666"  # general grey where needed

# Marker settings
MARKER_MACME = "o"
MARKER_LDES  = "o"

#latex
mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "axes.unicode_minus": False,
})

plt.rcParams.update({
    "font.size": 8,         
    "axes.labelsize": 8,    
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})
mpl.rcParams["pgf.preamble"] = r""

fig_width = 3.5
fig_height = 3


# ------------------------- Helpers: I/O & parsing ----------------------------

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.show()
    plt.close(fig)


def _parse_single_log(log_path: Path) -> pd.DataFrame:
    """
    Parse a single log.csv and extract: id, dates, horizon (date_range), number_reps, W_proxy.
    """
    df = pd.read_csv(log_path)

    # base columns presence
    required_cols = {"id", "dates", "date_range", "model_name", "tvp"}
    missing = required_cols - set(df.columns)
    if missing:
        # Allow some logs to be missing columns; fill with NaN
        for col in missing:
            df[col] = np.nan

    df = df.copy()

    # Horizon in years
    if "date_range" in df.columns:
        df["horizon"] = pd.to_numeric(df["date_range"], errors="coerce")
    else:
        df["horizon"] = np.nan

    # Extract number of representative periods (days)
    reps = df["model_name"].astype(str).str.extract(r"reps\s*=\s*(\d+)", expand=False)
    reps_k = df["model_name"].astype(str).str.extract(r"k\s*=\s*(\d+)", expand=False)
    df["number_reps"] = pd.to_numeric(reps.fillna(reps_k), errors="coerce")

    # Extract proxy weight (prefer explicit 'W_proxy', fallback to 'L=a/b')
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
    """
    Parse multiple logs and row-bind them (with de-duplication by id).
    """
    frames: List[pd.DataFrame] = []
    for p in log_paths:
        if p.exists():
            frames.append(_parse_single_log(p))
        else:
            print(f"[parse_logs] Missing log: {p}")
    if not frames:
        return pd.DataFrame(columns=["id", "dates", "horizon", "number_reps", "W_proxy", "model_name", "runtime"])
    df = pd.concat(frames, ignore_index=True)
    # drop duplicate ids, keep first occurrence
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df

def resolve_reference_nc(dates: str, tvp: Optional[str], models_dir: Path) -> Optional[Path]:
    """
    Return the first existing reference .nc Path matching (dates, tvp),
    trying several common naming conventions. Falls back to the standard reference.
    """
    years = re.findall(r"\d{4}", str(dates))
    if len(years) < 2:
        return None
    y0, y1 = years[0], years[1]

    candidates = []
    # tvp-specific candidates first (synthetic etc.)
    tvp_str = str(tvp).strip() if pd.notna(tvp) else ""
    if tvp_str and tvp_str.lower() not in {"", "nan", "none"}:
        candidates += [
            # models_dir / tvp_str / f"standard_{y0}_{y1}_reference.nc",
            models_dir / f"standard_{y0}_{y1}_reference.nc",
            models_dir / f"standard_{y0}_{y1}_GB_reference.nc",
            models_dir / f"{tvp_str}.nc",
        ]
    # default standard reference
    candidates.append(models_dir / f"standard_{y0}_{y1}_reference.nc")

    for c in candidates:
        if c.exists():
            return c
    return Exception(f'Reference doesnt exist for {tvp}')

# ------------------------- Model reading & metrics ----------------------------

def read_clustered_netcdf(path):
    """Open model .nc and strip lingering clustering from attrs."""
    p = Path(path).resolve()
    if not p.is_file():
        raise Exception(f"File does not exist at: {path}")
    with Dataset(p, "a") as nc:  # append mode to edit attrs
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)
        g.setncattr("config", yaml.safe_dump(cfg))

    return calliope.read_netcdf(path)


def get_capacities(m: calliope.Model):
    """Return power_cap and storage_cap Series indexed by techs."""
    # Power capacities
    df_power_caps = (
        m.results["flow_cap"]
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame("capacity")
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )

    # drop pure storage & demand from the power cap metric;
    # keep electrolyser only when carriers == hydrogen; others only when carriers == power
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

    # Storage (energy) capacities
    df_energy_caps = (
        m.results["storage_cap"]
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame("capacity")
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    df_energy_caps = df_energy_caps[df_energy_caps["capacity"] > 0]
    df_energy_caps.set_index("techs", inplace=True)

    return df_power_caps["capacity"], df_energy_caps["capacity"]

def relative_error(df_ref, df_test):
    e = (df_ref - df_test) / df_ref
    e_mean_abs = np.mean(np.abs(e))
    return e, e_mean_abs



# ------------------------- Cache builders ------------------------------------

def build_or_load_cem_cache(df_needed: pd.DataFrame,
                            models_dir: Path,
                            cache_path: Path) -> pd.DataFrame:
    """
    Build or load a cache with CEM metrics per model id.
    df_needed provides the union of IDs we care about (from both figure groups).

    Cache columns:
        id, horizon, number_reps, W_proxy, ldes_error, macme, abs_ldes_error
    """
    if cache_path.exists():
        df_cache = pd.read_csv(cache_path)
    else:
        df_cache = pd.DataFrame(columns=["id", "horizon", "number_reps", "W_proxy",
                                         "ldes_error", "macme", "abs_ldes_error"])

    cached_ids = set(df_cache["id"]) if not df_cache.empty else set()

    # Memoize reference capacities per date span to avoid repeated loads
    ref_cache: Dict[str, Tuple[float, float]] = {}

    rows = []

    num_reads = len(df_needed)
    counter = 0

    for _, row in df_needed.iterrows():
        mid = row["id"]
        counter += 1
        print(f'Reading {mid} for capacities, {counter}/{num_reads}')
        if pd.isna(mid) or (mid in cached_ids):
            continue

         # --- resolve paths ---------------------------------------------------
        test_nc = models_dir / f"{mid}.nc"
        if not test_nc.exists():
            continue

        dates = str(row.get("dates", ""))
        tvp   = row.get("tvp", np.nan)

        ref_key = (dates, str(tvp))
        if ref_key not in ref_cache:
            ref_nc = resolve_reference_nc(dates, tvp, models_dir)
            if ref_nc is None:
                print('- Reference identified at:', ref_nc)
                continue
            try:
                model_reference = calliope.read_netcdf(str(ref_nc))
                ref_cache[ref_key] = model_reference
            except Exception:
                continue

        # use the cached ref model
        model_reference = ref_cache[ref_key]

        # --- load test model (strip time_cluster first) ----------------------
        try:
            model_test = read_clustered_netcdf(test_nc)
        except Exception:
            continue

        # --- capacities & errors ---------------------------------------------
        try:
            # get_capacities returns (power_caps_series, energy_caps_series)
            power_caps_reference, energy_caps_reference = get_capacities(ref_cache[ref_key])
            power_caps_test,      energy_caps_test      = get_capacities(model_test)
        except Exception:
            continue

        # MACME: mean abs relative error on POWER capacities
        _, e_mean_power = relative_error(power_caps_reference, power_caps_test)

        # LDES cap error: tech-specific relative error from ENERGY capacities
        e_storage, _ = relative_error(energy_caps_reference, energy_caps_test)
        ldes_err = e_storage.get("h2_salt_cavern", np.nan)

        rows.append({
            "id": mid,
            "horizon": row.get("horizon", np.nan),
            "number_reps": row.get("number_reps", np.nan),
            "W_proxy": row.get("W_proxy", np.nan),
            "ldes_error": ldes_err,                 # single value (fraction)
            "macme": e_mean_power,                  # single value (fraction)
            "abs_ldes_error": abs(ldes_err) if pd.notna(ldes_err) else np.nan
        })

    if rows:
        df_new = pd.DataFrame(rows)
        df_cache = pd.concat([df_cache, df_new], ignore_index=True)

    df_cache.drop_duplicates(subset=["id"], keep="first", inplace=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_cache.to_csv(cache_path, index=False)

    return df_cache


def build_or_load_runtime_cache(df_needed: pd.DataFrame,
                                models_dir: Path,
                                cache_path: Path) -> pd.DataFrame:
    """
    Build or load runtime cache for IDs required by figs 4–5,
    including *tvp-specific* reference models resolved from (dates, tvp).

    Columns: id, runtime_min, horizon, number_reps, dates, tvp
    """
    cols = ["id", "runtime_min", "horizon", "number_reps", "dates", "tvp"]
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
        except Exception:
            df_cache = pd.DataFrame(columns=cols)
    else:
        df_cache = pd.DataFrame(columns=cols)

    cached_ids = set(df_cache["id"]) if not df_cache.empty else set()

    # Build the set of required IDs: the logged test ids + their tvp-aware reference ids (by path)
    needed = []  # list of (id, dates, tvp, nc_path)
    for _, r in df_needed.iterrows():
        mid   = r.get("id")
        dates = r.get("dates", "")
        tvp   = r.get("tvp", np.nan)

        if pd.notna(mid):
            nc = models_dir / f"{mid}.nc"
            needed.append((str(mid), dates, tvp, nc))

        # add tvp-aware reference
        ref_nc = resolve_reference_nc(dates, tvp, models_dir)
        if ref_nc is not None:
            needed.append((ref_nc.stem, dates, tvp, ref_nc))

    rows = []
    num_reads = len(needed)
    counter = 0
    for mid, dates, tvp, nc_path in needed:
        counter += 1
        print(f'Reading {mid} for runtimes, {counter}/{num_reads}')
        if mid in cached_ids:
            continue
        if not Path(nc_path).exists():
            continue

        try:
            m = read_clustered_netcdf(nc_path)
        except Exception:
            continue

        # runtime (minutes)
        runtime_min = np.nan
        try:
            rt = m.all_attrs.runtime.timings
            runtime_min = float(rt["solve_complete"] - rt["build_start"]) / 60.0
            if runtime_min <= 0:
                runtime_min = np.nan
        except Exception:
            pass

        # horizon + number_reps
        horizon = np.nan
        number_reps = np.nan
        try:
            if str(mid).startswith("standard_") and str(mid).endswith("_reference"):
                # reference: compute horizon from timesteps
                horizon = round(m.inputs.sizes["timesteps"] / (24 * 365.25))
                number_reps = np.nan
            else:
                # clustered: prefer parameters JSON
                param_path = PARAM_DIR / f"{mid}.json"
                if param_path.exists():
                    with open(param_path) as p:
                        d = json.load(p)
                    dr = d["calliope_params"]["date_range"]
                    horizon = round(dr[1] - dr[0] + 1)
                    number_reps = d["tsa_params"]["k_periods"]
                else:
                    # best-effort fallback from the log row
                    rr = df_needed[df_needed["id"] == mid].head(1)
                    if not rr.empty:
                        if pd.notna(rr.iloc[0].get("horizon", np.nan)):
                            horizon = int(rr.iloc[0]["horizon"])
                        if pd.notna(rr.iloc[0].get("number_reps", np.nan)):
                            number_reps = int(rr.iloc[0]["number_reps"])
        except Exception:
            pass

        rows.append({
            "id": mid,
            "runtime_min": runtime_min,
            "horizon": horizon,
            "number_reps": number_reps,
            "dates": dates,
            "tvp": tvp,
        })

    if rows:
        df_cache = pd.concat([df_cache, pd.DataFrame(rows)], ignore_index=True)

    df_cache.drop_duplicates(subset=["id"], keep="first", inplace=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_cache.to_csv(cache_path, index=False)
    return df_cache

# ------------------------- Figure builders -----------------------------------

def fig1_heatmap_10y(df_cem: pd.DataFrame, path: Path) -> None:
    df = df_cem.copy()
    df = df[df["horizon"] == 10]
    df = df.dropna(subset=["number_reps", "W_proxy", "abs_ldes_error"])
    

    if df.empty:
        print("[fig1] No data after filtering for horizon==10.")
        return

    pivot = (df.groupby(["number_reps", "W_proxy"])["abs_ldes_error"]
             .mean()
             .reset_index()
             .pivot(index="number_reps", columns="W_proxy", values="abs_ldes_error")
             .sort_index())

    # Ensure x-order for columns
    cols_sorted = sorted(pivot.columns.tolist())
    pivot = pivot[cols_sorted]

    import matplotlib.colors as mcolors
    # Discrete levels in 5% increments (0, 5%, ..., 100%)
    levels = np.arange(0.0, 0.5, 0.05)  # values are fractions 0..1
    cmap = plt.get_cmap("plasma", len(levels) - 1)  # plasma goes #0D0887 -> #F0F921
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot.values,
                   aspect="auto",
                   origin="lower",
                   cmap=cmap,
                   norm=norm)

    # tick labels from the actual row/column labels
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=0)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{int(i)}" for i in pivot.index])

    # Discrete colorbar with 5% tick labels
    cbar = fig.colorbar(im, ax=ax, ticks=levels)
    cbar.ax.set_yticklabels([f"{int(v*100)}%" for v in levels])
    cbar.set_label("Absolute LDES capacity error")

    ax.set_xlabel("Proxy weight $(W_P)$")
    ax.set_ylabel("Number of representative days")
    # ax.set_title("Absolute LDES capacity error (10-year models)")

    savefig(fig, path)



def _add_y0_line(ax: plt.Axes) -> None:
    ax.axhline(0, color="black", linewidth=1, zorder=0)


def fig2_box_by_reps(df_cem: pd.DataFrame, path: Path) -> None:
    df = df_cem.dropna(subset=["number_reps"])

    if df.empty:
        print("[fig2] No data to plot.")
        return

    x_vals = sorted(df["number_reps"].dropna().unique().tolist())

    df_filter = df[df['W_proxy']>0]

    data_ldes = [df_filter.loc[df_filter["number_reps"] == x, "ldes_error"].dropna().values for x in x_vals]
    data_mac  = [df_filter.loc[df_filter["number_reps"] == x, "macme"].dropna().values for x in x_vals]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    offsets = np.array([-0.15, 0.15])
    positions_ldes = np.arange(len(x_vals)) + offsets[0]
    positions_mac  = np.arange(len(x_vals)) + offsets[1]

    bp1 = ax.boxplot(data_ldes, positions=positions_ldes, widths=0.25, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_mac,  positions=positions_mac,  widths=0.25, patch_artist=True, showfliers=False)

    for elem in ["boxes", "caps", "whiskers", "medians"]:
        for patch in bp1[elem]:
            patch.set_color(COLOUR_LDES)
        for patch in bp2[elem]:
            patch.set_color(COLOUR_MACME)
    for patch in bp1["boxes"]:
        patch.set_facecolor(mcolors.to_rgba(COLOUR_LDES, 0.15))
    for patch in bp2["boxes"]:
        patch.set_facecolor(mcolors.to_rgba(COLOUR_MACME, 0.15))

    rng = np.random.default_rng(42)
    for i, x in enumerate(x_vals):
        sub = df[df["number_reps"] == x]

        # Only show markers where W_proxy == 0
        sub0 = sub[np.isclose(sub["W_proxy"], 0.0, atol=1e-9)]

        # LDES markers (round, hollow, edge = pink)
        xs = np.full(len(sub0), positions_ldes[i]) + rng.normal(0, 0.025, len(sub0))
        ys = sub0["ldes_error"].values
        ax.scatter(xs, ys, s=fig_height*4, marker="o",
                   facecolors="white", edgecolors=COLOUR_LDES, linewidths=0.9, zorder=3)

        # MACME markers (round, hollow, edge = blue/purple)
        xs = np.full(len(sub0), positions_mac[i]) + rng.normal(0, 0.025, len(sub0))
        ys = sub0["macme"].values
        ax.scatter(xs, ys, s=fig_height*4, marker="o",
                   facecolors="white", edgecolors=COLOUR_MACME, linewidths=0.9, zorder=3)


    _add_y0_line(ax)

    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([str(int(x)) for x in x_vals])
    ax.set_xlabel("Number of representative days")
    ax.set_ylabel("Relative error")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))

    legend_handles = [
        Line2D([0], [0], marker=MARKER_LDES, color="none",
               markerfacecolor=COLOUR_LDES, markeredgecolor=COLOUR_LDES,
               label="LDES capacity error", markersize=6),
        Line2D([0], [0], marker=MARKER_MACME, color="none",
               markerfacecolor=COLOUR_MACME, markeredgecolor=COLOUR_MACME,
               label="MACME", markersize=6),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="white", markeredgecolor=COLOUR_W0_EDGE,
               label="$W_P = 0$", markersize=6),
        # Line2D([0], [0], marker="o", color="none",
        #        markerfacecolor="black", markeredgecolor="black",
        #        label="$W_P = 1$", markersize=6),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    savefig(fig, path)
    print(f"[fig2] Saved {path}")


def fig3_box_by_proxy(df_cem: pd.DataFrame, path: Path) -> None:
    df = df_cem.copy()
    df = df[df["number_reps"] != 14]
    df = df.dropna(subset=["W_proxy"])

    x_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    df = df[df["W_proxy"].round(2).isin(x_levels)]

    if df.empty:
        print("[fig3] No data to plot after filtering.")
        return

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    data_ldes = [df.loc[np.isclose(df["W_proxy"], x), "ldes_error"].dropna().values for x in x_levels]
    data_mac  = [df.loc[np.isclose(df["W_proxy"], x), "macme"].dropna().values for x in x_levels]

    offsets = np.array([-0.15, 0.15])
    positions_ldes = np.arange(len(x_levels)) + offsets[0]
    positions_mac  = np.arange(len(x_levels)) + offsets[1]

    bp1 = ax.boxplot(data_ldes, positions=positions_ldes, widths=0.25, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_mac,  positions=positions_mac,  widths=0.25, patch_artist=True, showfliers=False)

    for elem in ["boxes", "caps", "whiskers", "medians"]:
        for patch in bp1[elem]:
            patch.set_color(COLOUR_LDES)
        for patch in bp2[elem]:
            patch.set_color(COLOUR_MACME)
    for patch in bp1["boxes"]:
        patch.set_facecolor(mcolors.to_rgba(COLOUR_LDES, 0.15))
    for patch in bp2["boxes"]:
        patch.set_facecolor(mcolors.to_rgba(COLOUR_MACME, 0.15))

    # rng = np.random.default_rng(7)
    # for i, x in enumerate(x_levels):
    #     sub = df[np.isclose(df["W_proxy"], x)]

    #     xs = np.full(len(sub), positions_ldes[i]) + rng.normal(0, 0.025, len(sub))
    #     ys = sub["ldes_error"].values
    #     w  = sub["W_proxy"].values
    #     face = [COLOUR_LDES if np.isclose(val, 1.0, atol=1e-9) else "white" if np.isclose(val, 0.0, atol=1e-9) else COLOUR_LDES for val in w]
    #     edge = [COLOUR_LDES if not np.isclose(val, 0.0, atol=1e-9) else COLOUR_W0_EDGE for val in w]
    #     ax.scatter(xs, ys, s=18, marker=MARKER_LDES, facecolors=face, edgecolors=edge, linewidths=0.8, zorder=3)

    #     xs = np.full(len(sub), positions_mac[i]) + rng.normal(0, 0.025, len(sub))
    #     ys = sub["macme"].values
    #     face = [COLOUR_MACME if np.isclose(val, 1.0, atol=1e-9) else "white" if np.isclose(val, 0.0, atol=1e-9) else COLOUR_MACME for val in w]
    #     edge = [COLOUR_MACME if not np.isclose(val, 0.0, atol=1e-9) else COLOUR_W0_EDGE for val in w]
    #     ax.scatter(xs, ys, s=18, marker=MARKER_MACME, facecolors=face, edgecolors=edge, linewidths=0.8, zorder=3)

    _add_y0_line(ax)

    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels([f"{x:g}" for x in x_levels])
    ax.set_xlabel("Proxy weight (W)")
    ax.set_ylabel("Relative error")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))

    legend_handles = [
        Line2D([0], [0], marker=MARKER_LDES, color="none",
               markerfacecolor=COLOUR_LDES, markeredgecolor=COLOUR_LDES,
               label="LDES capacity error", markersize=6),
        Line2D([0], [0], marker=MARKER_MACME, color="none",
               markerfacecolor=COLOUR_MACME, markeredgecolor=COLOUR_MACME,
               label="MACME", markersize=6),
        # Line2D([0], [0], marker="o", color="none",
        #        markerfacecolor="white", markeredgecolor=COLOUR_W0_EDGE,
        #        label="$W_P = 0$", markersize=6),
        # Line2D([0], [0], marker="o", color="none",
        #        markerfacecolor="black", markeredgecolor="black",
        #        label="$W_P = 1$", markersize=6),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    savefig(fig, path)
    print(f"[fig3] Saved {path}")


def fig4_error_vs_horizon(df_cem: pd.DataFrame, path: Path) -> None:
    # Filter to W in {0,1} and reps in [45, 60]
    df = df_cem.copy()
    df = df[(df["number_reps"] >= 60) & (df["number_reps"] <= 90)]
    df = df[df["W_proxy"].isin([0.0, 1.0])]
    df = df.dropna(subset=["horizon"])

    # Bucket horizons to '2', '5', '10' only
    mapping = {2: "2", 5: "5", 10: "10"}
    df["h_bucket"] = df["horizon"].round().astype(int).map(mapping)
    df = df[df["h_bucket"].isin(["2", "5", "10"])]

    if df.empty:
        print("[fig4] No data to plot after filtering.")
        return

    horizons = ["2", "5", "10"]
    x = np.arange(len(horizons))
    # Adjacent box positions: left = W0, right = W1
    pos_W0 = x - 0.15
    pos_W1 = x + 0.15
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(fig_width*2, fig_height), sharey=True)
    metrics = [
        ("ldes_error", "LDES capacity error", COLOUR_LDES, axes[0]),
        ("macme",      "MACME",              COLOUR_MACME, axes[1]),
    ]

    for colname, title, color, ax in metrics:
        # Collect data arrays per horizon and W
        data_W0 = [df[(df["h_bucket"] == h) & np.isclose(df["W_proxy"], 0.0)][colname].dropna().values
                   for h in horizons]
        data_W1 = [df[(df["h_bucket"] == h) & np.isclose(df["W_proxy"], 1.0)][colname].dropna().values
                   for h in horizons]

        # Plot W=0 (hollow) boxes
        bp0 = ax.boxplot(
            data_W0,
            positions=pos_W0,
            widths=width,
            patch_artist=True,
            showfliers=False
        )
        # Style W=0 boxes: white face, colored edges
        for patch in bp0["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
        for elem in ["caps", "whiskers", "medians"]:
            for p in bp0[elem]:
                p.set_color(color)

        # Plot W=1 (filled) boxes
        bp1 = ax.boxplot(
            data_W1,
            positions=pos_W1,
            widths=width,
            patch_artist=True,
            showfliers=False
        )
        # Style W=1 boxes: light fill of the same color
        for patch in bp1["boxes"]:
            patch.set_facecolor(mcolors.to_rgba(color, 0.4))
            patch.set_edgecolor(color)
        for elem in ["caps", "whiskers", "medians"]:
            for p in bp1[elem]:
                p.set_color(color)

        # Axis formatting
        _add_y0_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_xlabel("Horizon (years)")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Relative error")

    # Simple legend explaining W=0 vs W=1 encoding (hollow vs filled)
    legend_handles = [
        Line2D([0], [0], marker="s", color=COLOUR_LDES,
               markerfacecolor="white", markeredgecolor=COLOUR_GREY_MEAN,
               linestyle="", label="$W_P = 0$ (hollow)"),
        Line2D([0], [0], marker="s", color=COLOUR_LDES,
               markerfacecolor=mcolors.to_rgba(COLOUR_GREY_MEAN, 0.4),
               markeredgecolor=COLOUR_GREY_MEAN, linestyle="", label="$W_P = 1$ (filled)"),
    ]
    axes[1].legend(handles=legend_handles, frameon=False, loc="best")

    fig.tight_layout()
    savefig(fig, path)



def fig5_runtime_vs_horizon(df_runtime: pd.DataFrame, path: Path) -> None:
    df = df_runtime.copy()
    df = df.dropna(subset=["runtime_min", "horizon"])


    # Horizon buckets as strings '2','5','10' (others dropped)
    mapping = {2: "2", 5: "5", 10: "10"}
    df["h_bucket"] = df["horizon"].astype(int).map(mapping)
    df = df[df["h_bucket"].isin(["2", "5", "10"])]

    if df.empty:
        print("[fig5] No runtime data to plot.")
        return

    # Colour map by number_reps (k); references (NaN) in grey
    ks = df["number_reps"]
    unique_k = sorted([int(k) for k in ks.dropna().unique()])
    cmap = plt.get_cmap("plasma")
    # make a colour lookup for each k
    col_lut = {k: cmap(i / max(len(unique_k)-1, 1)) for i, k in enumerate(unique_k)}
    grey = "#808080"

    # jitter scatter per horizon bucket
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    rng = np.random.default_rng(23)
    horizons = ["2", "5", "10"]
    xloc = {h: i for i, h in enumerate(horizons)}

    for h in horizons:
        sub = df[df["h_bucket"] == h]
        if sub.empty:
            continue

        xbase = xloc[h]
        jitter = rng.normal(0, 0.06, len(sub))
        xs = xbase + jitter
        ys = sub["runtime_min"].values

        # choose colour by k; refs in grey
        cols = []
        for k in sub["number_reps"].values:
            if pd.isna(k):
                cols.append(grey)            # reference
            else:
                kk = int(k)
                cols.append(col_lut.get(kk, grey))

        ax.scatter(xs, ys, s=fig_height*2, marker="o", linewidths=0.2, edgecolors="black", alpha=0.9, c=cols, zorder=3)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel(r"Runtime ($\log_{10}$ minutes)")
    ax.set_yscale("log")

    # legend: build from unique k plus 'ref'
    handles = []
    for k in unique_k:
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=col_lut[k], label=f"{k}", markersize=6))
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=grey, label="ref", markersize=6))
    if handles:
        ax.legend(handles=handles, title="Rep. days (k)", frameon=False,
                  bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig(fig, path)



# ------------------------- Main ----------------------------------------------

def main():
    # Parse logs for figure groups
    df_f123 = parse_logs(LOGS_F123)  # for figs 1–3
    df_f45  = parse_logs(LOGS_F45)   # for figs 4–5

    # Union for CEM cache (ids needed for figs 1–4)
    df_union_cem = pd.concat([df_f123, df_f45], ignore_index=True).drop_duplicates(subset=["id"], keep="first")

    # Build caches once
    try:
        df_cem_cache = build_or_load_cem_cache(df_union_cem, MODELS_DIR, CACHE_CEM)
    except Exception as e:
        print(f"[main] CEM cache build failed: {e}")
        df_cem_cache = pd.DataFrame(columns=["id", "horizon", "number_reps", "W_proxy",
                                             "ldes_error", "macme", "abs_ldes_error"])

    try:
        # Only need runtime for the fig 4–5 ids
        df_runtime_cache = build_or_load_runtime_cache(df_f45, MODELS_DIR, CACHE_RUNTIME)
    except Exception as e:
        print(f"[main] Runtime cache build failed: {e}")
        df_runtime_cache = pd.DataFrame(columns=["id", "runtime_min", "horizon", "number_reps"])

    # Create view tables for each figure group by merging cache with the specific logs
    # (so filtering like horizon==10 or W sets works properly)
    df_f123_ready = df_f123.merge(df_cem_cache, on="id", suffixes=("", "_c"))
    df_f45_ready  = df_f45.merge(df_cem_cache, on="id", suffixes=("", "_c"))
    # Build the set of IDs we want in Fig 5: logged + their references
    req_ids = set(df_f45["id"].astype(str))
    ref_ids = []
    for dates in df_f45["dates"].astype(str):
        years = re.findall(r"\d{4}", dates)
        if len(years) >= 2:
            ref_ids.append(f"standard_{years[0]}_{years[1]}_reference")
    req_ids.update(ref_ids)

    # Pull those rows directly from the *runtime cache*
    keys = df_f45[["dates", "tvp"]].drop_duplicates()
    df_run_ready = df_runtime_cache.merge(keys, on=["dates", "tvp"], how="inner")

    # Generate figures
    fig1_heatmap_10y(df_f123_ready, FIG1_HEATMAP_10Y)
    fig2_box_by_reps(df_f123_ready, FIG2_ERR_VS_REPS)
    fig3_box_by_proxy(df_f123_ready, FIG3_ERR_VS_PROXY)
    fig4_error_vs_horizon(df_f45_ready,  FIG4_ERR_VS_HORIZON)
    fig5_runtime_vs_horizon(df_run_ready, FIG5_RUNTIME_VS_HOR)

    print("\nAll done.\n"
          f"  - {FIG1_HEATMAP_10Y}\n"
          f"  - {FIG2_ERR_VS_REPS}\n"
          f"  - {FIG3_ERR_VS_PROXY}\n"
          f"  - {FIG4_ERR_VS_HORIZON}\n"
          f"  - {FIG5_RUNTIME_VS_HOR}\n")


if __name__ == "__main__":
    main()
