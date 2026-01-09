#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import matplotlib as mpl

from matplotlib.offsetbox import (
    OffsetImage, TextArea, HPacker, AnchoredOffsetbox
)
import matplotlib.image as mpimg
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


import yaml
from netCDF4 import Dataset
import calliope

from helper_signals import build_signal_metrics


# ------------------------- Paths & constants ---------------------------------

SAVE_FIGURES = True

COUNTRIES = ["NL", "BE"]  # order = plotting order (top -> bottom)

LOGS_F123_BY_CC = {
    cc: [Path(f"SoC_proxy_TSA/data/notes/log_{cc}_rerun.csv")]
    for cc in COUNTRIES
}

# Keep these for your other figures if you still use them
LOGS_F45 = [Path("SoC_proxy_TSA/data/notes/log_horizons_runtimes.csv")]
LOGS_F6  = [Path("SoC_proxy_TSA/data/notes/log_NL_margins.csv")]

ICON_BY_CC = {cc: f"SoC_proxy_TSA/icons/{cc}.png" for cc in COUNTRIES}

MODELS_DIR = Path("SoC_proxy_TSA/data/calliope_models")
PARAM_DIR  = Path("SoC_proxy_TSA/data/parameters")
OUT_DIR    = Path(".")

CACHE_CEM     = Path("SoC_proxy_TSA/data/notes/cem_cache.csv")
CACHE_RUNTIME = Path("SoC_proxy_TSA/data/notes/runtime_cache.csv")

# Compound outputs for Figs 1–3
FIG1_HEATMAP_10Y_COMPOUND = OUT_DIR / "fig1_heatmap_ldes_error_10y_NL_BE.pdf"
FIG2_ERR_VS_REPS_COMPOUND = OUT_DIR / "fig2_error_box_vs_reps_NL_BE.pdf"
FIG3_ERR_VS_PROXY_COMPOUND = OUT_DIR / "fig3_error_box_vs_proxy_NL_BE.pdf"

# Your existing outputs for the rest (unchanged)
FIG4_ERR_VS_HORIZON = OUT_DIR / "fig4_error_vs_horizon_W00p5_combined.pdf"
FIG5_RUNTIME_VS_HOR = OUT_DIR / "fig5_runtime_vs_horizon.pdf"
FIG6_ERR_VS_MARGIN  = OUT_DIR / "fig6_error_box_vs_margin_GB_NL.pdf"

# Colours (preserve existing)
COLOUR_MACME = "#0D0887"
COLOUR_LDES  = "#CC4778"
COLOUR_W0_EDGE = "#666666"
COLOUR_GREY_MEAN = "#666666"

MARKER_MACME = "o"
MARKER_LDES  = "o"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})
mpl.rcParams["pgf.preamble"] = r""

fig_width = 3.5
fig_height = 3


# ------------------------- Helpers -------------------------------------------

def savefig(fig, path, use_tight_layout=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_tight_layout:
        fig.tight_layout()
    if SAVE_FIGURES:
        fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.show()
    plt.close(fig)



def _parse_single_log(log_path: Path) -> pd.DataFrame:
    df = pd.read_csv(log_path)

    required_cols = {"id", "dates", "date_range", "model_name", "tvp"}
    missing = required_cols - set(df.columns)
    for col in missing:
        df[col] = np.nan

    df = df.copy()

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
        return pd.DataFrame(columns=["id", "dates", "horizon", "number_reps", "W_proxy", "model_name", "runtime"])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["id"], keep="first")
    return df


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


# ------------------------- Cache builders (unchanged) -------------------------

def build_or_load_cem_cache(df_needed: pd.DataFrame, models_dir: Path, cache_path: Path) -> pd.DataFrame:
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


def build_or_load_runtime_cache(df_needed: pd.DataFrame,
                                models_dir: Path,
                                cache_path: Path) -> pd.DataFrame:
    """
    Build or load runtime cache for IDs required by Fig 5,
    including tvp-specific reference models resolved from (dates, tvp).

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
    needed: List[Tuple[str, str, str, Path]] = []
    for _, r in df_needed.iterrows():
        mid = r.get("id")
        dates = r.get("dates", "")
        tvp = r.get("tvp", np.nan)

        if pd.notna(mid):
            nc = models_dir / f"{mid}.nc"
            needed.append((str(mid), str(dates), str(tvp), nc))

        ref_nc = resolve_reference_nc(str(dates), tvp, models_dir)
        if ref_nc is not None:
            needed.append((ref_nc.stem, str(dates), str(tvp), ref_nc))

    rows = []
    num_reads = len(needed)
    counter = 0

    for mid, dates, tvp, nc_path in needed:
        counter += 1
        print(f"Reading {mid} for runtimes, {counter}/{num_reads}")

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
                horizon = round(m.inputs.sizes["timesteps"] / (24 * 365.25))
                number_reps = np.nan
            else:
                param_path = PARAM_DIR / f"{mid}.json"
                if param_path.exists():
                    with open(param_path) as p:
                        d = json.load(p)
                    dr = d["calliope_params"]["date_range"]
                    horizon = round(dr[1] - dr[0] + 1)
                    number_reps = d["tsa_params"]["k_periods"]
                else:
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


# ------------------------- Figures 4–5 (combined NL+BE) -----------------------

def fig4_error_vs_horizon_combined(df_cem: pd.DataFrame, path: Path) -> None:
    """
    Combined (NL+BE) version of your Fig 4:
      - two stacked subplots: LDES capacity error (top) and MACME (bottom)
      - within each subplot: boxplots for W_P in {0, 0.5}
    """
    df = df_cem.copy()
    df = df[(df["number_reps"] >= 60) & (df["number_reps"] <= 365)]
    df = df[df["W_proxy"].isin([0.0, 0.5])]
    df = df.dropna(subset=["horizon"])

    mapping = {2: "2", 5: "5", 10: "10"}
    df["h_bucket"] = df["horizon"].round().astype(int).map(mapping)
    df = df[df["h_bucket"].isin(["2", "5", "10"])]

    if df.empty:
        print("[fig4_combined] No data to plot after filtering.")
        return

    horizons = ["2", "5", "10"]
    x = np.arange(len(horizons))
    pos_W0  = x - 0.15
    pos_W05 = x + 0.15
    width = 0.28

    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height * 2.0), sharey=True)

    metrics = [
        ("ldes_error", "LDES capacity error", COLOUR_LDES, axes[0]),
        ("macme",      "MACME",              COLOUR_MACME, axes[1]),
    ]

    for colname, _, color, ax in metrics:
        data_W0 = [
            df[(df["h_bucket"] == h) & np.isclose(df["W_proxy"], 0.0)][colname].dropna().values
            for h in horizons
        ]
        data_W05 = [
            df[(df["h_bucket"] == h) & np.isclose(df["W_proxy"], 0.5)][colname].dropna().values
            for h in horizons
        ]

        # W=0 (hollow)
        bp0 = ax.boxplot(
            data_W0, positions=pos_W0, widths=width, patch_artist=True, showfliers=False
        )
        for patch in bp0["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
        for elem in ["caps", "whiskers", "medians"]:
            for p in bp0[elem]:
                p.set_color(color)

        # W=0.5 (filled)
        bp1 = ax.boxplot(
            data_W05, positions=pos_W05, widths=width, patch_artist=True, showfliers=False
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor(mcolors.to_rgba(color, 0.4))
            patch.set_edgecolor(color)
        for elem in ["caps", "whiskers", "medians"]:
            for p in bp1[elem]:
                p.set_color(color)

        _add_y0_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[1].set_xlabel("Horizon (years)")
    axes[0].set_ylabel("LDES capacity error")
    axes[1].set_ylabel("MACME")

    legend_handles = [
        Line2D([0], [0], marker="s", color=COLOUR_GREY_MEAN,
               markerfacecolor="white", markeredgecolor=COLOUR_GREY_MEAN,
               linestyle="", label="$W_P = 0$ (hollow)"),
        Line2D([0], [0], marker="s", color=COLOUR_GREY_MEAN,
               markerfacecolor=mcolors.to_rgba(COLOUR_GREY_MEAN, 0.4),
               markeredgecolor=COLOUR_GREY_MEAN, linestyle="", label="$W_P = 0.5$ (filled)"),
    ]
    axes[0].legend(handles=legend_handles, frameon=False, loc="best")

    savefig(fig, path, use_tight_layout=True)


def fig5_runtime_vs_horizon_combined(df_runtime: pd.DataFrame, path: Path) -> None:
    """
    Combined (NL+BE) version of your Fig 5 (single panel):
    scatter of mean runtime by horizon bucket and number_reps.
    """
    df = df_runtime.copy()
    df = df.dropna(subset=["runtime_min", "horizon"])

    mapping = {2: "2", 5: "5", 10: "10"}
    df["h_bucket"] = df["horizon"].astype(int).map(mapping)
    df = df[df["h_bucket"].isin(["2", "5", "10"])]

    if df.empty:
        print("[fig5_combined] No runtime data to plot.")
        return

    df_mean = (
        df.groupby(["h_bucket", "number_reps"], dropna=False, as_index=False)["runtime_min"]
          .mean()
          .rename(columns={"runtime_min": "runtime_mean"})
    )

    ks = df_mean["number_reps"]
    unique_k = sorted([int(k) for k in ks.dropna().unique()])
    cmap = plt.get_cmap("plasma")
    col_lut = {k: cmap(i / max(len(unique_k) - 1, 1)) for i, k in enumerate(unique_k)}
    grey = "#808080"

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    horizons = ["2", "5", "10"]
    xloc = {h: i for i, h in enumerate(horizons)}

    offset_value = 0.0
    if unique_k:
        offsets = np.linspace(-offset_value, offset_value, len(unique_k))
        k_offset = {k: offsets[i] for i, k in enumerate(unique_k)}
    else:
        k_offset = {}

    for h in horizons:
        sub = df_mean[df_mean["h_bucket"] == h]
        if sub.empty:
            continue

        xs, ys, cols = [], [], []
        for _, row in sub.iterrows():
            k = row["number_reps"]
            base_x = xloc[h]
            if pd.isna(k):
                xval = base_x
                c = grey
            else:
                k_int = int(k)
                xval = base_x + k_offset.get(k_int, 0.0)
                c = col_lut.get(k_int, grey)

            xs.append(xval)
            ys.append(row["runtime_mean"])
            cols.append(c)

        ax.scatter(
            xs, ys,
            s=fig_height * 4,
            marker="o",
            linewidths=0.2,
            edgecolors="black",
            alpha=0.9,
            c=cols,
            zorder=3,
        )

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(horizons)
    ax.set_xlim(-0.5, len(horizons) - 0.5)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Runtime (minutes, log scale)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mtick.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mtick.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # legend for k
    handles = []
    for k in unique_k:
        handles.append(Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=col_lut[k], markeredgecolor="black",
                              label=f"k={k}", markersize=5))
    if handles:
        ax.legend(handles=handles, frameon=False, loc="best", ncol=2)

    savefig(fig, path, use_tight_layout=True)



# ------------------------- Plot cosmetics ------------------------------------

def _add_y0_line(ax: plt.Axes) -> None:
    ax.axhline(0, color="black", linewidth=1, zorder=0)


def add_panel_label(ax, label, x=0.01, y=0.98):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=mpl.rcParams["axes.titlesize"],
        fontweight="bold"
    )

def add_flag_title(ax, countrycode: str, title: str,
                   flag_path: str, zoom: float = 0.045,
                   sep_px: int = 6, y: float = 1.08,
                   fontsize: int = 10, fontweight: str = "bold"):
    """
    Add a centered title composed of [flag][space][text] above an axes.

    - sep_px controls spacing in pixels between flag and text (consistent!)
    - y controls vertical placement in axes coords (1.0 is top of axes)
    """

    # Remove the default title (we’re replacing it)
    ax.set_title("")

    # Build image + text as one packed box
    flag = mpimg.imread(flag_path)
    img = OffsetImage(flag, zoom=zoom)

    txt = TextArea(
        title,
        textprops=dict(size=fontsize, weight=fontweight, va="center")
    )

    box = HPacker(children=[img, txt], align="center", pad=0, sep=sep_px)

    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=box,
        frameon=False,
        pad=0.0,
        bbox_to_anchor=(0.5, y),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    ax.add_artist(anchored)


# ------------------------- Figure builders (ax-aware) -------------------------

def fig1_heatmap_10y(df_cem: pd.DataFrame, ax: plt.Axes, countrycode: str):
    df = df_cem.copy()
    df = df[df["horizon"] == 10]

    value_col = "ldes_error"
    df = df.dropna(subset=["number_reps", "W_proxy", value_col])
    if df.empty:
        ax.text(0.5, 0.5, "No data (horizon==10)", ha="center", va="center")
        return

    pivot = (
        df.groupby(["number_reps", "W_proxy"])[value_col]
          .mean()
          .reset_index()
          .pivot(index="number_reps", columns="W_proxy", values=value_col)
          .sort_index()
    )
    cols_sorted = sorted(pivot.columns.tolist())
    pivot = pivot[cols_sorted]
    data = pivot.values

    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    step = 0.05
    vmin = np.min(np.floor(data_min / step) * step, 0)
    vmax = np.ceil(data_max / step) * step

    if vmin < 0 < vmax:
        vmid = (0 - vmin) / (vmax - vmin)
        colors = [(0.0, "#0d0887"), (vmid, "#ffffff"), (1.0, "#cc4778")]
    elif vmax > 0:
        colors = [(0.0, "#ffffff"), (1.0, "#cc4778")]
    else:
        colors = [(0.0, "#0d0887"), (1.0, "#ffffff")]

    cmap = mcolors.LinearSegmentedColormap.from_list("plasmaish_div", colors)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=0)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{int(i)}" for i in pivot.index])

    ax.set_xlabel("Proxy weight $(W_P)$")
    ax.set_ylabel("Number of representative days")
    title = "The Netherlands" if countrycode == "NL" else "Belgium" 
    # add_flag_title(ax, countrycode, title, ICON_BY_CC[countrycode])

    return im, vmin, vmax, step


def fig2_box_by_reps(df_cem: pd.DataFrame, ax: plt.Axes, countrycode: str):
    df = df_cem.dropna(subset=["number_reps"])
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    x_vals = sorted(df["number_reps"].dropna().unique().tolist())
    df_filter = df[df["W_proxy"] > 0]

    data_ldes = [df_filter.loc[df_filter["number_reps"] == x, "ldes_error"].dropna().values for x in x_vals]
    data_mac  = [df_filter.loc[df_filter["number_reps"] == x, "macme"].dropna().values for x in x_vals]

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
        sub0 = sub[np.isclose(sub["W_proxy"], 0.0, atol=1e-9)]

        xs = np.full(len(sub0), positions_ldes[i]) + rng.normal(0, 0.025, len(sub0))
        ax.scatter(xs, sub0["ldes_error"].values, s=fig_height*4, marker="o",
                   facecolors="white", edgecolors=COLOUR_LDES, linewidths=0.9, zorder=3)

        xs = np.full(len(sub0), positions_mac[i]) + rng.normal(0, 0.025, len(sub0))
        ax.scatter(xs, sub0["macme"].values, s=fig_height*4, marker="o",
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
        Line2D([0], [0], marker=MARKER_MACME, color="none",
               markerfacecolor="none", markeredgecolor=COLOUR_W0_EDGE,
               label="$W_P = 0$", markersize=6),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    title = "The Netherlands" if countrycode == "NL" else "Belgium" 
    # add_flag_title(ax, countrycode, title, ICON_BY_CC[countrycode])


def fig3_box_by_proxy(df_cem: pd.DataFrame, ax: plt.Axes, countrycode: str):
    df = df_cem.copy()
    df = df[df["number_reps"] >= 45]
    df = df[df["number_reps"] <= 180]
    df = df.dropna(subset=["W_proxy"])

    x_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    df = df[df["W_proxy"].round(2).isin(x_levels)]

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

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
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    title = "The Netherlands" if countrycode == "NL" else "Belgium" 
    # add_flag_title(ax, countrycode, title, ICON_BY_CC[countrycode])


# ------------------------- Compound figure wrappers ---------------------------




def fig1_compound_heatmap(df_by_cc, path):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height * 2.25),
        sharex=True
    )

    # Reserve space: more room above top axis, and more gap between axes
    fig.subplots_adjust(
        left=0.12, right=0.95,
        bottom=0.10, top=0.92,
        hspace=0.32
    )

    # Fixed scale: -30% to +40%, centered at 0
    vmin, vmax = -0.40, 0.40
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "plasmaish_div_fixed",
        [(0.0, "#0d0887"), (0.5, "#ffffff"), (1.0, "#cc4778")]
    )

    # Plot
    im_nl, *_ = fig1_heatmap_10y(df_by_cc["NL"], axes[0], "NL")
    im_be, *_ = fig1_heatmap_10y(df_by_cc["BE"], axes[1], "BE")

    # Force identical scaling on both heatmaps
    for im in (im_nl, im_be):
        im.set_cmap(cmap)
        im.set_norm(norm)

    # Replace titles with centered [flag + text] box, bumped up
    add_flag_title(
        axes[0], "NL", "The Netherlands", ICON_BY_CC["NL"],
        zoom=0.045, sep_px=6, y=1.15, fontsize=10
    )
    add_flag_title(
        axes[1], "BE", "Belgium", ICON_BY_CC["BE"],
        zoom=0.045, sep_px=6, y=1.15, fontsize=10
    )

    add_panel_label(axes[0], "(a)", y=1.11)
    add_panel_label(axes[1], "(b)", y=1.11)

    # Two colourbars, one per axes, identical ticks/labels
    ticks = [-0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40]
    ticklabels = [f"{int(t*100)}%" for t in ticks]

    for ax, im in zip(axes, (im_nl, im_be)):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        cbar.set_label("LDES capacity error")

    # Don’t tight_layout compound figs with custom artists
    savefig(fig, path, use_tight_layout=False)

def fig2_compound_by_reps(df_by_cc, path):
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height * 2.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.93, hspace=0.32)

    fig2_box_by_reps(df_by_cc["NL"], axes[0], "NL")
    fig2_box_by_reps(df_by_cc["BE"], axes[1], "BE")

    add_flag_title(axes[0], "NL", "The Netherlands", ICON_BY_CC["NL"], y=1.15)
    add_flag_title(axes[1], "BE", "Belgium", ICON_BY_CC["BE"], y=1.15)

    add_panel_label(axes[0], "(a)", y=1.11)
    add_panel_label(axes[1], "(b)", y=1.11)

    savefig(fig, path, use_tight_layout=False)


def fig3_compound_by_proxy(df_by_cc, path):
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height * 2.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.93, hspace=0.32)

    fig3_box_by_proxy(df_by_cc["NL"], axes[0], "NL")
    fig3_box_by_proxy(df_by_cc["BE"], axes[1], "BE")

    add_flag_title(axes[0], "NL", "The Netherlands", ICON_BY_CC["NL"], y=1.15)
    add_flag_title(axes[1], "BE", "Belgium", ICON_BY_CC["BE"], y=1.15)

    add_panel_label(axes[0], "(a)", y=1.11)
    add_panel_label(axes[1], "(b)", y=1.11)

    savefig(fig, path, use_tight_layout=False)



# ------------------------- Main ----------------------------------------------

def main():
    # Parse logs per country for figs 1–3
    df_f123_by_cc = {cc: parse_logs(LOGS_F123_BY_CC[cc]) for cc in COUNTRIES}

    # Union for CEM cache across both countries (+ your other figs if desired)
    df_union_cem = pd.concat(
        list(df_f123_by_cc.values()) + [parse_logs(LOGS_F45), parse_logs(LOGS_F6)],
        ignore_index=True
    ).drop_duplicates(subset=["id"], keep="first")

    # Build / update CEM cache once
    df_cem_cache = build_or_load_cem_cache(df_union_cem, MODELS_DIR, CACHE_CEM)

    # Prepare per-country df for figs 1–3
    df_f123_ready_by_cc = {
        cc: df_f123_by_cc[cc].merge(df_cem_cache, on="id", suffixes=("", "_c"))
        for cc in COUNTRIES
    }

    # --- Compound figs 1–3 ---
    fig1_compound_heatmap(df_f123_ready_by_cc, FIG1_HEATMAP_10Y_COMPOUND)
    fig2_compound_by_reps(df_f123_ready_by_cc, FIG2_ERR_VS_REPS_COMPOUND)
    fig3_compound_by_proxy(df_f123_ready_by_cc, FIG3_ERR_VS_PROXY_COMPOUND)

    # --- Fig 4–5 combined across NL+BE ---
    df_f45 = parse_logs(LOGS_F45)
    if not df_f45.empty:
        df_f45_ready = df_f45.merge(df_cem_cache, on="id", suffixes=("", "_c"))

        # runtime cache (includes tvp-aware references)
        df_runtime_cache = build_or_load_runtime_cache(df_f45, MODELS_DIR, CACHE_RUNTIME)
        keys = df_f45[["dates", "tvp"]].drop_duplicates()
        df_run_ready = df_runtime_cache.merge(keys, on=["dates", "tvp"], how="inner")

        fig4_error_vs_horizon_combined(df_f45_ready, FIG4_ERR_VS_HORIZON)
        fig5_runtime_vs_horizon_combined(df_run_ready, FIG5_RUNTIME_VS_HOR)
    else:
        print("[main] LOGS_F45 empty/missing, skipping Fig 4–5.")

    print(
        "\nCompound figs done.\n"
        f"  - {FIG1_HEATMAP_10Y_COMPOUND}\n"
        f"  - {FIG2_ERR_VS_REPS_COMPOUND}\n"
        f"  - {FIG3_ERR_VS_PROXY_COMPOUND}\n"
    )


if __name__ == "__main__":
    main()
