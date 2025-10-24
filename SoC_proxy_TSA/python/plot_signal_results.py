# plot_signal_results.py
# Combined signal analyses:
#   (A) LDES capacity error vs SoC-proxy metrics (Pearson r, RMSE, maxima timing/magnitude)
#   (B) Segmented (monthly) RMSE analysis per model, colored by LDES error
#   (C) Month-importance analysis: correlation between monthly (normalized) RMSE and LDES error across models
#
# Conventions:
# - Proxies are built hourly, with optional resampling (mean) AFTER proxy computation (e.g., daily).
# - Time alignment uses the intersection of test vs reference timestamps.
# - Paths and helper functions mirror your existing codebase.

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import calliope
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats

# --- Utilities from your codebase ---
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
from utility_functions.helper_timeseries_tools import (
    calliope_ts_to_pandas,
    extrapolate_ts_from_cluster_map,
)

# ------------------------------
# Config / paths
# ------------------------------
REFERENCE_NC = "SoC_proxy_TSA/data/calliope_models/standard_2010_2019_reference.nc"
REFERENCE_TIMESERIES_CSV = Path("SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv")

# Resample AFTER proxy build: 'D' for daily-mean metrics (recommended for LDES), or None for hourly
RESAMPLE_FREQ: str | None = "D"

# Normalization toggle for segmented RMSE (divide each model's monthly RMSE by its own mean RMSE)
NORMALIZE_SEGMENT_RMSE: bool = True

# Storage tech and proxy settings (copy/paste from your presentation file)
STORAGE_TECH = "h2_salt_cavern"
TS_WINDOW = ["2010-01-01", "2019-12-31"]
DEMAND_FIELD = "demand_power"
SOC_PROXY_PARAMS: Dict[str, Any] = {
    "capacity_weights": {"solar": 1, "onshore_wind": 0.5, "offshore_wind": 0.5},
    "storage_process_losses": {"charging_efficiency": 0.65 * 0.99, "discharging_efficiency": 0.56 * 0.99},
    "dispatchable_techs": {"known_dispatchable_capacity": 3300},
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}

# Colors
COLOUR_R = "#0D0887"  # Pearson r
COLOUR_E = "#CC4778"  # RMSE
COLOUR_EC = '#f89540' # Combined

# ------------------------------
# Path helpers
# ------------------------------
def path_nc(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/calliope_models/{model_id}.nc"

def path_cluster_map(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/cluster_maps/{model_id}.csv"

def path_timeseries(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/timeseries/{model_id}.csv"

def path_params(model_id: str) -> str:
    return f"SoC_proxy_TSA/data/parameters/{model_id}.json"

# ------------------------------
# Generic helpers
# ------------------------------
def _maybe_resample(series: pd.Series, freq: str | None) -> pd.Series:
    """If freq is provided (e.g., 'D'), resample with mean; otherwise return unchanged."""
    if freq is None:
        return series
    return series.resample(freq).mean()

def read_clustered_netcdf_with_attr_fix(path: str) -> calliope.Model:
    """Matches your attribute cleanup so Calliope doesn't think clustering is still 'active'."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File does not exist at: {path}")
    with Dataset(p, "a") as nc:  # append mode
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)
        g.setncattr("config", yaml.safe_dump(cfg))
    return calliope.read_netcdf(path)

# ------------------------------
# Parameter-file filtering (10y span, etc.)
# ------------------------------
def _extract_year(val):
    """Return an int year from int/str like 2010 or '2010-01-01'."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        m = re.search(r"\d{4}", val)
        if m:
            return int(m.group(0))
    return None

def _read_span_from_params(params_path: str) -> Tuple[int | None, int | None]:
    """Read (start_year, end_year) from params JSON if present."""
    try:
        with open(params_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, None

    cp = (data or {}).get("calliope_params", {})
    dr = cp.get("date_range")

    if isinstance(dr, (list, tuple)) and len(dr) >= 2:
        y0 = _extract_year(dr[0])
        y1 = _extract_year(dr[-1])
        return y0, y1

    if isinstance(dr, dict):
        for k_start, k_end in [("start", "end"), ("first_year", "last_year")]:
            if k_start in dr and k_end in dr:
                y0 = _extract_year(dr[k_start])
                y1 = _extract_year(dr[k_end])
                return y0, y1

    return None, None

def filter_ids_by_year_span(
    model_ids: Iterable[str], start_year: int = 2010, end_year: int = 2019
) -> List[str]:
    """Keep only model ids whose parameters/{id}.json date_range matches [start_year, end_year]."""
    kept: List[str] = []
    for mid in model_ids:
        p = Path(path_params(mid))
        if not p.is_file():
            continue
        y0, y1 = _read_span_from_params(str(p))
        if y0 is None or y1 is None:
            continue
        if y0 == start_year and y1 == end_year:
            kept.append(mid)
    return kept

# ------------------------------
# Capacity helpers (ported from plot_cem_results.py)
# ------------------------------
def _get_capacities(m: calliope.Model) -> Tuple[pd.Series, pd.Series]:
    df_power = (
        m.results["flow_cap"].fillna(0).to_series().dropna().to_frame("capacity").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    mask_drop = (
        df_power["techs"].isin(["battery", "h2_salt_cavern", "demand"])
        | df_power["techs"].str.startswith("demand")
    )
    df_power = df_power[
        ~mask_drop
        & (
            ((df_power["techs"] == "electrolyser") & (df_power["carriers"] == "hydrogen"))
            | ((df_power["techs"] != "electrolyser") & (df_power["carriers"] == "power"))
        )
    ]
    df_power.set_index("techs", inplace=True)

    df_energy = (
        m.results["storage_cap"].fillna(0).to_series().dropna().to_frame("capacity").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    df_energy = df_energy[df_energy["capacity"] > 0]
    df_energy.set_index("techs", inplace=True)

    return df_power["capacity"], df_energy["capacity"]

def _relative_error(ref: pd.Series, test: pd.Series) -> Tuple[pd.Series, float]:
    e = (ref - test) / ref
    e_mean_abs = float(np.mean(np.abs(e)))
    return e, e_mean_abs

# ------------------------------
# Proxy build & metrics
# ------------------------------
def _load_timeseries_reference(csv_path: Path, ts_window: List[str]) -> pd.DataFrame:
    df = calliope_ts_to_pandas(csv_path, ts_window[0], ts_window[1])
    df.set_index("timesteps", inplace=True)
    return df.sort_index()

def _load_timeseries_clustered(cluster_map_csv: str, csv_path: Path) -> pd.DataFrame:
    df, _ = extrapolate_ts_from_cluster_map(cluster_map_csv, csv_path)
    df.set_index("timesteps", inplace=True)
    return df.sort_index()

def _build_proxy(df: pd.DataFrame, demand_field: str, params: Dict[str, Any]) -> pd.Series:
    df_proxy, _, _ = generate_soc_proxy(
        df=df,
        demand_field=demand_field,
        renewables_fields_and_weights=params["capacity_weights"],
        dispatchable_techs=params["dispatchable_techs"],
        storage_process_losses=params["storage_process_losses"],
        soc_decomposition=params["soc_decomposition"],
        timestamp_col=None,
    )
    return df_proxy["soc_proxy_LDES"].rename("soc_proxy_LDES")

def _metrics_vs_reference_proxy(proxy_test: pd.Series, proxy_ref: pd.Series) -> Dict[str, float]:
    R, T = proxy_ref.align(proxy_test, join="inner")
    pearson_r = float(R.corr(T))
    nrmse = float(np.sqrt(np.mean(np.square(R - T))) / np.max(R))

    t_start, t_end = R.index.min(), R.index.max()
    horizon = t_end - t_start
    t_max_R, t_max_T = R.idxmax(), T.idxmax()
    t_max_delta = abs(t_max_R - t_max_T)
    max_R, max_T = float(R.loc[t_max_R]), float(T.loc[t_max_T])

    maxima_magnitude_error = np.abs((max_R - max_T) / max_R)
    maxima_timing_error = (t_max_delta / horizon) if horizon != pd.Timedelta(0) else np.nan

    return {
        "pearson_r": pearson_r,
        "rmse": nrmse,
        "e_timing_maxima": maxima_timing_error,
        "e_magnitude_maxima": maxima_magnitude_error,
    }

# ------------------------------
# Core baselines & per-model metrics
# ------------------------------
def compute_reference_baselines(resample_freq: str | None = None) -> Dict[str, Any]:
    """Load reference model + build reference proxy (optionally resampled)."""
    m_ref = read_clustered_netcdf_with_attr_fix(REFERENCE_NC)

    # Reference input TS (hourly) -> build proxy (hourly)
    df_ref_ts = _load_timeseries_reference(REFERENCE_TIMESERIES_CSV, TS_WINDOW)
    soc_ref_proxy = _build_proxy(df_ref_ts, DEMAND_FIELD, SOC_PROXY_PARAMS)

    # Optional resampling AFTER proxy (mean aggregation)
    soc_ref_proxy = _maybe_resample(soc_ref_proxy, resample_freq)

    power_caps_ref, energy_caps_ref = _get_capacities(m_ref)
    return {
        "soc_ref_proxy": soc_ref_proxy,
        "power_caps_ref": power_caps_ref,
        "energy_caps_ref": energy_caps_ref,
    }

def compute_ldes_error(model_id: str, energy_caps_ref: pd.Series) -> float:
    """Absolute relative error for LDES energy capacity (h2_salt_cavern)."""
    m_test = read_clustered_netcdf_with_attr_fix(path_nc(model_id))
    _, energy_caps_test = _get_capacities(m_test)
    e_storage, _ = _relative_error(energy_caps_ref, energy_caps_test)
    return float(np.abs(e_storage.get(STORAGE_TECH, np.nan)))

def compute_proxy_metrics(
    model_id: str, soc_ref_proxy: pd.Series, resample_freq: str | None = None
) -> Dict[str, float]:
    """Pearson r, normalized RMSE, and peak timing/magnitude errors vs reference proxy."""
    df_test = _load_timeseries_clustered(path_cluster_map(model_id), path_timeseries(model_id))
    soc_test_proxy = _build_proxy(df_test, DEMAND_FIELD, SOC_PROXY_PARAMS)
    soc_test_proxy = _maybe_resample(soc_test_proxy, resample_freq)
    return _metrics_vs_reference_proxy(soc_test_proxy, soc_ref_proxy)

def build_results(df_in: pd.DataFrame, resample_freq: str | None = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    df_in must contain at least:
      - 'id': list of model ids
      - 'x_axis': label/value used for plotting
    """
    baselines = compute_reference_baselines(resample_freq=resample_freq)
    soc_ref_proxy = baselines["soc_ref_proxy"]
    energy_caps_ref = baselines["energy_caps_ref"]

    rows = []
    for model_id, x_val in zip(df_in["id"], df_in["x_axis"]):
        ldes_err = compute_ldes_error(model_id, energy_caps_ref)
        metrics = compute_proxy_metrics(model_id, soc_ref_proxy, resample_freq=resample_freq)
        rows.append(
            {
                "id": model_id,
                "ldes_error": ldes_err,
                "pearson_r": metrics["pearson_r"],
                "rmse": metrics["rmse"],
                "e_timing_maxima": metrics["e_timing_maxima"],
                "e_magnitude_maxima": metrics["e_magnitude_maxima"],
                "e_combined": (
                    float(metrics["e_magnitude_maxima"])
                    + float(metrics["e_timing_maxima"])
                    + float(metrics["rmse"])
                    + (1.0 - float(metrics["pearson_r"]))
                )
                / 4.0,
            }
        )
    return pd.DataFrame(rows)

# ------------------------------
# Plot (A): LDES error vs proxy metrics
# ------------------------------
def plot_ldes_vs_socproxy_metrics(
    df: pd.DataFrame,
    x_label: str = r"$\epsilon^C_{\mathrm{LDES}}$ (abs. rel. error)",
    figtitle: str = "SoC Proxy vs LDES Capacity Error",
    savepath: str | None = None,
):
    """
    Expects columns: 'ldes_error', 'pearson_r', 'rmse', 'e_timing_maxima', 'e_magnitude_maxima', 'e_combined'
    """
    x = df["ldes_error"].values
    r = df["pearson_r"].values
    e = df["rmse"].values
    e_combined = df["e_combined"].values

    fig = plt.figure(figsize=(7, 4.2))
    ax_l = fig.add_subplot(1, 1, 1)

    ax_l.set_title(figtitle)
    ax_l.set_axisbelow(True)

    m_r, b_r, r_r, _, _ = scipy.stats.linregress(x, r)
    m_e, b_e, r_e, _, _ = scipy.stats.linregress(x, e)
    m_ec, b_ec, r_ec, _, _ = scipy.stats.linregress(x, e_combined)
    sorted_x = np.sort(x)

    # Left axis: Pearson r and RMSE (two series)
    ax_l.scatter(x, r, label="Pearson r (proxy vs ref)", edgecolors=COLOUR_R, linewidth=1.2)
    ax_l.plot(sorted_x,  m_r*sorted_x + b_r, label="_trend", color=COLOUR_R, linewidth=1.2)
    ax_l.annotate('r^2: ' + str("{:.2f}".format(r_r**2)), xy=(x.mean(),0.8*r.mean()))

    ax_l.scatter(x, e, label="RMSE (proxy vs ref)", color=COLOUR_E, linewidth=1.2)
    ax_l.plot(sorted_x,  m_e*sorted_x + b_e, label="_trend", color=COLOUR_E, linewidth=1.2)
    ax_l.annotate('r^2: ' + str("{:.2f}".format(r_e**2)), xy=(0.5*x.mean(),1.25*e.mean()))

    ax_l.scatter(x, e_combined, label="Combined Error", color=COLOUR_EC, linewidth=1.2)
    ax_l.plot(sorted_x,  m_ec*sorted_x + b_ec, label="_trend", color=COLOUR_EC, linewidth=1.2)
    ax_l.annotate('r^2: ' + str("{:.2f}".format(r_ec**2)), xy=(1.5*x.mean(),0.75*e_combined.mean()))


    ax_l.set_ylabel("Metric value")
    ax_l.set_xlabel(x_label)
    ax_l.yaxis.grid(True, which="major", linestyle=":", alpha=0.6)
    ax_l.legend(loc="best", frameon=False)
    ax_l.legend(loc='lower center', bbox_to_anchor=(0.5, 1))

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ==============================
# (B) Segmented (monthly) error analysis
# ==============================
def segmented_rmse_by_month(proxy_test: pd.Series, proxy_ref: pd.Series) -> pd.DataFrame:
    """
    Compute RMSE per calendar month segment (aligned on intersection).
    Returns columns: ['month', 'rmse'] where 'month' is a Period('M').
    """
    ref_aligned, test_aligned = proxy_ref.align(proxy_test, join="inner")
    if ref_aligned.empty or test_aligned.empty:
        return pd.DataFrame(columns=["month", "rmse"])

    df = pd.DataFrame({"ref": ref_aligned, "test": test_aligned})
    df["month"] = df.index.to_period("M")  # calendar-month segmentation
    grouped = df.groupby("month").apply(lambda g: float(np.sqrt(np.mean((g["ref"] - g["test"]) ** 2))))
    return grouped.to_frame("rmse").reset_index()

def build_segmented_results(
    model_ids: List[str],
    soc_ref_proxy: pd.Series,
    energy_caps_ref: pd.Series,
    resample_freq: str | None = RESAMPLE_FREQ,
    normalize_segment_rmse: bool = NORMALIZE_SEGMENT_RMSE,
) -> pd.DataFrame:
    """
    For each model id:
      - builds test proxy (hourly) -> optional resample (mean) after proxy
      - computes monthly RMSE vs resampled reference proxy
      - attaches LDES capacity error
      - optionally adds rmse_norm = rmse / mean_rmse_for_that_model
    Returns a tidy DataFrame with:
      ['model_id','month','rmse','rmse_norm','ldes_error','month_midpoint_ts']
    """
    all_rows = []
    for mid in model_ids:
        # Build test proxy hourly + optional resample AFTER proxy
        df_test = _load_timeseries_clustered(path_cluster_map(mid), path_timeseries(mid))
        soc_test_proxy = _build_proxy(df_test, DEMAND_FIELD, SOC_PROXY_PARAMS)
        soc_test_proxy = _maybe_resample(soc_test_proxy, resample_freq)

        # Monthly RMSE
        df_m = segmented_rmse_by_month(soc_test_proxy, soc_ref_proxy)
        if df_m.empty:
            continue

        # Add model id
        df_m["model_id"] = mid

        # Optional per-model normalization
        if normalize_segment_rmse:
            mean_rmse = df_m["rmse"].mean()
            df_m["rmse_norm"] = df_m["rmse"] / mean_rmse if mean_rmse and not np.isnan(mean_rmse) else np.nan
        else:
            df_m["rmse_norm"] = np.nan  # placeholder for consistent schema

        # LDES capacity error (scalar per model)
        ldes_err = compute_ldes_error(mid, energy_caps_ref)
        df_m["ldes_error"] = float(ldes_err)

        # Month midpoint timestamps for plotting on a continuous x-axis
        ts = df_m["month"].dt.to_timestamp(how="start")
        df_m["month_midpoint_ts"] = ts + pd.to_timedelta(15, unit="D")

        all_rows.append(df_m)

    if not all_rows:
        return pd.DataFrame(columns=["model_id", "month", "rmse", "rmse_norm", "ldes_error", "month_midpoint_ts"])
    return pd.concat(all_rows, ignore_index=True)

def plot_monthly_rmse_with_ref_proxy(
    df_seg: pd.DataFrame,
    proxy_ref: pd.Series,
    title: str = "Monthly RMSE vs Reference SoC Proxy (colour = LDES capacity error)",
    savepath: str | None = None,
    cmap: str = "plasma",
    show_proxy_strip: bool = True,
    height_ratios: tuple[int, int] = (4, 1),  # main : strip → strip is 4x smaller than main
    use_normalized: bool = NORMALIZE_SEGMENT_RMSE,
):
    """
    Top panel: monthly (normalized) RMSE scatter (color = LDES capacity error), continuous datetime x-axis.
    Bottom panel (optional): thin SoC Proxy strip sharing the x-axis (no y-axis).
    """
    if df_seg.empty:
        print("No segmented data to plot.")
        return

    # Select y column
    y_col = "rmse_norm" if use_normalized and "rmse_norm" in df_seg.columns else "rmse"

    # === Figure & axes (now single y on LEFT; proxy only in strip)
    if show_proxy_strip:
        fig, (ax_main, ax_strip) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6),
            gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.05}
        )
    else:
        fig, ax_main = plt.subplots(figsize=(12, 5))
        ax_strip = None

    # Prepare data
    x = pd.to_datetime(df_seg["month_midpoint_ts"].values)
    y = df_seg[y_col].values
    c = df_seg["ldes_error"].values

    # Main panel: RMSE scatter (left axis)
    norm = Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))
    sc = ax_main.scatter(x, y, c=c, cmap=cmap, norm=norm, alpha=0.9, edgecolors="none", s=24, label="Monthly RMSE")

    ax_main.set_ylabel("Monthly RMSE" + (" (normalized)" if y_col == "rmse_norm" else ""))
    ax_main.xaxis.set_major_locator(mdates.YearLocator())
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_main.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))  # two minor ticks per year
    ax_main.grid(True, which="major", axis="x", linestyle=":", alpha=0.5)

    # Compact colorbar inset
    cax = inset_axes(ax_main, width="2.2%", height="60%", loc="upper left", borderpad=1)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"LDES capacity error ($|\epsilon^C_{\mathrm{LDES}}|$)")

    # Strip panel: thin SoC proxy under the main chart
    if show_proxy_strip and ax_strip is not None:
        ref_series = proxy_ref  # scale if needed for visuals, e.g., /1e6
        ax_strip.plot(ref_series.index, ref_series.values, color="grey", linewidth=1.0)
        ax_strip.set_xlabel("")  # add bottom xlabel if you prefer
        ax_strip.yaxis.set_visible(False)
        ax_strip.spines["left"].set_visible(False)
        ax_strip.spines["right"].set_visible(False)
        ax_strip.spines["top"].set_visible(False)
        ax_strip.grid(False)

    fig.suptitle(title)
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ==============================
# (C) Month-importance across models
# ==============================
def month_importance_by_corr(
    df_seg: pd.DataFrame, use_normalized: bool = NORMALIZE_SEGMENT_RMSE
) -> pd.DataFrame:
    """
    For each calendar month across models, compute the Pearson correlation between
    (monthly RMSE or normalized RMSE) and LDES capacity error. Higher |corr| implies
    that deviations in that month are more associated with larger LDES errors.

    Returns columns:
      ['month', 'corr', 'n_models', 'month_midpoint_ts']
    """
    if df_seg.empty:
        return pd.DataFrame(columns=["month", "corr", "n_models", "month_midpoint_ts"])

    y_col = "rmse_norm" if use_normalized and "rmse_norm" in df_seg.columns else "rmse"

    # group by 'month' (Period) and compute correlation across models
    def _corr_for_month(g: pd.DataFrame) -> float:
        if g[y_col].notna().sum() < 3 or g["ldes_error"].notna().sum() < 3:
            return np.nan
        try:
            return float(pd.Series(g[y_col]).corr(pd.Series(g["ldes_error"])))
        except Exception:
            return np.nan

    grouped = df_seg.groupby("month").apply(lambda g: pd.Series({
        "corr": _corr_for_month(g),
        "n_models": g["model_id"].nunique()
    })).reset_index()

    ts = grouped["month"].dt.to_timestamp(how="start")
    grouped["month_midpoint_ts"] = ts + np.timedelta64(14,"D")  # ~mid-month

    return grouped

def plot_month_importance_bar(
    df_imp: pd.DataFrame,
    title: str = "Month importance (corr between monthly RMSE and LDES error across models)",
    savepath: str | None = None,
):
    """
    Simple bar/stem-like chart over time showing correlation per month.
    X-axis: continuous datetime (year ticks), Y-axis: Pearson correlation ([-1,1]).
    """
    if df_imp.empty:
        print("No month-importance data to plot.")
        return

    x = pd.to_datetime(df_imp["month_midpoint_ts"].values)
    y = df_imp["corr"].values

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axhline(0, color="lightgrey", linewidth=1)
    ax.stem(x, y, linefmt="-", markerfmt="o", basefmt=" ")

    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title(title)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ------------------------------
# Runner
# ------------------------------
if __name__ == "__main__":
    from os import walk

    # 1) Discover model ids from cluster_maps
    ids_all: List[str] = []
    for (dirpath, dirnames, filenames) in walk("SoC_proxy_TSA/data/cluster_maps"):
        ids_all.extend(fn.removesuffix(".csv") for fn in filenames if fn.endswith(".csv"))
        break

    # 2) Filter to 10-year models (change years here as needed)
    ids_10y = filter_ids_by_year_span(ids_all, start_year=2010, end_year=2019)

    # 3) Baselining
    baselines = compute_reference_baselines(resample_freq=RESAMPLE_FREQ)
    soc_ref_proxy = baselines["soc_ref_proxy"]
    energy_caps_ref = baselines["energy_caps_ref"]

    # 4A) Original overall chart (LDES error vs metrics)
    df_in = pd.DataFrame({"id": ids_10y, "x_axis": ids_10y})
    df_overall = build_results(df_in, resample_freq=RESAMPLE_FREQ)
    plot_ldes_vs_socproxy_metrics(df_overall, savepath="soc_proxy_vs_ldes_error.pdf")

    # 4B) Segmented monthly RMSE pathway (with optional normalization)
    df_seg = build_segmented_results(
        ids_10y,
        soc_ref_proxy,
        energy_caps_ref,
        resample_freq=RESAMPLE_FREQ,
        normalize_segment_rmse=NORMALIZE_SEGMENT_RMSE,
    )
    plot_monthly_rmse_with_ref_proxy(
        df_seg,
        soc_ref_proxy,
        title="Monthly RMSE vs Reference SoC Proxy (colour = LDES capacity error)"
              + (" [normalized]" if NORMALIZE_SEGMENT_RMSE else ""),
        savepath="monthly_rmse_vs_ref_proxy_coloured.pdf",
        use_normalized=NORMALIZE_SEGMENT_RMSE,
    )

    # 4C) Month-importance: which months' deviations are most associated with LDES errors?
    df_imp = month_importance_by_corr(df_seg, use_normalized=NORMALIZE_SEGMENT_RMSE)
    plot_month_importance_bar(
        df_imp,
        title="Month importance (corr between monthly "
              + ("normalized " if NORMALIZE_SEGMENT_RMSE else "")
              + "RMSE and LDES error across models)",
        savepath="month_importance_corr.pdf",
    )
