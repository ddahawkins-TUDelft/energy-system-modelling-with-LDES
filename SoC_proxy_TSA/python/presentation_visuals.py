# --- SoC & Proxy for REFERENCE + CLUSTERED, LaTeX/PGF export, metrics, maxima labels ---

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

# Utilities you already have
from utility_functions.helper_calliope import read_clustered_netcdf
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
from utility_functions.helper_timeseries_tools import calliope_ts_to_pandas, extrapolate_ts_from_cluster_map

# ------------------------------
# Matplotlib + LaTeX (PGF)
# ------------------------------
# mpl.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "axes.unicode_minus": False,
# })
# mpl.rcParams["pgf.preamble"] = r""  # inherit fonts from your Elsevier doc

# ------------------------------
# Inputs / Paths
# ------------------------------
# cluster_id = '145cfcdba374fdda32e5' #no proxy
cluster_id = '1dff92aac973a8452fdc' 

# Reference (unclustered/full) model
reference_nc_path = "SoC_proxy_TSA/data/calliope_models/standard_2010_2019_reference.nc"
reference_timeseries_csv_path = Path("SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv")

# Clustered model (fill these 3 to match your run artifacts)
clustered_nc_path = f"SoC_proxy_TSA/data/calliope_models/{cluster_id}.nc"               # <-- set me
cluster_map_csv_path = f"SoC_proxy_TSA/data/cluster_maps/{cluster_id}.csv"                         # <-- set me
clustered_timeseries_csv_path = Path(f"SoC_proxy_TSA/data/timeseries/{cluster_id}.csv")  # <-- set me if different

# Window and tech ids
ts_window = ['2010-01-01', '2019-12-31']
storage_tech = "h2_salt_cavern"

# Proxy params (used if not embedded)
soc_proxy_params = {
    'capacity_weights': {'solar': 1, 'onshore_wind': 0.5, 'offshore_wind': 0.5},
    'storage_process_losses': {'charging_efficiency': 0.65 * 0.99, 'discharging_efficiency': 0.56 * 0.99},
    'dispatchable_techs': {'known_dispatchable_capacity': 3300},
    'soc_decomposition': {'method': 'fft_lowpass', 'time_horizon_hours': 24},
}
demand_field = 'demand_power'

# Outputs
pgf_out = "soc_vs_proxy_ref_vs_clustered.pgf"
pdf_out = "soc_vs_proxy_ref_vs_clustered.pdf"

# Colors per *model* (same color for its solid/dashed pair)
colour_ref = '#0D0887'   # reference actual + proxy (solid/dashed)
colour_clu = '#CC4778'   # clustered actual + proxy (solid/dashed)

# ------------------------------
# Helpers
# ------------------------------
def _soc_series_unclustered(model, storage_tech: str) -> pd.Series:
    """SoC from a full/unclustered Calliope model (a.k.a. reference)."""
    df = (model.results['storage'].fillna(0).to_series().dropna()
          .to_frame('soc').reset_index())
    df = df[df['techs'] == storage_tech]
    df = df.set_index('timesteps').sort_index()
    return df['soc']

def _soc_series_clustered(model, cluster_map_csv: str, storage_tech: str) -> pd.Series:
    """
    Reconstruct SoC for clustered model by combining inter-cluster + intra-cluster
    and re-expanding to full timestamps using the cluster map.
    Mirrors the approach in your uploaded adapters.
    """
    # cluster map: columns like 'timesteps' (original) and 'PeriodNum' (mapped day index)
    df_map = pd.read_csv(cluster_map_csv)
    # normalize names to what we use below
    df_map = df_map.rename(columns={'timesteps': 'datesteps', 'PeriodNum': 'mapped_datesteps'})
    df_map['datesteps'] = pd.to_datetime(df_map['datesteps'])
    df_map['mapped_datesteps'] = pd.to_datetime(df_map['mapped_datesteps'])

    # intra-cluster (within representative periods)
    df_intra = (model.results['storage'].fillna(0).to_series().dropna()
                .to_frame('intra').reset_index())
    df_intra = df_intra[df_intra['techs'] == storage_tech]
    # map intra day → a "mapped date"
    df_intra['mapped_datesteps'] = pd.to_datetime(df_intra['timesteps']).dt.normalize()  # date only
    t_only = pd.to_datetime(df_intra['timesteps']).dt.time                                  # time-of-day

    # inter-cluster (day-to-day jump)
    df_inter = (model.results['storage_inter_cluster'].fillna(0).to_series().dropna()
                .to_frame('inter').reset_index())
    df_inter = df_inter[df_inter['techs'] == storage_tech]

    # merge inter with map (to get actual dates), then bring in intra by mapped date
    df = df_inter.merge(df_map, on='datesteps', how='left')
    df = df.merge(df_intra[['mapped_datesteps', 'intra', 'timesteps']], on='mapped_datesteps', how='left')

    # reconstruct full timestamps: actual-date + time-of-day from intra timesteps
    ts = df['datesteps'].dt.normalize() + pd.to_timedelta(pd.to_datetime(df['timesteps']).dt.time.astype(str))
    soc = (df['inter'].fillna(0) + df['intra'].fillna(0)).rename('soc')
    soc.index = pd.DatetimeIndex(ts, name='timesteps')
    return soc.sort_index()

def _load_timeseries_reference(csv_path: Path, ts_window: List[str]) -> pd.DataFrame:
    df = calliope_ts_to_pandas(csv_path, ts_window[0], ts_window[1])
    df.set_index('timesteps', inplace=True)
    return df.sort_index()

def _load_timeseries_clustered(cluster_map_csv: str, csv_path: Path) -> pd.DataFrame:
    """
    Re-expand clustered exogenous timeseries back to the full timeline
    so the proxy is computed in the original time domain.
    """
    df, _ = extrapolate_ts_from_cluster_map(cluster_map_csv, csv_path)
    df.set_index('timesteps', inplace=True)
    return df.sort_index()

def _build_proxy(df: pd.DataFrame, demand_field: str, params: Dict[str, Any]) -> pd.Series:
    df_proxy, _, _ = generate_soc_proxy(
        df=df,
        demand_field=demand_field,
        renewables_fields_and_weights=params['capacity_weights'],
        dispatchable_techs=params['dispatchable_techs'],
        storage_process_losses=params['storage_process_losses'],
        soc_decomposition=params['soc_decomposition'],
        timestamp_col=None,
    )
    return df_proxy['soc_proxy_LDES'].rename('soc_proxy_LDES')

def _daily_twh(series: pd.Series) -> pd.Series:
    """Daily mean and convert from MWh→TWh if input is Wh-based; adjust if yours is already in Wh."""
    # You were dividing by 1e6 previously (MWh→TWh). Keep that here.
    return (series.resample("D").mean()) / 1e6

def _metrics(actual: pd.Series, proxy: pd.Series, label: str):
    A, P = actual.align(proxy, join="inner")
    pearson_r = A.corr(P)
    errors = P - A
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    nrmse_max = rmse / float(A.max()) if A.max() != 0 else np.nan

    t_start, t_end = A.index.min(), A.index.max()
    horizon = t_end - t_start
    t_max_A, t_max_P = A.idxmax(), P.idxmax()
    max_A, max_P = float(A.loc[t_max_A]), float(P.loc[t_max_P])

    delta = abs(t_max_P - t_max_A)
    timing_frac = (delta / horizon) if horizon != pd.Timedelta(0) else np.nan
    magnitude_err_at_peak = (max_A - max_P) / max_A if max_A != 0 else np.nan

    print(f"[Metrics: {label}]")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  RMSE: {rmse:.4f} TWh   (nRMSE_max: {nrmse_max:.2%})")
    print(f"  Peak actual: {max_A:.3f} TWh on {t_max_A.date()}")
    print(f"  Peak proxy : {max_P:.3f} TWh on {t_max_P.date()}")
    print(f"  Magnitude error at peak: {magnitude_err_at_peak:.2%} (proxy vs actual)")
    print(f"  Timing error: {timing_frac:.2%} of horizon\n")

    return {
        "pearson_r": pearson_r,
        "rmse": rmse,
        "nrmse_max": nrmse_max,
        "t_max_actual": t_max_A, "max_actual": max_A,
        "t_max_proxy": t_max_P,  "max_proxy": max_P,
    }

# ------------------------------
# Load models & series
# ------------------------------
# Reference
m_ref = read_clustered_netcdf(reference_nc_path)    # “reference” is still a calliope.Model
soc_ref_actual = _soc_series_unclustered(m_ref, storage_tech=storage_tech)
df_ref = _load_timeseries_reference(reference_timeseries_csv_path, ts_window)
soc_ref_proxy = _build_proxy(df_ref, demand_field, soc_proxy_params)

# Clustered
m_clu = read_clustered_netcdf(clustered_nc_path)
soc_clu_actual = _soc_series_clustered(m_clu, cluster_map_csv_path, storage_tech=storage_tech)
df_clu = _load_timeseries_clustered(cluster_map_csv_path, clustered_timeseries_csv_path)
soc_clu_proxy = _build_proxy(df_clu, demand_field, soc_proxy_params)

# Align to same window + daily TWh
soc_ref_actual, soc_ref_proxy = soc_ref_actual.align(soc_ref_proxy, join="inner")
soc_ref_actual_d = _daily_twh(soc_ref_actual)
soc_ref_proxy_d  = _daily_twh(soc_ref_proxy)

soc_clu_actual, soc_clu_proxy = soc_clu_actual.align(soc_clu_proxy, join="inner")
soc_clu_actual_d = _daily_twh(soc_clu_actual)
soc_clu_proxy_d  = _daily_twh(soc_clu_proxy)

# ------------------------------
# Metrics
# ------------------------------
metrics_ref = _metrics(soc_ref_actual_d, soc_ref_proxy_d, label="Reference")
metrics_clu = _metrics(soc_clu_actual_d, soc_clu_proxy_d, label="Clustered")

# ------------------------------
# Plot
# ------------------------------
fig = plt.figure(figsize=(12, 4))  # wide × short
ax = fig.add_subplot(1, 1, 1)
ax.set_axisbelow(True)

# Reference model (same color; solid = actual, dashed = proxy)
ax.plot(soc_ref_actual_d.index, soc_ref_actual_d.values,
        label='Reference: SoC (CEM)', color=colour_ref, linewidth=1.2)
ax.plot(soc_ref_proxy_d .index, soc_ref_proxy_d.values,
        label='Reference: SoC Proxy', color=colour_ref, linewidth=1.2, linestyle='dotted')

# # Clustered model (same color; solid = actual, dashed = proxy)
# ax.plot(soc_clu_actual_d.index, soc_clu_actual_d.values,
#         label='Clustered: SoC (CEM)', color=colour_clu, linewidth=1.2)
# ax.plot(soc_clu_proxy_d .index, soc_clu_proxy_d .values,
#         label='Clustered: SoC Proxy', color=colour_clu, linewidth=1.2, linestyle='dotted')

ax.set_ylabel('State of Charge (TWh)')
ax.set_xlabel('Time')

# vertical grid
ax.xaxis.grid(True, which='major', linestyle=':', alpha=0.6)

# Maxima annotations (keep boxes small; offset a bit differently per model)
def _annotate_max(series: pd.Series, color: str, label: str, dy: int = 12):
    tmax = series.idxmax()
    vmax = float(series.loc[tmax])
    ax.scatter([tmax], [vmax], color=color, s=14, zorder=5)
    ax.annotate(
        f"{label}\n{vmax:.1f} TWh\n{pd.to_datetime(tmax).date()}",
        xy=(tmax, vmax),
        xytext=(5, dy), textcoords='offset points',
        fontsize=8, ha='left', va='bottom',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8, alpha=0.95),
        arrowprops=dict(arrowstyle='-', lw=0.7, color=color, alpha=0.85),
    )

_annotate_max(soc_ref_actual_d, colour_ref, "Ref max (CEM)", dy=12)
_annotate_max(soc_ref_proxy_d,  colour_ref, "Ref max (Proxy)", dy=28)
# _annotate_max(soc_clu_actual_d, colour_clu, "Clu max (CEM)", dy=-26)
# _annotate_max(soc_clu_proxy_d,  colour_clu, "Clu max (Proxy)", dy=-10)

ax.legend(loc='best', frameon=False)
fig.tight_layout()

# ------------------------------
# Save
# ------------------------------
plt.savefig(pgf_out)                       # PGF for \input{...}
plt.savefig(pdf_out, bbox_inches='tight')  # quick preview
plt.show()

