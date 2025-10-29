import calliope
import pandas as pd
import matplotlib.pyplot as plt

from utility_functions.helper_SoC_proxy_fast_compute_test import generate_soc_proxy
from plot_signal_results import _load_timeseries_reference


ref_path, TS_WINDOW = 'SoC_proxy_TSA/data/calliope_models/standard_2010_2019_reference.nc', ["2010-01-01", "2019-12-31"] #'SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc'
ref_path, TS_WINDOW = 'SoC_proxy_TSA/data/calliope_models/standard_2006_2015_reference.nc', ["2006-01-01", "2015-12-31"]


tvp_csv = 'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv'
DEMAND_FIELD = "demand_power"

SOC_PROXY_PARAMS = {
    "capacity_weights": {"solar": 1, "onshore_wind": 0.5, "offshore_wind": 0.5},
    "storage_process_losses": {"charging_efficiency": 0.65 * 0.99, "discharging_efficiency": 0.56 * 0.99},
    "dispatchable_techs": {"known_dispatchable_capacity": 3300},
    "soc_decomposition": {"method": "fft_lowpass", "time_horizon_hours": 24},
}

ref_model = calliope.read_netcdf(ref_path)

# Colors
COLOUR_R = "#0d0887"   # Pearson r
COLOUR_E = "#6a00a8"   # RMSE
COLOUR_EC = "#b12a90"  # Combined
COLOUR_TM = "#e16462"  # Timing of maxima
COLOUR_5 = "#fca636"  
COLOUR_MM = "#f0f921" # Magnitude of maxima

def _build_proxy(df: pd.DataFrame, demand_field: str, params) -> pd.Series:
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

df_ref_ts = _load_timeseries_reference(tvp_csv, TS_WINDOW)
soc_ref_proxy = _build_proxy(df_ref_ts, DEMAND_FIELD, SOC_PROXY_PARAMS)

df_ref_ts['soc_proxy_LDES'] = soc_ref_proxy

df_storage = (
        ref_model.results["storage"].fillna(0).to_series().dropna().to_frame("soc").reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )

df_storage=df_storage[df_storage['techs']=='h2_salt_cavern']
df_storage.set_index('timesteps', inplace=True)


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_axisbelow(True)

ax.plot(df_ref_ts.index, df_ref_ts["soc_proxy_LDES"],
           label="proxy",
           color=COLOUR_TM, linewidth=1.2)
ax.plot(df_storage.index, df_storage["soc"],
           label="soc",
           color=COLOUR_R, linewidth=1.2)

ax.set_ylabel('SoC')
ax.set_xlabel('Time')
ax.yaxis.grid(True, which='major', linestyle=':', alpha=0.6)
ax.legend(loc='best', frameon=False)
fig.tight_layout()
plt.show()