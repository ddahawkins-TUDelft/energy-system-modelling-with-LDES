import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import calliope 
import numpy as np

from pathlib import Path
from utility_functions.helper_plot_tsa_timeseries import plot_cluster_vs_reference_timeseries, get_series_for_models
from utility_functions.helper_calliope import read_clustered_netcdf


from typing import Optional, Dict, Any, Tuple, List
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
from utility_functions.helper_timeseries_tools import calliope_ts_to_pandas

#--------------------------------------------------------------------------------------------------

#                                            Time Series Error

#--------------------------------------------------------------------------------------------------


cluster_id = '5d9a9d1d629c409ea74f'
reference_path = "SoC_proxy_TSA/data/calliope_models/standard_2016_2017_reference.nc"

# Minimal: make the comparison figure
# fig = plot_cluster_vs_reference_timeseries(
#     cluster_id=cluster_id,
#     reference_nc_path=reference_path,
#     storage_tech="h2_salt_cavern",   # change if needed
#     solar_tech="solar"               # change if your tech id differs (e.g., "pv")
# )
# fig.show()

# Or, grab the series if you want to do your own plotting
solar_c, solar_r, soc_c, soc_r = get_series_for_models(
    cluster_id=cluster_id,
    reference_nc_path=reference_path,
    storage_tech="h2_salt_cavern",
    solar_tech="solar"
)

var =  'ldes'

# fig = plt.figure(figsize=(9, 3.5))
# colour1='#0D0887'
# colour2='#CC4778'

# if var == 'solar':
#     solar_r = solar_r.resample("D").agg('mean')
#     solar_c = solar_c.resample("D").agg('mean')
#     plt.scatter(solar_c.index, solar_c.values, label=f"Clustered ({cluster_id})", color=colour2)
#     plt.scatter(solar_r.index, solar_r.values, label="Reference", color=colour1)
#     plt.xlabel('Time')
#     plt.ylabel('Solar Generation (MW)')
#     plt.title('Solar Generation vs. Time, for Reference and Clustered Models')
# else:
#     plt.plot(soc_c.index, soc_c.values, label=f"Clustered ({cluster_id})", color=colour2)
#     plt.plot(soc_r.index, soc_r.values, label="Reference", color=colour1)
#     plt.xlabel('Time')
#     plt.ylabel('State of Charge (MWh)')
#     plt.title('LDES State of Charge vs. Time, for Reference and Clustered Models')

# plt.grid(True)
# # plt.legend()
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()


#--------------------------------------------------------------------------------------------------

#                                           SoC vs. Proxy

#--------------------------------------------------------------------------------------------------




def _reference_soc_series(model, storage_tech: str) -> pd.Series:
    """Extract actual LDES SoC from a full (unclustered) Calliope model."""
    df = (model.results['storage'].fillna(0).to_series().dropna().to_frame('soc').reset_index())
    df = df[df['techs'] == storage_tech]
    df = df.set_index('timesteps').sort_index()
    return df['soc']

def _load_reference_timeseries_df(timeseries_csv_path: Optional[Path], ts_window: List[str]) -> pd.DataFrame:
    if timeseries_csv_path is None:
        raise FileNotFoundError('timeseries_csv_path not provided. Please pass the reference timeseries CSV used to build the model.')
    df = calliope_ts_to_pandas(timeseries_csv_path, ts_window[0], ts_window[1])
    df.set_index('timesteps', inplace=True)
    return df

def _load_soc_proxy_params(model, explicit_params: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Return (params_dict, demand_field_name)."""
    demand_field = 'demand_power'
    if explicit_params is not None:
        return explicit_params, demand_field
    if hasattr(model, 'params') and isinstance(model.params, dict) and 'soc_proxy_params' in model.params:
        return model.params['soc_proxy_params'], demand_field
    try:
        if 'soc_proxy_params' in model.attrs:
            return model.attrs['soc_proxy_params'], demand_field
    except Exception:
        pass
    raise KeyError("Could not find 'soc_proxy_params' on the model. Pass soc_proxy_params=... explicitly.")


reference_nc_path=reference_path
# Point this to the reference timeseries CSV used to build the model
timeseries_csv_path=Path("SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv")
ts_window=['2016-01-01','2017-12-31']
storage_tech="h2_salt_cavern"     # adjust if your LDES tech id differs
# Optional: pass soc_proxy_params explicitly if not embedded in the model
soc_proxy_params = {
    'capacity_weights': {
        'solar': 1,
        'onshore_wind': .5, # making the baseline assumption of an even distribution between solar and wind -based products i.e. the sum of onshore and offshore wind equals solar
        'offshore_wind': .5 
    },
    'storage_process_losses': {
        'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
        'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
    },
    'dispatchable_techs': {
        'known_dispatchable_capacity_portion_mean_demand': .25 #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
    },
    'soc_decomposition': {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    }
} 
title="Reference SoC vs Proxy, with deltas"

# Load model & actual SoC
model = read_clustered_netcdf(reference_nc_path)
soc_actual = _reference_soc_series(model, storage_tech=storage_tech)
# Build df for proxy
df = _load_reference_timeseries_df(timeseries_csv_path, ts_window)
params, demand_field = _load_soc_proxy_params(model, soc_proxy_params)
df_proxy, _, _ = generate_soc_proxy(
    df=df,
    demand_field=demand_field,
    renewables_fields_and_weights=params['capacity_weights'],
    dispatchable_techs=params['dispatchable_techs'],
    storage_process_losses=params['storage_process_losses'],
    soc_decomposition=params['soc_decomposition'],
    timestamp_col=None,
)
soc_proxy = df_proxy['soc_proxy_LDES'].rename('soc_proxy_LDES')
surplus = df_proxy['surplus_LDES'].rename('surplus_LDES')
soc_actual, soc_proxy = soc_actual.align(soc_proxy, join='inner')
_, surplus = soc_actual.align(surplus, join='inner')

soc_proxy = soc_proxy.resample("D").agg('mean')
soc_actual = soc_actual.resample("D").agg('mean')
surplus = surplus.resample("D").agg('mean')


fig = plt.figure(figsize=(9, 8))
colour1='#0D0887'
colour2='#CC4778'
colour3='#EBB5C9'
ax1 = fig.add_subplot(1,1,1)
ax1.set_ylabel('∆SoC Proxy')


ax1.bar(surplus.index, surplus.values, label='∆SoC Proxy', color=colour3, zorder=1)
ax2 = ax1.twinx()
ax2.plot(soc_actual.index, soc_actual.values, label='Actual SoC', color=colour1, zorder=2)
ax2.plot(soc_proxy.index, soc_proxy.values, label='SoC Proxy', color=colour2, zorder=3)
ax2.set_ylabel('State of Charge')
ax2.set_xlabel('Time')
ax2.grid(True)

y1 = np.r_[soc_actual.values, soc_proxy.values]  #this is just to line up the two charts
y2 = surplus.values       
ax2.set_ylim(-abs(y1).max()*0.5, abs(y1).max()*1.1)
ax1.set_ylim(-abs(y2).max()*1, abs(y2).max()*2.2)

h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper right')
if title: ax1.set_title(title)
fig.tight_layout()

plt.show()
