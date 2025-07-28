from utility_functions.helper_SoC_proxy_fast_compute_with_curtailment_correction import generate_soc_proxy
import calliope
import re
import utility_functions.helper_timeseries_tools as tt
import time


# import timeseries using helper function that also filters over date range


capacity_weights = {
    'solar': 2,
    'onshore_wind': 1,
    'offshore_wind': 1
}

storage_process_losses = {
    'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
    'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
}

dispatchable_techs = {
        'known_dispatchable_capacity_portion_mean_demand': .25 #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
        # 'relative_dispatchable': 0 #TODO: add in capacity for relative dispatchable
    }


path = 'C:/Users/dhawkins/Offline Workspace/Development/energy-system-modelling-with-LDES/SoC_proxy_TSA/results/multi_year_soc_proxy_effectiveness/standard_2015_2019_reference.netcdf'
reference_model = calliope.read_netcdf(path)
ref_model = re.search(r'20\d{2}_20\d{2}', path).group()

#compute soc proxy of reference model
ref_df_soc_proxy = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',f"{ref_model[:4]}-01-1",f"{ref_model[-4:]}-12-31")

dictionary_costs = {
    'storage': {
        'capex': 0.003190,
        'opex': 0
    },
    'charging': {
        'capex': 1.2,
        'opex': 0
    },
    'discharging': {
        'capex': 0.093,
        'opex': 0
    },
    'solar': {
        'capex': 0.56,
        'opex': 0
    },
    'onshore_wind': {
        'capex': 2.12,
        'opex': 0
    },
    'offshore_wind': {
        'capex': 1.11,
        'opex': 0
    },
}

start_time = time.time()

df, cap_fac, installed_caps_nominal = generate_soc_proxy(
    df=ref_df_soc_proxy,
    demand_field='demand_power',
    renewables_fields_and_weights={
        'solar': 2,
        'onshore_wind': 1,
        'offshore_wind': 1
        }, 
    dispatchable_techs=dispatchable_techs,
    storage_process_losses={
        'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
        'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
        },
    soc_decomposition = {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
        },
    timestamp_col= 'timesteps',
    dictionary_costs = dictionary_costs
    )
end_time = time.time()
runtime = end_time - start_time

print(f"Runtime: {runtime*1000:.5f} milliseconds")
print('done')