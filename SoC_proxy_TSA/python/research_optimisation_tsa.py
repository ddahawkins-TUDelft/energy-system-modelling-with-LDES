import pandas as pd
import calliope 
import numpy as np
from utility_functions.class_tsa_model import tsa_model


m = tsa_model(tsa_type='optimisation', path_timeseries='SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv')

calliope_params = {
        'type': 'clustered',
        'config_yaml_name': 'model',
        'date_range': [2015,2019],
        'calliope_full_log': [False, False],
}

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
        },
}

tsa_params = {
    'k_periods': 37,
    'matrix_weights': {
            'renewables': 1,
            'demand': 1,
            'proxy': 1,
        },
    'resample_to_daily_resolution': False,
    'names_renewables': list(soc_proxy_params['capacity_weights'].keys()),
    'distance_matrix_metric': 'euclidean',
    'name_demand': ['demand_power'],
        'soc_proxy': {
            'use_soc_proxy': True,
            'proxy_inputs_to_consider': ['surplus_LDES'],
            'proxy_window': ['2016-01-01','2017-12-31'] 
        }
}



m.soc_proxy.set_params(soc_proxy_params)
m.tsa.set_params(tsa_params)
m.calliope_model.set_params(calliope_params)
m.compute_id()
m.assign_paths(directory='SoC_proxy_TSA/data')


#TSA FUNCTIONS -------------------------------------------------------------------------------------------------

m.compute_features_dataframe()
m.compute_distance_matrix()
m.apply_tsa()

#CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

# m.configure_calliope()
# m.build_calliope()
# m.solve_and_save_calliope()

print('done')