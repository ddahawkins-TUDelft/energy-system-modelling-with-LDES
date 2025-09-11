from utility_functions.class_tsa_model import tsa_model
from utility_functions.helper_visualise import visualise
import os
import calliope
import yaml
from copy import deepcopy
from utility_functions.helper_calliope import read_clustered_netcdf, hotfix_unify_clusters_universe, hotfix_normalize_timestep_selectors

def temp_function(m):
    ds = m.calliope_model.model.inputs  # or m.model.backend.inputs depending on your handle

    print("coords:", list(ds.coords))
    print("data vars (subset):", [k for k in ds.data_vars if "cluster" in k])

    # 1) The clusters coordinate (the valid labels)
    clusters = ds.coords.get("clusters", None)
    print("clusters coord dtype:", clusters.dtype if clusters is not None else None)
    print("clusters (first 20):", clusters.values if clusters is not None else None)

    # 2) The mapping arrays
    tc = ds.get("timestep_cluster", None)
    ldc = ds.get("lookup_datestep_cluster", None)
    print("timestep_cluster:", None if tc is None else (tc.dims, tc.dtype, tc.shape))
    print("lookup_datestep_cluster:", None if ldc is None else (ldc.dims, ldc.dtype, ldc.shape))

    # 3) Sanity checks: must be scalar INT labels, no NaNs
    import numpy as np, pandas as pd
    if tc is not None:
        print("timestep_cluster NaNs:", pd.isna(tc).sum().item())
        # show a few labels paired with timesteps
        print(pd.DataFrame({
            "timestep": tc.timesteps.values[:10],
            "cluster":  tc.values[:10],
        }))

    if ldc is not None:
        print("lookup_datestep_cluster NaNs:", pd.isna(ldc).sum().item())
        print(pd.DataFrame({
            "datestep": ldc.datesteps.values[:10],
            "cluster":  ldc.values[:10],
        }))

    # 4) Label compatibility: every requested label must exist in the clusters coord
    if clusters is not None and tc is not None:
        want = np.unique(tc.values[~pd.isna(tc.values)])
        have = set(clusters.values.tolist())
        missing = [x for x in want.tolist() if x not in have]
        print("MISSING (timestep_cluster -> clusters):", missing[:20])

    if clusters is not None and ldc is not None:
        want = np.unique(ldc.values[~pd.isna(ldc.values)])
        have = set(clusters.values.tolist())
        missing = [x for x in want.tolist() if x not in have]
        print("MISSING (lookup_datestep_cluster -> clusters):", missing[:20])



def run(calliope_params, soc_proxy_params, tsa_params, tsa_type):

    #MODEL SETUP FUNCTIONS -------------------------------------------------------------------------------------------------

    m = tsa_model(tsa_type=tsa_type, path_timeseries='SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv')


    m.soc_proxy.set_params(soc_proxy_params)
    m.tsa.set_params(tsa_params)
    m.calliope_model.set_params(calliope_params)
    m.compute_id()
    m.assign_paths(directory='SoC_proxy_TSA/data')
    m.save_params()

    if os.path.exists(m.paths['calliope_model']):
        
        print(f'> Model: {m.paths['calliope_model']} already exists. Reading file...')
        # m.calliope_model.model = calliope.read_netcdf(m.paths['calliope_model'])
        m.calliope_model.model = read_clustered_netcdf(m.paths['calliope_model'])

    else:
        #TSA FUNCTIONS -------------------------------------------------------------------------------------------------

        m.configure_tsa()
        m.apply_tsa() 
            
        #CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

        m.configure_calliope()
        # hotfix_unify_clusters_universe(m)
        # hotfix_normalize_timestep_selectors(m)
        m.build_calliope()
        m.solve_and_save_calliope()

    #CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

    m.generate_soc_proxy_expost()

    #RETURN FUNCTIONS -------------------------------------------------------------------------------------------------

    return m

calliope_params = {
        'type': 'cluster',
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
    # 'resample_to_daily_resolution': True,
    'names_renewables': list(soc_proxy_params['capacity_weights'].keys()),
    # 'distance_matrix_metric': 'euclidean',
    'name_demand': ['demand_power'],
    'soc_proxy': {
        'use_soc_proxy': False,
        'proxy_inputs_to_consider': ['surplus_LDES'], #Options: surplus_LDES, soc_proxy_LDES
        'proxy_window': None
    },
    'cluster_method': 'hierarchical', #Options: k_medoids, k_means, hierarchical
    'representation_method': 'distributionRepresentation',  #Options: medoidRepresentation, meanRepresentation, distributionRepresentation
    'hours_per_period': 24,
    'soc_features': {},
    'extremes_spec': {},
    'soft_prune': False,
    'post_cluster_optimisation_params': {}
    
}


#DISPATCH CONFIGURATION  -------------------------------------------------------------------------------------------------

#open yaml config file which defines the model runs
with open('SoC_proxy_TSA/model_config/batch_run_config.yaml','r') as f:
    batch_config = yaml.safe_load(f)

scenarios = ['no_proxy','proxy_baseline','soc_features'] # 'no_proxy', 'proxy_baseline','proxy_weights','soc_features', 'soc_feature_combinations' , 'cluster_with_optimisation_baseline'

#EXECUTION FUNCTIONS -------------------------------------------------------------------------------------------------

#loop over the model runs, update parameters, and run
list_model_dict = []
for scenario_name, scenario_batch in batch_config.items():
    if scenario_name in scenarios:
        print(f'> Dispatch: running scenario {scenario_name}')
        for model_name, config in scenario_batch.items():
            
            calliope_p = deepcopy(calliope_params)
            soc_proxy_p = deepcopy(soc_proxy_params)
            tsa_p = deepcopy(tsa_params)

            if config['calliope_params']:
                for key, value in config['calliope_params'].items():
                    if value:
                        calliope_p[key] = value
            if config['soc_proxy_params']:
                for key, value in config['soc_proxy_params'].items():
                    if value:
                        soc_proxy_p[key] = value
            if config['tsa_params']:
                for key, value in config['tsa_params'].items():
                    if value:
                        tsa_p[key] = value


            m=run(calliope_p, soc_proxy_p, tsa_p, tsa_type='cluster')

            list_model_dict.append({
                'model': m,
                'name': f'id={m.id[:4]}... {model_name}', # proxy_wt={tsa_params['matrix_weights']['proxy']} {'with proxy' if use_proxy else 'without proxy'}, k={m.tsa.params['k_periods']}, agg={cluster_method}, rep={rep_method}'
                'params': {},
            })

#VISUALISATION FUNCTIONS -------------------------------------------------------------------------------------------------
print(f'> Dispatch: Loading reference standard_{calliope_params['date_range'][0]}_{calliope_params['date_range'][-1]}_reference.netcdf')
#add the reference case
list_model_dict.append({
    'model': calliope.read_netcdf(f'SoC_proxy_TSA/data/calliope_models/standard_{calliope_params['date_range'][0]}_{calliope_params['date_range'][-1]}_reference.netcdf'),
    'name': 'reference',
    'params': {
        'date_range': calliope_params['date_range'],
        'path_timeseries': 'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv',
        'soc_proxy_params': soc_proxy_params
    },
})
print('> Dispatch: Visualising results')
visualise(
    list_model_dict=list_model_dict,
    x_field='Time', #'Time'
    y_field='State of Charge', #'State of Charge', 'SoC Proxy'
    colour_field='MAGMe'
)

