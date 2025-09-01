from utility_functions.class_tsa_model import tsa_model
from utility_functions.helper_visualise import visualise_soc, visualise
import os
import calliope


def run(calliope_params, soc_proxy_params, tsa_params, tsa_type):

    #Model FUNCTIONS -------------------------------------------------------------------------------------------------

    m = tsa_model(tsa_type=tsa_type, path_timeseries='SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv')


    m.soc_proxy.set_params(soc_proxy_params)
    m.tsa.set_params(tsa_params)
    m.calliope_model.set_params(calliope_params)
    m.compute_id()
    m.assign_paths(directory='SoC_proxy_TSA/data')
    m.save_params()

    if os.path.exists(m.paths['calliope_model']):
        

        print(f'> Model: {m.paths['calliope_model']} already exists. Skipping...')
        m.calliope_model.model = calliope.read_netcdf(m.paths['calliope_model'])

    else:
        #TSA FUNCTIONS -------------------------------------------------------------------------------------------------

        m.configure_tsa()
        m.apply_tsa() 
            
        #CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

        m.configure_calliope()
        m.build_calliope()
        m.solve_and_save_calliope()
        
    #VISUALISATION FUNCTIONS -------------------------------------------------------------------------------------------------

    # print(f'> Visual: LDES Dispatch for {m.id}, k={m.tsa.params['k_periods']}, {'with SoC Proxy' if m.tsa.params['soc_proxy']['use_soc_proxy'] else ''}')
    # visualise_soc(
    #     model=m.calliope_model.model,
    #     cluster_params={
    #         'path_cluster_map': m.paths['cluster_map'],
    #         'path_timeseries': m.paths['timeseries']
    #     })

    #RETURN FUNCTIONS -------------------------------------------------------------------------------------------------

    return m

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
    # 'resample_to_daily_resolution': True,
    'names_renewables': list(soc_proxy_params['capacity_weights'].keys()),
    # 'distance_matrix_metric': 'euclidean',
    'name_demand': ['demand_power'],
    'soc_proxy': {
        'use_soc_proxy': True,
        'proxy_inputs_to_consider': ['surplus_LDES'], #Options: surplus_LDES, soc_proxy_LDES
        'proxy_window': None
    },
    'cluster_method': 'k_medoids', #Options: k_medoids, k_means, hierarchical
    'representation_method': 'medoidRepresentation',  #Options: medoidRepresentation, meanRepresentation, distributionRepresentation
    'hours_per_period': 24
}

period_length = 1826

options_compressions = [
    round(period_length*0.01),
    # round(period_length*0.02),
    round(period_length*0.03),
    # round(period_length*0.04),
    round(period_length*0.05),
]

options_use_proxy = [True, False]

list_model_dict = []

for compression in options_compressions:
    # for use_proxy in options_use_proxy:
    for rep_method in ['medoidRepresentation','meanRepresentation','distributionRepresentation']:
        for agg_method in ['hierarchical']:
            for opt in [True, False]:
                tsa_params['k_periods'] = compression
                tsa_params['cluster_method']=agg_method
                tsa_params['representation_method']=rep_method
                tsa_params['soc_proxy']['use_soc_proxy']=opt

                m=run(calliope_params, soc_proxy_params, tsa_params, tsa_type='cluster')
                list_model_dict.append({
                    'model': m.calliope_model.model,
                    'type': m.calliope_model.params['type'],
                    'name': f'k={m.tsa.params['k_periods']}, agg={agg_method}, rep={rep_method}',
                    'cluster_params': {'path_cluster_map': m.paths['cluster_map']}
                })

visualise(
    list_model_dict=list_model_dict,
    x_field='Time',
    y_field='State of Charge'
)



