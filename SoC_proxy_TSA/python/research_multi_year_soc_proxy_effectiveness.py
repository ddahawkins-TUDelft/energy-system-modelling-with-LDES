import pandas as pd
import os
import json
import calliope
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
import utility_functions.helper_timeseries_tools as tt
import numpy as np
import matplotlib.pyplot as plt

import re


#_______________________________________________________________________________________________
# 
#                                  Generate Reference
#_______________________________________________________________________________________________


# # #configure reference model, standard single-year model for 2010

# date_lower_bound = '2010-01-01'
# date_upper_bound = '2012-12-31'

# # # script configuration
# params = {
#         'output_directory_name': 'multi_year_soc_proxy_effectiveness',
#         'output_model_name': 'reference',
#         'config_yaml_name': 'model',
#         'horizon_start':  date_lower_bound,
#         'horizon_end':  date_upper_bound,
#         'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
#         'calliope_full_log': [False, True],
#         'path_to_cluster_csv': 'SoC_proxy_TSA/cache/outputdata.csv'
#     }

# #construct model using standardised constructor
# print(f"  --- Configuring: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
# standard_model, filename_standard_model = standardised_model_config(params)

# #build and solve standard model
# print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")

# # auto_config calliope runtime 
# calliope.set_log_verbosity('ERROR', include_solver_output=params['calliope_full_log'][1]) 

# #build 
# standard_model.build()

# #solve
# print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
# standard_model.solve()

# #print results
# print(f"  ----- Results: Obj. Function: {standard_model.results.cost.sum().item():e}, Solve Time: {round(standard_model.results.timestamp_solve_complete - standard_model.results.timestamp_solve_start,1)}s")

# #auto-save, first checks if full directory tree exists (if not, creates it)
# output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir,filename_standard_model)
# standard_model.to_netcdf(output_path)
# print(f"  ----- Saved: Model saved to: {output_path}")

# #_______________________________________________________________________________________________
# # 
# #                                  Generate Reference
# #_______________________________________________________________________________________________

#config
set_model_paths = [
    # 'SoC_proxy_TSA/results/multi_year_soc_proxy_effectiveness/standard_2010_2014_reference.netcdf',
    'SoC_proxy_TSA/results/multi_year_soc_proxy_effectiveness/standard_2015_2019_reference.netcdf'
]

set_n_representative_days = [
    # 10,
    # 20,
    # 30,
    # 40,
    50
]

set_methods = [
    'hierarchical'
]

# #model parameters
# params = {
#         'output_directory_name': 'multi_year_soc_proxy_effectiveness',
#         'output_model_name': 'standard',
#         'config_yaml_name': 'model',
#         'horizon_start':  '2010-01-01',
#         'horizon_end':  '2010-12-31',
#         'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
#         'calliope_full_log': [False, False],
#         # 'dict_additional_overrides': {},
#         'path_to_cluster_csv': 'SoC_proxy_TSA/cache/outputdata.csv'
#     }

# #loop over models
# for path in set_model_paths:
#     model = calliope.read_netcdf(path)
#     ref_model = re.search(r'20\d{2}_20\d{2}', path).group()
#     params['horizon_start']= f"{ref_model[:4]}-01-01"
#     params['horizon_end']= f"{ref_model[-4:]}-12-31"

#     #loop pver n_repdays
#     for n_days in set_n_representative_days:

#         #loop over clustering methods
#         for method in set_methods:
#             params['output_model_name'] = f"clustered_{ref_model}_n_{n_days}_method_{method}"
#             params['path_to_cluster_csv'] = f"SoC_proxy_TSA/cache/cluster_maps/map_{ref_model}_n_{n_days}_method_{method}.csv"
#             apply_tsam_to_calliope_timeseries(model,n_days,24,method,params['path_to_cluster_csv'])
#             clustered_model, filename_clustered_model = clustered_model_config(params)

#             #build
#             print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
#             clustered_model.build()

#             #solve
#             print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
#             clustered_model.solve()

#             #print results
#             print(f"  ----- Results: Obj. Function: {clustered_model.results.cost.sum().item():e}, Solve Time: {round(clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,1)}s")

#             #auto-save, first checks if full directory tree exists (if not, creates it)
#             output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
#             os.makedirs(output_dir, exist_ok=True)
#             output_path = os.path.join(output_dir,filename_clustered_model)
#             clustered_model.to_netcdf(output_path)
#             print(f"  ----- Saved: Model saved to: {output_path}")

#_______________________________________________________________________________________________
# 
#                                  Apply SoC Proxy
#_______________________________________________________________________________________________

#function to get ldes capacity and power capacities from calliope result file
def get_capacities(calliope_model, type: str = 'standard'):
    df_storage_caps = (   
            (calliope_model.results['storage_cap'].fillna(0))
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame('storage_cap')
            .reset_index()
        )

    df_energy_caps = (   
            (calliope_model.results['flow_cap'].fillna(0))
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame('flow_cap')
            .reset_index()
        ) 
    
    if type == 'standard':
        df_state_of_charge = (   
            (calliope_model.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('soc')
            .reset_index()
        )
    else:
        df_state_of_charge = (   
            (calliope_model.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('soc')
            .reset_index()
        )
        # df_state_of_charge = (   
        #     (calliope_model.results['storage_inter_cluster'].fillna(0))
        #     .to_series()
        #     # .where(lambda x: x != 0)
        #     .dropna()
        #     .to_frame('soc')
        #     .reset_index()
        # )

        #TODO: compute the SoC profile for clustered models as per inter-cluster storage maths
        # https://calliope.readthedocs.io/en/v0.7.0.dev5/math/storage_inter_cluster/?h=inter+c#storage


    #process SoC
    df_state_of_charge=df_state_of_charge[df_state_of_charge['techs'] == 'h2_salt_cavern']
    # df_state_of_charge['timesteps']=tt.convert_to_hour_of_year(df_state_of_charge['timesteps'])
    df_state_of_charge = df_state_of_charge.set_index('timesteps')
    # df_state_of_charge['soc'] = df_state_of_charge['soc']-df_state_of_charge['soc'].iloc[0] #baseline soc to 0 starting point
    # df_state_of_charge['soc'] = df_state_of_charge['soc'] /np.mean(df_state_of_charge['soc'] ) #normalise by mean

    ldes_capacity = df_storage_caps.loc[df_storage_caps['techs'] == 'h2_salt_cavern', 'storage_cap'].iloc[0] #storage cap
    capacities = np.array(df_energy_caps.loc[df_energy_caps['carriers'] == 'power', 'flow_cap']) #array of power caps

    return ldes_capacity, capacities, df_state_of_charge


def extract_soc_from_clustered_model(ref_model, n_days, method):

    model = calliope.read_netcdf(f"SoC_proxy_TSA/results/multi_year_soc_proxy_effectiveness/clustered_{ref_model}_clustered_{ref_model}_n_{n_days}_method_{method}.netcdf")
    
    #process the clustering map
    cluster_map = pd.read_csv(f"SoC_proxy_TSA/cache/cluster_maps/map_{ref_model}_n_{n_days}_method_{method}.csv")
    cluster_map = cluster_map.rename(columns={
    'timesteps': 'datesteps',
    'PeriodNum': 'mapped_datesteps'
    })
    cluster_map['datesteps'] = pd.to_datetime(cluster_map['datesteps'], format='%Y-%m-%d')
    cluster_map['mapped_datesteps'] = pd.to_datetime(cluster_map['mapped_datesteps'], format='%Y-%m-%d')

    #pull the intracluster soc
    df_intracluster_soc = (   
            (model.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('intra_soc')
            .reset_index()
        )
    df_intracluster_soc=df_intracluster_soc[df_intracluster_soc['techs'] == 'h2_salt_cavern']
    df_intracluster_soc['mapped_datesteps'] = pd.to_datetime(df_intracluster_soc['timesteps'], format='%Y-%m-%d')


    #pull the intercluster soc
    df_intercluster_soc = (   
        (model.results['storage_inter_cluster'].fillna(0))
        .to_series()
        # .where(lambda x: x != 0)
        .dropna()
        .to_frame('inter_soc')
        .reset_index()
    )
    df_intercluster_soc=df_intercluster_soc[df_intercluster_soc['techs'] == 'h2_salt_cavern']
    df_intracluster_soc['mapped_datesteps'] = df_intracluster_soc['mapped_datesteps'].dt.normalize()
    
    #merge everything and filter
    df_result = df_intercluster_soc.merge(cluster_map, on='datesteps', how='left')
    df_result = df_result.merge(df_intracluster_soc, on='mapped_datesteps', how='left')
    df_result = df_result[['datesteps','timesteps','inter_soc','intra_soc']]

    #create a proper measure of timestamps
    time_only = df_result['timesteps'].dt.time
    df_result['full_timestamp'] = df_result['datesteps'].dt.normalize() + pd.to_timedelta(time_only.astype(str))    
    df_result = df_result.set_index('full_timestamp')


    #compute a comprehensive SoC, combining intracluster variatinos and intercluster variations
    df_result['soc_proxy_LDES'] = df_result['inter_soc']+df_result['intra_soc']

    return df_result

#define capacity weights for proxy
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

soc_decomposition = {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
        }

list_df_clustered = []
list_labels = []

#loop over models
for path in set_model_paths:

    #reference model
    reference_model = calliope.read_netcdf(path)
    ref_model = re.search(r'20\d{2}_20\d{2}', path).group()
    ref_ldes_capacity, ref_capacities, ref_soc = get_capacities(reference_model)
    val = np.max(ref_soc['soc'])

    #compute soc proxy of reference model
    ref_df_soc_proxy = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',f"{ref_model[:4]}-01-1",f"{ref_model[-4:]}-12-31")
    ref_df_soc_proxy, ref_capacity_factors, ref_nominal_capacities = generate_soc_proxy(
            df=ref_df_soc_proxy,
            demand_field='demand_power',
            renewables_fields_and_weights=capacity_weights, 
            dispatchable_techs=dispatchable_techs,
            storage_process_losses=storage_process_losses,
            soc_decomposition = soc_decomposition
        )
    

    #re-index
    ref_df_soc_proxy = ref_df_soc_proxy.set_index('timesteps')

    #loop pver n_repdays
    for n_days in set_n_representative_days:

        #loop over clustering methods
        for method in set_methods:
            print('Processing:', f"{ref_model}_n_{n_days}_method_{method}")
            cluster_model = calliope.read_netcdf(f"SoC_proxy_TSA/results/multi_year_soc_proxy_effectiveness/clustered_{ref_model}_clustered_{ref_model}_n_{n_days}_method_{method}.netcdf")
            cluster_df_soc = extract_soc_from_clustered_model(ref_model,n_days,method)
            cluster_ldes_capacity, cluster_capacities, cluster_soc = get_capacities(cluster_model, 'clustered')

            #compute soc proxy of given model model
            cluster_df_soc_proxy,s_cluster_datetimes = tt.extrapolate_ts_from_cluster_map(f"SoC_proxy_TSA/cache/cluster_maps/map_{ref_model}_n_{n_days}_method_{method}.csv",'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv')
            cluster_df_soc_proxy, cluster_capacity_factors, cluster_nominal_capacities = generate_soc_proxy(
                df=cluster_df_soc_proxy,
                demand_field='demand_power',
                renewables_fields_and_weights=capacity_weights, 
                dispatchable_techs=dispatchable_techs,
                storage_process_losses=storage_process_losses,
                soc_decomposition = soc_decomposition
            )
            cluster_df_soc_proxy = cluster_df_soc_proxy.set_index('timesteps')

            #append the real profile
            list_df_clustered.append(cluster_df_soc)
            list_labels.append(f"Cluster SoC, soc_peak={np.max(cluster_df_soc['soc_proxy_LDES']):.1e}")
            
            #append the proxy
            list_df_clustered.append(cluster_df_soc_proxy)
            list_labels.append(f"Cluster Proxy,soc_peak={np.max(cluster_df_soc_proxy['soc_proxy_LDES']):.1e}")

#plot and compare
#assign pastel colours
colours = plt.cm.Pastel1.colors
colours = ['blue', 'cyan']

#exploring the clustered soc
df_daily = cluster_df_soc.groupby('datesteps').agg({
    'intra_soc': 'last',
    'inter_soc': 'last',
    'soc_proxy_LDES': 'last'
}).reset_index()
df_daily['delta_inter_soc'] = df_daily['inter_soc'].diff()

plt.figure(figsize=(15, 5))
# Plotting 'soc_proxy' vs 'timesteps' for reference data 

for x in range(len(list_df_clustered)):
    plt.plot(list_df_clustered[x].index, list_df_clustered[x]['soc_proxy_LDES'], label=list_labels[x], color=colours[x % len(colours)], zorder=1)

plt.plot(ref_df_soc_proxy.index, ref_df_soc_proxy['soc_proxy_LDES'], label=f"Reference Proxy,soc_peak={np.max(ref_df_soc_proxy['soc_proxy_LDES']):.1e}", color='grey', zorder=101)   
# plt.plot(cluster_soc.index, cluster_soc['soc'], label=f"Cluster SoC, soc_peak={np.max(cluster_soc['soc']):.1e}, ldes={cluster_ldes_capacity:.1e}",color='gray', zorder=103)   
plt.plot(ref_soc.index, ref_soc['soc'], label=f"Reference SoC, soc_peak={np.max(ref_soc['soc']):.1e}",color='black', zorder=102)   
#*np.mean(ref_df_soc_proxy['soc_proxy'])/np.mean(ref_soc['soc'])
plt.xlabel('Time')
plt.ylabel('State of Charge')
plt.title('Storage State of Charge Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()
        