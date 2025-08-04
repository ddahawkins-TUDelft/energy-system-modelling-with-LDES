import pandas as pd
import os
import json
import calliope
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
import utility_functions.helper_timeseries_tools as tt
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #avoids the annoying Qt errors on windows
import matplotlib.pyplot as plt
from utility_functions.helper_tsam_calliope import apply_tsam_to_calliope, apply_tsam_to_calliope_with_soc_proxy
from utility_functions.helper_model_config import clustered_model_config, reconstructured_full_model_config
import tkinter as tk
import re

# ----------------------------------------------------------------------------
#
#                               Configuration
#
# ----------------------------------------------------------------------------

#general model parameters
params = {
        'output_directory_name': 'soc_aware_tsa_clustering',
        'output_model_name': 'standard',
        'config_yaml_name': 'model',
        'horizon_start':  '',
        'horizon_end':  '',
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
        # 'dict_additional_overrides': {},
        'path_to_cluster_csv': 'SoC_proxy_TSA/cache/outputdata.csv',
        'path_to_new_timeseries': ''
    }

set_model_paths = [
    'SoC_proxy_TSA/results/soc_aware_tsa_clustering/standard_2015_2019_reference.netcdf'
]

set_n_representative_days = [
    # round(0.01*(5*365)), # a c.98% compression
    round(0.02*(5*365)), # a c.98% compression
    # round(0.03*(5*365)), # a c.98% compression
    # round(0.04*(5*365)), # a c.96% compression
    # round(0.05*(5*365)), # a c.95% compression
]

set_clustering_method = [
    # 'k_means',
    # 'exact k_medoid',
    'hierarchical',
]

set_representative_methods = [
    'medoidRepresentation',
    # 'distributionRepresentation',
    # 'minmaxmeanRepresentation'
]


proxy_parameters = {
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
        # 'relative_dispatchable': 0 #TODO: add in capacity for relative dispatchable
    },
    'soc_decomposition': {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    }

}
# ----------------------------------------------------------------------------
#
#                               Model Runs
#
# ----------------------------------------------------------------------------

for path in set_model_paths:
    model = calliope.read_netcdf(path)
    ref_model = re.search(r'20\d{2}_20\d{2}', path).group()
    params['horizon_start']= f"{ref_model[:4]}-01-01"
    params['horizon_end']= f"{ref_model[-4:]}-12-31"

    #loop pver n_repdays
    for n_days in set_n_representative_days:

        #loop over clustering methods
        for method in set_clustering_method:

            #loop over representation methods
            for rep_method in set_representative_methods:

                print('>> Processing: dates=',ref_model,"| n=",n_days,'| methods:',method,'+',rep_method)

                #creates output directory if it does not exist
                output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
                os.makedirs(output_dir, exist_ok=True)

                #create save paths for output models
                path_netcdf_clustered_model = os.path.join(output_dir,f"clustered_{ref_model}_n_{n_days}_{method}_{rep_method}.netcdf")
                path_netcdf_reconstructed_model = os.path.join(output_dir,f"reconstructed_{ref_model}_n_{n_days}_{method}_{rep_method}.netcdf")

                #update params parameters
                params['output_model_name'] = f"clustered_{ref_model}_n_{n_days}_{method}_{rep_method}"
                params['path_to_cluster_csv'] = f"SoC_proxy_TSA/cache/cluster_maps/{ref_model}_n_{n_days}_{method}_{rep_method}"
                params['path_to_new_timeseries'] = f"SoC_proxy_TSA/cache/clustered_timeseries/{ref_model}_n_{n_days}_{method}_{rep_method}"

                #call tsam to apply TSA clustering method to the timeseries inputs
                apply_tsam_to_calliope_with_soc_proxy(model,n_days,24,method, rep_method, f"{params['path_to_cluster_csv']}_with_soc_proxy.csv", f"{params['path_to_new_timeseries']}_with_soc_proxy.csv", proxy_parameters)
                apply_tsam_to_calliope(model,n_days,24,method, rep_method, f"{params['path_to_cluster_csv']}.csv", f"{params['path_to_new_timeseries']}.csv")

                path_netcdf_cluster = os.path.join(output_dir,f"{ref_model}_n_{n_days}_{method}_{rep_method}.netcdf")
                path_netcdf_reconstructed = os.path.join(output_dir,f"reconstructed_{ref_model}_n_{n_days}_{method}_{rep_method}.netcdf")
                path_netcdf_cluster_with_soc = os.path.join(output_dir,f"{ref_model}_n_{n_days}_{method}_{rep_method}_with_soc_proxy.netcdf")
                path_netcdf_reconstructed_with_soc = os.path.join(output_dir,f"reconstructed_{ref_model}_n_{n_days}_{method}_{rep_method}_with_soc_proxy.netcdf")

                params['path_to_cluster_csv'] = f"SoC_proxy_TSA/cache/cluster_maps/{ref_model}_n_{n_days}_{method}_{rep_method}.csv"
                params['path_to_new_timeseries'] = f"SoC_proxy_TSA/cache/clustered_timeseries/{ref_model}_n_{n_days}_{method}_{rep_method}.csv"

                #run the clustered model
                if os.path.exists(path_netcdf_cluster):
                    print(f">>> Skipping: {path_netcdf_cluster} already exists.")
                else:
                    
                    clustered_model = clustered_model_config(params)

                    #build, solve, save clustered model
                    print(f">> Building: {path_netcdf_cluster}")
                    clustered_model.build()
                    #solve
                    print(f">>> Solving: {path_netcdf_cluster}")
                    clustered_model.solve()
                    #print results
                    print(f">>> Results: Obj. Function: {clustered_model.results.cost.sum().item():e}, Solve Time: {round(clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,1)}s")
                    #auto-save, first checks if full directory tree exists (if not, creates it)
                    clustered_model.to_netcdf(path_netcdf_cluster)
                    print(f">>> Saved: {path_netcdf_cluster}")

                #run the reconstructed full model
                # if os.path.exists(path_netcdf_reconstructed):
                #     print(f">>> Skipping: {path_netcdf_reconstructed} already exists.")
                # else:
                #     full_model_reconstructed_from_clustered_timeseries = reconstructured_full_model_config(params)

                #     #build, solve, save clustered model
                #     print(f">> Building: {path_netcdf_reconstructed}")
                #     full_model_reconstructed_from_clustered_timeseries.build()
                #     #solve
                #     print(f">>> Solving: {path_netcdf_reconstructed}")
                #     full_model_reconstructed_from_clustered_timeseries.solve()
                #     #print results
                #     print(f">>> Results: Obj. Function: {full_model_reconstructed_from_clustered_timeseries.results.cost.sum().item():e}, Solve Time: {round(full_model_reconstructed_from_clustered_timeseries.results.timestamp_solve_complete - full_model_reconstructed_from_clustered_timeseries.results.timestamp_solve_start,1)}s")
                #     #auto-save, first checks if full directory tree exists (if not, creates it)
                #     full_model_reconstructed_from_clustered_timeseries.to_netcdf(path_netcdf_reconstructed)
                #     print(f">>> Saved: {path_netcdf_reconstructed}")
                
                #update params

                params['path_to_cluster_csv'] = f"SoC_proxy_TSA/cache/cluster_maps/{ref_model}_n_{n_days}_{method}_{rep_method}_with_soc_proxy.csv"
                params['path_to_new_timeseries'] = f"SoC_proxy_TSA/cache/clustered_timeseries/{ref_model}_n_{n_days}_{method}_{rep_method}_with_soc_proxy.csv"

                #run the clustered model
                if os.path.exists(path_netcdf_cluster_with_soc):
                    print(f">>> Skipping: {path_netcdf_cluster_with_soc} already exists.")
                else:
                    
                    clustered_model = clustered_model_config(params)

                    #build, solve, save clustered model
                    print(f">> Building: {path_netcdf_cluster_with_soc}")
                    clustered_model.build()
                    #solve
                    print(f">>> Solving: {path_netcdf_cluster_with_soc}")
                    clustered_model.solve()
                    #print results
                    print(f">>> Results: Obj. Function: {clustered_model.results.cost.sum().item():e}, Solve Time: {round(clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,1)}s")
                    #auto-save, first checks if full directory tree exists (if not, creates it)
                    clustered_model.to_netcdf(path_netcdf_cluster_with_soc)
                    print(f">>> Saved: {path_netcdf_cluster_with_soc}")

                #run the reconstructed full model
                # if os.path.exists(path_netcdf_reconstructed_with_soc):
                #     print(f">>> Skipping: {path_netcdf_reconstructed_with_soc} already exists.")
                # else:
                #     full_model_reconstructed_from_clustered_timeseries = reconstructured_full_model_config(params)

                #     #build, solve, save clustered model
                #     print(f">> Building: {path_netcdf_reconstructed_with_soc}")
                #     full_model_reconstructed_from_clustered_timeseries.build()
                #     #solve
                #     print(f">>> Solving: {path_netcdf_reconstructed_with_soc}")
                #     full_model_reconstructed_from_clustered_timeseries.solve()
                #     #print results
                #     print(f">>> Results: Obj. Function: {full_model_reconstructed_from_clustered_timeseries.results.cost.sum().item():e}, Solve Time: {round(full_model_reconstructed_from_clustered_timeseries.results.timestamp_solve_complete - full_model_reconstructed_from_clustered_timeseries.results.timestamp_solve_start,1)}s")
                #     #auto-save, first checks if full directory tree exists (if not, creates it)
                #     full_model_reconstructed_from_clustered_timeseries.to_netcdf(path_netcdf_reconstructed_with_soc)
                #     print(f">>> Saved: {path_netcdf_reconstructed_with_soc}")


# ----------------------------------------------------------------------------
#
#                               Plot Results
#
# ----------------------------------------------------------------------------

list_df_clustered = []
list_labels = []

for path in set_model_paths:
    model = calliope.read_netcdf(path)
    ref_model = re.search(r'20\d{2}_20\d{2}', path).group()
    params['horizon_start']= f"{ref_model[:4]}-01-01"
    params['horizon_end']= f"{ref_model[-4:]}-12-31"

    #reference model
    reference_model = calliope.read_netcdf(path)
    ref_model = re.search(r'20\d{2}_20\d{2}', path).group()
    _, _, ref_soc = tt.get_capacities(reference_model)

    #compute soc proxy of reference model
    ref_df_soc_proxy = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',f"{ref_model[:4]}-01-1",f"{ref_model[-4:]}-12-31")
    ref_df_soc_proxy, _, _ = generate_soc_proxy(
            df=ref_df_soc_proxy,
            demand_field='demand_power',
            renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
            dispatchable_techs=proxy_parameters['dispatchable_techs'],
            storage_process_losses=proxy_parameters['storage_process_losses'],
            soc_decomposition = proxy_parameters['soc_decomposition'],
            timestamp_col='timesteps'
    )
    ref_df_soc_proxy = ref_df_soc_proxy.set_index('timesteps')

    #loop pver n_repdays
    for n_days in set_n_representative_days:

        #loop over clustering methods
        for method in set_clustering_method:

            #loop over representation methods
            for rep_method in set_representative_methods:

                print('>> Extracting Results: dates=',ref_model,"| n=",n_days,'| methods:',method,'+',rep_method)

                #creates output directory if it does not exist
                output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
                os.makedirs(output_dir, exist_ok=True)

                #create save paths for output models
                dictionary_paths = {
                    'clustered with proxy': os.path.join(output_dir,f"{ref_model}_n_{n_days}_{method}_{rep_method}_with_soc_proxy.netcdf"),
                    # 'reconstructed with proxy': os.path.join(output_dir,f"reconstructed_{ref_model}_n_{n_days}_{method}_{rep_method}_with_soc_proxy.netcdf")
                    'clustered': os.path.join(output_dir,f"{ref_model}_n_{n_days}_{method}_{rep_method}.netcdf"),
                    # 'reconstructed': os.path.join(output_dir,f"reconstructed_{ref_model}_n_{n_days}_{method}_{rep_method}.netcdf"),
                }

                #loop over the four versions of each clustered, reconstructed, clustered with soc proxy, reconstructed with soc proxy
                for key,path in dictionary_paths.items():

                    cluster_model = calliope.read_netcdf(path)
                    
                    cluster_df_soc_proxy = pd.DataFrame() #initialising these to be save
                    cluster_df_soc = pd.DataFrame() #initialising these to be save

                    map_with_soc_proxy = False
                    if re.match(r'^clustered with proxy\b', key):
                        map_with_soc_proxy = True

                    #if clustered use the cluster route, otherwise extract the full soc
                    if re.match(r'^clustered\b', key):
                        cluster_df_soc = tt.extract_soc_from_clustered_model(path,ref_model,n_days,method, rep_method, map_with_soc_proxy)
                        cluster_df_soc_proxy,_ = tt.extrapolate_ts_from_cluster_map(
                            f"SoC_proxy_TSA/cache/cluster_maps/{ref_model}_n_{n_days}_{method}_{rep_method}.csv",
                            f"SoC_proxy_TSA/cache/clustered_timeseries/{ref_model}_n_{n_days}_{method}_{rep_method}.csv",
                        )

                    else:
                        _, _, cluster_df_soc = tt.get_capacities(cluster_model)
                        cluster_df_soc_proxy = tt.calliope_ts_to_pandas(f"SoC_proxy_TSA/cache/clustered_timeseries/{ref_model}_n_{n_days}_{method}_{rep_method}.csv") #'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',f"{ref_model[:4]}-01-1",f"{ref_model[-4:]}-12-31" #TODO: should this be the recomputed timeseries?

                    #compute soc proxy of given model model
                    cluster_df_soc_proxy, _, _ = generate_soc_proxy(
                        df=cluster_df_soc_proxy,
                        demand_field='demand_power',
                        renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
                        dispatchable_techs=proxy_parameters['dispatchable_techs'],
                        storage_process_losses=proxy_parameters['storage_process_losses'],
                        soc_decomposition = proxy_parameters['soc_decomposition'],
                        timestamp_col='timesteps'
                    )
                    cluster_df_soc_proxy = cluster_df_soc_proxy.set_index('timesteps')


                    #save down the soc output of the model
                    list_df_clustered.append(cluster_df_soc)
                    list_labels.append(f"SoC, {key}, {n_days}, {method}_{rep_method}, soc_peak={np.max(cluster_df_soc['soc']):.1e} on {cluster_df_soc['soc'].idxmax().strftime('%Y-%m-%d')}")

                    #save down the soc proxy of the model
                    list_df_clustered.append(cluster_df_soc_proxy)
                    list_labels.append(f"SoC Proxy, {key}, {n_days}, {method}_{rep_method}, soc_peak={np.max(cluster_df_soc_proxy['soc_proxy_LDES']):.1e} on {cluster_df_soc_proxy['soc_proxy_LDES'].idxmax().strftime('%Y-%m-%d')}")

#plot and compare
n_plots = len(list_df_clustered)
plasma_colors = plt.cm.plasma(np.linspace(0, 1, n_plots))


plt.figure(figsize=(12, 6))

# Plotting 'soc_proxy' vs 'timesteps' for reference data 
plt.plot(ref_soc.index, ref_soc['soc'], label=f"SoC, Reference, soc_peak={np.max(ref_soc['soc']):.1e} on {ref_soc['soc'].idxmax().strftime('%Y-%m-%d')}",color='black', zorder=102)  
plt.plot(ref_df_soc_proxy.index, ref_df_soc_proxy['soc_proxy_LDES'], label=f"Proxy, Reference, soc_peak={np.max(ref_df_soc_proxy['soc_proxy_LDES']):.1e} on {ref_df_soc_proxy['soc_proxy_LDES'].idxmax().strftime('%Y-%m-%d')}", color='grey', zorder=101)   

# === Plotting logic ===
show_soc = True
show_proxy = True

# Gather filtered curves based on flags
plot_data = []


# include soc in output
if show_soc:
    soc_filtered = [
        (df, label, 'soc')
        for i, (df, label) in enumerate(zip(list_df_clustered, list_labels))
        if i % 2 == 0 #and re.match(r'^SoC, clustered with proxy\b', label)
    ]
    plot_data.extend(soc_filtered)

# include proxy in output
if show_proxy:
    proxy_filtered = [
        (df, label, 'soc_proxy_LDES')
        for i, (df, label) in enumerate(zip(list_df_clustered, list_labels))
        if i % 2 == 1
    ]
    plot_data.extend(proxy_filtered)

# Generate consistent colors across all plotted curves
plasma_colors = plt.cm.plasma(np.linspace(0, 1, len(plot_data)))

# Plot all
for (df, label, column), color in zip(plot_data, plasma_colors):
    plt.plot(df.index, df[column], label=label, color=color, zorder=1)

plt.xlabel('Time')
plt.ylabel('State of Charge')
plt.title('Storage State of Charge Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()