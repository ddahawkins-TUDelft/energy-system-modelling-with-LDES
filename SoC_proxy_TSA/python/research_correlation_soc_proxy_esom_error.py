import pandas as pd
import os
import json
import calliope
from utility_functions.helper_SoC_proxy import generate_SoC_proxy
import utility_functions.helper_timeseries_tools as tt
from utility_functions.helper_plot_soc_comparison import figure_compare_SoC, figure_compare_muliple_SoCs
from utility_functions.helper_model_config import standardised_model_config, clustered_model_config
from utility_functions.helper_tsam_calliope import apply_tsam_to_calliope_timeseries
import numpy as np
from utility_functions.helper_seasonal_sampling import monte_carlo_tsa



# #_______________________________________________________________________________________________
# # 
# #                                  Generate Reference
# #_______________________________________________________________________________________________


# #configure reference model, standard single-year model for 2010

date_lower_bound = '2010-01-01'
date_upper_bound = '2010-12-31'

# # script configuration
params = {
        'output_directory_name': 'correlation_soc_proxy_esom_error',
        'output_model_name': 'reference',
        'config_yaml_name': 'model',
        'horizon_start':  date_lower_bound,
        'horizon_end':  date_upper_bound,
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
        'path_to_cluster_csv': 'SoC_proxy_TSA/cache/outputdata.csv'
    }

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

#_______________________________________________________________________________________________
# 
#                                  Generate Test Models
#_______________________________________________________________________________________________

#procedurally generate random test models
#strategy, monte-carlo esque random selection of representative days but ensuring that days are evenly distributed across seasons.

n_days = 40
n_samples = 10
year = 2010

tsa_samples = {}

clustering_source_data = 'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv'

for i in range(n_samples):
    
    assignment = monte_carlo_tsa(
        clustering_source_data,
        date_lower_bound=date_lower_bound,
        date_upper_bound=date_upper_bound,
        feature_cols=['demand_power', 'onshore_wind', 'offshore_wind', 'solar'],
        n_rep_days=40,
        seed=i
    )

    tsa_samples[i] = assignment
    assignment_to_save = assignment[['timesteps', 'PeriodNum']]
    assignment_to_save.to_csv(f"SoC_proxy_TSA/cache/cluster_maps/n_days_{n_days}_seed_{i}.csv", index=False)

#_______________________________________________________________________________________________
# 
#                                  Compare Ex-ante SoC Proxies
#_______________________________________________________________________________________________

# df_reference = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv','2010-01-01','2010-12-31')

# n_days = 40
# n_samples = 10

# capacity_weights = {
#     'solar': 1,
#     'onshore_wind': 1,
#     'offshore_wind': 1
# }

# df_reference,threshold_SoC_reference = generate_SoC_proxy(df_reference,'demand_power',capacity_weights)

# list_df_clustered = []
# list_labels = []

# for i in range(n_samples):
#     df_clustered,s_cluster_datetimes = tt.extrapolate_ts_from_cluster_map(f"SoC_proxy_TSA/cache/cluster_maps/n_days_{n_days}_seed_{i}.csv",'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv')
#     df, alpha = generate_SoC_proxy(df_clustered,'demand_power',capacity_weights)
#     list_df_clustered.append(df)
#     list_labels.append(f"α={round(alpha,4)}")

# #plot and compare
# figure_compare_muliple_SoCs(
#     df_reference=df_reference,
#     list_of_test_dfs=list_df_clustered,
#     label_graph_1='Reference',
#     list_of_test_labels=list_labels
# )

#_______________________________________________________________________________________________
# 
#                                  Run the sample models
#_______________________________________________________________________________________________

n_days = 40
n_samples = 10
for i in range(n_samples):

    ref = f"n_days_{n_days}_seed_{i}"
    params['path_to_cluster_csv'] = f"SoC_proxy_TSA/cache/cluster_maps/{ref}.csv"

    #configure clustered model
    clustered_model, filename_clustered_model = clustered_model_config(params)

    #build
    print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
    clustered_model.build()

    #solve
    print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
    clustered_model.solve()

    #auto-save, first checks if full directory tree exists (if not, creates it)
    output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,ref,'.netcdf')
    clustered_model.to_netcdf(output_path)
    print(f"  ----- Saved: Model saved to: {output_path}")