
# import copy
import os
import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
import calliope
from utility_functions.helper_model_config import standardised_model_config, clustered_model_config
from utility_functions.helper_tsam_calliope import apply_tsam_to_calliope_timeseries
import json

#issues with matplotlib backend and vsc. Set this config to avoid.
# matplotlib.use('Agg')

results_df = pd.DataFrame(columns=["model_name", "run_time", "objective_function", "capacities", "storage_capacities"])

# script configuration
params = {
        'output_directory_name': 'reference',
        'output_model_name': 'standard',
        'config_yaml_name': 'model',
        'horizon_start':  '2010-01-01',
        'horizon_end':  '2010-12-31',
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
        # 'dict_additional_overrides': {},
        'path_to_cluster_csv': 'SoC_proxy_TSA/cache/outputdata.csv'
    }



#construct model using standardised constructor
print(f"  --- Configuring: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
standard_model, filename_standard_model = standardised_model_config(params)

#build and solve standard model
print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")

# auto_config calliope runtime 
calliope.set_log_verbosity('ERROR', include_solver_output=params['calliope_full_log'][1]) 

#build 
standard_model.build()

#solve
print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
standard_model.solve()

#print results
print(f"  ----- Results: Obj. Function: {standard_model.results.cost.sum().item():e}, Solve Time: {round(standard_model.results.timestamp_solve_complete - standard_model.results.timestamp_solve_start,1)}s")


#auto-save, first checks if full directory tree exists (if not, creates it)
output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir,filename_standard_model)
standard_model.to_netcdf(output_path)
print(f"  ----- Saved: Model saved to: {output_path}")

#extracting key results and saving
results_df.loc[len(results_df)] = {
    "model_name": params['output_model_name'],
    "run_time": standard_model.results.timestamp_solve_complete - standard_model.results.timestamp_solve_start,
    "objective_function": standard_model.results.cost.sum().item(),
    "capacities": json.dumps(standard_model.results.flow_cap.values.tolist()),
    "storage_capacities": json.dumps(standard_model.results.storage_cap.values.tolist())
}


#_______________________________________________________________________________________________
#                                  CLUSTERING
#_______________________________________________________________________________________________


clustering_iterations = [
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1
    ]

for i in clustering_iterations:
    if i > 1:
        raise Exception(f"Invalid compression, a compression value of {i} is too high.")
    if i<=0:
        raise Exception(f"Invalid compression, a compression value of {i} is too low.")

    number_days = round(i*365)
    compression_method = 'hierarchical'
    params['output_model_name'] = f"{compression_method}_{number_days}days"
    print(f"  --- Configuring: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
    #apply TSAM to calliope
    apply_tsam_to_calliope_timeseries(standard_model,number_days,24,compression_method,params['path_to_cluster_csv'])
    #configure clustered model
    clustered_model, filename_clustered_model = clustered_model_config(params)

    #build
    print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
    clustered_model.build()

    #solve
    print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
    clustered_model.solve()

    #print results
    print(f"  ----- Results: Obj. Function: {clustered_model.results.cost.sum().item():e}, Solve Time: {round(clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,1)}s")

    #auto-save, first checks if full directory tree exists (if not, creates it)
    output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,filename_clustered_model)
    standard_model.to_netcdf(output_path)
    print(f"  ----- Saved: Model saved to: {output_path}")

    #extracting key results and saving
    results_df.loc[len(results_df)] = {
        "model_name": params['output_model_name'],
        "run_time": clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,
        "objective_function": clustered_model.results.cost.sum().item(),
        "capacities": json.dumps(clustered_model.results.flow_cap.values.tolist()),
        "storage_capacities": json.dumps(clustered_model.results.storage_cap.values.tolist())
    }

results_df.to_csv("results_highlights.csv", index=False)