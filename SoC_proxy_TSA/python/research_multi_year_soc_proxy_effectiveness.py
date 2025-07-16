import pandas as pd
import os
import json
import calliope
from utility_functions.helper_SoC_proxy import generate_SoC_proxy, standardised_profile_comparison
import utility_functions.helper_timeseries_tools as tt
from utility_functions.helper_plot_soc_comparison import  figure_compare_muliple_SoCs
from utility_functions.helper_model_config import standardised_model_config, clustered_model_config
from utility_functions.helper_normalise import normalise_by_method
import numpy as np
from utility_functions.helper_seasonal_sampling import monte_carlo_tsa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


# #_______________________________________________________________________________________________
# # 
# #                                  Generate Reference
# #_______________________________________________________________________________________________


# #configure reference model, standard single-year model for 2010

date_lower_bound = '2015-01-01'
date_upper_bound = '2018-12-31'

# # script configuration
params = {
        'output_directory_name': 'multi_year_soc_proxy_effectiveness',
        'output_model_name': 'reference',
        'config_yaml_name': 'model',
        'horizon_start':  date_lower_bound,
        'horizon_end':  date_upper_bound,
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
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