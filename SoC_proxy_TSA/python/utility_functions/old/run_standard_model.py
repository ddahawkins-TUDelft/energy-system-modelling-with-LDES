import calliope
import os
# print(os.getcwd())
from SoC_proxy_TSA.python.utility_functions.helper_model_config import standardised_model_config as standard_model, filename_standard

# script configuration
params = {
        'output_directory_name': 'reference',
        'output_model_name': 'full_horizon',
        'config_yaml_name': 'model',
        'horizon_start':  '2010-01-01',
        'horizon_end':  '2010-12-31',
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'scenario_name': 'standard',
        'calliope_full_log': True,
        # 'dict_additional_overrides': {},
    }

# auto_config calliope runtime 
if params['calliope_full_log']:
    calliope.set_log_verbosity("INFO", include_solver_output=True) 

#construct model using standardised constructor
model = standard_model(params)

#build
model.build()
print(f"  --- Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']}, Scenario: {params['scenario_name']} --- ")

#solve
model.solve()

#auto-save, first checks if full directory tree exists (if not, creates it)
output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
os.makedirs(output_dir, exist_ok=True)
model.to_netcdf(os.path.join(output_dir,filename_standard(params, 'netcdf')))