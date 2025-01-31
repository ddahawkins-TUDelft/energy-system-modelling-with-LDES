import calliope

import pandas as pd
import results_viewer as rv
import script_utilities as util #custom module to abstract away some frequently used functions

# script configuration
model_type = "reference" # model type
results_directory = f"weather_year_clustering_LDES_study/results/{model_type}" #save directory for results
calliope.set_log_verbosity("INFO", include_solver_output=True) #if True, shows calliope logs as output

save_path = results_directory+'/results_2010_to_2019.netcdf'

model = calliope.Model(
    'weather_year_clustering_LDES_study/model_config/model.yaml',
    scenario='single_year_runs_plan',
    override_dict={'config.init.time_subset': ["2010-01-01", "2019-12-31"], 'techs.h2_salt_cavern.number_year_cycles': 10}
    )

model.build()
model.solve()
model.to_netcdf(save_path)

rv.chart_bar_flow_cap(model)
print('breakpoint, in case something goes wrong with save that I can retain the variables to manually save... dont want to lose something after a 17hr run')
print('breakpoint')