import calliope
import pandas as pd
import plotly.express as px
import script_utilities as util #custom module to abstract away some frequently used functions

# script configuration
model_type = "single_year_cluster" # model type
results_directory = f"weather_year_clustering_LDES_study/results/{model_type}" #save directory for results
# calliope.set_log_verbosity("INFO", include_solver_output=True) #if True, shows calliope logs as output


# import weather years
df_casestudy = (util.read_years_weights(model_type)) #df of years and weights, only years relevant for single year model

#loop over the case study years, generate and solve models for each, saving the results as netcdf
for index, values in df_casestudy.iterrows():

    print('Running: ', index, values['years'])
    save_path = results_directory+f'/results_year_{values['years']}.netcdf'

    model = calliope.Model(
        'weather_year_clustering_LDES_study/model_config/model.yaml',
        scenario='single_year_runs_plan',
        override_dict={'config.init.time_subset': [f"{values['years']}-01-01", f"{values['years']}-12-31"], 'techs.h2_salt_cavern.number_year_cycles': 1}
        )

    model.build()
    model.solve()
    model.to_netcdf(save_path)

