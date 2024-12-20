import calliope
import datetime
import pandas as pd
import plotly.express as px

#Run Parameters
plan_year = 2019
operate_year = 2015
number_years = 1

#results save path
plan_save_path= 'simple_weather-year_ldes-model/results/single_year_runs/results_plan_'+str(plan_year)+'_'+str(int(datetime.datetime.now().timestamp()))+'.netcdf'
operate_save_path= 'simple_weather-year_ldes-model/results/single_year_runs/results_operate_'+str(operate_year)+'_plan_'+str(plan_year)+'_'+str(int(datetime.datetime.now().timestamp()))+'.netcdf'

#generate calliope config variables
plan_start_date = str(plan_year)+'-01-01'
plan_end_date = str(plan_year)+'-12-31'
operate_start_date = str(plan_year)+'-01-01'
operate_end_date = str(plan_year)+'-12-31'

# run the build model for the build year
model = calliope.Model(
    'simple_weather-year_ldes-model/model.yaml',
    scenario='single_year_runs_plan',
    override_dict={'config.init.time_subset': [str(plan_start_date), str(plan_end_date)], 'techs.h2_salt_cavern.number_year_cycles': number_years}
)

calliope.set_log_verbosity("INFO", include_solver_output=True)
model.build()
model.solve()
model.to_netcdf(plan_save_path)
#model.backend.to_lp('simple_weather-year_ldes-model/results/lp files/lp_export_'+str(int(datetime.datetime.now().timestamp()))+'.lp')

# next, run the dispatch model for the dispatch year taking the build year outputs as capacity inputs

