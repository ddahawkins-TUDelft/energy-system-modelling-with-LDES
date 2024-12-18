import calliope
import datetime
import pandas as pd


model = calliope.Model('simple_weather-year_ldes-model/model.yaml', scenario='fixed_min_capacities')
calliope.set_log_verbosity("INFO", include_solver_output=True)
model.build()
model.backend.to_lp('simple_weather-year_ldes-model/results/lp files/my_saved_model.lp')
print(model.applied_math)
# model.to_csv('simple_weather-year_ldes-model/results/results_'+str(int(datetime.datetime.now().timestamp()))+'.csv')
