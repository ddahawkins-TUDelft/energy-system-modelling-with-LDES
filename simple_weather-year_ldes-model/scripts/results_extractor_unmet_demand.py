import calliope
import datetime
import pandas as pd
import plotly.express as px
import re
import os
import plotly.graph_objects as go

#load individual dispatch year models.
#save annual unmet demand

min_year = 2010
max_year = 2019

plan_year = []
op_year = []
unmet_demand = []
number_hours_unmet_threshold_10_percent = []
daily_average_unmet_demand = []

save_folder = 'simple_weather-year_ldes-model/results/single_year_runs'

model = {}

for py in range(min_year,max_year+1):
    for oy in range(min_year,max_year+1):
        plan_year.append(py)
        op_year.append(oy)

        import_file_path = ""

        filename_re = r"results_operate_"+str(oy)+r"_plan_"+str(py)+r"_\d+\.netcdf$"
        for filename in os.listdir(save_folder):
            if re.search(filename_re, filename):
                # print(f"Identified an existing plan file: {filename}. This run will be imported as the plane model.")
                import_file_path = save_folder+'/'+filename

        if import_file_path:
            model = calliope.read_netcdf(import_file_path)
            # print(import_file_path)
        else:
            raise Exception(f"Sorry, no results file found for op year {oy} and plan year {py}")
        
        df_unmet_demand_hourly = (
                    (model.results.unmet_demand.fillna(0))
                    .sel(carriers="power")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Unmet Demand (MWh)")
                    .reset_index()
                )
        
        df_demand_hourly = (
                    (model.inputs.sink_use_equals.fillna(0))
                    .sel(techs="demand_power")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Demand (MWh)")
                    .reset_index()
                )
        
        #count number of hours with unmet demand of more than 10% demand 
        demand_CF_hourly = df_unmet_demand_hourly['Unmet Demand (MWh)']/df_demand_hourly['Demand (MWh)']
        number_hours_gte_10_percent_unmet_demand = (demand_CF_hourly >= 0.1).sum()
        number_hours_unmet_threshold_10_percent.append(number_hours_gte_10_percent_unmet_demand)

        #sum the unmet demand in year
        annual_sum_demand = df_unmet_demand_hourly['Unmet Demand (MWh)'].sum()
        unmet_demand.append(annual_sum_demand)

        

        if not df_unmet_demand_hourly.empty:
            var_daily_unmet = df_unmet_demand_hourly.groupby([df_unmet_demand_hourly['timesteps'].dt.date])['Unmet Demand (MWh)'].sum().mean()
        else:
            var_daily_unmet = 0
        
        daily_average_unmet_demand.append(var_daily_unmet)

        print(f"PY: {py}, OY: {oy}, Annuam UD: {int(annual_sum_demand)}, Hours>10%: {number_hours_gte_10_percent_unmet_demand}, Daily Mean UD: {int(var_daily_unmet)}")

# export csv

export_dict= {
                'plan_year' : plan_year, 
                'op_year' : op_year, 
                'unmet_demand' : unmet_demand, 
                'number_hours_unmet_threshold_10_percent' : number_hours_unmet_threshold_10_percent,
                'daily_average_unmet_demand' : daily_average_unmet_demand
              }

pd.DataFrame(export_dict).to_csv('simple_weather-year_ldes-model/export/single_year_unmet_demand_results.csv')
