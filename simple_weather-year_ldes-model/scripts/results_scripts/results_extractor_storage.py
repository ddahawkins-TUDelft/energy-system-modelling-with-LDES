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

df_storage_hss = {}

for py in range(min_year,max_year+1):

    import_file_path = ""

    filename_re = r"results_operate_"+str(py)+r"_plan_"+str(py)+r"_\d+\.netcdf$"
    for filename in os.listdir(save_folder):
        if re.search(filename_re, filename):
            # print(f"Identified an existing plan file: {filename}. This run will be imported as the plane model.")
            import_file_path = save_folder+'/'+filename

    if import_file_path:
        model = calliope.read_netcdf(import_file_path)
        # print(import_file_path)
    else:
        raise Exception(f"Sorry, no results file found for op year {oy} and plan year {py}")
    
    if py==min_year:
        df_storage_hss = (
            (model.results.storage.fillna(0))
            # .sel(techs="hydrogen_storage_system")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .rename(columns={'Storage (MWh)':'Storage (GWh)'})
            .mul(0.001) #convert MWh to GWh
            .reset_index()
        )
    else:
        df_storage_hss = pd.concat(
            [df_storage_hss,
            (
            (model.results.storage.fillna(0))
            # .sel(techs="hydrogen_storage_system")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .rename(columns={'Storage (MWh)':'Storage (GWh)'})
            .mul(0.001) #convert MWh to GWh
            .reset_index()
            )
            ]
            )

node_order = df_storage_hss.nodes.unique()

fig = px.area(
        df_storage_hss,
        x="timesteps",
        y="Storage (GWh)",
        facet_row="nodes",
        category_orders={"nodes": node_order},
    )

full_model=calliope.read_netcdf('simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf')

df_storage_hss_full = (
            (full_model.results.storage.fillna(0))
            # .sel(techs="hydrogen_storage_system")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .rename(columns={'Storage (MWh)':'Storage (GWh)'})
            .mul(0.001) #convert MWh to GWh
            .reset_index()
            
        )

for idx, node in enumerate(node_order[::-1]):
            full_val = df_storage_hss_full.loc[
                df_storage_hss_full.nodes == node, "Storage (GWh)"
            ]
            if not full_val.empty:
                fig.add_scatter(
                    x=df_storage_hss_full.loc[
                        df_storage_hss_full.nodes == node, "timesteps"
                    ],
                    y=1 * full_val,
                    row=idx + 1,
                    col="all",
                    marker_color="black",
                    marker_size=1,
                    name="Full Model Storage",
                    legendgroup="Full Model",
                    mode='markers'
                )


fig.update_yaxes(matches=None)
fig.update_xaxes(tickvals=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
fig.write_html("simple_weather-year_ldes-model/export/result_storage_full_vs_singles.html", auto_open=True)

    
#     df_demand_hourly = (
#                 (model.inputs.sink_use_equals.fillna(0))
#                 .sel(techs="demand_power")
#                 .to_series()
#                 .where(lambda x: x != 0)
#                 .dropna()
#                 .to_frame("Demand (MWh)")
#                 .reset_index()
#             )
    
#     #count number of hours with unmet demand of more than 10% demand 
#     demand_CF_hourly = df_unmet_demand_hourly['Unmet Demand (MWh)']/df_demand_hourly['Demand (MWh)']
#     number_hours_gte_10_percent_unmet_demand = (demand_CF_hourly >= 0.1).sum()
#     number_hours_unmet_threshold_10_percent.append(number_hours_gte_10_percent_unmet_demand)

#     #sum the unmet demand in year
#     annual_sum_demand = df_unmet_demand_hourly['Unmet Demand (MWh)'].sum()
#     unmet_demand.append(annual_sum_demand)

    

#     if not df_unmet_demand_hourly.empty:
#         var_daily_unmet = df_unmet_demand_hourly.groupby([df_unmet_demand_hourly['timesteps'].dt.date])['Unmet Demand (MWh)'].sum().mean()
#     else:
#         var_daily_unmet = 0
    
#     daily_average_unmet_demand.append(var_daily_unmet)

#     print(f"PY: {py}, OY: {oy}, Annuam UD: {int(annual_sum_demand)}, Hours>10%: {number_hours_gte_10_percent_unmet_demand}, Daily Mean UD: {int(var_daily_unmet)}")

# # export csv

# export_dict= {
#                 'plan_year' : plan_year, 
#                 'op_year' : op_year, 
#                 'unmet_demand' : unmet_demand, 
#                 'number_hours_unmet_threshold_10_percent' : number_hours_unmet_threshold_10_percent,
#                 'daily_average_unmet_demand' : daily_average_unmet_demand
#               }

# pd.DataFrame(export_dict).to_csv('simple_weather-year_ldes-model/export/single_year_unmet_demand_results.csv')
