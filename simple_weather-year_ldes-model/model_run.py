import calliope
import datetime
import pandas as pd
import plotly.express as px
import yaml
import re
import os

#function for generating plotly graphs of run results
def visualise_SOC(model, run_type):
    df_storage = (
        (model.results.storage.fillna(0))
        # .sel(techs="hydrogen_storage_system")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Storage (MWh)")
        .rename(columns={'Storage (MWh)':'Storage (GWh)'})
        .mul(0.001) #convert kWh to GWh
        .reset_index()
    )

    print(df_storage.head())

    node_order = df_storage.nodes.unique()

    df_storage_hss = df_storage[df_storage.techs == "h2_salt_cavern"]
    df_storage_batt = df_storage[df_storage.techs == "battery"]

    print("Battery Storage Capacity: "+str(df_storage_batt["Storage (GWh)"].max())+ "GWh")
    print("Hydrogen Storage Capacity: "+str(df_storage_hss["Storage (GWh)"].max())+ "GWh")

    fig = px.area(
        df_storage_hss,
        x="timesteps",
        y="Storage (GWh)",
        facet_row="nodes",
        category_orders={"nodes": node_order},
        height=1000,
    )

    showlegend = True
    # we reverse the node order (`[::-1]`) because the rows are numbered from bottom to top.
    for idx, node in enumerate(node_order[::-1]):
        storage_val = df_storage_batt.loc[
            df_storage_batt.nodes == node, "Storage (GWh)"
        ]
        if not storage_val.empty:
            fig.add_scatter(
                x=df_storage_batt.loc[
                    df_storage_batt.nodes == node, "timesteps"
                ],
                y=1 * storage_val,
                row=idx + 1,
                col="all",
                marker_color="black",
                name="Storage",
                legendgroup="storage",
                showlegend=showlegend,
            )
            showlegend = False
    
    if run_type == 'operate':

        df_unmet_demand = (
            (model.results.unmet_demand.fillna(0))
            # .sel(techs="hydrogen_storage_system")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Unmet Demand (MWh)")
            .reset_index()
        )

        for idx, node in enumerate(node_order[::-1]):
            unmet_val = df_unmet_demand.loc[
                df_unmet_demand.nodes == node, "Unmet Demand (MWh)"
            ]
            if not unmet_val.empty:
                fig.add_scatter(
                    x=df_unmet_demand.loc[
                        df_unmet_demand.nodes == node, "timesteps"
                    ],
                    y=1 * unmet_val,
                    row=idx + 1,
                    col="all",
                    marker_color="red",
                    name="Unmet Demand",
                    legendgroup="Demand",
                    showlegend=showlegend,
                    mode='markers'
                )
                showlegend = False
            
    fig.update_yaxes(matches=None)
    fig.write_html("simple_weather-year_ldes-model/results/result_storage"+str(run_type)+".html", auto_open=True)


run_id =str(int(datetime.datetime.now().timestamp()))
save_folder = 'simple_weather-year_ldes-model/results/single_year_runs'
plan_save_path= save_folder+'/results_full_horizon_2010_2019.netcdf'


#generate calliope config variables
plan_start_date = str(2010)+'-01-01'
plan_end_date = str(2019)+'-12-31' #TODO: set to December again
number_years = 10
# run the build model for the build year
# calliope.set_log_verbosity("INFO", include_solver_output=True)
calliope.set_log_verbosity("INFO", include_solver_output=True)



model = calliope.Model(
'simple_weather-year_ldes-model/model.yaml',
scenario='single_year_runs_plan',
override_dict={'config.init.time_subset': [str(plan_start_date), str(plan_end_date)], 'techs.h2_salt_cavern.number_year_cycles': number_years}
)

model.build()
model.solve()
model.to_netcdf(plan_save_path)

visualise_SOC(model, 'plan')