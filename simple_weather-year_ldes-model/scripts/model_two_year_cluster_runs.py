import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px

#calliope.set_log_verbosity("info", include_solver_output=False)


#plan
# Load original timeseries
# generate new timeseries .csv for cluster's weather years
#    - probably need to rewrite dates as fake dates so calliope can accept non-consecutive WYs
#    - need to track the mapping function to convert back later
# function for assigning weights to weather years

#parameters
anchor_weather_year = 2010
appended_weather_year = 2011
weight_anchor_weather_year = 0.1  #TODO: Make decisions around these weights
weight_appended_weather_year = 0.9 #TODO: Make decisions around these weights

def hours_in_year(year):
    """
    Determines the number of hours in a given year.
    
    Parameters:
    year (int): The input year
    
    Returns:
    int: Number of hours in the year
    """
    # Check if the year is a leap year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # Calculate hours
    hours = 366 * 24 if is_leap else 365 * 24
    
    return hours

#resamples the anchor and appended weather year timeseries to new consecutive timeseries beginning 01/01/2010
def generate_timeseries(anchor_weather_year,appended_weather_year):

    #load the original timeseries data
    df_original_timeseries = pd.read_csv(
        "simple_weather-year_ldes-model/data_tables/time_varying_parameters.csv",
        names=["timesteps","var1","var2","var3","var4"]
        )

    #retain structural pointers for calliope to be re-added later
    df_headers = df_original_timeseries.head(5)

    #isolate timesteps content for manipulation
    df_timesteps_values = df_original_timeseries.drop([0,1,2,3,4])

    #isolate anchor weather year
    df_anchor_weather_year = df_timesteps_values[(df_timesteps_values['timesteps']<f"{anchor_weather_year+1}-01-01")&(df_timesteps_values['timesteps']>=f"{anchor_weather_year}-01-01")]
    # df_anchor_weather_year['timestep_weights']=anc_weight
    #isolate appended weather year
    df_appended_weather_year = df_timesteps_values[(df_timesteps_values['timesteps']<f"{appended_weather_year+1}-01-01")&(df_timesteps_values['timesteps']>=f"{appended_weather_year}-01-01")]
    # df_appended_weather_year['timestep_weights']=app_weight

    df_clustered_timeseries_data = pd.concat([df_anchor_weather_year,df_appended_weather_year])

    #generate timeseries to replace original timedates to retain continuity for calliope inputs
    start_ts = pd.Timestamp('2010-01-01')  #arbitrary start date for clusters
    periods = int(df_clustered_timeseries_data.count()['timesteps']) #calculate the length of timeseries to be resampled
    dti = pd.date_range(start=start_ts, periods=int(df_clustered_timeseries_data.count()['timesteps']), freq='h') #generate new timeseries. Method accoutns for leap years
    df_clustered_timeseries_data['timesteps'] = dti

    #time format
    df_clustered_timeseries_data['timesteps']=df_clustered_timeseries_data['timesteps'].dt.strftime("%Y/%m/%d %H:%M")

    df_result = pd.concat([df_headers,df_clustered_timeseries_data])
    df_result.reset_index(drop=True, inplace=True)

    #date formatting

    
    df_result.to_csv(
        path_or_buf=f"simple_weather-year_ldes-model/data_tables/clustered_weather_years/cluster_timeseries_cluster_{anchor_weather_year}_{appended_weather_year}.csv", 
        index=False, 
        header=False,
        )
    
    return df_result
    
#generates timestep weights given anchor/appended weather years and associated weights
def generate_timestep_weights(anchor_weather_year, appended_weather_year, anc_weight, app_weight):

    generate_timeseries(anchor_weather_year,appended_weather_year)
    start_ts = pd.Timestamp('2010-01-01')  #arbitrary start date for clusters
    periods = int(hours_in_year(anchor_weather_year)+hours_in_year(appended_weather_year)) #calculate the length of timeseries to be resampled
    dti = pd.date_range(start=start_ts, periods=periods, freq='h',name='timesteps') #generate new timeseries. Method accoutns for leap years
    
    df_result = dti.to_frame().drop(columns=['timesteps']) #drop avoids duplication of index and dedicated column
    df_result['timestep_weights'] = np.concatenate([np.full(hours_in_year(anchor_weather_year), anc_weight), np.full(hours_in_year(appended_weather_year), app_weight)])

    df_result.to_csv(
        path_or_buf=f"simple_weather-year_ldes-model/data_tables/clustered_weather_years/cluster_timestep_weights_cluster_{anchor_weather_year}_{anc_weight}_{appended_weather_year}_{app_weight}.csv", 
        )
    
    df_result = pd.read_csv(f"simple_weather-year_ldes-model/data_tables/clustered_weather_years/cluster_timestep_weights_cluster_{anchor_weather_year}_{anc_weight}_{appended_weather_year}_{app_weight}.csv")

    return df_result

#build a model and modify the timestep weights
def run_model_with_timestep_weights(anc_wy,app_wy,anc_weight,app_weight):

    # df_timestep_weights = pd.read_csv(path_timestep_weights_csv)
    df_timestep_weights=generate_timestep_weights(anc_wy, app_wy, anc_weight, app_weight)
    
    model = calliope.Model(
        'simple_weather-year_ldes-model/model_cluster_with_timestep_weights.yaml',
        scenario='single_year_runs_plan',
        override_dict={
             'config.init.time_subset': [df_timestep_weights['timesteps'].min(), df_timestep_weights['timesteps'].max()], 
             'techs.h2_salt_cavern.number_year_cycles': 2,
             'data_tables.time_varying_parameters.data': f"data_tables/clustered_weather_years/cluster_timeseries_cluster_{anc_wy}_{app_wy}.csv",
             }
        )

    model.inputs.timestep_weights.data=df_timestep_weights['timestep_weights'].to_numpy()
    print(f'Building Model [{anc_wy},{app_wy}] with weights: [{model.inputs.timestep_weights.data[0]},{model.inputs.timestep_weights.data[-1]}]')
    model.build()
    model.solve()

    model.to_netcdf(f"simple_weather-year_ldes-model/results/two_year_cluster_runs/results_plan_cluster_{anc_wy}_{app_wy}_weights_{anc_weight}_{app_weight}.netcdf")

    return model

# print(generate_timestep_weights(anchor_weather_year, appended_weather_year, weight_anchor_weather_year, weight_appended_weather_year))
def visualise_costs(anc_wy,app_wy,anc_weight,app_weight):

    #run the clustered weather year runs
    model_cluster = run_model_with_timestep_weights(anc_wy,app_wy,anc_weight,app_weight)
    # full_temp = calliope.read_netcdf("simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf")
    # #extract system costs
    # df_costs = (
    #     (model_cluster.results.cost.fillna(0))
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Costs Million EURO")
    #     .reset_index()
    # )
    
    # df_storage_hss = (
    #         (model_cluster.results.storage.fillna(0))
    #         .sel(techs="h2_salt_cavern")
    #         .to_series()
    #         .where(lambda x: x != 0)
    #         .dropna()
    #         .to_frame("Storage (MWh)")
    #         .rename(columns={'Storage (MWh)':'Storage (GWh)'})
    #         .mul(0.001) #convert MWh to GWh
    #         .reset_index()
    #     )
    # node_order = df_storage_hss.nodes.unique()
    # df_storage_hss_full = (
    #         (full_temp.results.storage.fillna(0))
    #         .sel(techs="h2_salt_cavern")
    #         .to_series()
    #         .where(lambda x: x != 0)
    #         .dropna()
    #         .to_frame("Storage (MWh)")
    #         .rename(columns={'Storage (MWh)':'Storage (GWh)'})
    #         .mul(0.001) #convert MWh to GWh
    #         .reset_index()
            
    #     )
    
    #create area chart of storage distros over time
    # fig_storage = px.area(
    #     df_storage_hss,
    #     x="timesteps",
    #     y="Storage (GWh)",
    #     facet_row="nodes",
    #     height=1000,
    #     category_orders={"nodes": node_order},
    # )
    # for idx, node in enumerate(node_order[::-1]):
    #         full_val = df_storage_hss_full.loc[
    #             df_storage_hss_full.nodes == node, "Storage (GWh)"
    #         ]
    #         if not full_val.empty:
    #             fig_storage.add_scatter(
    #                 x=df_storage_hss_full.loc[
    #                     df_storage_hss_full.nodes == node, "timesteps"
    #                 ],
    #                 y=1 * full_val,
    #                 row=idx + 1,
    #                 col="all",
    #                 marker_color="black",
    #                 marker_size=1,
    #                 name="Full Model Storage",
    #                 legendgroup="Full Model",
    #                 mode='markers'
    #             )
    # fig_storage.update_yaxes(matches=None)
    # fig_storage.update_xaxes(tickvals=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    # fig_storage.write_html("simple_weather-year_ldes-model/results/result_cluster_storage.html", auto_open=True)
    # #create bar figure of costs
    # fig_costs = px.bar(
    #         df_costs,
    #         x="techs",
    #         y="Costs Million EURO",
    #         height=1000,
    #     )
    # df_costs_full = (
    #     (full_temp.results.cost.fillna(0))
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Costs Million EURO")
    #     .reset_index()
    # )
    # fig_costs.add_bar(
    #             x=df_costs_full["techs"],
    #             y=df_costs_full["Costs Million EURO"],
    #             name=f"Model Full",
    #             alignmentgroup="group",
    #         )
    # fig_costs.update_layout(
    #     xaxis_title="techs",
    #     yaxis_title="Costs",
    #     legend_title="Dataset",
    #     barmode="group"  # Ensures bars are grouped side by side
    # )
    # # fig_costs.write_html("simple_weather-year_ldes-model/results/result_cluster_costs.html", auto_open=True)
    # #create bar chart to compare capacity mixes
    # df_flow_cap_cluster = (
    #                 (model_cluster.results.flow_cap.fillna(0))
    #                 # .sel(carriers="power")
    #                 .to_series()
    #                 .where(lambda x: x != 0)
    #                 .dropna()
    #                 .to_frame("Flow Capacity (MW)")
    #                 .reset_index()
    #             )
    # df_flow_cap_full = (
    #                 (full_temp.results.flow_cap.fillna(0))
    #                 # .sel(carriers="power")
    #                 .to_series()
    #                 .where(lambda x: x != 0)
    #                 .dropna()
    #                 .to_frame("Flow Capacity (MW)")
    #                 .reset_index()
    #             )
    # fig_capacities = px.bar(
    #         df_flow_cap_cluster,
    #         x="techs",
    #         y="Flow Capacity (MW)",
    #         height=1000,
    #     )
    # fig_capacities.add_bar(
    #             x=df_flow_cap_full["techs"],
    #             y=df_flow_cap_full["Flow Capacity (MW)"],
    #             name=f"Model Full",
    #             alignmentgroup="group",
    #         )
    # fig_capacities.update_layout(
    #     xaxis_title="techs",
    #     yaxis_title="Flow Caps",
    #     legend_title="Dataset",
    #     barmode="group"  # Ensures bars are grouped side by side
    # )
    # # fig_capacities.write_html("simple_weather-year_ldes-model/results/result_cluster_caps.html", auto_open=True)
    # print(f"Cluster Cost: {df_costs["Costs Million EURO"].sum()}, Full Cost: {df_costs_full["Costs Million EURO"].sum()}")

visualise_costs(2012,2014,1,9)
visualise_costs(2012,2015,1,9)
visualise_costs(2012,2016,1,9)
visualise_costs(2012,2017,1,9)
visualise_costs(2012,2018,1,9)
visualise_costs(2012,2019,1,9)

for i in range(2013,2019+1):
    for j in range (2011,2019+1):
        if j != i+1:
            if j!= i:
                visualise_costs(i,j,1,9)
                print(f'Optimisation for [{i},2010] complete')