import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px

#plan
# Load original timeseries
# generate new timeseries .csv for cluster's weather years
#    - probably need to rewrite dates as fake dates so calliope can accept non-consecutive WYs
#    - need to track the mapping function to convert back later
# function for assigning weights to weather years

#parameters
anchor_weather_year = 2010
appended_weather_year = 2011
weight_anchor_weather_year = 0.2  #TODO: Make decisions around these weights
weight_appended_weather_year = 1.8 #TODO: Make decisions around these weights

def generate_timeseries(anchor_weather_year,appended_weather_year,anc_weight,app_weight):

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
    
    # --------------------------------------------------------------------------------------------------------
    
    ## experimental
    # model = calliope.read_netcdf('simple_weather-year_ldes-model/results/single_year_runs/results_full_horizon_2010_2019.netcdf')

    plan_start_date = str(anchor_weather_year)+'-01-01'
    plan_end_date = str(appended_weather_year)+'-12-31' #TODO: set to December again

    calliope.set_log_verbosity("INFO", include_solver_output=True)

    # TODO:timesubset should reflect the max mins of the custom timeseries not the function values currently in
    # TODO: modify the model to take in the bespoke timeseries
    model = calliope.Model(
            'simple_weather-year_ldes-model/model.yaml',
            scenario='single_year_runs_plan',
            override_dict={'config.init.time_subset': [str(plan_start_date), str(plan_end_date)], 'techs.h2_salt_cavern.number_year_cycles': 2}
            )

    # model.build()

    #generate the timestep weights array

    new_weights_array = np.concatenate([np.full(len(df_anchor_weather_year.index), weight_anchor_weather_year), np.full(len(df_appended_weather_year.index), weight_appended_weather_year)])
    data_array_weights = xr.DataArray(new_weights_array, dims=["index"], coords={"index": np.arange(len(df_anchor_weather_year.index)+len(df_appended_weather_year.index))})
    # print(data_array_weights)

    # ts_pull = model.inputs.timestep_weights
    ts_pull = model.backend.get_parameter("timestep_weights", as_backend_objs=False)
    ts_update = ts_pull.copy()
    ts_update.data = new_weights_array
    model.backend.update_parameter("timestep_weights", ts_update)
    checker = model.backend.get_parameter("timestep_weights", as_backend_objs=False)
    model.solve()
    return model

    #-- func end

model1 = generate_timeseries(anchor_weather_year,appended_weather_year,1,1)
# model1.to_netcdf('simple_weather-year_ldes-model/debug/model_1_1.netcdf')
# model2 = generate_timeseries(anchor_weather_year,appended_weather_year,0.2,0.8)
# model2.to_netcdf('simple_weather-year_ldes-model/debug/model_0.2_1.8.netcdf')

model1 = calliope.read_netcdf('simple_weather-year_ldes-model/debug/model_1_1.netcdf')
model2 = calliope.read_netcdf('simple_weather-year_ldes-model/debug/model_0.2_1.8.netcdf')

df_costs_1 = (
        (model1.results.cost_investment.fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Costs Million EURO")
        .reset_index()
    )

df_costs_2 = (
        (model2.results.cost_investment.fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Costs Million EURO")
        .reset_index()
)

df_costs_3 = (
        (model3.results.cost_investment.fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Costs Million EURO")
        .reset_index()
    )

df_costs_4 = (
        (model4.results.cost_investment.fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Costs Million EURO")
        .reset_index()
    )


fig = px.bar(
        df_costs_1,
        x="techs",
        y="Costs Million EURO",
        height=1000,
    )

fig.add_bar(
        x=df_costs_2["techs"],
        y=df_costs_2["Costs Million EURO"],
        name='Model2',
        alignmentgroup="group"
    )

fig.add_bar(
        x=df_costs_3["techs"],
        y=df_costs_3["Costs Million EURO"],
        name='Model3',
        alignmentgroup="group"
    )

fig.add_bar(
        x=df_costs_3["techs"],
        y=df_costs_3["Costs Million EURO"],
        name='Model4',
        alignmentgroup="group"
    )

fig.update_layout(
    xaxis_title="techs",
    yaxis_title="Costs",
    legend_title="Dataset",
    barmode="group"  # Ensures bars are grouped side by side
)

fig.show()