import datetime
import pandas as pd
import calliope

#plan
# Load original timeseries
# generate new timeseries .csv for cluster's weather years
#    - probably need to rewrite dates as fake dates so calliope can accept non-consecutive WYs
#    - need to track the mapping function to convert back later
# function for assigning weights to weather years

#parameters
anchor_weather_year = 2010
appended_weather_year = 2016
weight_anchor_weather_year = 0.5  #TODO: Make decisions around these weights
weight_aappended_weather_year = 0.5 #TODO: Make decisions around these weights

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

    #isolate appended weather year
    df_appended_weather_year = df_timesteps_values[(df_timesteps_values['timesteps']<f"{appended_weather_year+1}-01-01")&(df_timesteps_values['timesteps']>=f"{appended_weather_year}-01-01")]

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

generate_timeseries(anchor_weather_year,appended_weather_year)

## experimental
model = calliope.read_netcdf('simple_weather-year_ldes-model/results/single_year_runs/results_full_horizon_2010_2019.netcdf')
print('hello')

#based on above, can assign year weights via the model.inputs.timestep_weights
#  model.inputs.timestep_weights.data is an array([]) of weights for each hour of the model, set to 1 by default.
# a representative model can be generated perhaps, by manipulating the relative weights of each year
# for example. Anchor Year (weight=0.1) + append year (weight=0.9) representing 1 extreme (system defining) year and 9 normal years

