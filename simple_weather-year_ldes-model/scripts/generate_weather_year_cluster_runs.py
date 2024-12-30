import datetime
import pandas as pd

#plan
# Load original timeseries
# generate new timeseries .csv for cluster's weather years
#    - probably need to rewrite dates as fake dates so calliope can accept non-consecutive WYs
#    - need to track the mapping function to convert back later
# function for assigning weights to weather years

#parameters
anchor_weather_year = 2014
appended_weather_year = 2015


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

df_result = pd.concat([df_headers,df_clustered_timeseries_data])
df_result.reset_index(drop=True, inplace=True)

df_result.to_csv(
    path_or_buf=f"simple_weather-year_ldes-model/data_tables/clustered_weather_years/cluster_timeseries_anchor{anchor_weather_year}_app{appended_weather_year}.csv", 
    index=False, 
    header=False
    ) 