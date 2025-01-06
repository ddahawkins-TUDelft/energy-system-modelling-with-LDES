import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px
import re
import os

df_data = pd.read_csv(
    'simple_weather-year_ldes-model/data_tables/time_varying_parameters.csv',
    )
df_data.columns = ['timesteps','demand','solar','offshore_wind','onshore_wind']
df_data = df_data[4:]
# df_data.reset_index().set_index('timesteps')

#change data types
df_data['timesteps'] = pd.to_datetime(df_data['timesteps'], format='%Y/%m/%d %H:%M')
df_data['demand'] = pd.to_numeric(df_data['demand'])
df_data['solar'] = pd.to_numeric(df_data['solar'])
df_data['offshore_wind'] = pd.to_numeric(df_data['offshore_wind'])
df_data['onshore_wind'] = pd.to_numeric(df_data['onshore_wind'])

#non dimensionalise demand (>%)
factor_demand_max = df_data['demand'].max()
df_data['demand'] = df_data['demand'] / factor_demand_max

#filtering
df_data = df_data[(df_data['timesteps'] >= '2010-01-01') & (df_data['timesteps'] < '2020-01-01')]

#aggregate RES trackers
df_data['mean_RES'] = (df_data['solar']+df_data['offshore_wind']+df_data['onshore_wind'])/3


#RES normalised to demand
df_data['solar_norm_demand'] = df_data['solar'] / df_data['demand']
df_data['offshore_norm_demand'] = df_data['offshore_wind'] / df_data['demand']
df_data['onshore_norm_demand'] = df_data['onshore_wind'] / df_data['demand']
df_data['mean_RES_norm_demand'] = df_data['mean_RES'] / df_data['demand']

#resample
# df = df_data.groupby([df_data['timesteps'].dt.date]).mean()
df = df_data.resample('ME', on='timesteps').mean()
df.reset_index(inplace=True)

#Rolling average
df['rolling_mean_RES_norm_demand'] = df['mean_RES_norm_demand'].rolling(window=6).mean()
df['rolling_mean_RES'] = df['mean_RES'].rolling(window=6).mean()


#generate figure
fig = px.area(
        df,
        x="timesteps",
        y="demand",
        height=800,
    )

fig.add_scatter(
    x=df['timesteps'],
    y=df['mean_RES_norm_demand'],
    name = 'mean_RES_norm_demand'
)

fig.add_scatter(
    x=df['timesteps'],
    y=df['rolling_mean_RES_norm_demand'],
    name = 'rolling_mean_RES_norm_demand'
)

fig.add_scatter(
    x=df['timesteps'],
    y=df['rolling_mean_RES'],
    name = 'rolling_mean_RES'
)

fig.update_layout(yaxis_range=[0,1])

fig.write_html('simple_weather-year_ldes-model/debug/ts_analysis.html', auto_open=True)

print(df)

