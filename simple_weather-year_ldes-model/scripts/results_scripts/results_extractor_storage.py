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
            .sel(techs="h2_salt_cavern")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .rename(columns={'Storage (MWh)':'Storage (TWh)'})
            .mul(0.000001) #convert MWh to GWh
            .reset_index()
        )
    else:
        df_storage_hss = pd.concat(
            [df_storage_hss,
            (
            (model.results.storage.fillna(0))
            .sel(techs="h2_salt_cavern")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .rename(columns={'Storage (MWh)':'Storage (TWh)'})
            .mul(0.000001) #convert MWh to GWh
            .reset_index()
            )
            ]
            )

    

node_order = df_storage_hss.nodes.unique()


full_model=calliope.read_netcdf('simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf')

df_storage_hss_full = (
            (full_model.results.storage.fillna(0))
            .sel(techs="h2_salt_cavern")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .rename(columns={'Storage (MWh)':'Storage (TWh)'})
            .mul(0.000001) #convert MWh to GWh
            .reset_index()
            
        )


# Roll hourly data into daily data for visualisation
df_storage_hss = df_storage_hss.groupby([df_storage_hss['timesteps'].dt.date]).max()
df_storage_hss_full = df_storage_hss_full.groupby([df_storage_hss_full['timesteps'].dt.date]).max()

# annotations
id_max_storage_full = df_storage_hss_full['Storage (TWh)'].idxmax()
SOC_max_full = df_storage_hss_full.loc[id_max_storage_full,'Storage (TWh)']
Date_max_full = df_storage_hss_full.loc[id_max_storage_full,'timesteps']

id_max_storage = df_storage_hss['Storage (TWh)'].idxmax()
SOC_max = df_storage_hss.loc[id_max_storage,'Storage (TWh)']
Date_max = df_storage_hss.loc[id_max_storage,'timesteps']

colour_1 = 'rgb(102, 153, 255)'
colour_1_darker = 'rgb(51, 76, 128)'
colour_2 = 'rgb(255, 153, 102)'
colour_2_darker = 'rgb(128, 76, 51)'


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_storage_hss_full['timesteps'],
    y=df_storage_hss_full['Storage (TWh)'],
    fill='tozeroy',
    mode='none',
    line_color=colour_1,
    name='Reference Model'
))

fig.add_trace(go.Scatter(
    x=df_storage_hss['timesteps'],
    y=df_storage_hss['Storage (TWh)'],
    fill='tozeroy',
    mode='none',
    line_color=colour_2,
    name='Concatenated Single Weather-Year Models'
))

fig.add_trace(go.Scatter(
    x=[Date_max_full],
    y=[SOC_max_full],
    mode="markers+text",
    name="Maximum SOC Full",
    line_color = 'dark gray',
    text=[f"Maximum SOC for Reference Model : {round(SOC_max_full,2)} TWh "],
    textfont=dict(
        size=16,
        color="light grey"
    ),
    textposition="middle left",
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[Date_max],
    y=[SOC_max],
    mode="markers+text",
    name="Maximum SOC",
    line_color = 'dark gray',
    text=[f" Maximum SOC across Single-Year Models: {round(SOC_max,2)} TWh"],
    textfont=dict(
        size=24,
        color="light grey"
    ),
    textposition="middle right",
    showlegend=False
))

fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        title=dict(text=f"SOC for hydrogen storage over time for the reference model and concatenated single weather-year models."),
        yaxis = dict(
            title = dict(
                text='State of Charge, TWh',
                font=dict(size=24)
            )
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        boxmode='group',
        yaxis_tickformat=".2f",
    )

# fig.update_yaxes(matches=None)
fig.update_xaxes(
    tickvals=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey'
)
fig.update_yaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(255, 255, 255, 0)'
)
fig.write_html("simple_weather-year_ldes-model/export/result_storage_full_vs_singles.html", auto_open=True)
