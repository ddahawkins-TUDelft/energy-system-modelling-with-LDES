import pandas as pd
import calliope
import plotly.graph_objects as go

#function that extracts the target parameter as a DataFrame for a saved netcdf of a model
def model_results_extractor(model, target):

    df_result  =  (   
        (model.results[target].fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame(target)
        .reset_index()
    )

    output = df_result

    return output

#generates a bar chart of the capacity mix.
def chart_bar_flow_cap(model):

    df = model_results_extractor(model, 'flow_cap')
    
    fig =  go.Figure()

    fig.add_trace(go.Bar(
        x= df['techs'],
        y=df['flow_cap']
    ))

    fig.write_html('simple_weather-year_ldes-model/export/results.html', auto_open=True)

#  scatter chart of SOC for H2 and Battery
def char_scatter_SOC(model):

    df = model_results_extractor(model, 'storage')
    
    fig =  go.Figure()

    fig.add_trace(go.Scatter(
        x= df['timesteps'],
        y=df['storage'],
        mode='markers'
    ))

    fig.write_html('simple_weather-year_ldes-model/export/results.html', auto_open=True)

model = calliope.read_netcdf('weather_year_clustering_LDES_study/results/three_year_cluster/with_intracluster_cycle_condition/results_years_[2012,2016,2017]_weight_[8,1,1].netcdf')
char_scatter_SOC(model)
model = calliope.read_netcdf('weather_year_clustering_LDES_study/results/three_year_cluster/with_surplus_tracking/results_years_[2012,2016,2017]_weight_[8,1,1].netcdf')
char_scatter_SOC(model)
model = calliope.read_netcdf('weather_year_clustering_LDES_study/results/three_year_cluster/unmodified/results_years_[2012,2016,2017]_weight_[8,1,1].netcdf')
char_scatter_SOC(model)