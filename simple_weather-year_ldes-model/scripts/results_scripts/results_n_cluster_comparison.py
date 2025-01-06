import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px
import re
import os


directory = 'simple_weather-year_ldes-model\\results\\n_year_cluster_runs'
file = 'model_cluster_2010_2011_2012_1_1_8.netcdf'

filepath = 'simple_weather-year_ldes-model/results/n_year_cluster_runs/model_cluster_2010_2011_2012_1_1_8.netcdf'


def extract_years_weights(filepath: str):
    
    model = filepath.split("\\")[-1]

    years = [int(model[14:18]),int(model[19:23]),int(model[24:28])]
    weights = [int(model[29]),int(model[31]),int(model[33])]
    
    output = {
        'years': years,
        'weights': weights
        }
    
    return output

def review_highlights(path):

    output = {}
    model = calliope.read_netcdf(path)
    full_model = calliope.read_netcdf('simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf')
    
    df_storage_cap = (
                    (model.results.storage_cap.fillna(0))
                    # .sel(techs="h2_salt_cavern")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Hydrogen Storage Capacity (MWh)")
                    .reset_index()
    )['Hydrogen Storage Capacity (MWh)']

    df_storage_cap_full = (
                    (full_model.results.storage_cap.fillna(0))
                    # .sel(techs="h2_salt_cavern")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Hydrogen Storage Capacity (MWh)")
                    .reset_index()
    )['Hydrogen Storage Capacity (MWh)']
    
    df_cost = (
                    (model.results.cost.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Annualised Cost €M")
                    .reset_index()
    )["Annualised Cost €M"].sum()

    df_cost_full = (
                    (full_model.results.cost.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Annualised Cost €M")
                    .reset_index()
    )["Annualised Cost €M"].sum()

    output = {
        'Parameter': [
            'H2 Storage (GWh)',
            'Battery Storage (GWh)',
            "Annualised Cost €M",
            ],
        'Model Value': [
            df_storage_cap[1],
            df_storage_cap[0],
            df_cost,
            ],
        'Reference Value': [
            df_storage_cap_full[1],
            df_storage_cap_full[0],
            df_cost_full,
            ],
        'Error': [
            f"{round(100*(df_storage_cap[1]/df_storage_cap_full[1]-1),1)}%",
            f"{round(100*(df_storage_cap[0]/df_storage_cap_full[0]-1),1)}%",
            f"{round(100*(df_cost/df_cost_full-1),1)}%",
            ]
        }
    
    techs_of_interest = ['solar','offshore_wind','electrolyser','h2_elec_conversion','battery']
    df_flow_cap_vals = []
    df_flow_cap_full_vals = []
    for tech in techs_of_interest:
        df_flow_cap = (
            (model.results.flow_cap.fillna(0))
            .sel(techs=tech)
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow Capacity (MW)")
            .reset_index()
        )
        df_flow_cap_full = (
            (full_model.results.flow_cap.fillna(0))
            .sel(techs=tech)
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow Capacity (MW)")
            .reset_index()
        )
        output['Parameter'].append(f"{tech} Flow Cap (MW)")
        output['Model Value'].append(int(df_flow_cap["Flow Capacity (MW)"][0]))
        output['Reference Value'].append(int(df_flow_cap_full["Flow Capacity (MW)"][0])) 
        output['Error'].append(f"{round(((df_flow_cap["Flow Capacity (MW)"][0]/(df_flow_cap_full["Flow Capacity (MW)"][0])-1)*100).mean(),1)}%")

    return pd.DataFrame.from_dict(output).set_index('Parameter')

def generate_results_table(filepath):

    print('Extracting results for: ',filepath)
    df_design_performance_results = review_highlights(filepath)
    mean_abs_system_error = (abs(df_design_performance_results['Model Value'] / df_design_performance_results['Reference Value'] - 1)).mean()
    # print(extract_years_weights(filepath))

    output = extract_years_weights(filepath)
    output['Mean Abs. Error'] = [mean_abs_system_error,mean_abs_system_error,mean_abs_system_error]
    
    return pd.DataFrame(output)

# print(generate_results_table(filepath))

def example_function():
    result = pd.DataFrame
        
    base_colour = 'rgba(150, 150, 255, .9)'
    anchor_colour = 'rgba(50, 50, 150, 1)'

    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".netcdf") and filename.startswith('model_cluster_'):
            full_path = (os.path.join(directory, filename))
            yrs_array = extract_years_weights(full_path)['years']
            # if yrs_array[1]==yrs_array[1]:
            if yrs_array[1]==yrs_array[0]+1 and yrs_array[2]==yrs_array[1]+1:
                if result.empty:
                    output = generate_results_table(full_path)
                    result = output.copy()
                    anchor_years = result[result['weights']==1]
                    result['colour'] = [base_colour,base_colour,base_colour]
                    fig = px.line(
                        result,
                        x='years',
                        y='Mean Abs. Error',
                        color='colour'                    
                    )
                    fig.add_scatter(
                        x=anchor_years['years'],
                        y=anchor_years['Mean Abs. Error'],
                        name= f"[{",".join(str(x) for x in anchor_years['years'])}],[{",".join(str(x) for x in anchor_years['weights'])}]",
                        mode='lines',
                        marker_color=anchor_colour
                    )
                else:
                    result = generate_results_table(full_path)
                    output = pd.concat([output,result])
                    anchor_years = result[result['weights']==1]
                    fig.add_scatter(
                        x=result['years'],
                        y=result['Mean Abs. Error'],
                        name= f"[{",".join(str(x) for x in result['years'])}],[{",".join(str(x) for x in result['weights'])}]",
                        mode='lines',
                        marker_color=base_colour
                    )
                    fig.add_scatter(
                        x=anchor_years['years'],
                        y=anchor_years['Mean Abs. Error'],
                        name= f"[{",".join(str(x) for x in result['years'])}],[{",".join(str(x) for x in result['weights'])}]",
                        mode='lines',
                        marker_color=anchor_colour
                    )
                
    fig.update_layout(showlegend=False, yaxis_tickformat="5%")
    fig.write_html('simple_weather-year_ldes-model/debug/n_cluster_comparison.html', auto_open=True)

print(extract_years_weights('simple_weather-year_ldes-model/results/two_year_cluster_runs/results_plan_cluster_2010_2012_weights_1_9.netcdf'))

