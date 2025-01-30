import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px
import re
import os

import plotly.graph_objects as go

#debug params
n_1_sample = 'simple_weather-year_ldes-model\\results\\single_year_runs\\results_plan_2010_1735331400.netcdf'
n_2_sample = 'simple_weather-year_ldes-model\\results\\two_year_cluster_runs\\results_plan_cluster_2019_2016_weights_1_9.netcdf'
n_3_sample = 'simple_weather-year_ldes-model\\results\\n_year_cluster_runs\\model_cluster_2011_2012_2013_1_1_8.netcdf'

# Function to extract model parameters from filename
def extract_years_weights(filepath: str, model_type: int):
    
    if model_type == 3:
        model = filepath.split("\\")[-1]

        years = [int(model[14:18]),int(model[19:23]),int(model[24:28])]
        weights = [int(model[29]),int(model[31]),int(model[33])]
        
        output = {
            'years': years,
            'weights': weights
            }
    if model_type == 2:
        model = filepath.split("\\")[-1]

        years = [int(model[21:25]),int(model[26:30])]
        weights = [int(model[39]),int(model[41])]
        
        output = {
            'years': years,
            'weights': weights
            }
    if model_type == 1:
        model = filepath.split("\\")[-1]

        years = [int(model[13:17])]
        weights = [1]
        
        output = {
            'years': years,
            'weights': weights
            }

    return output

# function to extract key result parameters
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

# function to generate a results table in the format required for chart
def generate_results_table(filepath, model_type):

    print('Extracting results for: ',filepath)
    df = review_highlights(filepath)
    #in the case of the unweighted single year model, multiply cost value by 10 to achieve equivalent cost over 10yr model
    if model_type == 1:
        cost_error = df.loc['Annualised Cost €M']['Model Value'] *10 / df.loc['Annualised Cost €M']['Reference Value'] -1
    else:
        cost_error = df.loc['Annualised Cost €M']['Model Value'] / df.loc['Annualised Cost €M']['Reference Value'] -1
    mean_capacity_mix_error = ((df.drop('Annualised Cost €M'))['Model Value'] / (df.drop('Annualised Cost €M'))['Reference Value'] -1).mean()
    mean_ab_capacity_mix_error = abs((df.drop('Annualised Cost €M'))['Model Value'] / (df.drop('Annualised Cost €M'))['Reference Value'] -1).mean()
    
    # TODO: Decision around weighting of compound error. Should capacity mix and cost errors be weighted equally?
    abs_compound_error = (abs(cost_error) + mean_ab_capacity_mix_error)/2

    output = extract_years_weights(filepath,model_type)

    # Populate results arrays, which depends on whether its a 1, 2, 3 year run
    arr_cost_err = []
    arr_mean_cap_err = []
    arr_mean_ab_cap_err = []
    arr_abs_comp_err = []
    for i in output['years']:
        arr_cost_err.append(cost_error)
        arr_mean_cap_err.append(mean_capacity_mix_error)
        arr_mean_ab_cap_err.append(mean_ab_capacity_mix_error)
        arr_abs_comp_err.append(abs_compound_error)

    output['Cost Error'] = arr_cost_err
    output['Mean Capacity Mix Error'] = arr_mean_cap_err
    output['Mean Abs Capacity Mix Error'] = arr_mean_ab_cap_err
    output['Abs Compound Error'] = arr_abs_comp_err

    if model_type ==1:
        output['Run'] = f"[{",".join(str(x) for x in output['years'])}]"
    else:
        output['Run'] = f"[{",".join(str(x) for x in output['years'])}]_[{":".join(str(x) for x in output['weights'])}]"
    
    return pd.DataFrame(output).set_index('Run')

# function that aggregates all results from a directory given a regex query
def results_in_dir(directory, regex_query, model_type):

    output = pd.DataFrame

    # search through the provided directory
    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        full_path = directory+"\\"+filename
        # print(full_path)
        if re.match(regex_query,filename):
            #if output is not yet initialised, initialise it
            if output.empty:
                output = generate_results_table(full_path,model_type)
            else:
                output = pd.concat([output,generate_results_table(full_path,model_type)])

    return output

#function that executes the above, with appropriate regex and dir pointers
def run():
    result_n_1 = results_in_dir('simple_weather-year_ldes-model/results/single_year_runs',r"^results_plan_\d{4}_[a-zA-Z0-9]*\.netcdf$",1)
    result_n_2 = results_in_dir('simple_weather-year_ldes-model/results/two_year_cluster_runs',r"^results_plan_cluster_\d{4}_\d{4}_[a-zA-Z0-9_]*\.netcdf$",2)
    result_n_3 = results_in_dir('simple_weather-year_ldes-model/results/n_year_cluster_runs',r"^model_cluster_\d{4}_\d{4}_\d{4}_[a-zA-Z0-9_]*\.netcdf$",3)

    result = pd.concat([result_n_1,result_n_2, result_n_3])
    result.to_csv('simple_weather-year_ldes-model/export/all_runs_results.csv')

#function for plotting a basic box plot of results
def compound_box_plot(result):
   
 # grouped = result.groupby('Run').mean()
    result.drop_duplicates(subset=['Run'], inplace=True)

    result_n_1 = result[result['Run'].str.match(r"\[\d{4}\]")]
    result_n_2 = result[result['Run'].str.match(r"^\[\d{4},\d{4}\]")]
    result_n_3 = result[result['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]

    fig = go.Figure()

    fig.add_trace(go.Box(
        name='1 WY',
        y=result_n_1['Abs Compound Error'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        ))
    fig.add_trace(go.Box(
        name='2 WYs',
        y=result_n_2['Abs Compound Error'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        ))
    fig.add_trace(go.Box(
        name='3 WYs',
        y=result_n_3['Abs Compound Error'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        ))
    fig.update_layout(yaxis_tickformat="5%")

    fig.show()

def graphic_bar_featured_year_errors(result, parameter):
    arr_years = sorted(result['years'].unique())
    arr_errors_n_1 = []
    arr_errors_n_2 = []
    arr_errors_n_3 = []

    result_n_1 = result[result['Run'].str.match(r"\[\d{4}\]")]
    result_n_2 = result[result['Run'].str.match(r"^\[\d{4},\d{4}\]")]
    result_n_3 = result[result['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]


    for y in arr_years:
        arr_errors_n_1.append((result_n_1[result_n_1['years']==y][parameter]).mean())
        arr_errors_n_2.append((result_n_2[result_n_2['years']==y][parameter]).mean())
        arr_errors_n_3.append((result_n_3[result_n_3['years']==y][parameter]).mean())
    
    output = pd.DataFrame({
        'years': arr_years,
        '1WY Error': arr_errors_n_1,
        '2WY Error': arr_errors_n_2,
        '3WY Error': arr_errors_n_3
    })

    fig = go.Figure()

    WY1_colour = 'rgb(102, 153, 255)'
    WY2_colour = 'rgb(51, 204, 204)'
    WY3_colour = 'rgb(51, 51, 153)'

    fig.add_trace(go.Bar(
        name='1WY Error',
        y=output['1WY Error'],
        x=output['years'],
        marker_color=WY1_colour
        ))
    fig.add_trace(go.Bar(
        name='2WY Error',
        y=output['2WY Error'],
        x=output['years'],
        marker_color=WY2_colour
        ))
    fig.add_trace(go.Bar(
        name='3WY Error',
        y=output['3WY Error'],
        x=output['years'],
        marker_color=WY3_colour
        ))
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        title=dict(text=f"{parameter}, by Cluster Size"),
        yaxis = dict(
            title = dict(
                text=parameter,
                font=dict(size=16)
            )
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap = 0.15,
        bargroupgap=0.1,
        yaxis_tickformat="2%",
    )
    

    fig.write_html('simple_weather-year_ldes-model/export/bar_featured_year_error.html', auto_open=True)


    return output

# function plots box plot of results grouped by error type
def graphic_box_plot_cluster_errors(result):

    result.drop_duplicates(subset=['Run'], inplace=True)

    #separate and group Design Error and Cost Error
    str_cost = 'Cost Error'
    str_abs_design = 'Mean Abs Capacity Mix Error'
    # str_design = 'Mean Capacity Mix Error'
    df_cost_design_unpivoted = result[['Run',str_cost,str_abs_design]]
    df_cost_design_unpivoted=pd.melt(df_cost_design_unpivoted, id_vars=['Run'], var_name='Parameter', value_name='Error')  

    result_n_1 = df_cost_design_unpivoted[df_cost_design_unpivoted['Run'].str.match(r"\[\d{4}\]")]
    result_n_2 = df_cost_design_unpivoted[df_cost_design_unpivoted['Run'].str.match(r"^\[\d{4},\d{4}\]")]
    result_n_3 = df_cost_design_unpivoted[df_cost_design_unpivoted['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]

    WY1_colour = 'rgb(102, 153, 255)'
    WY2_colour = 'rgb(51, 204, 204)'
    WY3_colour = 'rgb(51, 51, 153)'

    fig = go.Figure()

    fig.add_trace(go.Box(
        name='1 WY',
        y=result_n_1['Error'],
        x=result_n_1['Parameter'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=WY1_colour,
        # line=dict(width=0),
        ))
    fig.add_trace(go.Box(
        name='2 WYs',
        y=result_n_2['Error'],
        x=result_n_2['Parameter'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=WY2_colour,
        # line=dict(width=0),
        ))
    fig.add_trace(go.Box(
        name='3 WYs',
        y=result_n_3['Error'],
        x=result_n_3['Parameter'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=WY3_colour,
        # line=dict(width=0),
        ))

    fig.update_layout(
        title=dict(text=f"Distribution of Error Types by Cluster Size for different runs"),
        yaxis = dict(
            title = dict(
                text='Error',
                font=dict(size=16)
            )
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        boxmode='group',
        yaxis_tickformat="5.0%",
    )

    fig.write_html('simple_weather-year_ldes-model/export/box_plot_cluster_error.html', auto_open=True)

#function that extracts model runs that meet performance threshold
def identify_design_candidates(result,cost_err_threshold,mean_abs_cap_mix_threshold):
    
    #drops duplicates, (due to previous functionins breaking down years)
    result.drop_duplicates(subset=['Run'], inplace=True)

    #apply filters and sort
    df = (result[(abs(result['Cost Error'])<=cost_err_threshold) & (result['Mean Abs Capacity Mix Error']<=mean_abs_cap_mix_threshold)]).sort_values(by='Abs Compound Error', ascending=True)
 
    return df

# function that pulls 4 digit numbers from a string and returns the first that does not match an array of exclusion numbers
def extract_numbers(s, year_couple):
    numbers = re.findall(r'\b\d{4}\b', s)  # Find all 4-digit numbers
    numbers.remove(str(year_couple[0]))
    numbers.remove(str(year_couple[1]))
    return int(numbers[0])  # Join numbers into a single string (optional)

def review_year_couples_in_n_3_runs(result,year_couple: list[int]):
    
    #drops duplicates, (due to previous functionins breaking down years)
    result.drop_duplicates(subset=['Run'], inplace=True)

    # array of regex identifiers
    pattern=[]
    pattern.append(fr"\[{year_couple[0]},{year_couple[1]},[0-9]{{4}}\]_\[1:1:8\]")
    pattern.append(fr"\[[0-9]{{4}},{year_couple[0]},{year_couple[1]}\]_\[8:1:1\]")
    pattern.append(fr"\[{year_couple[1]},[0-9]{{4}},{year_couple[0]}\]_\[1:8:1\]")
    
    #identify results where year-couple prepend the operation year
    pre_anchor_result = result[result['Run'].str.match(pattern[0])]
    #modify the output df such that the year column corresponds with the operation year
    pre_anchor_result['years'] = pre_anchor_result['Run'].apply(lambda x: extract_numbers(x, year_couple))
    
    #identify results where year-couple postfix the operation year, and modify as above
    post_anchor_result = result[result['Run'].str.match(pattern[1])]
    post_anchor_result['years'] = post_anchor_result['Run'].apply(lambda x: extract_numbers(x, year_couple))
    
    #identify results where year-couple top and tail the op year, and modify as above
    mid_anchor_result = result[result['Run'].str.match(pattern[2])]
    mid_anchor_result['years'] = mid_anchor_result['Run'].apply(lambda x: extract_numbers(x, year_couple))


    #concat the results filtered by the above regex queries 
    output = pd.concat(
        [pre_anchor_result,
         mid_anchor_result,
         post_anchor_result,
        ]
    )
    
    print(output)

    colour_1 = 'rgb(102, 153, 255)'
    colour_2 = 'rgb(51, 204, 204)'
    colour_3 = 'rgb(51, 51, 153)'

    fig = go.Figure()

    if not pre_anchor_result.empty:
        fig.add_trace(go.Bar(
            x=pre_anchor_result['years'],
            y=pre_anchor_result['Abs Compound Error'],
            name='Prefixed System Defining WYs',
            marker_color = colour_1
        ))
    if not post_anchor_result.empty:
        fig.add_trace(go.Bar(
            x=post_anchor_result['years'],
            y=post_anchor_result['Abs Compound Error'],
            name='Postfixed System Defining WYs',
            marker_color = colour_2
        ))
    if not mid_anchor_result.empty:
        fig.add_trace(go.Bar(
            x=mid_anchor_result['years'],
            y=mid_anchor_result['Abs Compound Error'],
            name='Top and Tailed System Defining WYs',
            marker_color = colour_3
        ))
    
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(
        title=dict(text=f"Comparing Results for 3-yr Clusters Anchored by {year_couple[0]} and {year_couple[1]}"),
        yaxis = dict(
            title = dict(
                text='Absolute Compound Error',
                font=dict(size=16)
            )
        ),
        xaxis = dict(
            title = dict(
                text='Appended (Operation) Year',
                font=dict(size=16)
            )
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap = 0.15,
        bargroupgap=0.1,
        yaxis_tickformat="2%",
    )

    fig.write_html('simple_weather-year_ldes-model/export/year_couple_results.html', auto_open=True)

result = pd.read_csv('simple_weather-year_ldes-model/export/all_runs_results.csv')

parameter='Mean Abs Capacity Mix Error' # Options:'Mean Capacity Mix Error' 'Mean Abs Capacity Mix Error' 'Abs Compound Error' 'Cost Error'

graphic_box_plot_cluster_errors(result)
graphic_bar_featured_year_errors(result,'Abs Compound Error')
review_year_couples_in_n_3_runs(result,[2016,2017])