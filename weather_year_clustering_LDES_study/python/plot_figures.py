import calliope
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import re
from plotly.subplots import make_subplots

# Config
variant = 'with_surplus_tracking' #with_intracluster_cycle_condition #with_surplus_tracking #two cluster conditions were tested, here can switch between them
figure_directory = 'weather_year_clustering_LDES_study/results/figures'

# function based colour palette, takes id and opacity factor
def colour(id: int, opacity: float = 1):
    colours = [
        '0,0,0', #0 black
        '240, 249, 33', #1 yellow
        '253, 180, 47', #2 gold
        '237, 121, 83', #3 orange
        '204, 71, 120', #4 pink 
        '156, 23, 158', #5 purple
        '92, 1, 166', #6 indigo 
        '13, 8, 135', #7 blue
        '255,255,255', #8 white
    ]
    return f'rgba({colours[id]},{opacity})'

plasma_scale = [
    'rgb(240, 249, 33)',
    'rgb(252, 166, 54)',
    'rgb(225, 100, 98)',
    'rgb(177, 42, 144)',
    'rgb(106, 0, 168)',
    'rgb(13, 8, 135)',
    'rgb(0, 0, 62)',
]

plasma_discrete_scale = [
    [0,plasma_scale[5]], #0-5%
    [1/6,plasma_scale[5]],

    [1/6,plasma_scale[4]], #5%-10%
    [2/6,plasma_scale[4]],

    [2/6,plasma_scale[3]], #10%-15%
    [3/6,plasma_scale[3]],

    [3/6,plasma_scale[2]], #15%-20%
    [4/6,plasma_scale[2]],

    [4/6,plasma_scale[1]], #20%-25%
    [5/6,plasma_scale[1]],

    [5/6,plasma_scale[0]], #25%-30%
    [1,plasma_scale[0]],


]

#paths to ref model and cluster model directories
path_ref_model = 'weather_year_clustering_LDES_study/results/reference/results_2010_to_2019.netcdf' #TODO: Update with latest run
path_directory_single_year_cluster = 'weather_year_clustering_LDES_study/results/single_year_cluster'
path_variants = {
    'unmodified': {
        'two_year' : 'weather_year_clustering_LDES_study/results/two_year_cluster/unmodified',
        'three_year' : 'weather_year_clustering_LDES_study/results/three_year_cluster/unmodified',
    },
    'with_surplus': {
        'two_year' : 'weather_year_clustering_LDES_study/results/two_year_cluster/with_surplus_tracking',
        'three_year' : 'weather_year_clustering_LDES_study/results/three_year_cluster/with_surplus_tracking',
    },
    'cycle_condition': {
        'two_year' : 'weather_year_clustering_LDES_study/results/two_year_cluster/with_intracluster_cycle_condition',
        'three_year' : 'weather_year_clustering_LDES_study/results/three_year_cluster/with_intracluster_cycle_condition',
    }
}

# 
#               HELPER FUNCTIONS
# 

# Function to extract model parameters from filename
def extract_years_weights(filepath: str, model_type: int):
    
    if model_type == 3:
        model = filepath.split("/")[-1]
        years = [int(model[15:19]),int(model[20:24]),int(model[25:29])]
        weights = [int(model[39]),int(model[41]),int(model[43])]

        output = {
            'years': years,
            'weights': weights,
            'contiguous': True if (abs(years[0]-years[1]) ==1 and abs(years[1]-years[2])) else False
            }
    if model_type == 2:
        model = filepath.split("/")[-1]

        years = [int(model[15:19]),int(model[20:24])]
        weights = [int(model[34]),int(model[36])]
        
        output = {
            'years': years,
            'weights': weights,
            'contiguous': True if (abs(years[0]-years[1]) ==1) else False
            
            }
    if model_type == 1:
        model = filepath.split("/")[-1]

        years = [int(model[13:17])]
        weights = [1]
        
        output = {
            'years': years,
            'weights': weights,
            'contiguous': True
            }

    return output

# Function to extract results from model
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

# function to extract flow capacity mix results, across power and hydrogen
def extract_plan_results(model_name, import_file_path):
    
    #read model
    model = calliope.read_netcdf(import_file_path)
    
    #get power flow capacities
    df_flow_cap_power = (
                    (model.results.flow_cap.fillna(0))
                    .sel(carriers="power")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Flow Capacity (Power) (MW)")
                    .reset_index()
                )
    
    #get hydrogen flow capacities
    df_flow_cap_hydrogen = (
                    (model.results.flow_cap.fillna(0))
                    .sel(carriers="hydrogen")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Flow Capacity (Hydrogen) (MW)")
                    .reset_index()
                )
    
    results_dict = {
        'Model': model_name,
        'Solar': df_flow_cap_power.loc[df_flow_cap_power['techs']=='solar']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Offshore Wind': df_flow_cap_power.loc[df_flow_cap_power['techs']=='offshore_wind']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Onshore Wind': df_flow_cap_power.loc[df_flow_cap_power['techs']=='onshore_wind']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Nuclear': df_flow_cap_power.loc[df_flow_cap_power['techs']=='nuclear']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Battery (Power)': df_flow_cap_power.loc[df_flow_cap_power['techs']=='battery']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Electrolyser': df_flow_cap_hydrogen.loc[df_flow_cap_hydrogen['techs']=='electrolyser']["Flow Capacity (Hydrogen) (MW)"].values[0]/1000,
        'Demand': df_flow_cap_power.loc[df_flow_cap_power['techs']=='demand_power']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Hydrogen CCGT': df_flow_cap_power.loc[df_flow_cap_power['techs']=='h2_elec_conversion']["Flow Capacity (Power) (MW)"].values[0]/1000,
        }
    
    df_result = pd.DataFrame(results_dict, index=[model_name])
    
    return df_result

# function to extract key result parameters for figure 3
def review_highlights(path_test_model, path_reference):

    #load respective models for test case and reference
    model = calliope.read_netcdf(path_test_model)
    full_model = calliope.read_netcdf(path_reference)
    
    #extract storage capacities
    df_storage_cap_test = model_results_extractor(model, 'storage_cap')
    df_storage_cap_reference = model_results_extractor(full_model, 'storage_cap')
    #extract annualised costs
    df_cost_test = model_results_extractor(model, 'cost')
    df_cost_reference = model_results_extractor(full_model, 'cost')

    #initialise and populate output dictionary
    output = {
        'Parameter': [
            'H2 Storage (GWh)',
            'Battery Storage (GWh)',
            "Annualised Cost €M",
            ],
        'Model Value': [
            df_storage_cap_test.loc[1]['storage_cap'], #save absolute values from test
            df_storage_cap_test.loc[0]['storage_cap'],
            df_cost_test['cost'].sum(),
            ],
        'Reference Value': [
            df_storage_cap_reference.loc[1]['storage_cap'], #save absolute values from reference
            df_storage_cap_reference.loc[0]['storage_cap'],
            df_cost_reference['cost'].sum(),
            ],
        'Error': [
            f"{round(100*(df_storage_cap_test.loc[1]['storage_cap']/df_storage_cap_reference.loc[1]['storage_cap']-1),1)}%", #compute errors as per paper
            f"{round(100*(df_storage_cap_test.loc[0]['storage_cap']/df_storage_cap_reference.loc[0]['storage_cap']-1),1)}%",
            f"{round(100*(df_cost_test['cost'].sum()/df_cost_reference['cost'].sum()-1),1)}%",
            ]
        }

    #list of techs with flow capacities
    techs = ['solar','offshore_wind','electrolyser','h2_elec_conversion','battery']
    #for each tech, extract the flow capacity from the test model and the reference, compute the error and save to the output dictionary
    for tech in techs:
        df_flow_cap_test = model_results_extractor(model, 'flow_cap')
        df_flow_cap_test = df_flow_cap_test[df_flow_cap_test['techs']==tech]
        df_flow_cap_reference = model_results_extractor(full_model, 'flow_cap')
        df_flow_cap_reference = df_flow_cap_reference[df_flow_cap_reference['techs']==tech]

        output['Parameter'].append(f"{tech} Flow Cap (MW)")
        output['Model Value'].append(int(df_flow_cap_test.iloc[0]["flow_cap"]))
        output['Reference Value'].append(int(df_flow_cap_reference.iloc[0]["flow_cap"])) 
        output['Error'].append(f"{round(((df_flow_cap_test.iloc[0]["flow_cap"]/(df_flow_cap_reference.iloc[0]["flow_cap"])-1)*100).mean(),1)}%")
    
    return pd.DataFrame.from_dict(output).set_index('Parameter')

# function to generate a results table in the format required for chart
def generate_results_table(path_test_model, test_model_type, path_reference):

    print('Extracting results for: ',path_test_model)
    df = review_highlights(path_test_model, path_reference)
    #in the case of the unweighted single year model, multiply cost value by 10 to achieve equivalent cost over 10yr model
    if test_model_type == 1:
        cost_error = df.loc['Annualised Cost €M']['Model Value'] *10 / df.loc['Annualised Cost €M']['Reference Value'] -1
    else:
        cost_error = df.loc['Annualised Cost €M']['Model Value'] / df.loc['Annualised Cost €M']['Reference Value'] -1
    mean_capacity_mix_error = ((df.drop('Annualised Cost €M'))['Model Value'] / (df.drop('Annualised Cost €M'))['Reference Value'] -1).mean()
    mean_ab_capacity_mix_error = abs((df.drop('Annualised Cost €M'))['Model Value'] / (df.drop('Annualised Cost €M'))['Reference Value'] -1).mean()
    h2_capacity_error = (df[df.index=='H2 Storage (GWh)']['Model Value']/(df[df.index=='H2 Storage (GWh)']['Reference Value']).mean()).mean()-1
    
    # TODO: Decision around weighting of compound error. Should capacity mix and cost errors be weighted equally?
    abs_compound_error = (abs(cost_error) + mean_ab_capacity_mix_error)/2

    output = extract_years_weights(path_test_model,test_model_type)

    # Populate results arrays, which depends on whether its a 1, 2, 3 year run
    arr_cost_err = []
    arr_mean_cap_err = []
    arr_mean_ab_cap_err = []
    arr_abs_comp_err = []
    arr_h2_cap_err = []
    for i in output['years']:
        arr_cost_err.append(cost_error)
        arr_mean_cap_err.append(mean_capacity_mix_error)
        arr_mean_ab_cap_err.append(mean_ab_capacity_mix_error)
        arr_abs_comp_err.append(abs_compound_error)
        arr_h2_cap_err.append(h2_capacity_error)

    output['Cost Error'] = arr_cost_err
    output['Mean Capacity Mix Error'] = arr_mean_cap_err
    output['Mean Abs Capacity Mix Error'] = arr_mean_ab_cap_err
    output['Abs Compound Error'] = arr_abs_comp_err
    output['LDES Capacity Error'] = arr_h2_cap_err

    if test_model_type ==1:
        output['Run'] = f"[{",".join(str(x) for x in output['years'])}]"
    else:
        output['Run'] = f"[{",".join(str(x) for x in output['years'])}]_[{":".join(str(x) for x in output['weights'])}]"
    
    return pd.DataFrame(output).set_index('Run')

# function to compute error between two series
def compute_error(df_original,df_artificial,parameter: str):

    dc_original = (df_original[parameter].sort_values(ascending=False).values).astype(np.float32)
    dc_artificial = (df_artificial[parameter].sort_values(ascending=False).values).astype(np.float32)

    abs_deviation = np.divide(abs(dc_original-dc_artificial), dc_original, out=np.zeros_like(abs(dc_original-dc_artificial)), where=dc_original!=0)

    return abs_deviation.mean()

# duration curve error function
def compute_duration_curve_error(path_timeseries_ref: str, path: str, model_type: int):

    #extract cluster parameters
    cluster_params = extract_years_weights(path, model_type)
    # load original timeseries
    df_ts_original = pd.read_csv(path_timeseries_ref)
    df_ts_original.columns = ['timesteps','demand','solar_cf','offshore_cf','onshore_cf']
    df_ts_original = df_ts_original[4:]
    # Conversion of value formats
    df_ts_original['timesteps'] = pd.to_datetime(df_ts_original['timesteps'])
    df_ts_original['demand'] = pd.to_numeric(df_ts_original['demand'])
    df_ts_original['solar_cf'] = pd.to_numeric(df_ts_original['solar_cf'])
    df_ts_original['offshore_cf'] = pd.to_numeric(df_ts_original['offshore_cf'])
    df_ts_original['onshore_cf'] = pd.to_numeric(df_ts_original['onshore_cf'])

    # filter timerseries to 2010-2019 inclusive
    df_ts_original=df_ts_original[df_ts_original['timesteps'].dt.year >= 2010]
    # remove leap days due to series length mismatches
    df_ts_original=df_ts_original[~((df_ts_original['timesteps'].dt.month == 2) &(df_ts_original['timesteps'].dt.day == 29))]


    df_ts_artificial = pd.DataFrame
    columns_to_apply_weights = ['demand','solar_cf','offshore_cf','onshore_cf']

    for i in range(0,len(cluster_params['years'])):
        #filter the original ts to the given year
        df_temp = df_ts_original[df_ts_original['timesteps'].dt.year == cluster_params['years'][i]]

        #assign/append to artificial timeseries
        if df_ts_artificial.empty:
            df_ts_artificial = df_temp
        else:
            df_ts_artificial = pd.concat([df_ts_artificial,df_temp])

        if cluster_params['weights'][i] > 1:
            # duplicate the timeseries for that year based on its weighting. i.e. for a year weighted as 8, we get that year's timeseries 8 times.
            for j in range(1,(cluster_params['weights'][i]+1)-1):
                df_ts_artificial = pd.concat([df_ts_artificial,df_temp])

    # compute duration curve and duration curve error for each parameter
    demand_error = compute_error(df_ts_original,df_ts_artificial,'demand')
    solar_error = compute_error(df_ts_original,df_ts_artificial,'solar_cf')
    offshore_error = compute_error(df_ts_original,df_ts_artificial,'offshore_cf')
    onshore_error = compute_error(df_ts_original,df_ts_artificial,'onshore_cf')
    combined_absolute_error = (demand_error+solar_error+offshore_error+onshore_error)/4

    return  {
        'demand_error': demand_error,
        'solar_cf_error': solar_error,
        'offshore_cf_error': offshore_error,
        'onshore_cf_error': onshore_error,
        'compound_error': combined_absolute_error
    }

# function that aggregates all timeseries errors for models within a directory given a regex query
def timeseries_error_in_dir(path_timeseries_ref,directory, regex_query, model_type):

    output = pd.DataFrame

    # search through the provided directory
    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        full_path = directory+"/"+filename
        # print(full_path)
        if re.match(regex_query,filename):
            #if output is not yet initialised, initialise it
            print('Extracting Results:',filename)
            data = pd.DataFrame(compute_duration_curve_error(path_timeseries_ref,full_path,model_type),index=[0])
            cluster_params = extract_years_weights(filename,model_type)
            # data['Years'] = ",".join(str(x) for x in cluster_params['years'])
            # data['Weights'] = ",".join(str(x) for x in cluster_params['weights'])
            data['Run'] = f"[{",".join(str(x) for x in cluster_params['years'])}]_[{":".join(str(x) for x in cluster_params['weights'])}]"
            if output.empty:
                output = data
            else:
                output = pd.concat([output,data])

    return output

#  checks if model wys are sequential
def is_sequential(number_string: str):
    # Convert string to array of numbers
    numbers = list(map(int, number_string.split(',')))
    diffs = [abs(numbers[0]-numbers[1]),abs(numbers[1]-numbers[2]),abs(numbers[2]-numbers[0])]
    if diffs.count(1) != 2:
        return False
    return True
    # check these are consecutive
    # return all(numbers[i]+1 == numbers[i+1] for i in range(len(numbers)-1))

#  checks if model wys acnhor weather years are a particular set
def includes_system_defining_years(number_string: str, check_array: list[int] =[2016,2017]):
    # Convert string to array of numbers
    numbers = list(map(int, number_string.split(',')))

    if numbers[-1] == check_array[-1] and numbers[-2] == check_array[-2]:
        return True
    else:
        return False



# 
#          FIGURE FUNCTIONS
# 

# function plots the 1st figure of the paper, comparing single year run SOCs with reference SOC
def plot_fig1_area_chart_of_SOC(path_ref_model,path_directory_single_year_cluster):
    print('Plotting Fig 1')

    #load ref model, TODO: after reference finished
    df_reference= model_results_extractor(calliope.read_netcdf(path_ref_model), 'storage')

    #establish empty dataframe with results
    df_single_years=pd.DataFrame()

    #iterate over the files and save the SOC results, aggregating them into a single super df
    for file in os.listdir(os.fsencode(path_directory_single_year_cluster)):
        model = calliope.read_netcdf(path_directory_single_year_cluster+'/'+os.fsdecode(file))
        if df_single_years.empty:
            df_single_years = model_results_extractor(model, 'storage')
        else:
            df_single_years = pd.concat([df_single_years,model_results_extractor(model, 'storage')])
    
    #filter to only report results for LDES (i.e. remove battery SOC)
    df_single_years = df_single_years[df_single_years['techs']=='h2_salt_cavern']
    df_reference = df_reference[df_reference['techs']=='h2_salt_cavern'] 

    #show SOC in TWh not MWh
    df_single_years['storage'] = df_single_years['storage']/1e6
    df_reference['storage'] = df_reference['storage']/1e6


    #create figure object using Graph_objects
    fig = go.Figure()

    #add trace with ref, 
    fig.add_trace(go.Scatter(
        x=df_reference['timesteps'],
        y=df_reference['storage'],
        fill='tozeroy',
        mode='none',
        fillcolor=colour(2,0.8),
        name='Reference Model'
    ))

    #add traces with single wy results, 
    fig.add_trace(go.Scatter(
        x=df_single_years['timesteps'],
        y=df_single_years['storage'],
        fill='tozeroy',
        mode='none',
        fillcolor=colour(5,0.8),
        name='Concatenated Single Weather-Year Models'
    ))

    fig.add_vline(x='2015-11-01 00:00:00', line_width=2, line_dash="dash", line_color=colour(6,0.5))
    fig.add_vline(x='2017-02-01 00:00:00', line_width=2, line_dash="dash", line_color=colour(6,0.5))


    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        # title=dict(text=f"SOC for hydrogen storage over time for the reference model and concatenated single weather-year models."),
        yaxis = dict(
            title = dict(
                text='State of Charge, TWh',
                # font=dict(size=50)
            ),
            # tickfont = dict(size=48),
        ),
        xaxis = dict(
            title = dict(
                text='Date',
                # font=dict(size=48)
            ),
            # tickfont = dict(size=48),
        ),
        legend=dict(
            x=0.01,
            y=1.0,
            bgcolor = 'rgba(255,255,255,1)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        font=dict(
        family="Times New Roman",
        size=48,
        # color="RebeccaPurple"
        ),
        boxmode='group',
        yaxis_tickformat=".0f",
    )

    # fig.update_yaxes(matches=None)
    fig.update_xaxes(
        tickvals=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        tickformat="%Y",
        title_standoff = 36
    )
    fig.update_yaxes(
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='rgba(255, 255, 255, 0)',
        title_standoff = 36
    )
    # fig.write_html(f"{figure_directory}/fig1.html", auto_open=True)
    fig.write_image(f"{figure_directory}/fig1.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig1.jpeg", width=1920, height=1920, scale=1)

# function plots the 2nd figure of the paper, capacity mixes for the refernce vs single year runs
def plot_fig2_bar_chart_of_capacity_mix(path_ref_model,path_directory_single_year_cluster):
    print('Plotting Fig 2')
    #load ref model, TODO: after reference finished
    df_results= extract_plan_results('Reference',path_ref_model)

    #iterate over the files and save the SOC results, aggregating them into a single super df
    for file in os.listdir(os.fsencode(path_directory_single_year_cluster)):
        file_path = path_directory_single_year_cluster+'/'+os.fsdecode(file)
        df_results = pd.concat([df_results,extract_plan_results(str(extract_years_weights(file_path,1)['years'][0]),file_path)])

    fig = go.Figure(data=[
              go.Bar(name='Nuclear', x=df_results['Model'], y=df_results['Nuclear'],marker_color=colour(1,1)),
              go.Bar(name='Onshore Wind', x=df_results['Model'], y=df_results['Onshore Wind'], marker_color=colour(2,1)),
              go.Bar(name='Offshore Wind', x=df_results['Model'], y=df_results['Offshore Wind'], marker_color=colour(3,1)),
              go.Bar(name='Solar', x=df_results['Model'], y=df_results['Solar'], marker_color=colour(4,1)),
              go.Bar(name='Hydrogen CCGT', x=df_results['Model'], y=df_results['Hydrogen CCGT'], marker_color=colour(5,1)),
              go.Bar(name='Battery', x=df_results['Model'], y=df_results['Battery (Power)'], marker_color=colour(6,1)),
              go.Bar(name='Electrolyser', x=df_results['Model'], y=df_results['Electrolyser'],marker_color=colour(7,1)),
    ]
    )

    fig.update_layout(
        barmode='stack',
        plot_bgcolor='rgba(255, 255, 255, 0)',
        yaxis = dict(
            title = dict(
                text='Capacity, GW',
            )
        ),
        xaxis = dict(
            title = dict(
                text='Model',
            )
        ),
        font=dict(
        family="Times New Roman",
        size=48,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=.95,
            xanchor="center",
            x=0.5,
            bgcolor = 'rgba(255,255,255,1)',
            
        ),
    )

    fig.update_xaxes(
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        title_standoff = 36
    )
    fig.update_yaxes(
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='rgba(255, 255, 255, 0)',
        title_standoff = 36
    )

    # fig.write_html(f"{figure_directory}/fig2.html", auto_open=True)
    fig.write_image(f"{figure_directory}/fig2.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig2.jpeg", width=1920, height=1920, scale=1)

# function generates data for the 3rd figure
def generate_data_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster,variant):
    print('Extracting Data for Fig 3')

    #initialise empty df
    df = pd.DataFrame
    
    #single_year_results
    for file in os.listdir(os.fsencode(path_directory_single_year_cluster)):
        filepath = (path_directory_single_year_cluster+'/'+os.fsdecode(file))
        if df.empty:
            df = generate_results_table(filepath,1,path_ref_model)
        else:
            df = pd.concat([df,generate_results_table(filepath,1,path_ref_model)])

    #two_year_results
    for file in os.listdir(os.fsencode(path_directory_two_year_cluster)):
        filepath = (path_directory_two_year_cluster+'/'+os.fsdecode(file))
        df = pd.concat([df,generate_results_table(filepath,2,path_ref_model)])

    #three_year_results
    for file in os.listdir(os.fsencode(path_directory_three_year_cluster)):
        filepath = (path_directory_three_year_cluster+'/'+os.fsdecode(file))
        df = pd.concat([df,generate_results_table(filepath,3,path_ref_model)])

    df.to_csv(f'simple_weather-year_ldes-model/export/data_cluster_box_plots_{variant}.csv')

# function plots the 3rd figure, a box plot of errors against reference
def plot_fig3_1_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster, variant):
    print('Plotting Fig 3')

    # generate_data_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster,variant)
    df = pd.read_csv(f'simple_weather-year_ldes-model/export/data_cluster_box_plots_{variant}.csv')
    df.drop_duplicates(subset=['Run'], inplace=True)
    df['Cost Error'] = -1* df['Cost Error']
    df['LDES Capacity Error'] = -1* df['LDES Capacity Error']
    #separate and group Design Error and Cost Error
    str_cost = 'Cost<br>Error'
    str_abs_design = 'Mean Abs<br>Capacity Mix<br>Error'
    str_compound_err = 'Abs Compound<br>Error'
    ldes_err = 'LDES Capacity<br>Error'
    df[str_cost] = df['Cost Error']
    df[ldes_err] = df['LDES Capacity Error']
    df[str_abs_design] = df['Mean Abs Capacity Mix Error']
    df[str_compound_err] = df['Abs Compound Error']

    df_cost_design_unpivoted = df[['Run',str_cost,str_abs_design,ldes_err]]
    df_cost_design_unpivoted=pd.melt(df_cost_design_unpivoted, id_vars=['Run'], var_name='Parameter', value_name='Error')  

    result_n_1 = df_cost_design_unpivoted[df_cost_design_unpivoted['Run'].str.match(r"\[\d{4}\]")]
    result_n_2 = df_cost_design_unpivoted[df_cost_design_unpivoted['Run'].str.match(r"^\[\d{4},\d{4}\]")]
    result_n_3 = df_cost_design_unpivoted[df_cost_design_unpivoted['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]

    fig = go.Figure()

    fig.add_trace(go.Box(
        name='Single Weather Year',
        y=result_n_1['Error'],
        x=result_n_1['Parameter'],
        text=result_n_1['Run'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=colour(7,1),
        line=dict(width=3),
        ))
    fig.add_trace(go.Box(
        name='Two Weather Years',
        y=result_n_2['Error'],
        x=result_n_2['Parameter'],
        text=result_n_2['Run'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=colour(5,1),
        line=dict(width=3),
        ))
    fig.add_trace(go.Box(
        name='Three Weather Years',
        y=result_n_3['Error'],
        x=result_n_3['Parameter'],
        text=result_n_3['Run'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=colour(3,1),
        line=dict(width=3),
        ))
    fig.update_yaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0.2)',
    
    )
    fig.update_xaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0)'
    )
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        yaxis = dict(
            title = dict(
                text='Error',
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,1)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        boxmode='group',
        yaxis_tickformat=".0%",
        font=dict(
        family="Times New Roman",
        size=48,
        ),
    )

    # fig.write_html(f"{figure_directory}/fig3_1.html", auto_open=True)
    fig.write_image(f"{figure_directory}/fig3_1.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig3_1.jpeg", width=1920, height=1920, scale=1)

# function plots an alternative 3rd figure, putting error matrics on two axes
def plot_fig3_2_scatter_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster, variant):
    print('Plotting Fig 3.2')


    # generate_data_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster,variant)
    df = pd.read_csv(f'simple_weather-year_ldes-model/export/data_cluster_box_plots_{variant}.csv')
    df.drop_duplicates(subset=['Run'], inplace=True)
    df['Cost Error'] = -1* df['Cost Error']
    df['LDES Capacity Error'] = -1* df['LDES Capacity Error']
    #separate and group Design Error and Cost Error
    result_n_1 = df[df['Run'].str.match(r"\[\d{4}\]")]
    result_n_2 = df[df['Run'].str.match(r"^\[\d{4},\d{4}\]")]
    result_n_3 = df[df['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]

    y_val = 'Cost Error'
    z_val = 'Mean Abs Capacity Mix Error' #'Mean Abs Capacity Mix Error'   #'LDES Capacity Error'
    x_val = 'LDES Capacity Error'

    line_width = 2
    marker_sz = 32

    fig =  make_subplots(
        rows=3, cols=1,
        subplot_titles=('<b>Single Weather Year<b>','<b>Two Weather Years<b>','<b>Three Weather Years<b>'),
        # shared_xaxes=True
        )


    fig.add_trace(go.Scatter(
        name='Single Weather Year',
        y=result_n_1[y_val],
        x=result_n_1[x_val],
        text=result_n_1['Run'],
        mode='markers',
        marker_size = marker_sz,
        marker=dict(
            color = result_n_1[z_val],
            coloraxis = 'coloraxis',
            line=dict(
                color = colour(0,0.4),
                width=line_width
            )),
        customdata=np.stack(([result_n_1[z_val]]),axis=-1),
        hovertemplate =
        f'<i>{y_val}</i>: '+'%{y:.2%}'+
        f'<br><b>{x_val}</b>: '+'%{x:.2%}<br>'+
        f'<i>{z_val}</i>: '+'%{customdata[0]:.2%}<br>'+
        '<b>%{text}</b>',
        ),
        row=1,col=1)
    
    fig.add_trace(go.Scatter(
        name='Two Weather Years',
        y=result_n_2[y_val],
        x=result_n_2[x_val],
        text=result_n_2['Run'],
        mode='markers',
        marker_size = marker_sz,
        marker=dict(
            color = result_n_2[z_val],
            coloraxis = 'coloraxis',
            # showscale=True,
            line=dict(
                color = colour(0,0.4),
                width=line_width
            )),
        customdata=np.stack(([result_n_2[z_val]]),axis=-1),
        hovertemplate =
        f'<i>{y_val}</i>: '+'%{y:.2%}'+
        f'<br><b>{x_val}</b>: '+'%{x:.2%}<br>'+
        f'<i>{z_val}</i>: '+'%{customdata[0]:.2%}<br>'+
        '<b>%{text}</b>',
        ),
        row=2,col=1)
    
    #ARTIFICIAL DATA TO FORCE COLOURBAR TO SHOW PROPERLY. DATA POINT DOES NOT SHOW ON CHART.
    fig.add_trace(go.Scatter(
        name='Artificial',
        y=[result_n_3[y_val].iloc[0],result_n_3[y_val].iloc[1]],
        x=[result_n_3[x_val].iloc[0],result_n_3[x_val].iloc[1]],
        text=result_n_3['Run'],    
        mode='markers',
        marker_size = marker_sz,
        marker=dict(
            color = [0,0.3],
            coloraxis = 'coloraxis',
            line=dict(
                color = colour(0,0.0),
                width=line_width
            )),
        ),
        row=3,col=1)
    
    # REAL DATA
    fig.add_trace(go.Scatter(
        name='Three Weather Years',
        y=result_n_3[y_val],
        x=result_n_3[x_val],
        text=result_n_3['Run'],    
        mode='markers',
        marker_size = marker_sz,
        marker=dict(
            color = result_n_3[z_val],
            coloraxis = 'coloraxis',
            line=dict(
                color = colour(0,0.4),
                width=line_width
            )),
        customdata=np.stack(([result_n_3[z_val]]),axis=-1),
        hovertemplate =
        f'<i>{y_val}</i>: '+'%{y:.2%}'+
        f'<br><b>{x_val}</b>: '+'%{x:.2%}<br>'+
        f'<i>{z_val}</i>: '+'%{customdata[0]:.2%}<br>'+
        '<b>%{text}</b>',
        ),
        row=3,col=1)
    
    for i in range(1,3+1):
        fig.update_yaxes(
        row=i,col=1,
        ticks='outside',
        tickmode = 'array',
        tickvals = [-0.05,0,0.05,0.1,0.15],
        showline=True,
        linecolor='black',
        gridcolor='rgba(0, 0, 0, 0.2)',
        tickformat='.0%',
        zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        title=y_val,
        )
        fig.update_xaxes(
        row=i,col=1,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='rgba(0, 0, 0, 0)',
        tickformat='.0%',
        zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        title=x_val,
        range=[-0.07,0.83]
        )
        
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        showlegend=False,
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        font=dict(
        family="Times New Roman",
        size=40,
        ),
        coloraxis = dict(
            colorbar = dict(
            title = dict(text='MACME'),
            tickformat = '.0%',
            tickmode='array',
            tickvals = [0,.05,.1,.15,.2,.25,.3],
            ticktext = ['0%','5%','10%','15%','20%','25%',''],
            ),
            colorscale=plasma_discrete_scale,
            ),
    )
    fig.update_annotations(font_size=48)
    
    fig.write_html(f"{figure_directory}/fig3_2.html", auto_open=True)
    fig.write_image(f"{figure_directory}/fig3_2.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig3_2.jpeg", width=1920, height=1920, scale=1)

#  function plots an alternative 3rd figure, exploring difference between model variants
def plot_fig3_3_scatter_of_errors():
    print('Plotting Fig 3.2')

    df_w_surplus = pd.read_csv('simple_weather-year_ldes-model/export/data_cluster_box_plots_with_surplus.csv')
    df_cycle_condition  = pd.read_csv('simple_weather-year_ldes-model/export/data_cluster_box_plots_cycle_condition.csv')
    df_unmodified  = pd.read_csv('simple_weather-year_ldes-model/export/data_cluster_box_plots_unmodified.csv')

    df_w_surplus.drop_duplicates(subset=['Run'], inplace=True)
    df_cycle_condition.drop_duplicates(subset=['Run'], inplace=True)
    df_unmodified.drop_duplicates(subset=['Run'], inplace=True)

    df_w_surplus['Cost Error'] = -1* df_w_surplus['Cost Error']
    df_w_surplus['LDES Capacity Error'] = -1* df_w_surplus['LDES Capacity Error']
    df_cycle_condition['Cost Error'] = -1* df_cycle_condition['Cost Error']
    df_cycle_condition['LDES Capacity Error'] = -1* df_cycle_condition['LDES Capacity Error']
    df_unmodified['Cost Error'] = -1* df_unmodified['Cost Error']
    df_unmodified['LDES Capacity Error'] = -1* df_unmodified['LDES Capacity Error']

    #separate and group Design Error and Cost Error
    df_w_surplus = df_w_surplus[df_w_surplus['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]
    df_cycle_condition = df_cycle_condition[df_cycle_condition['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]
    df_unmodified = df_unmodified[df_unmodified['Run'].str.match(r"^\[\d{4},\d{4},\d{4}\]")]


    fig = go.Figure()

    y_val = 'Cost Error'
    x_val = 'Mean Abs Capacity Mix Error'

    fig.add_trace(go.Scatter(
        name='Surplus Tracking',
        y=df_w_surplus[y_val],
        x=df_w_surplus[x_val],
        text=df_w_surplus['Run'],
        marker_color=colour(6,.8),
        mode='markers',
        marker_size = 40,
        # line=dict(width=0),
        ))
    fig.add_trace(go.Scatter(
        name='Cycle Condition',
        y=df_cycle_condition[y_val],
        x=df_cycle_condition[x_val],
        text=df_cycle_condition['Run'],
        marker_color=colour(4,.8),
        mode='markers',
        marker_size = 40,
        # line=dict(width=0),
        ))
    fig.add_trace(go.Scatter(
        name='Unmodified',
        y=df_unmodified[y_val],
        x=df_unmodified[x_val],
        text=df_unmodified['Run'],
        marker_color=colour(5,.8),
        mode='markers',
        marker_size = 40,
        # line=dict(width=0),
        ))
    fig.update_yaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0.2)',
    
    )
    fig.update_xaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0)'
    )
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        yaxis = dict(
            title = dict(
                text=y_val,
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        xaxis = dict(
            title = dict(
                text=x_val,
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=.98,
            xanchor="center",
            x=0.5,
            bgcolor = 'rgba(255,255,255,1)',    
        ),
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        font=dict(
        family="Times New Roman",
        size=48,
        ),
    )
    
    # fig.write_html("simple_weather-year_ldes-model/export/results_variant_comparison.html", auto_open=True)

# function generates data for the 4th figure
def generate_data_fig4_scatter_of_timeseries_duration_curve_error(path_directory_three_year_cluster):


    result_n_3 = timeseries_error_in_dir('weather_year_clustering_LDES_study/data_tables/original_timeseries/time_varying_parameters.csv',path_directory_three_year_cluster,r"^results_years_\[\d{4},\d{4},\d{4}\]_weight_\[\d{1},\d{1},\d{1}\]*\.netcdf$",3)

    result_n_3.to_csv('simple_weather-year_ldes-model/export/timeseries_duration_curve_error.csv')

#  function plots the 4th figure, the timeseries duration curve error
def plot_fig4_scatter_of_timeseries_duration_curve_error(path_timeseries_error,path_cluster_error):
    df_timeseries_error = pd.read_csv(path_timeseries_error)
    df_cluster_error = pd.read_csv(path_cluster_error)
    df_timeseries_error = df_timeseries_error.drop_duplicates(subset=['Run'])
    df_cluster_error = df_cluster_error.drop_duplicates(subset=['Run'])

    # Join Results into single PD
    df_timeseries_error.set_index('Run')
    df_cluster_error.set_index('Run')
    df_combined = pd.merge(df_timeseries_error, df_cluster_error, how="left", on=["Run"])

    df_combined = df_combined[['Run','compound_error','Cost Error','Mean Abs Capacity Mix Error','Abs Compound Error','LDES Capacity Error']]
    df_combined['LDES Capacity Error'] = -1* df_combined['LDES Capacity Error']
    df_combined['Cost Error'] = -1* df_combined['Cost Error']
    df_combined['Years'] = df_combined['Run'].str[1:15]
    df_combined['Check'] = df_combined['Years'].apply(includes_system_defining_years)

    df_set_1 =df_combined[df_combined['Check'] == True]
    df_set_2 =df_combined[df_combined['Check']== False]

    fig = go.Figure()

    #PLOTLY has issues with scaling the colour bar correctly, have to produce artificial hidden datapoints. DOES NOT SHOW ON FINAL GRAPH!!
    

    fig.add_trace(go.Scatter(
            name='Artificial Data',
            y=[df_combined['compound_error'][0],df_combined['compound_error'][1]],
            x=[df_combined['LDES Capacity Error'][0],df_combined['LDES Capacity Error'][1]],
            text=['data to enforce colourbar','data to enforce colourbar'],
            mode='markers',
            marker=dict(
                color=[0,0.3],
                coloraxis='coloraxis',
                size=32,
                line=dict(
                    color=colour(0,.0),
                    width=4
                )),
            showlegend=False
        ))

    # REAL DATA

    fig.add_trace(go.Scatter(
        name='Three Year Clusters',
        y=df_combined['compound_error'],
        x=df_combined['LDES Capacity Error'],
        text=df_combined['Run'],
        mode='markers',
        marker=dict(
            color=df_combined['Mean Abs Capacity Mix Error'],
            coloraxis='coloraxis',
            size=32,
            line=dict(
                color=colour(0,.4),
                width=4
            )),
        customdata=np.stack(([df_combined['Mean Abs Capacity Mix Error']]),axis=-1),
        hovertemplate =
        f'<i>Ex Ante</i>: '+'%{y:.2%}'+
        f'<br><b>LDES</b>: '+'%{x:.2%}<br>'+
        f'<i>MACME</i>: '+'%{customdata[0]:.2%}<br>'+
        '<b>%{text}</b>',
        showlegend=False
    ))

    
    

    fig.update_yaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0.2)',
    title_standoff = 36,
    # tick0=-0.05, 
    # dtick=0.05
    )
    fig.update_xaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0.2)',
    title_standoff = 36,
    rangemode='tozero',
    # tick0=0, 
    # dtick=0.05
    )
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        # title=dict(text="Preoptimisation Duration Curve Error versus Postoptimisation Absolute Compound Error"),
        yaxis = dict(
            title = dict(
                text='Duration Curve Error (Ex-Ante)',
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        xaxis = dict(
            title = dict(
                text='LDES Capacity Error (Ex-Post)',
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=.98,
            xanchor="center",
            x=0.5,
            bgcolor = 'rgba(255,255,255,1)',    
        ),
        font=dict(
        family="Times New Roman",
        size=48,
        ),
        # boxmode='group',
        yaxis_tickformat=".0%",
        xaxis_tickformat=".0%",
        coloraxis = dict(
            colorbar = dict(
            title = dict(text='MACME<br>(Ex-Post)'),
            tickformat = '.0%',
            tickmode='array',
            tickvals = [0,.05,.1,.15,.2,.25,.3],
            ticktext = ['0%','5%','10%','15%','20%','25%',''],
            ),
            colorscale=plasma_discrete_scale,
            
            ),
    )

    fig.write_html(f"{figure_directory}/fig4.html", auto_open=True)
    fig.write_image(f"{figure_directory}/fig4.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig4.jpeg", width=1920, height=1920, scale=1)




    print('something')

plot_fig1_area_chart_of_SOC(path_ref_model, path_directory_single_year_cluster)
plot_fig2_bar_chart_of_capacity_mix(path_ref_model, path_directory_single_year_cluster)
generate_data_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_variants['unmodified']['two_year'],path_variants['unmodified']['three_year'],'unmodified')
# # plot_fig3_1_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_variants['unmodified']['two_year'],path_variants['unmodified']['three_year'],'unmodified')
plot_fig3_2_scatter_of_errors(path_ref_model,path_directory_single_year_cluster,path_variants['unmodified']['two_year'],path_variants['unmodified']['three_year'],'unmodified')
generate_data_fig4_scatter_of_timeseries_duration_curve_error('weather_year_clustering_LDES_study/results/three_year_cluster/unmodified')
plot_fig4_scatter_of_timeseries_duration_curve_error('simple_weather-year_ldes-model/export/timeseries_duration_curve_error.csv','simple_weather-year_ldes-model/export/data_cluster_box_plots_unmodified.csv')

