import calliope
import pandas as pd
import plotly.graph_objects as go
import os

# Config
variant = 'with_surplus_tracking' #with_intracluster_cycle_condition #with_surplus_tracking #two cluster conditions were tested, here can switch between them
figure_directory = 'weather_year_clustering_LDES_study/results/figures'

# function based colour palette, takes id and opacity factor
def colour(id: int, opacity: float = 1):
    colours = [
        '0,0,0', #0 black
        '88, 139, 139', #1 blue_green
        '101, 107, 157', #2 pale_red_blue
        '255, 213, 194', #3 peach
        '242, 143, 59', #4 orange
        '200, 85, 61', #5 burnt orange
        '45, 48, 71', #6 dark_red_blue
        '147, 183, 190', #7 pale_blue_green
        '255,255,255', #8 white
    ]

    return f'rgba({colours[id]},{opacity})'

#paths to ref model and cluster model directories
path_ref_model = 'weather_year_clustering_LDES_study/results/reference/results_2010_to_2019.netcdf' #TODO: Update with latest run
path_directory_single_year_cluster = 'weather_year_clustering_LDES_study/results/single_year_cluster'
path_variants = {
    'umodified': {
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
        fillcolor=colour(6,0.8),
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
    # fig.write_html("simple_weather-year_ldes-model/export/result_storage_full_vs_singles.html", auto_open=True)
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

    # fig.write_html("simple_weather-year_ldes-model/export/result_storage_full_vs_singles.html", auto_open=True)
    fig.write_image(f"{figure_directory}/fig2.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig2.jpeg", width=1920, height=1920, scale=1)

# function plots the 3rd figure, exploring model error against reference
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

def plot_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster, variant):
    print('Plotting Fig 3')

    generate_data_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster,variant)
    df = pd.read_csv(f'simple_weather-year_ldes-model/export/data_cluster_box_plots_{variant}.csv')
    df.drop_duplicates(subset=['Run'], inplace=True)
    df['Cost Error'] = -1* df['Cost Error']
    df['LDES Capacity Error'] = -1* df['LDES Capacity Error']
    #separate and group Design Error and Cost Error
    str_cost = 'Cost Error'
    str_abs_design = 'Mean Abs Capacity Mix Error'
    # str_design = 'Mean Capacity Mix Error'
    str_compound_err = 'Abs Compound Error'
    ldes_err = 'LDES Capacity Error'
    df_cost_design_unpivoted = df[['Run',str_cost,str_abs_design,ldes_err,str_compound_err]]
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
        marker_color=colour(4,1),
        # line=dict(width=0),
        ))
    fig.add_trace(go.Box(
        name='Two Weather Years',
        y=result_n_2['Error'],
        x=result_n_2['Parameter'],
        text=result_n_2['Run'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=colour(5,1),
        # line=dict(width=0),
        ))
    fig.add_trace(go.Box(
        name='Three Weather Years',
        y=result_n_3['Error'],
        x=result_n_3['Parameter'],
        text=result_n_3['Run'],
        boxpoints='all', # can also be outliers, or suspectedoutliers, or False
        jitter=0.3, # add some jitter for a better separation between points
        marker_color=colour(6,1),
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
                text='Error',
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        boxmode='group',
        yaxis_tickformat=".2%",
        font=dict(
        family="Times New Roman",
        size=48,
        ),
    )

    fig.write_image(f"{figure_directory}/fig3_{variant}.svg", width=1920, height=1920, scale=1)
    fig.write_image(f"{figure_directory}/fig3_{variant}.jpeg", width=1920, height=1920, scale=1)

# function plots the 4th figure, timeseries duration curve error across 3-yr clusters
def plot_fig4_timeseries_duration_curve_error(path_ref_model,path_directory_three_year_cluster):
    print('Plotting Fig 4')


# plot_fig1_area_chart_of_SOC(path_ref_model, path_directory_single_year_cluster)
# plot_fig2_bar_chart_of_capacity_mix(path_ref_model, path_directory_single_year_cluster)
# generate_data_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_directory_two_year_cluster,path_directory_three_year_cluster)
plot_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_variants['cycle_condition']['two_year'],path_variants['cycle_condition']['three_year'],'cycle_condition')
plot_fig3_box_plot_of_errors(path_ref_model,path_directory_single_year_cluster,path_variants['with_surplus']['two_year'],path_variants['with_surplus']['three_year'],'with_surplus')
# plot_fig4_timeseries_duration_curve_error(path_ref_model,path_directory_three_year_cluster)