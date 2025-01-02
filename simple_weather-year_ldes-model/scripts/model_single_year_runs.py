import calliope
import datetime
import pandas as pd
import plotly.express as px
import yaml
import re
import os


#functions
#function for generating plotly graphs of run results
def visualise_SOC(model, run_type):
    df_storage = (
        (model.results.storage.fillna(0))
        # .sel(techs="hydrogen_storage_system")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Storage (MWh)")
        .rename(columns={'Storage (MWh)':'Storage (GWh)'})
        .mul(0.001) #convert kWh to GWh
        .reset_index()
    )

    print(df_storage.head())

    node_order = df_storage.nodes.unique()

    df_storage_hss = df_storage[df_storage.techs == "h2_salt_cavern"]
    df_storage_batt = df_storage[df_storage.techs == "battery"]

    print("Battery Storage Capacity: "+str(df_storage_batt["Storage (GWh)"].max())+ "GWh")
    print("Hydrogen Storage Capacity: "+str(df_storage_hss["Storage (GWh)"].max())+ "GWh")

    fig = px.area(
        df_storage_hss,
        x="timesteps",
        y="Storage (GWh)",
        facet_row="nodes",
        category_orders={"nodes": node_order},
        height=1000,
    )

    showlegend = True
    # we reverse the node order (`[::-1]`) because the rows are numbered from bottom to top.
    for idx, node in enumerate(node_order[::-1]):
        storage_val = df_storage_batt.loc[
            df_storage_batt.nodes == node, "Storage (GWh)"
        ]
        if not storage_val.empty:
            fig.add_scatter(
                x=df_storage_batt.loc[
                    df_storage_batt.nodes == node, "timesteps"
                ],
                y=1 * storage_val,
                row=idx + 1,
                col="all",
                marker_color="black",
                name="Storage",
                legendgroup="storage",
                showlegend=showlegend,
            )
            showlegend = False
    
    if run_type == 'operate':

        df_unmet_demand = (
            (model.results.unmet_demand.fillna(0))
            # .sel(techs="hydrogen_storage_system")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Unmet Demand (MWh)")
            .reset_index()
        )

        for idx, node in enumerate(node_order[::-1]):
            unmet_val = df_unmet_demand.loc[
                df_unmet_demand.nodes == node, "Unmet Demand (MWh)"
            ]
            if not unmet_val.empty:
                fig.add_scatter(
                    x=df_unmet_demand.loc[
                        df_unmet_demand.nodes == node, "timesteps"
                    ],
                    y=1 * unmet_val,
                    row=idx + 1,
                    col="all",
                    marker_color="red",
                    name="Unmet Demand",
                    legendgroup="Demand",
                    showlegend=showlegend,
                    mode='markers'
                )
                showlegend = False
            
    fig.update_yaxes(matches=None)
    fig.write_html("simple_weather-year_ldes-model/results/result_storage"+str(run_type)+".html")

#function for printing run results to terminal
def print_capacities(model):

    df_storage_cap = (
            (model.results.storage_cap.fillna(0))
            # .sel(techs="hydrogen_storage_system")
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Storage (MWh)")
            .reset_index()
        )

    df_capacity = (
        model.results.flow_cap
        .where(model.results.techs != "demand_power")
        # .sel(carriers="power")
        .to_series()
        #.where(lambda x: x != 0)
        .dropna()
        .to_frame("Flow capacity (MW)")
        .reset_index()
    )

    # df_capacity=df_capacity.sort_values(by=['Flow capacity (MW)'],ascending=True)
    print('    ')
    print(df_storage_cap)
    print('    ')
    print(df_capacity)

#function that creates and runs a plan year which feeds into an operate year
def generate_runs(plan_year, operate_year, number_years):

#results save path
    run_id =str(int(datetime.datetime.now().timestamp()))
    save_folder = 'simple_weather-year_ldes-model/results/single_year_runs'
    plan_save_path= save_folder+'/results_plan_'+str(plan_year)+'_'+run_id+'.netcdf'
    operate_save_path= save_folder+'/results_operate_'+str(operate_year)+'_plan_'+str(plan_year)+'_'+run_id+'.netcdf'


    #generate calliope config variables
    plan_start_date = str(plan_year)+'-01-01'
    plan_end_date = str(plan_year)+'-12-31' #TODO: set to December again
    operate_start_date = str(operate_year)+'-01-01' #TODO: set to December again
    operate_end_date = str(operate_year)+'-12-31'
    days_for_foresight = (datetime.date(operate_year,12,31)-datetime.date(operate_year,1,1)).days
    print('Operate Foresight Horizon set to: '+str(days_for_foresight))
    # run the build model for the build year
    # calliope.set_log_verbosity("INFO", include_solver_output=True)
    calliope.set_log_verbosity("INFO", include_solver_output=True)

    import_file_path = ""

    filename_re = f"results_plan_{plan_year}_\d+\.netcdf$"
    for filename in os.listdir(save_folder):
        if re.search(filename_re, filename):
            print(f"Identified an existing plan file: {filename}. This run will be imported as the plane model.")
            import_file_path = save_folder+'/'+filename
    
    if import_file_path:
        model = calliope.read_netcdf(import_file_path)
    else:
        model = calliope.Model(
        'simple_weather-year_ldes-model/model.yaml',
        scenario='single_year_runs_plan',
        override_dict={'config.init.time_subset': [str(plan_start_date), str(plan_end_date)], 'techs.h2_salt_cavern.number_year_cycles': number_years}
        )

        model.build()
        model.solve()
        model.to_netcdf(plan_save_path)



    # model = calliope.read_netcdf('simple_weather-year_ldes-model/results/single_year_runs/results_plan_2016_1735039132.netcdf')
    # op_model = calliope.read_netcdf('simple_weather-year_ldes-model/results/single_year_runs/results_operate_2016_plan_2016_1735039132.netcdf')

    # print_capacities(model)
    # print_capacities(op_model)

    #dictionaries for assembling op model constraints
    techs_flow_dict = {}
    techs_storage_dict = {}

    # REDUNDANT USED FOR DEBUGGING
    # model = calliope.read_netcdf('simple_weather-year_ldes-model/results/single_year_runs/results_plan_2015_1734782728.netcdf')

    df_storage_cap = (
                (model.results.storage_cap.fillna(0))
                # .sel(techs="hydrogen_storage_system")
                .to_series()
                .where(lambda x: x != 0)
                .dropna()
                .to_frame("Storage (MWh)")
                .reset_index()
            )

    df_capacity = (
            model.results.flow_cap
            .where(model.results.techs != "demand_power")
            # .sel(carriers="power")
            .to_series()
            #.where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow capacity (MW)")
            .reset_index()
            .sort_values(by=['carriers'],ascending=False)
        )

    # -----------------------------------------------------------------------------------
    #      Process Carrier Specific flow capacities
    # -----------------------------------------------------------------------------------

    series_techs = df_capacity.groupby('techs').apply(lambda x: x['techs'].unique())
    series_flow_caps = df_capacity.groupby('techs').apply(lambda x: x['Flow capacity (MW)'].unique())
    # df_temp_renamed = df_temp.rename('techs','tech')
    # print(series_capacity)
    # print(series_capacity['battery'])

    min_max_diff_factor = 1.00001 #factor produces a real but marginal offset between the min and max values to avoid infeasibility errors during the optimisation

    for tech in series_techs:
        # temp_string = tech[0] + ": " + str(series_flow_caps[tech].iloc[0])
        
        val = series_flow_caps[tech[0]].flatten().tolist()
        if len(val) > 1:
            techs_flow_dict[f"techs.{tech[0]}.flow_cap_max.data"] = [i*(min_max_diff_factor) for i in val] #marginal offset to provide buffer such that the algorithm does not find the problem infeasible.
            techs_flow_dict[f"techs.{tech[0]}.flow_cap_min.data"] = val
            # temp_str = {f"techs.{tech[0]}.flow_cap_max.data": val} #for debug
        else:
            techs_flow_dict[f"techs.{tech[0]}.flow_cap_max"] = val[0]*min_max_diff_factor
            techs_flow_dict[f"techs.{tech[0]}.flow_cap_min"] = val[0]
            # temp_str = {f"techs.{tech[0]}.flow_cap_max": val[0]} #for debug
        
    # print(techs_flow_dict)
        
        # print(tech +": " + series_flow_caps[tech])

    # for index, row in df_capacity.iterrows():
    #     # dict_str = 'nodes.'+row['nodes']+'.techs.'+row['techs']+'.flow_cap'
    #     # techs_flow_dict[dict_str+'_max'] = row['Flow capacity (MW)']
    #     # techs_flow_dict[dict_str+'_min'] = row['Flow capacity (MW)']

    #     dict_str = f"techs.{row['techs']}.flow_cap_max.data"
    #     techs_flow_dict[dict_str] = row['Flow capacity (MW)']

    # -----------------------------------------------------------------------------------
    #      Process Storage Capacities
    # -----------------------------------------------------------------------------------   

    for index, row in df_storage_cap.iterrows():
        dict_str = 'nodes.'+row['nodes']+'.techs.'+row['techs']+'.storage_cap'
        techs_storage_dict[dict_str+'_max'] = row['Storage (MWh)']
        techs_storage_dict[dict_str+'_min'] = row['Storage (MWh)']
    # # run the operate model for the operate year

    override_dictionary = {'config.init.time_subset': [str(operate_start_date), str(operate_end_date)], 'techs.h2_salt_cavern.number_year_cycles': number_years}
    override_dictionary.update(techs_storage_dict)
    override_dictionary.update(techs_flow_dict)
    # print(override_dictionary)

    op_model = calliope.Model(
        'simple_weather-year_ldes-model/model_test.yaml',
        scenario='single_year_runs_operate_perfect_foresight',
        override_dict=override_dictionary
    )

    op_model.build()
    op_model.solve()
    op_model.to_netcdf(operate_save_path)

    visualise_SOC(model, 'plan')
    visualise_SOC(op_model, 'operate')

    print('-----------------------------------------------------------')
    print(f"Completed batch runs for plan year {plan_year}, and dispatch year {operate_year}. Run Id: {run_id}")
    print('-----------------------------------------------------------')

#Generate Runs
plan_year = 2019
operate_year = 2019

for i in range(2013,2019+1): #TODO: already run for 2010-2011
    for j in range(2010,2019+1):
        if i != j:
            generate_runs(i,j,1)

