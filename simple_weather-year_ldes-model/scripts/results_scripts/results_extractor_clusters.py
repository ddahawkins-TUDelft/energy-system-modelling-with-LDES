import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px
import re
import os
import plotly.graph_objects as go

# Plan
# function takes a directory
# load each model (inc. Full Model)
# extract: Flow Capacities
# extract: Storage Capacities
# extract: Costs (Cost, Cost Investment, Cost Investment Annualised, Cost Op Fixed, Cost Op Variable)

target_dir = 'simple_weather-year_ldes-model/results/two_year_cluster_runs'
pd_results = pd.DataFrame()


def cluster_results(path, result_category):
        
    full_model = calliope.read_netcdf('simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf')
    df_result = model_results_extractor(full_model, result_category)
    df_result['source_model'] = 'reference_full_horizon_2010_2019'

    #search through the provided directory and identify all netcdf models 
    filename_re = r"^results_plan_cluster.*\.netcdf$"
    for filename in os.listdir(path):
        if re.search(filename_re, filename):
            import_file_path = path+'/'+filename
            print('Extracting results from: ',filename)
            #load the model
            model = calliope.read_netcdf(import_file_path)
            #extract the relevant category of data
            df_extract = model_results_extractor(model, result_category)
            #add column reference to source model
            df_extract['source_model'] = filename
            #append to full model array for export
            df_result = pd.concat([df_result,df_extract])
    print(df_result.head())
    df_result.to_csv(f"{path}/export/cluster_results_{result_category}.csv")

def model_results_extractor(model, target):

    df_result  =  (   
        (model.results[target].fillna(0))
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Value")
        .reset_index()
    )


    # df_flow_cap = (   
    #     (model.results.flow_cap.fillna(0))
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Flow Capacity (Hydrogen) (MW)")
    #     .reset_index()
    # )

    # df_storage_cap = (
    #     (model.results.storage_cap.fillna(0))
    #     # .sel(carriers="hydrogen")
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Storage Capacity (MWh)")
    #     .reset_index()
    # )

    # df_cost_annualised = (
    #     (model.results.cost.fillna(0))
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Cost Investment")
    #     .reset_index()
    # )

    # df_cost_investment_annualised = (
    #     (model.results.cost_investment_annualised.fillna(0))
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Cost Investment Annualised")
    #     .reset_index()
    # )

    # df_cost_annual_operation_fixed = (
    #     (model.results.cost_operation_fixed.fillna(0))
    #     .to_series()
    #     .where(lambda x: x != 0)
    #     .dropna()
    #     .to_frame("Cost Fixed Opex")
    #     .reset_index()
    # )

    # output = {
    #     'df_flow_cap' : df_flow_cap,
    #     'df_storage_cap' : df_storage_cap,
    #     'df_cost_annualised' : df_cost_annualised,
    #     'df_cost_investment_annualised' : df_cost_investment_annualised,
    #     'df_cost_annual_operation_fixed' : df_cost_annual_operation_fixed,
    # }

    output = df_result

    return output

cluster_results(target_dir,'flow_cap')
