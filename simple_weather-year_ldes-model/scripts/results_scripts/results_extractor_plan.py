import calliope
import datetime
import pandas as pd
import plotly.express as px
import re
import os
import plotly.graph_objects as go
from IPython.display import Image

# -----------
#    plan 
# -----------

# load op_model
# extract:
#    operate_results:
#       unmet_demand_hourly
#       storage_level_hourly
#       system_cost: perhaps this is bad, as it uses bigM which is made up
#    plan_results:
#       technology:
#       capacity:
#       cost_fixed:
#       cost_variable:


#common vars
save_folder = 'simple_weather-year_ldes-model/results/single_year_runs'

# -----------
#    PLAN RESULTS 
# -----------

model = {}

pd_results = pd.DataFrame()

def extract_plan_results(plan_year):
    import_file_path = ""

    # checks if input year is added, otherwise imports the full model results
    if isinstance(plan_year, int):
        filename_re = r"results_plan_"+str(plan_year)+r"_\d+\.netcdf$"
        for filename in os.listdir(save_folder):
            if re.search(filename_re, filename):
                # print(f"Identified an existing plan file: {filename}. This run will be imported as the plane model.")
                import_file_path = save_folder+'/'+filename

        if import_file_path:
            model = calliope.read_netcdf(import_file_path)
        else:
            raise Exception(f"Sorry, no results file found for plan year: {plan_year}")
    else:
        model = calliope.read_netcdf("simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf")

    
    
    df_flow_cap_power = (
                    (model.results.flow_cap.fillna(0))
                    .sel(carriers="power")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Flow Capacity (Power) (MW)")
                    .reset_index()
                )
    
    df_flow_cap_hydrogen = (
                    (model.results.flow_cap.fillna(0))
                    .sel(carriers="hydrogen")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Flow Capacity (Hydrogen) (MW)")
                    .reset_index()
                )
    
    df_storage_cap_hydrogen = (
                    (model.results.storage_cap.fillna(0))
                    # .sel(carriers="hydrogen")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Storage Capacity (MWh)")
                    .reset_index()
                )
    
    df_cost_investment = (
                    (model.results.cost_investment.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Cost Investment")
                    .reset_index()
                )

    df_cost_investment_annualised = (
                    (model.results.cost_investment_annualised.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Cost Investment Annualised")
                    .reset_index()
                )
    
    df_cost_investment_flow_cap = (
                    (model.results.cost_investment_flow_cap.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Cost Investment Flow Cap")
                    .reset_index()
                )
    
    df_cost_investment_storage_cap = (
                    (model.results.cost_investment_storage_cap.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Cost Investment Storage Cap")
                    .reset_index()
                )
    
    df_cost_operation_fixed = (
                    (model.results.cost_operation_fixed.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Cost Fixed Opex")
                    .reset_index()
                )

    df_techs = (
                    (model.results.techs.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("technologies")
                    .reset_index()
                    # .drop('techs')
                )

    df_year = pd.DataFrame([plan_year,plan_year,plan_year,plan_year,plan_year,plan_year,plan_year,plan_year,plan_year], columns=['year'])
   
    df_flow_out_carrier = (
                    (model.inputs.carrier_out)
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Flow Out Carrier")
                    .reset_index()
                    # .drop('techs')
                )
    
    results_dict = {
        'Model': str(plan_year),
        'Solar': df_flow_cap_power.loc[df_flow_cap_power['techs']=='solar']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Offshore Wind': df_flow_cap_power.loc[df_flow_cap_power['techs']=='offshore_wind']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Onshore Wind': df_flow_cap_power.loc[df_flow_cap_power['techs']=='onshore_wind']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Nuclear': df_flow_cap_power.loc[df_flow_cap_power['techs']=='nuclear']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Battery (Power)': df_flow_cap_power.loc[df_flow_cap_power['techs']=='battery']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Electrolyser': df_flow_cap_hydrogen.loc[df_flow_cap_hydrogen['techs']=='electrolyser']["Flow Capacity (Hydrogen) (MW)"].values[0]/1000,
        'LDES': df_storage_cap_hydrogen.loc[df_storage_cap_hydrogen['techs']=='h2_salt_cavern']["Storage Capacity (MWh)"].values[0]/1000,
        'Battery Storage': df_storage_cap_hydrogen.loc[df_storage_cap_hydrogen['techs']=='battery']["Storage Capacity (MWh)"].values[0]/1000,
        'Demand': df_flow_cap_power.loc[df_flow_cap_power['techs']=='demand_power']["Flow Capacity (Power) (MW)"].values[0]/1000,
        'Hydrogen CCGT': df_flow_cap_power.loc[df_flow_cap_power['techs']=='h2_elec_conversion']["Flow Capacity (Power) (MW)"].values[0]/1000,
        }
    
    df_result = pd.DataFrame(results_dict, index=[plan_year])
    
    return df_result
         
#full horizon results

df_results = extract_plan_results('Full Horizon')

for i in range(2010,2019+1):
    print(f"Importing results for: {i}")
    df_results = pd.concat([df_results,extract_plan_results(i)])

colour_1 = 'rgb(0, 51, 102)'
colour_2 = 'rgb(102, 0, 204)'
colour_3 = 'rgb(102, 153, 255)'
colour_4 = 'rgb(51, 204, 204)'
colour_5 = 'rgb(51, 51, 153)'
colour_6 = 'rgb(102, 153, 153)'
colour_7 = 'rgb(51, 102, 153)'



fig = go.Figure(data=[
              go.Bar(name='Nuclear', x=df_results['Model'], y=df_results['Nuclear'],marker_color=colour_1),
              go.Bar(name='Onshore Wind', x=df_results['Model'], y=df_results['Onshore Wind'], marker_color=colour_5),
              go.Bar(name='Offshore Wind', x=df_results['Model'], y=df_results['Offshore Wind'], marker_color=colour_2),
              go.Bar(name='Solar', x=df_results['Model'], y=df_results['Solar'], marker_color=colour_7),
              go.Bar(name='Hydrogen CCGT', x=df_results['Model'], y=df_results['Hydrogen CCGT'], marker_color=colour_3),
              go.Bar(name='Battery', x=df_results['Model'], y=df_results['Battery (Power)'], marker_color=colour_4),
              go.Bar(name='Electrolyser', x=df_results['Model'], y=df_results['Electrolyser'],marker_color=colour_6),
]
)

# fig = go.Figure(data=[
              
#               go.Bar(name='LDES', x=df_results['Model'], y=df_results['LDES']),
#               go.Bar(name='Battery', x=df_results['Model'], y=df_results['Battery Storage']),
# ]
# )
fig.update_layout(
        barmode='stack',
        plot_bgcolor='rgba(255, 255, 255, 0)',
        title=dict(text=f"Power capacity mix across single year models and the reference model."),
        yaxis = dict(
            title = dict(
                text='Capacity, GW',
                font=dict(size=24)
            )
        ),
        xaxis = dict(
            title = dict(
                text='Model',
                font=dict(size=24)
            )
        ),
        legend=dict(
            x=0.9,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        # yaxis_tickformat=".2f",
    )

# fig.update_yaxes(matches=None)
fig.update_xaxes(
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
# fig.write_html("simple_weather-year_ldes-model/export/result_single_year_caps.html", auto_open=True)
fig.write_image("simple_weather-year_ldes-model/export/result_single_year_caps.png", format="png", width=2000, height=2000, scale=1)


# print(pd_results[['technologies','Storage Capacity (MWh)']].dropna().head())