import calliope
import datetime
import pandas as pd
import plotly.express as px
import re
import os
import plotly.graph_objects as go

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

def extract_plan_results(pd_results, plan_year):
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

    result = (
                df_techs
                .join(df_year).set_index('techs')
                .join(df_flow_out_carrier.set_index('techs')).drop(['Flow Out Carrier','nodes'], axis=1).rename(columns={'carriers':'flow_out_carrier'})
                .join(df_flow_cap_power.set_index('techs')).drop(['nodes'], axis=1)
                .join(df_storage_cap_hydrogen.set_index('techs')).drop(['nodes'], axis=1)
                .join(df_flow_cap_hydrogen.set_index('techs')).drop(['nodes'], axis=1)
                .join(df_cost_investment.set_index('techs')).drop(['nodes','costs'], axis=1)
                .join(df_cost_investment_annualised.set_index('techs')).drop(['nodes','costs'], axis=1)
                .join(df_cost_investment_flow_cap.set_index('techs')).drop(['nodes','costs'], axis=1)
                .join(df_cost_investment_storage_cap.set_index('techs')).drop(['nodes','costs'], axis=1)
                .join(df_cost_operation_fixed.set_index('techs')).drop(['nodes','costs'], axis=1)
            )

    if pd_results.empty:
        pd_results = result
    else:
        pd_results = pd.concat([pd_results,result])
    
    return pd_results
         
#full horizon results


for i in range(2010,2019+1):
    print(f"Importing results for: {i}")
    pd_results = extract_plan_results(pd_results,i)
print(f"Importing results for: Full: 2010-2019") 
pd_results = extract_plan_results(pd_results,'Full: 2010-2019')

pd_results.to_csv('simple_weather-year_ldes-model/export/plan_results.csv')

df_hydrogen_caps = (pd_results
                               .where(pd_results['flow_out_carrier'] == pd_results['carriers'])
                               .where(pd_results['flow_out_carrier'] == 'hydrogen')
                               .dropna(how='all')
                               .rename(columns={"Flow Capacity (Hydrogen) (MW)": 'Flow Capacity (MW)'})
                            )
# df_hydrogen_caps = df_output_carrier_flow_caps[df_output_carrier_flow_caps['carriers'] == 'hydrogen']

df_power_caps = (pd_results
                               .where(pd_results['flow_out_carrier'] == pd_results['carriers'])
                               .where(pd_results['flow_out_carrier'] == 'power')
                               .dropna(how='all')
                               .rename(columns={"Flow Capacity (Power) (MW)": 'Flow Capacity (MW)'})
                            )

df_cap_out_flows = pd.concat([df_hydrogen_caps[['year','technologies','carriers',"Flow Capacity (MW)"]],df_power_caps[['year','technologies','carriers',"Flow Capacity (MW)"]]])
# print(df_cap_out_flows)

#TODO: FIXME: Datasets prepared above, they need to be imported into box plot below and grouped by carrier

fig = go.Figure()

fig.add_trace(go.Box(
    y=df_power_caps['Flow Capacity (MW)'].where(df_power_caps['year'] != 'Full: 2010-2019'),
    x=df_power_caps['technologies'],
    name='power',
    marker_color='#3D9970',
    boxpoints='all'
))

df_full_year = df_cap_out_flows.where(df_cap_out_flows['year'] == 'Full: 2010-2019')

fig.add_trace(go.Scatter(
    y=df_full_year['Flow Capacity (MW)'],
    x=df_full_year['technologies'],
    name='Reference (Full Horizon)',
    mode='markers',
    marker_color='black',
    marker_symbol ='diamond',
    # boxpoints='all'
))

fig.add_trace(go.Box(
    y=df_hydrogen_caps['Flow Capacity (MW)'],
    x=df_hydrogen_caps['technologies'],
    name='hydrogen',
    marker_color='#FF851B',
    boxpoints='all'
))

fig.update_layout(
    yaxis=dict(
        title=dict(
            text='Outflow Capacity (MW)')
    ),
    boxmode='group' # group together boxes of the different traces for each value of x
)

# fig = px.box(pd_results[['technologies',"Flow Capacity (MW)"]].dropna(), 
#              x="technologies", 
#              y="Flow Capacity (MW)",
#              points=False
#              )
# fig.show()
fig.write_html("simple_weather-year_ldes-model/export/result_caps.html", auto_open=True)

df_storage_results = pd_results.where(pd_results['year'] != 'Full: 2010-2019')
df_storage_results_full_year = pd_results[['technologies','Storage Capacity (MWh)']].where(pd_results['year'] == 'Full: 2010-2019').dropna()

fig = px.box(df_storage_results[['technologies','Storage Capacity (MWh)']].dropna(), 
             x="technologies", 
             y="Storage Capacity (MWh)",
             points='all'
             )

fig.add_trace(go.Scatter(
    y=df_storage_results_full_year['Storage Capacity (MWh)'],
    x=df_storage_results_full_year['technologies'],
    name='Reference (Full Horizon)',
    mode='markers',
    marker_color='black',
    marker_symbol ='diamond',
    # boxpoints='all'
))

fig.write_html("simple_weather-year_ldes-model/export/result_storage.html", auto_open=True)

# print(pd_results[['technologies','Storage Capacity (MWh)']].dropna().head())