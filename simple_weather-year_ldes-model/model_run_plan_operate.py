import calliope
import datetime
import pandas as pd
import plotly.express as px


#functions
def visualise_SOC(model):
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
    fig.update_yaxes(matches=None)
    fig.write_html("simple_weather-year_ldes-model/results/result_storage.html", auto_open=True)

#Run Parameters
plan_year = 2015
operate_year = 2016
number_years = 1

#results save path
run_id =str(int(datetime.datetime.now().timestamp()))
plan_save_path= 'simple_weather-year_ldes-model/results/single_year_runs/results_plan_'+str(plan_year)+'_'+run_id+'.netcdf'
operate_save_path= 'simple_weather-year_ldes-model/results/single_year_runs/results_operate_'+str(operate_year)+'_plan_'+str(plan_year)+'_'+run_id+'.netcdf'


#generate calliope config variables
plan_start_date = str(plan_year)+'-01-01'
plan_end_date = str(plan_year)+'-01-31' #TODO: set to December again
operate_start_date = str(plan_year)+'-01-01' #TODO: set to December again
operate_end_date = str(plan_year)+'-01-31'

# run the build model for the build year
# calliope.set_log_verbosity("INFO", include_solver_output=True)

model = calliope.Model(
    'simple_weather-year_ldes-model/model.yaml',
    scenario='single_year_runs_plan',
    override_dict={'config.init.time_subset': [str(plan_start_date), str(plan_end_date)], 'techs.h2_salt_cavern.number_year_cycles': number_years}
)

model.build()
model.solve()
model.to_netcdf(plan_save_path)



# run the operate model for the operate year

op_model = model

operate_mode_dict = {'config.build.ensure_feasibility': True,'config.build.mode': 'operate', 'config.build.operate_use_cap_results': True,'config.build.operate_horizon': '48h','config.build.operate_window': '24h','config.init.time_subset': [str(operate_start_date), str(operate_end_date)], 'techs.h2_salt_cavern.number_year_cycles': number_years}
print(operate_mode_dict)

op_model.build(operate_mode_dict)
print(op_model.inputs)


op_model.solve(force=True)
op_model.to_netcdf(operate_save_path)



# model = calliope.read_netcdf(plan_save_path)

# visualise_SOC(model)
# visualise_SOC(op_model)

#visualise results

# visualising SOC

