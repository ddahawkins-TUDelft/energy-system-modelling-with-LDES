import calliope
import datetime
import pandas as pd
import plotly.express as px

model = calliope.Model('simple_weather-year_ldes-model/model.yaml', scenario='fixed_min_capacities')
calliope.set_log_verbosity("INFO", include_solver_output=True)
model.build()
model.solve()
model.to_csv('simple_weather-year_ldes-model/results/results_'+str(int(datetime.datetime.now().timestamp()))+'.csv')
#model.backend.to_lp('simple_weather-year_ldes-model/results/lp files/lp_export_'+str(int(datetime.datetime.now().timestamp()))+'.lp')

lcoes = model.results.systemwide_levelised_cost.to_series().dropna()
lcoes

colors = model.inputs.color.to_series().to_dict()

#plotting flows



#plotting capacities

df_capacity = (
    model.results.flow_cap.where(model.results.techs != "demand_power")
    .sel(carriers="power")
    .to_series()
    #.where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow capacity (MW)")
    .mul(0.001) #convert MW to GW
    .rename(columns={'Flow capacity (MW)':'Flow Capacity (GW)'})
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_capacity,
    x="nodes",
    y="Flow Capacity (GW)",
    color="techs",
    color_discrete_map=colors,
)
fig.show()

# visualising flows

df_electricity = (
    (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
    .sel(carriers="power")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow in/out (MWh)")
    .rename(columns={'Flow in/out (MWh)':'Flow in/out (GWh)'})
    .mul(0.001) #convert kWh to GWh
    .reset_index()
)
df_electricity_demand = df_electricity[df_electricity.techs == "demand_power"]
df_electricity_other = df_electricity[df_electricity.techs != "demand_power"]

print(df_electricity.head())

node_order = df_electricity.nodes.unique()

fig = px.bar(
    df_electricity_other,
    x="timesteps",
    y="Flow in/out (GWh)",
    facet_row="nodes",
    color="techs",
    category_orders={"nodes": node_order},
    height=1000,
    color_discrete_map=colors,
)

showlegend = True
# we reverse the node order (`[::-1]`) because the rows are numbered from bottom to top.
for idx, node in enumerate(node_order[::-1]):
    demand_ = df_electricity_demand.loc[
        df_electricity_demand.nodes == node, "Flow in/out (GWh)"
    ]
    if not demand_.empty:
        fig.add_scatter(
            x=df_electricity_demand.loc[
                df_electricity_demand.nodes == node, "timesteps"
            ],
            y=-1 * demand_,
            row=idx + 1,
            col="all",
            marker_color="black",
            name="Demand",
            legendgroup="demand",
            showlegend=showlegend,
        )
        showlegend = False
fig.update_yaxes(matches=None)
fig.show()

# visualising SOC

df_storage = (
    (model.results.storage.fillna(0))
    .sel(techs="hydrogen_storage_system")
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

fig = px.bar(
    df_storage,
    x="timesteps",
    y="Storage (GWh)",
    facet_row="nodes",
    category_orders={"nodes": node_order},
    height=1000,
)

showlegend = True
# we reverse the node order (`[::-1]`) because the rows are numbered from bottom to top.
for idx, node in enumerate(node_order[::-1]):
    storage_val = df_storage.loc[
        df_storage.nodes == node, "Storage (GWh)"
    ]
    if not storage_val.empty:
        fig.add_scatter(
            x=df_storage.loc[
                df_storage.nodes == node, "timesteps"
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
fig.show()