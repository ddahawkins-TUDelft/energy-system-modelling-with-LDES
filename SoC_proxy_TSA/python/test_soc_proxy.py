from utility_functions.helper_SoC_proxy import generate_SoC_proxy
import utility_functions.helper_timeseries_tools as tt
from utility_functions.helper_plot_soc_comparison import figure_compare_SoC
# import timeseries using helper function that also filters over date range
df_reference = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv','2010-01-01','2010-12-31')

df_clustered,s_cluster_datetimes = tt.extrapolate_ts_from_cluster_map('SoC_proxy_TSA/cache/outputdata.csv','SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv')

capacity_weights = {
    'solar': 1,
    'onshore_wind': 1,
    'offshore_wind': 1
}

df_reference,threshold_SoC_reference = generate_SoC_proxy(df_reference,'demand_power',capacity_weights)
df_clustered,threshold_SoC_clustered = generate_SoC_proxy(df_clustered,'demand_power',capacity_weights)

#plot and compare
figure_compare_SoC(
    df_reference=df_reference,
    df_test=df_clustered,
    label_graph_1 = f"Reference, α={round(threshold_SoC_reference,4)}",
    label_graph_2 = f"Clustered, α={round(threshold_SoC_clustered,4)}"
    )