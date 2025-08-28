import pandas as pd
import os
import json
import calliope
from utility_functions.helper_SoC_proxy import generate_SoC_proxy, standardised_profile_comparison
import utility_functions.helper_timeseries_tools as tt
from utility_functions.helper_plot_soc_comparison import  figure_compare_muliple_SoCs
from utility_functions.helper_model_config import standardised_model_config, clustered_model_config
from utility_functions.helper_normalise import normalise_by_method
import numpy as np
from utility_functions.helper_seasonal_sampling import monte_carlo_tsa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity



# #_______________________________________________________________________________________________
# # 
# #                                  Generate Reference
# #_______________________________________________________________________________________________


# #configure reference model, standard single-year model for 2010

date_lower_bound = '2010-01-01'
date_upper_bound = '2010-12-31'

# # script configuration
params = {
        'output_directory_name': 'correlation_soc_proxy_esom_error',
        'output_model_name': 'reference',
        'config_yaml_name': 'model',
        'horizon_start':  date_lower_bound,
        'horizon_end':  date_upper_bound,
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
        'path_to_cluster_csv': 'SoC_proxy_TSA/cache/outputdata.csv'
    }

# #construct model using standardised constructor
# print(f"  --- Configuring: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
# standard_model, filename_standard_model = standardised_model_config(params)

# #build and solve standard model
# print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")

# # auto_config calliope runtime 
# calliope.set_log_verbosity('ERROR', include_solver_output=params['calliope_full_log'][1]) 

# #build 
# standard_model.build()

# #solve
# print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
# standard_model.solve()

# #print results
# print(f"  ----- Results: Obj. Function: {standard_model.results.cost.sum().item():e}, Solve Time: {round(standard_model.results.timestamp_solve_complete - standard_model.results.timestamp_solve_start,1)}s")

# #auto-save, first checks if full directory tree exists (if not, creates it)
# output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir,filename_standard_model)
# standard_model.to_netcdf(output_path)
# print(f"  ----- Saved: Model saved to: {output_path}")

# #_______________________________________________________________________________________________
# # 
# #                                  Generate Test Models
# #_______________________________________________________________________________________________

#procedurally generate random test models
#strategy, monte-carlo esque random selection of representative days but ensuring that days are evenly distributed across seasons.

n_days = 40
n_samples = 100
year = 2010

# tsa_samples = {}

# clustering_source_data = 'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv'

# for i in range(n_samples):
    
#     assignment = monte_carlo_tsa(
#         clustering_source_data,
#         date_lower_bound=date_lower_bound,
#         date_upper_bound=date_upper_bound,
#         feature_cols=['demand_power', 'onshore_wind', 'offshore_wind', 'solar'],
#         n_rep_days=40,
#         seed=i
#     )

#     tsa_samples[i] = assignment
#     assignment_to_save = assignment[['timesteps', 'PeriodNum']]
#     assignment_to_save.to_csv(f"SoC_proxy_TSA/cache/cluster_maps/n_days_{n_days}_seed_{i}.csv", index=False)

# # #_______________________________________________________________________________________________
# # # 
# # #                                  Run the sample models
# # #_______________________________________________________________________________________________


# for i in range(n_samples):

#     ref = f"n_days_{n_days}_seed_{i}"
#     params['path_to_cluster_csv'] = f"SoC_proxy_TSA/cache/cluster_maps/{ref}.csv"

#     #configure clustered model
#     clustered_model, filename_clustered_model = clustered_model_config(params)

#     #build
#     print(f"  ---- Building: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
#     clustered_model.build()

#     #solve
#     print(f"  ---- Solving: Model Descr: {params['output_directory_name']}/{params['output_model_name']}, Time Horizon: {params['horizon_start']} until {params['horizon_end']} --- ")
#     clustered_model.solve()

#     #auto-save, first checks if full directory tree exists (if not, creates it)
#     output_dir = os.path.join("SoC_proxy_TSA", "results", params["output_directory_name"])
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir,ref+'.netcdf')
#     clustered_model.to_netcdf(output_path)
#     print(f"  ----- Saved: Model saved to: {output_path}")

# #_______________________________________________________________________________________________
# # 
# #                                  Pull LDES results
# #_______________________________________________________________________________________________


# results_df = pd.DataFrame(columns=["model_name", "run_time", "objective_function", "mean_absolute_capacity_error", "ldes_capacity_error"])

# m_reference = calliope.read_netcdf('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/standard_2010_2010_reference.netcdf')

# df_storage_caps = (   
#         (m_reference.results['storage_cap'].fillna(0))
#         .to_series()
#         .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('storage_cap')
#         .reset_index()
#     )

# df_energy_caps = (   
#         (m_reference.results['flow_cap'].fillna(0))
#         .to_series()
#         .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('flow_cap')
#         .reset_index()
#     ) 

# ldes_capacity_reference = df_storage_caps.loc[df_storage_caps['techs'] == 'h2_salt_cavern', 'storage_cap'].iloc[0] #storage cap
# capacities_reference = np.array(df_energy_caps.loc[df_energy_caps['carriers'] == 'power', 'flow_cap']) #array of power caps


# for i in range(n_samples):
#     ref = f"n_days_{n_days}_seed_{i}"
#     path_results = f"SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/{ref}.netcdf"

#     m = calliope.read_netcdf(path_results)
#     #extracting key results and saving

#     df_storage_caps = (   
#         (m.results['storage_cap'].fillna(0))
#         .to_series()
#         .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('storage_cap')
#         .reset_index()
#     )

#     df_energy_caps = (   
#         (m.results['flow_cap'].fillna(0))
#         .to_series()
#         .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('flow_cap')
#         .reset_index()
#     ) 

#     #compute the errors
#     ldes_capacity_error = (ldes_capacity_reference-df_storage_caps.loc[df_storage_caps['techs'] == 'h2_salt_cavern', 'storage_cap'].iloc[0])/ldes_capacity_reference
#     mean_abs_capacity_error = np.mean(np.abs((capacities_reference-np.array(df_energy_caps.loc[df_energy_caps['carriers'] == 'power', 'flow_cap']))/capacities_reference))

#     results_df.loc[len(results_df)] = {
#         "model_name": ref,
#         "run_time": m.results.timestamp_solve_complete - m.results.timestamp_solve_start,
#         "objective_function": m.results.cost.sum().item(),
#         "mean_absolute_capacity_error": mean_abs_capacity_error,
#         "ldes_capacity_error": ldes_capacity_error
#     }

# results_df.to_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/comparison_results.csv')


#_______________________________________________________________________________________________
# # 
# #                                  Compare Ex-ante SoC Proxies
# #_______________________________________________________________________________________________

# df_reference = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv','2010-01-01','2010-12-31')


# capacity_weights = {
#     'solar': 1,
#     'onshore_wind': 1,
#     'offshore_wind': 1
# }

# df_reference,threshold_SoC_reference = generate_SoC_proxy(df_reference,'demand_power',capacity_weights)
# df_reference = df_reference.set_index('timesteps')

# list_df_clustered = []
# list_labels = []

# results_df = pd.DataFrame(columns=["model_name", "rmse","rmse_normalised","alpha_error", "pearson_r","composite_error", "max_SoC_error", "max_SoC_datetime_error",'amplitude_error'])

# idx_best_rmse = 0
# best_rmse = 100000
# idx_best_composite = 0
# best_composite = 100000

# #calculate reference amplitude
# reference_peak_amplitude = df_reference['soc_proxy'].max()-df_reference['soc_proxy'].min()


# for i in range(n_samples):
#     print(f"Computing statistics for sample {i}")
#     df_clustered,s_cluster_datetimes = tt.extrapolate_ts_from_cluster_map(f"SoC_proxy_TSA/cache/cluster_maps/n_days_{n_days}_seed_{i}.csv",'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv')
#     df, alpha = generate_SoC_proxy(df_clustered,'demand_power',capacity_weights)
#     df = df.set_index('timesteps')
#     list_df_clustered.append(df)
#     list_labels.append(f"i={i}, α={round(alpha,4)}")

#     df_aligned = pd.concat([df_reference['soc_proxy'], df['soc_proxy']], axis=1).dropna()
#     df_aligned.columns = ['ref', 'test']

#     #calculate amplitude
#     clustered_peak_amplitude = df['soc_proxy'].max()-df['soc_proxy'].min()

#     #compute error statistics
#     e_rmse = np.sqrt((((df_aligned['ref']-df_aligned['test'])**2).mean()))
#     e_rmse_normalised = np.abs(e_rmse / np.mean(df_aligned['ref']))
#     r_value, p_value = pearsonr(df_aligned['ref'], df_aligned['test'])
#     e_composite = (1 - (r_value**2)) + e_rmse_normalised
#     e_soc_peak_value = (df_reference['soc_proxy'].max()-df['soc_proxy'].max())/df_reference['soc_proxy'].max()
#     e_soc_peak_datetime = (tt.hour_of_year_from_timestamp(df_reference['soc_proxy'].idxmax())-tt.hour_of_year_from_timestamp(df['soc_proxy'].idxmax()))/tt.hour_of_year_from_timestamp(df_reference['soc_proxy'].idxmax())
#     e_soc_amplitude = (reference_peak_amplitude-clustered_peak_amplitude)/reference_peak_amplitude

#     if e_rmse<best_rmse:
#         idx_best_rmse = i
#         best_rmse = e_rmse
    
#     if e_composite<best_composite:
#         idx_best_composite = i
#         best_composite = e_composite

#     #generate results output
#     results_df.loc[len(results_df)] = {
#         "model_name": f"n_days_{n_days}_seed_{i}",
#         "rmse": e_rmse,
#         "rmse_normalised": e_rmse_normalised,
#         'pearson_r': r_value,
#         'composite_error': e_composite,
#         "alpha_error": (threshold_SoC_reference-alpha)/threshold_SoC_reference,
#         "max_SoC_error": e_soc_peak_value,
#         "max_SoC_datetime_error": e_soc_peak_datetime,
#         "amplitude_error": e_soc_amplitude,
#     }

# results_df.to_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/soc_comparison_results.csv')

# print('Top RMSE performer:', idx_best_rmse, '. Top Composite Performer:',idx_best_composite)

# #plot and compare
# figure_compare_muliple_SoCs(
#     df_reference=df_reference,
#     list_of_test_dfs=list_df_clustered,
#     label_graph_1=f"Reference,α={round(threshold_SoC_reference,4)}",
#     list_of_test_labels=list_labels
# )

# #_______________________________________________________________________________________________
# # 
# #                                  Ex-ante vs Ex-post error
# #_______________________________________________________________________________________________

# #load results
# df_exante = pd.read_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/soc_comparison_results.csv')
# df_expost = pd.read_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/comparison_results.csv')
# #create an absolute rmse from rmse

# # df_exante['max_hour'] = tt.convert_to_hour_of_year(df_exante['max_datetime'])
# # df_exante['min_hour'] = tt.convert_to_hour_of_year(df_exante['min_datetime'])


# #df_exante: rmse, rmse_normalised, pearson_r, alpha_error, max_SoC_error, max_SoC_datetime_error, amplitude_error
# #df_expost: run_time,objective_function,mean_absolute_capacity_error,ldes_capacity_error


# def plotter(graph_list, df_x, df_y):
#     plt.figure(figsize=(10, 5))
#     x_var = graph_list[0]
#     x_data = df_x[x_var]
#     plt.xlabel(x_var)

#     for i in range(len(graph_list)-1):

        
#         y_var = graph_list[i+1]
        
#         y_data = df_y[y_var]

#         plt.scatter(x_data, y_data, label=f"{y_var} vs. {x_var}")

#         coeffs = np.polyfit(x_data, y_data, deg=1)
#         trendline = np.poly1d(coeffs)
#         plt.plot(x_data, trendline(x_data),  label='Trendline')

#     plt.grid(True)
#     plt.legend()
#     # plt.tight_layout()
#     plt.axhline(0, color='black', linewidth=1.5)  # y = 0 line
#     plt.axvline(0, color='black', linewidth=1.5)  # x = 0 line
#     plt.show()

        
# plotter(['ldes_capacity_error','max_SoC_datetime_error','amplitude_error','rmse_normalised'], df_expost, df_exante)

# #_______________________________________________________________________________________________
# # 
# #                                  Exploring Capacity Weights in SoC Proxy
# #_______________________________________________________________________________________________

# def plot_SoC_exante_expost(n_days, i,list_cw):

#     capacity_weights = {
#     'solar': 1,
#     'onshore_wind': 1,
#     'offshore_wind': 1
#     }


#     # capacity_weights = {
#     # 'solar': 1,
#     # 'onshore_wind': 1,
#     # 'offshore_wind': 1
#     # }

#     #reference model
#     m_reference = calliope.read_netcdf('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/standard_2010_2010_reference.netcdf')
#     df_ref_soc = (   
#         (m_reference.results['storage'].fillna(0))
#         .to_series()
#         # .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('storage')
#         .reset_index()
#     )
#     df_ref_cf = (   
#         (m_reference.results['capacity_factor'].fillna(0))
#         .to_series()
#         # .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('cf')
#         .reset_index()
#     )

#     df_ref_capacities = (   
#         (m_reference.results['flow_cap'].fillna(0))
#         .to_series()
#         # .where(lambda x: x != 0)
#         .dropna()
#         .to_frame('cf')
#         .reset_index()
#     )

#     #calculate a weighted mean capacity factor
#     renewable_fields = [str(key) for key in capacity_weights.keys()]
#     #first the demand power offset
#     #filtering
#     df_ref_cf = df_ref_cf[df_ref_cf['carriers']=='power']
#     df_ref_cf = df_ref_cf[df_ref_cf['techs'].isin(renewable_fields)]
#     df_ref_capacities = df_ref_capacities[df_ref_capacities['techs'].isin(renewable_fields)]
#     df_ref_capacities = df_ref_capacities[df_ref_capacities['carriers']=='power']
   
#     #group
#     mean_cf_by_tech = df_ref_cf.groupby('techs', as_index=False)['cf'].mean()
#     df_ref_capacities['cf'] = normalise_by_method(df_ref_capacities['cf'], method='sum')
#     df_ref_capacities = df_ref_capacities.set_index('techs')
#     mean_cf_by_tech = mean_cf_by_tech.set_index('techs')
#     weighted_mean_cf = 0
#     for tech in renewable_fields:
#         cf = mean_cf_by_tech.loc[tech, 'cf']
#         cap = df_ref_capacities.loc[tech, 'cf']
#         weighted_mean_cf += cf * cap

#     #extract the expost state of charge profile
#     df_ref_soc=df_ref_soc[df_ref_soc['techs'] == 'h2_salt_cavern']
#     df_ref_soc['timesteps']=tt.convert_to_hour_of_year(df_ref_soc['timesteps'])
#     df_ref_soc = df_ref_soc.set_index('timesteps')
#     df_ref_soc['storage'] = df_ref_soc['storage']-df_ref_soc['storage'].iloc[0] #baseline soc to 0 starting point
#     df_ref_soc['storage'] = df_ref_soc['storage'] /np.mean(df_ref_soc['storage'] ) #normalise
    

#     plt.figure(figsize=(10, 5))
#     x_data = df_ref_soc.index
#     y1_data= df_ref_soc['storage']
#     plt.xlabel('hour')
#     plt.plot(x_data,y1_data,label=f"ExPost, weight_mean_cf={weighted_mean_cf:.3e}")
    
    
#     plt.axhline(0, color='black', linewidth=1.5)  # y = 0 line
#     plt.axvline(0, color='black', linewidth=1.5)  # x = 0 line

#     for i in list_cw:
#         capacity_weights = {
#         'solar': i[0],
#         'onshore_wind': i[1],
#         'offshore_wind': i[2]
#         }

#         # reference soc proxy plot
#         df_ref_soc_proxy = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv','2010-01-01','2010-12-31')
#         df_ref_soc_proxy,threshold_SoC_reference = generate_SoC_proxy(df_ref_soc_proxy,'demand_power',capacity_weights)
#         df_ref_soc_proxy['timesteps']=tt.convert_to_hour_of_year(df_ref_soc_proxy['timesteps'])
#         df_ref_soc_proxy = df_ref_soc_proxy.set_index('timesteps')
#         # proxy_peak = np.max(df_ref_soc_proxy['soc_proxy'])
#         # ref_peak = np.max(df_ref_soc['storage'])
#         # proxy_demand_peak = np.max(df_ref_soc_proxy['demand_power'])/1000 #divide by 1000 because 
#         # threshold_SoC_reference = threshold_SoC_reference/proxy_peak*proxy_demand_peak#offset the alpha by the scale difference between SoC and SoC Proxy
#         df_ref_soc_proxy['soc_proxy'] = df_ref_soc_proxy['soc_proxy']/np.max(df_ref_soc_proxy['soc_proxy'])*np.max(df_ref_soc['storage']) #offset the SoC Proxy by the scale difference between SoC and SoC Proxy


#         y2_data= df_ref_soc_proxy['soc_proxy']
#         plt.plot(x_data,y2_data,label=f"s:{i[0]}, ons_w:{i[1]}, offs_w:{i[2]}, α={threshold_SoC_reference:.3e}")
    


#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()    

#     return df_ref_soc, df_ref_soc_proxy



# m_reference = calliope.read_netcdf('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/standard_2010_2010_reference.netcdf')
# df_ref_capacities = (   
#     (m_reference.results['flow_cap'].fillna(0))
#     .to_series()
#     # .where(lambda x: x != 0)
#     .dropna()
#     .to_frame('capacity')
#     .reset_index()
# )

# df_ref_capacities = df_ref_capacities[df_ref_capacities['techs'].isin(['solar','offshore_wind','onshore_wind'])]
# df_ref_capacities = df_ref_capacities[df_ref_capacities['carriers']=='power']
# df_ref_capacities = df_ref_capacities.set_index('techs')

# samples = [
#     # [1,1,1],
#     # [1,1,2],
#     # [1,2,1],
#     # [2,1,1],
#     # [5,2,3],
#     [round(df_ref_capacities.loc['solar','capacity']/1000,0),round(df_ref_capacities.loc['offshore_wind','capacity']/1000,0),round(df_ref_capacities.loc['onshore_wind','capacity']/1000,0)] #this case represents the outcome of an iterative approach where the tsa is evaluated and recomputed following a run
# ]

# plot_SoC_exante_expost(40,42,samples)

#what does this teach?
# capacity weights affect the profile, but apparently not the location of the peak
# all SoC profiles exhibit similar charge/decision decision timings to that of the ESOM output, but the scales are off which leads to the error.
# Perhaps amplitude is not the best determiner
# Peak location in time should factor into the process
# perhaps a statistical function that considers the ups and downs without scale?

# #_______________________________________________________________________________________________
# # 
# #                                  Evaluating ex-ante metrics
# #_______________________________________________________________________________________________

# # metrics to consider:
# # peak-lag (use difference in time of peak, or even normalised cross correlation)
# # pearson R (Provided the two signals are normalised, Pearson R captures the correlation between fluctuations i.e shape)
# # cosine similarity (captures shape correlation regardless of magnitudes, a backup measur for pearson)

# df_ref_soc, df_ref_soc_proxy = plot_SoC_exante_expost(40,42,[[2,1,1]]) 

# # [39,26,16]    Pearson R: 0.9439 Cosine similarity: 0.8244 First-Derivative correlation: 0.8125 Peak Timing Correlation: 0.9994
# # [1,1,1]       Pearson R: 0.4107 Cosine similarity: 0.1603 First-Derivative correlation: 0.7813 Peak Timing Correlation: 1.0043
# # [2,1,1]       Pearson R: 0.9430 Cosine similarity: 0.8265 First-Derivative correlation: 0.8158 Peak Timing Correlation: 0.9994

# soc_ref = df_ref_soc['storage']
# soc_proxy = df_ref_soc_proxy['soc_proxy']

# # compute statistical measures comparing the two signals, focusing on shape
# e_peak_time,e_pearson_r,e_cos_sim, e_fd_corr = standardised_profile_comparison(soc_ref,soc_proxy)

# print(f"Pearson R: {e_pearson_r:.4f}", f"Cosine similarity: {e_cos_sim:.4f}", f"First-Derivative correlation: {e_fd_corr:.4f}", f"Peak Timing Correlation: {e_peak_time:.4f}")

# _______________________________________________________________________________________________

#                                  Evaluating ex-ante metrics
# _______________________________________________________________________________________________

df_reference = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv','2010-01-01','2010-12-31')

capacity_weights = {
    'solar': 39,
    'onshore_wind': 26,
    'offshore_wind': 16
}

df_reference,threshold_SoC_reference,demand_scaling_factor = generate_SoC_proxy(df_reference,'demand_power',capacity_weights)
df_reference = df_reference.set_index('timesteps')

list_df_clustered = []
list_labels = []

results_df = pd.DataFrame(columns=["model_name","alpha_error", "standardised_pearson_r","fd_correlation", "cosine_similarity", "peak_time_error"])


for i in [93,90,80,77,99,82,86,5]:
    print(f"Computing statistics for sample {i}")
    df_clustered,s_cluster_datetimes = tt.extrapolate_ts_from_cluster_map(f"SoC_proxy_TSA/cache/cluster_maps/n_days_{n_days}_seed_{i}.csv",'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv')
    df, alpha, scalar = generate_SoC_proxy(df_clustered,'demand_power',capacity_weights)
    df = df.set_index('timesteps')

    list_df_clustered.append(df)
    list_labels.append(f"i={i}, α={alpha:.4e}")


    #identify the key timeseries
    y_true = df_reference['soc_proxy'] #"true" being abstract, proxy of original data
    y_pred = df['soc_proxy'] #proxy of clustered data

    

    #compute stastical comparison
    e_peak_time,e_pearson_r,e_cos_sim, e_fd_corr = standardised_profile_comparison(y_true,y_pred)
    alpha_correlation= (threshold_SoC_reference-alpha)/threshold_SoC_reference

    df_aligned = pd.concat([df_reference['soc_proxy'], df['soc_proxy']], axis=1).dropna()
    df_aligned.columns = ['ref', 'test']

    #generate results output
    results_df.loc[len(results_df)] = {
        "model_name": f"n_days_{n_days}_seed_{i}",
        "alpha_error": alpha_correlation,
        "standardised_pearson_r": e_pearson_r,
        "fd_correlation": e_fd_corr,
        "cosine_similarity": e_cos_sim,
        "peak_time_error": e_peak_time
    }

results_df.to_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/soc_comparison_results.csv')

#plot and compare
figure_compare_muliple_SoCs(
    df_reference=df_reference,
    list_of_test_dfs=list_df_clustered,
    label_graph_1=f"Reference,α={threshold_SoC_reference:.4e}",
    list_of_test_labels=list_labels
)

# #_______________________________________________________________________________________________
# # 
# #                                  Evaluating Performance
# #_______________________________________________________________________________________________

#load results
df_exante = pd.read_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/soc_comparison_results.csv')
df_expost = pd.read_csv('SoC_proxy_TSA/results/correlation_soc_proxy_esom_error/comparison_results.csv')



def plotter(graph_list, df_x, df_y):
    plt.figure(figsize=(10, 5))
    x_var = graph_list[0]
    x_data = df_x[x_var]
    plt.xlabel(x_var)

    for i in range(len(graph_list)-1):

        
        y_var = graph_list[i+1]
        
        y_data = df_y[y_var]

        plt.scatter(x_data, y_data, label=f"{y_var} vs. {x_var}")

        coeffs = np.polyfit(x_data, y_data, deg=1)
        trendline = np.poly1d(coeffs)
        plt.plot(x_data, trendline(x_data),  label='Trendline')

    plt.grid(True)
    plt.legend()
    # plt.tight_layout()
    plt.axhline(0, color='black', linewidth=1.5)  # y = 0 line
    plt.axvline(0, color='black', linewidth=1.5)  # x = 0 line
    plt.show()

        
plotter(['peak_time_error','ldes_capacity_error','mean_absolute_capacity_error'], df_exante, df_expost)

#Outcomes
# no clear correlations for randomly sampled compressed single year models. Will move onto multi-year models and/or a TSA implementation