import pandas as pd
import calliope 
import numpy as np
from scipy.spatial.distance import cdist
import pyomo.environ as pyo
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
import utility_functions.helper_timeseries_tools as tt
import matplotlib
matplotlib.use('TkAgg') #avoids the annoying Qt errors on windows
import matplotlib.pyplot as plt
from utility_functions.helper_model_config import clustered_model_config
import os
from sklearn.preprocessing import StandardScaler
from utility_functions.helper_compare_models import compare_models
import time
import multiprocessing
max_threads = multiprocessing.cpu_count()

def extract_timeseries_from_calliope(model: calliope.Model):
        #extract timeseries from calliope model
    raw_data = (
    model.inputs[[
        k for k, v in model.inputs.data_vars.items()
        if "timesteps" in v.dims and len(v.dims) > 1
    ]]
    .to_dataframe()
    .stack()
    .unstack("timesteps")
    .T
    )
    

    raw_data.columns = [col[1] if isinstance(col, tuple) else col for col in raw_data.columns]

    # #apply the soc proxy for input into TSAM clustering
    # raw_data, capacity_factors, nominal_capacities = generate_soc_proxy(
    #         df=raw_data,
    #         demand_field='demand_power',
    #         renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
    #         dispatchable_techs=proxy_parameters['dispatchable_techs'],
    #         storage_process_losses=proxy_parameters['storage_process_losses'],
    #         soc_decomposition = proxy_parameters['soc_decomposition'],
    # )

    # #now drop the capacity factor, general surplus, dynamic soc, and SDES fields, instead retaining only the LDES Surplus field which serves as the static input: SoC stresses
    # raw_data.drop(columns=['mean_capacity_factor','surplus','surplus_SDES', 'soc_proxy_LDES', 'soc_proxy_SDES'], inplace=True)
    # raw_data.rename(columns={'surplus_LDES': 'soc_stresses'}, inplace=True)

    return raw_data

def generate_daily_feature_matrix(df: pd.DataFrame):
    """
    Restructure hourly time series data into a daily feature matrix where each row
    corresponds to a single day, and columns represent 24-hour profiles for each variable.

    Parameters:
    ----------
    df : pd.DataFrame
        Hourly time series with datetime index and one column per variable
        (e.g., demand_power, solar, onshore_wind, offshore_wind)

    Returns:
    -------
    pd.DataFrame
        Daily feature matrix with shape (n_days, n_features),
        where each row contains concatenated 24-hour profiles for each variable.
        Columns are named like 'demand_power_h00', ..., 'solar_h23'.
    """
    
    # Sanity check: ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    # Get the list of variables (columns)
    variables = df.columns.tolist()

    # Create a container for daily data
    daily_profiles = []

    # Group by day
    for day, group in df.groupby(df.index.date):
        if len(group) != 24:
            # Skip incomplete days (edge cases at start/end of dataset)
            continue
        day_vector = []
        for var in variables:
            # Add the full 24-hour profile for this variable
            day_vector.extend(group[var].values.tolist())
        daily_profiles.append(day_vector)

    # Build column names
    col_names = []
    for var in variables:
        for h in range(24):
            col_names.append(f"{var}_h{h:02d}")

    # Build DataFrame
    daily_df = pd.DataFrame(daily_profiles, columns=col_names)

    # Optional: Add date index (not required for MILP TSA)
    valid_days = [day for day, group in df.groupby(df.index.date) if len(group) == 24]
    daily_df.index = pd.to_datetime(valid_days)

    return daily_df

def aggregate_to_daily_features(df: pd.DataFrame, agg_func: str = "mean") -> pd.DataFrame:
    """
    Aggregate an hourly timeseries DataFrame into daily scalar features for TSA.

    Parameters:
    ----------
    df : pd.DataFrame
        Hourly timeseries with datetime index and multiple variable columns.

    agg_func : str
        Aggregation function to use per variable ('mean', 'sum', 'std', etc.).

    Returns:
    -------
    pd.DataFrame
        One row per day, with one aggregated value per variable.
        Columns are automatically inferred from the input.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    # Select only numeric columns (e.g., drop metadata if present)
    df_numeric = df.select_dtypes(include=[np.number])

    # Group by day and apply aggregation
    daily_df = df_numeric.resample("D").agg(agg_func)

    return daily_df

def compute_distance_matrix(feature_df: pd.DataFrame, metric: str = "euclidean"):
    """
    Compute the pairwise distance matrix between all daily feature vectors.

    Parameters:
    ----------
    feature_df : pd.DataFrame
        Daily feature matrix where each row corresponds to one day
        and columns are 24-hour profiles for each variable.

    metric : str
        Distance metric to use (any supported by scipy.spatial.distance.cdist).
        E.g., 'euclidean', 'manhattan', 'cosine'

    Returns:
    -------
    np.ndarray
        A 2D array of shape (n_days, n_days) representing pairwise distances.
    """

    X = feature_df.values  # shape: (n_days, n_features)
    distance_matrix = cdist(X, X, metric=metric)
    return distance_matrix

def compute_proxy_distance_matrix(feature_df: pd.DataFrame, proxy_parameters: dict, matrix_weights: list = [1,1,1], metric: str = "euclidean"):
   
    source_cols = [col for col in feature_df.columns if any(col.startswith(prefix) for prefix in ['solar', 'onshore_wind', 'offshore_wind'])]
    demand_cols = [col for col in feature_df.columns if col.startswith('demand_power')]
    surplus_cols = [col for col in feature_df.columns if col.startswith('surplus_LDES')]

    renewables = feature_df[source_cols].copy()
    demand = feature_df[demand_cols].copy()
    surplus = feature_df[surplus_cols].copy()

    # Scale each group
    scaler_renew = StandardScaler()
    scaler_surplus = StandardScaler()
    scaler_demand = StandardScaler()

    X_renew_scaled = scaler_renew.fit_transform(renewables)
    X_surplus_scaled = scaler_surplus.fit_transform(surplus)
    X_demand_scaled = scaler_demand.fit_transform(demand)

    #rescales the signals based on number of fields i.e. downscale renewables because there are multiple techs within that signal
    X_renew_scaled = scaler_renew.fit_transform(renewables) / np.sqrt(X_renew_scaled.shape[1])
    X_demand_scaled = scaler_demand.fit_transform(demand) / np.sqrt(X_demand_scaled.shape[1])
    X_surplus_scaled = scaler_surplus.fit_transform(surplus) / np.sqrt(X_surplus_scaled.shape[1])


    # Weight them (optional)
    weight_sum = np.sum(matrix_weights)
    weight_renew = matrix_weights[0]/weight_sum
    weight_surplus = matrix_weights[1]/weight_sum  #TODO: results may have been better when this was 0.4 and the others 0.3. Will try.
    weight_demand = matrix_weights[2]/weight_sum

    X_combined = np.hstack([
        weight_renew * X_renew_scaled,
        weight_surplus * X_surplus_scaled,
        weight_demand * X_demand_scaled
    ])

    # Print norms of the signals as a guide on their strength of influence on the distancing matrix
    # print("Feature matrix shape:", X_combined.shape)
    # print("Renewable contribution (norm):", np.linalg.norm(weight_renew * X_renew_scaled))
    # print("Demand contribution (norm):", np.linalg.norm(weight_demand * X_demand_scaled))
    # print("Surplus contribution (norm):", np.linalg.norm(weight_surplus * X_surplus_scaled))

    # Compute distance matrix
    distance_matrix = cdist(X_combined, X_combined, metric=metric)

    return distance_matrix 

def solve_standard_milp_tsa(
    distance_matrix: np.ndarray,
    k: int,
    solver: str = "gurobi",
    mipgap: float = 0.01,
    verbose: bool = True
):
    """
    Solve a MILP time series aggregation problem using Pyomo and a dense distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Dense symmetric matrix of shape (n_days, n_days) with pairwise distances.
    k : int
        Number of representative days to select.
    solver : str
        MILP solver to use (e.g., "gurobi", "cbc", "glpk").
    mipgap : float
        Relative MIP optimality gap (e.g., 0.01 = 1%).
    verbose : bool
        Whether to print log output.

    Returns
    -------
    dict
        {
            'selected_days': list of representative day indices,
            'assignments': dict mapping each day to its representative,
            'model': Pyomo model object,
            'results': Solver result object
        }
    """

    if verbose:
        print(">>>>> Building Pyomo model")

    n_days = distance_matrix.shape[0]
    I = range(n_days)
    J = range(n_days)

    model = pyo.ConcreteModel()

    # Sets
    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    # Parameters #TODO: implement a threshold here to encourage sparsity e.g. all values <0.01*Matrix Rangeare set to 0. We can explore the impact of this on model run times vs. outcome accuracy.
    D_dict = {
        (i, j): float(distance_matrix[i, j])
        for i in I for j in J
    }

    model.D = pyo.Param(model.I, model.J, initialize=D_dict, within=pyo.NonNegativeReals)

    # Decision variables
    model.y = pyo.Var(model.I, domain=pyo.Binary)
    model.x = pyo.Var(model.I, model.J, domain=pyo.Binary)

    # Objective: Minimize total assignment cost
    def obj_rule(m):
        return sum(m.x[i, j] * m.D[i, j] for i in m.I for j in m.J)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints

    # Each day assigned to exactly one representative
    def assign_once_rule(m, j):
        return sum(m.x[i, j] for i in m.I) == 1
    model.assign_once = pyo.Constraint(model.J, rule=assign_once_rule)

    # Linking constraint: x[i,j] only active if y[i] is selected
    def link_x_y_rule(m, i, j):
        return m.x[i, j] <= m.y[i]
    model.link_x_y = pyo.Constraint(model.I, model.J, rule=link_x_y_rule)

    # Exactly k representatives
    def num_reps_rule(m):
        return sum(m.y[i] for i in m.I) == k
    model.num_reps = pyo.Constraint(rule=num_reps_rule)

    if verbose:
        print(">>>>> Solving MILP with", solver)

    solver_obj = pyo.SolverFactory(solver)
    results = solver_obj.solve(
        model,
        tee=verbose,
        options={
            'TimeLimit': 3000,
            'MipGap': mipgap,
            'Threads': min(max_threads-2 if max_threads>2 else 1, 16), #give gurobi as many as it can take, 16 is where efficiency drops 
            'LogToConsole': int(verbose),
            # Optional: Uncomment if needed
            # 'Presolve': 1,
            # 'Heuristics': 0.5,
            # 'Cuts': 2,
        }
    )

    # Extract solution
    selected_days = [i for i in model.I if pyo.value(model.y[i]) > 0.5]
    assignments = {
        j: next(i for i in model.I if pyo.value(model.x[i, j]) > 0.5)
        for j in model.J
    }

    return {
        "selected_days": selected_days,
        "assignments": assignments,
        "model": model,
        "results": results
    }


def save_milp_result_to_calliope_csv(
    result: dict,
    df_daily: pd.DataFrame,
    output_path: str
):
    """
    Save the MILP TSA result in the Calliope-compatible format:
    timesteps,PeriodNum

    Parameters
    ----------
    result : dict
        Output of `solve_standard_milp_tsa`, including 'assignments' and 'selected_days'.
    original_df : pd.DataFrame
        DataFrame used to generate the feature matrix, must have a DatetimeIndex.
    df_daily : pd.DataFrame
        DataFrame used to capture daily information.
    output_path : str
        Path to the CSV file to write.
    """
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise ValueError("df_daily must have a DatetimeIndex")

    # Convert index to list so we can use integer positions
    daily_dates = df_daily.index.to_list()

    # Map day index → representative day
    day_to_rep = {
        daily_dates[day_idx]: daily_dates[result['assignments'][day_idx]]
        for day_idx in result['assignments']
    }

    # Build final dataframe
    mapping_df = pd.DataFrame.from_dict(day_to_rep, orient="index", columns=["PeriodNum"])
    mapping_df.index.name = "timesteps"
    mapping_df = mapping_df.reset_index()

    # Format as strings for Calliope compatibility
    mapping_df["timesteps"] = mapping_df["timesteps"].dt.strftime("%Y-%m-%d")
    mapping_df["PeriodNum"] = mapping_df["PeriodNum"].dt.strftime("%Y-%m-%d")

    # Save to CSV
    mapping_df.to_csv(output_path, index=False)
    print(f"Saved daily representative mapping to {output_path}")

def run_calliope_model_on_cluster(ref_model, id_string, reference_model, path_cluster_map, proxy_parameters):

    _, _, ref_soc = tt.get_capacities(reference_model)

    params = {
        'output_directory_name': 'tsa_dev',
        'output_model_name': 'clustered_2015_2019_customMilpTSA',
        'config_yaml_name': 'model',
        'horizon_start':  '2015-01-01',
        'horizon_end':  '2019-12-31',
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
        # 'dict_additional_overrides': {},
        'path_to_cluster_csv': path_cluster_map,
        'path_to_new_timeseries': 'original'
    }

    #compute soc proxy of reference model
    ref_df_soc_proxy = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',f"{ref_model[:4]}-01-1",f"{ref_model[-4:]}-12-31")
    ref_df_soc_proxy, _, _ = generate_soc_proxy(
            df=ref_df_soc_proxy,
            demand_field='demand_power',
            renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
            dispatchable_techs=proxy_parameters['dispatchable_techs'],
            storage_process_losses=proxy_parameters['storage_process_losses'],
            soc_decomposition = proxy_parameters['soc_decomposition'],
            timestamp_col='timesteps'
    )
    ref_df_soc_proxy = ref_df_soc_proxy.set_index('timesteps')
    save_path = f"SoC_proxy_TSA/results/tsa_dev/clustered_{id_string}.netcdf"

    if os.path.exists(save_path):
        print(f"Skipping solving clustered model: {save_path} already exists.")
        clustered_model = calliope.read_netcdf(save_path)
    else:

        clustered_model = clustered_model_config(params)

        #build, solve, save clustered model
        print(f">> Building: {path_cluster_map}")
        clustered_model.build()
        #solve
        print(f">>> Solving: {save_path}")
        clustered_model.solve()
        #print results
        print(f">>> Results: Obj. Function: {clustered_model.results.cost.sum().item():e}, Solve Time: {round(clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,1)}s")
        #auto-save, first checks if full directory tree exists (if not, creates it)
        clustered_model.to_netcdf(save_path)
        print(f">>> Saved: {save_path}")

    


    comparison_result = compare_models(
        model_reference = reference_model, 
        model_test =clustered_model, 
        df_clustermap_test_model=pd.read_csv(path_cluster_map))
    
    return comparison_result

def run():


    #load model
    path_full_model = 'SoC_proxy_TSA/results/tsa_dev/standard_2015_2019_reference.netcdf'
    full_model = calliope.read_netcdf(path_full_model)

    proxy_parameters = {
    'capacity_weights': {
        'solar': 1,
        'onshore_wind': .5, # making the baseline assumption of an even distribution between solar and wind -based products i.e. the sum of onshore and offshore wind equals solar
        'offshore_wind': .5 
    },
    'storage_process_losses': {
        'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
        'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
    },
    'dispatchable_techs': {
        'known_dispatchable_capacity_portion_mean_demand': .25 #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
        # 'relative_dispatchable': 0 #TODO: add in capacity for relative dispatchable
    },
    'soc_decomposition': {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    }

    }

    cluster_map_path = 'SoC_proxy_TSA/cache/cluster_maps/2015_2019_n_37_custom_milp_weights_1_1_1_hourly_features.csv'
    milp_string = "custom_milp_weights_1_1_1_hourly_features"
    id_string = f"{2015}_{2019}_k_{37}_{milp_string}"

    if os.path.exists(cluster_map_path):
        print(f"Skipping MILP: {cluster_map_path} already exists.")
    else:

        #extract timeseries
        df_raw = extract_timeseries_from_calliope(full_model)
        # print(df_raw.head())   # See a sample of the output

        feature_df,_,_ = generate_soc_proxy(
            df=df_raw,
            demand_field='demand_power',
            renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
            dispatchable_techs=proxy_parameters['dispatchable_techs'],
            storage_process_losses=proxy_parameters['storage_process_losses'],
            soc_decomposition = proxy_parameters['soc_decomposition'],
            timestamp_col=None
        )
        feature_df.drop(['mean_capacity_factor', 'surplus','surplus_SDES','soc_proxy_LDES','soc_proxy_SDES'], axis=1, inplace=True)

        #restructure as daily feature sets
        # df_daily = aggregate_to_daily_features(df_raw, agg_func="mean") #alternative that agrgegates to daily and drops the hourly info
        df_daily = generate_daily_feature_matrix(df_raw)
        # print(df_daily.shape)

        #compute euclidiean distances
        # distance_matrix = compute_distance_matrix(df_daily, metric="euclidean")
        distance_matrix = compute_proxy_distance_matrix(df_daily, proxy_parameters, metric='euclidean')
        # print(distance_matrix.shape)  
        # print(distance_matrix[:3, :3])  # See a sample of the output
        print("Running MILP TSA...")
        result = solve_standard_milp_tsa(distance_matrix, k=37)
        save_milp_result_to_calliope_csv(
            result=result,
            df_daily=df_daily,  # this has hourly timesteps
            output_path=cluster_map_path
        )

    run_calliope_model_on_cluster('2015_2019', id_string, full_model, cluster_map_path, proxy_parameters)
    
    
def batch_run():

    time_start = time.time()

    #loop over years
    for year in set_year_range:

        #load model
        path_reference_model = f"SoC_proxy_TSA/results/tsa_dev/standard_{year[0]}_{year[1]}_reference.netcdf"
        model_reference = calliope.read_netcdf(path_reference_model)

        #loop over number of rep periods
        for compression in set_k_periods_as_percent_compression:

            k = int(round(compression*(1+year[1]-year[0])*365.25))

            #loop over distance matrix weights
            for matrix_weights in set_distance_matrix_weights:

                milp_string = f"custom_milp_weights_{matrix_weights[0]}_{matrix_weights[1]}_{matrix_weights[2]}"
                id_string = f"{year[0]}_{year[1]}_k_{k}_{milp_string}"

                cluster_map_path = f"SoC_proxy_TSA/cache/cluster_maps/{id_string}.csv"

                print(f">>>  Processing {id_string}")

                if os.path.exists(cluster_map_path):
                    print(f">>>>>  Skipping MILP: {cluster_map_path} already exists.")
                else:
                    
                    #extract timeseries
                    df_raw = extract_timeseries_from_calliope(model_reference)

                    feature_df,_,_ = generate_soc_proxy(
                        df=df_raw,
                        demand_field='demand_power',
                        renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
                        dispatchable_techs=proxy_parameters['dispatchable_techs'],
                        storage_process_losses=proxy_parameters['storage_process_losses'],
                        soc_decomposition = proxy_parameters['soc_decomposition'],
                        timestamp_col=None
                    )

                    #restructure as daily feature sets
                    df_daily = aggregate_to_daily_features( #alternative that agrgegates to daily and drops the hourly info
                        df_raw, 
                        agg_func="mean"
                        ) 
                    # df_daily = generate_daily_feature_matrix(df_raw)
                    
                    #compute euclidiean distances
                    print(f">>>  Computing the distance matrix @t= {time.time()-time_start:.2f}")
                    distance_matrix = compute_proxy_distance_matrix(
                        feature_df=df_daily, 
                        proxy_parameters=proxy_parameters, 
                        matrix_weights=matrix_weights, 
                        metric='euclidean'
                        )

                    print(f">>>  Running MILP TSA @t= {time.time()-time_start:.2f}")
                    result = solve_standard_milp_tsa(distance_matrix, k=k)
                    save_milp_result_to_calliope_csv(
                        result=result,
                        df_daily=df_daily,  # this has hourly timesteps
                        output_path=cluster_map_path
                    )

                print(f">>>  Running calliope on clustered model @t= {time.time()-time_start:.2f}")
                run_calliope_model_on_cluster(f"{year[0]}_{year[1]}", id_string, model_reference, cluster_map_path, proxy_parameters)


def batch_review():

    time_start = time.time()

    list_results = []
    list_labels = []

    #loop over years
    for year in set_year_range:

        #load model
        path_reference_model = f"SoC_proxy_TSA/results/tsa_dev/standard_{year[0]}_{year[1]}_reference.netcdf"
        model_reference = calliope.read_netcdf(path_reference_model)

        #loop over number of rep periods
        for compression in set_k_periods_as_percent_compression:

            k = int(round(compression*(1+year[1]-year[0])*365.25))

            #loop over distance matrix weights
            for matrix_weights in set_distance_matrix_weights:

                milp_string = f"custom_milp_weights_{matrix_weights[0]}_{matrix_weights[1]}_{matrix_weights[2]}"
                id_string = f"{year[0]}_{year[1]}_k_{k}_{milp_string}"

                path_clustered_model = f"SoC_proxy_TSA/results/tsa_dev/clustered_{id_string}.netcdf"
                path_cluster_map = f"SoC_proxy_TSA/cache/cluster_maps/{id_string}.csv"

                print(f">>>  Processing {id_string}")

                if not os.path.exists(path_clustered_model):
                    raise Exception(f"{id_string} does not exist.")
                else:
                    
                    model_clustered = calliope.read_netcdf(path_clustered_model)

                    result = compare_models(
                            model_reference = model_reference, 
                            model_test =model_clustered, 
                            df_clustermap_test_model=pd.read_csv(path_cluster_map)
                        )[0]
                    
                    result['id']={
                        'k_period': k,
                        'id_string:': id_string,
                        'matrix_weights': matrix_weights
                    }

                    list_results.append(result)

                    list_labels.append(id_string)
    
    df_standard_milp_soc = compare_models(
        model_reference=model_reference,
        model_test =calliope.read_netcdf('SoC_proxy_TSA/results/tsa_dev/clustered_2015_2019_k_37_custom_milp_standard.netcdf'), 
        df_clustermap_test_model=pd.read_csv('SoC_proxy_TSA/cache/cluster_maps/2015_2019_n_37_custom_milp_standard.csv')
    )[0]['df_soc']

    df_standard_hourly_milp_soc = compare_models(
        model_reference=model_reference,
        model_test =calliope.read_netcdf('SoC_proxy_TSA/results/tsa_dev/clustered_2015_2019_k_37_custom_milp_standard_hourly.netcdf'), 
        df_clustermap_test_model=pd.read_csv('SoC_proxy_TSA/cache/cluster_maps/2015_2019_n_37_custom_milp_standard_hourly.csv')
    )[0]['df_soc']

    df_test_hourly_milp_soc = compare_models(
        model_reference=model_reference,
        model_test =calliope.read_netcdf('SoC_proxy_TSA/results/tsa_dev/clustered_2015_2019_k_37_custom_milp_weights_1_1_1_hourly_features.netcdf'), 
        df_clustermap_test_model=pd.read_csv('SoC_proxy_TSA/cache/cluster_maps/2015_2019_n_37_custom_milp_weights_1_1_1_hourly_features.csv')
    )[0]['df_soc']
    

    df_reference_soc = compare_models(
                            model_reference = model_reference, 
                            model_test =model_clustered, 
                            df_clustermap_test_model=pd.read_csv(path_cluster_map)
                        )[1]
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_reference_soc.index, df_reference_soc['soc'], label="SoC, Reference",color='black', zorder=102)  
    plt.plot(df_standard_milp_soc.index, df_standard_milp_soc['soc'], label="SoC, Standard MILP",color='grey', zorder=101)  
    plt.plot(df_standard_hourly_milp_soc.index, df_standard_hourly_milp_soc['soc'], label="SoC, Hourly MILP",color='blue', zorder=101)
    plt.plot(df_test_hourly_milp_soc.index, df_test_hourly_milp_soc['soc'], label="SoC, Hourly MILP with proxy",color='orange', zorder=101)  

    for key, result in enumerate(list_results):
        if result['id']['k_period']:
            plt.plot(result['df_soc'].index, result['df_soc']['soc'], label=list_labels[key], zorder=101)   
    plt.xlabel('Time')
    plt.ylabel('State of Charge')
    plt.title('Storage State of Charge Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


proxy_parameters = {
    'capacity_weights': {
        'solar': 1,
        'onshore_wind': .5, # making the baseline assumption of an even distribution between solar and wind -based products i.e. the sum of onshore and offshore wind equals solar
        'offshore_wind': .5 
    },
    'storage_process_losses': {
        'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
        'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
    },
    'dispatchable_techs': {
        'known_dispatchable_capacity_portion_mean_demand': .25 #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
    },
    'soc_decomposition': {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    }
}


set_year_range = [
    [2015,2019]
]

set_k_periods_as_percent_compression = [
    # 0.01,
    0.02,
    # 0.03,
    # 0.04,
    # 0.05
]

set_distance_matrix_weights = [
    # renewables : state of charge : demand

    [100,100,100],
    # [1,0,1],
    # # [1,0.1,1],
    # # [1,0.2,1],
    # # [1,0.3,1],
    # # [1,0.4,1],
    # [1,0.5,1],
    # [1,0.6,1],
    # [1,0.7,1],
    # [1,0.8,1],
    # [1,0.9,1],
    [1,1,1],
    # [1,1.1,1],
    # [1,1.2,1],        
    # [1,1.25,1],
    # [1,1.3,1],
    # [1,1.4,1],
    # [1,1.5,1],
    # # [1,1.6,1],
    # [1,1.7,1],
    # [1,1.75,1],
    # [1,1.8,1],
    # [1,1.9,1],
    # [1,2,1],
] 

# run()              
batch_run()
batch_review()