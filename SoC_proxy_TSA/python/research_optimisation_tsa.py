import pandas as pd
import calliope 
import numpy as np
from scipy.spatial.distance import cdist
import pyomo.environ as pyo

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
    save_columns = raw_data.columns
    old_names  = save_columns.names
    save_columns = pd.MultiIndex.from_tuples(
    [('comment',) + col for col in save_columns]
    )   
    save_columns.names = ['comment','nodes','techs','parameters']
    save_index = raw_data.index

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

def solve_standard_milp_tsa(distance_matrix: np.ndarray, k: int, solver: str = "gurobi"):
    """
    Solve a MILP time series aggregation problem using Pyomo and a given distance matrix.

    Parameters:
    ----------
    distance_matrix : np.ndarray
        A symmetric matrix of shape (n_days, n_days) representing pairwise distances.
    k : int
        The number of representative days to select.
    solver : str
        MILP solver to use (e.g., "gurobi", "cbc", "glpk").

    Returns:
    -------
    dict
        Dictionary with:
            'selected_days': list of indices selected as representative days
            'assignments': dict mapping each day to its assigned representative
            'model': the Pyomo model instance (for inspection)
    """
    n_days = distance_matrix.shape[0]
    I = range(n_days)
    J = range(n_days)

    # Create model
    model = pyo.ConcreteModel()

    # Sets
    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    # Parameters
    model.D = pyo.Param(model.I, model.J, initialize=lambda model, i, j: distance_matrix[i, j])

    # Decision variables
    print(f'creating {n_days} binary variables for y[i]')
    model.y = pyo.Var(model.I, domain=pyo.Binary)  # y[i] = 1 if day i is selected
    print(f'creating {n_days*n_days} binary variables for x[i,j]')
    model.x = pyo.Var(model.I, model.J, domain=pyo.Binary)  # x[i,j] = 1 if day j is assigned to rep i

    # Objective: Minimize total distance
    print('Assigning objective function and constraints')
    def obj_rule(model):
        return sum(model.x[i, j] * model.D[i, j] for i in model.I for j in model.J)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints

    # Each day j must be assigned to exactly one representative i
    def assign_once_rule(model, j):
        return sum(model.x[i, j] for i in model.I) == 1
    model.assign_once = pyo.Constraint(model.J, rule=assign_once_rule)

    # x[i,j] can only be 1 if y[i] is selected
    def assign_only_if_selected_rule(model, i, j):
        return model.x[i, j] <= model.y[i]
    model.link_x_y = pyo.Constraint(model.I, model.J, rule=assign_only_if_selected_rule)

    # Limit number of representatives
    def num_reps_rule(model):
        return sum(model.y[i] for i in model.I) == k
    model.num_reps = pyo.Constraint(rule=num_reps_rule)

    # Solve
    print('Building and solving the model...')
    solver_obj = pyo.SolverFactory(solver)

    results = solver_obj.solve(model,
        tee=True,
        options={
            'LogToConsole': 1,     # Force Gurobi to show log
            'TimeLimit': 600,      
            # 'MIPGap': 0.01         
        })

    # Extract solution
    selected_days = [i for i in model.I if pyo.value(model.y[i]) > 0.5]
    assignments = {j: next(i for i in model.I if pyo.value(model.x[i, j]) > 0.5) for j in model.J}

    return {
        "selected_days": selected_days,
        "assignments": assignments,
        "model": model,
        "results": results
    }

def run():


    #load model
    path_full_model = 'SoC_proxy_TSA/results/tsa_dev/standard_2015_2019_reference.netcdf'
    full_model = calliope.read_netcdf(path_full_model)

    #extract timeseries
    df_raw = extract_timeseries_from_calliope(full_model)
    # print(df_raw.head())   # See a sample of the output

    #restructure as daily feature sets
    df_daily = generate_daily_feature_matrix(df_raw)
    # print(df_daily.shape)

    #compute euclidiean distances
    distance_matrix = compute_distance_matrix(df_daily, metric="euclidean")
    # print(distance_matrix.shape)  
    # print(distance_matrix[:3, :3])  # See a sample of the output

    print("time to solve...")
    #run the milp tsa
    result = solve_standard_milp_tsa(distance_matrix, k=36)
    print("Selected representative days:")
    print(result['selected_days'][:5])  # show a sample

    print("Example assignments:")
    print(list(result['assignments'].items())[:5])
    
    
    print('done')
run()