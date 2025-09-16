import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing
import pyomo.environ as pyo
import json
import re

max_threads = multiprocessing.cpu_count()

def distance_matrix(
    feature_df: pd.DataFrame,
    *,
    matrix_weights: dict = {"renewables": 1.0, "demand": 1.0, "proxy": 1.0},
    metric: str = "euclidean",
    column_prefixes_renewables=['solar','onshore_wind','offshore_wind'],
    column_prefixes_demand=['demand_power'],
    column_prefixes_proxy=[],
    proxy_window: tuple[str, str] | None = None,  # (start, end) dates; inclusive
) -> np.ndarray:
    """
    Returns a weighted distance matrix D = wR*D_R + wD*D_D + wP*(M ⊙ D_P),
    where M masks the proxy term so it only contributes for rows (days) within proxy_window.
    """

    # --- collect columns by prefix
    def cols_by_prefix(prefixes):
        return [c for c in feature_df.columns if any(c.startswith(p) for p in prefixes)]

    renew_cols = cols_by_prefix(column_prefixes_renewables)
    dem_cols   = cols_by_prefix(column_prefixes_demand)
    prox_cols  = cols_by_prefix(column_prefixes_proxy)

    # --- helper to scale block with feature-count normalization (keeps blocks comparable)
    def scaled_block(cols):
        if not cols:
            return None
        X = feature_df[cols].to_numpy()
        Xz = StandardScaler().fit_transform(X)
        # divide by sqrt(n_features) so block magnitude is roughly comparable across sizes
        return Xz / np.sqrt(len(cols))

    X_R = scaled_block(renew_cols)
    X_D = scaled_block(dem_cols)
    X_P = scaled_block(prox_cols) if prox_cols else None

    # --- compute per-block pairwise distances
    D_R = cdist(X_R, X_R, metric=metric) if X_R is not None else 0.0
    D_D = cdist(X_D, X_D, metric=metric) if X_D is not None else 0.0
    D_P = cdist(X_P, X_P, metric=metric) if X_P is not None else 0.0

    # --- weights (normalize so totals are comparable even if proxy missing)
    present = {
        "renewables": X_R is not None,
        "demand":     X_D is not None,
        "proxy":      X_P is not None,
    }
    denom = sum(matrix_weights[k] for k, ok in present.items() if ok) or 1.0
    wR = (matrix_weights["renewables"] if present["renewables"] else 0.0) / denom
    wD = (matrix_weights["demand"]     if present["demand"]     else 0.0) / denom
    wP = (matrix_weights["proxy"]      if present["proxy"]      else 0.0) / denom

    # --- build row mask for proxy term
    # If proxy_window = (start, end), only rows with index in that range get proxy cost.
    if wP == 0.0:
        M = 0.0
    else:
        if proxy_window is None or proxy_window == 'None':
            # proxy applies everywhere
            M = 1.0
        else:
            str_start, str_end = proxy_window
            start = pd.Timestamp(str_start)
            end = pd.Timestamp(str_end)
            
            # Ensure DatetimeIndex
            if not isinstance(feature_df.index, pd.DatetimeIndex):
                raise ValueError("feature_df.index must be a DatetimeIndex to use proxy_window.")
            in_window = (feature_df.index >= pd.to_datetime(start)) & (feature_df.index <= pd.to_datetime(end))
            # Row mask → shape (n,1); broadcast across columns to weight rows only
            m = np.asarray(in_window, dtype=float).reshape(-1, 1)
            M = m @ np.ones((1, feature_df.shape[0]))  # (n,n) mask: 1 for rows in window, else 0

    # --- combine
    D = wR * D_R + wD * D_D

    # Only add proxy distances if they matter
    if wP > 0:
        if np.isscalar(M):
            if M != 0.0:
                D += wP * D_P
        else:
            # if no rows are in-window, mask is all zeros → skip
            if np.any(M):
                D += wP * (D_P * M)

    return D

def milp_tsa(
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
        print(">>> MILP: Building Pyomo model")

    n_days = distance_matrix.shape[0]
    I = range(n_days)
    J = range(n_days)

    model = pyo.ConcreteModel()

    # Sets
    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    # Parameters #TODO: implement a threshold here to encourage sparsity e.g. all values <0.01*Matrix Range are set to 0. We can explore the impact of this on model run times vs. outcome accuracy.
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
        print(">>> MILP: Solving with", solver)

    solver_obj = pyo.SolverFactory(solver)
    results = solver_obj.solve(
        model,
        tee=verbose,
        options={
            'TimeLimit': 3600,
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

def save_milp_result_to_cluster_map(
    result: dict,
    dates_index: pd.DatetimeIndex,
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
    if not isinstance(dates_index, pd.DatetimeIndex):
        raise ValueError("df_daily must have a DatetimeIndex")

    # Convert index to list so we can use integer positions
    daily_dates = dates_index.to_list()

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


def ORDO(
    df_features: pd.DataFrame,
    k: int, #can be different to k of pre_cluster if the MILP is intended to further shrink the number of rep days
    path_cluster_map: str, #if none, perform milp on all days, if clustermap provided (path) then only use the pre-clustered days as candidates
    use_endogenous_soc_proxy: bool = False,
    is_pre_clustered: bool = False,
    solver: str = "gurobi",
    mipgap: float = 0.01,
    verbose: bool = True
    ):

    if verbose:
        print("[TSA] Building Pyomo model")

    

    pyomo_model = pyo.ConcreteModel()

    #TODO: if is_pre_clustered then one of thesse becomes len(np.unique(df_clustermap))
    n_days = distance_matrix.shape[0]
    I = range(n_days)
    J = range(n_days)

    # Sets
    pyomo_model.I = pyo.Set(initialize=I)
    pyomo_model.J = pyo.Set(initialize=J)

    # Parameters #TODO: implement a threshold here to encourage sparsity e.g. all values <0.01*Matrix Range are set to 0. We can explore the impact of this on model run times vs. outcome accuracy.
    D_dict = {
        (i, j): float(distance_matrix[i, j])
        for i in I for j in J
    }

    

    pyomo_model.D = pyo.Param(pyomo_model.I, pyomo_model.J, initialize=D_dict, within=pyo.NonNegativeReals)

    # Decision variables
    pyomo_model.y = pyo.Var(pyomo_model.I, domain=pyo.Binary)
    pyomo_model.x = pyo.Var(pyomo_model.I, pyomo_model.J, domain=pyo.Binary)

    # Objective: Minimize total assignment cost
    def obj_rule(m):
        return sum(m.x[i, j] * m.D[i, j] for i in m.I for j in m.J)
    if use_endogenous_soc_proxy:
        pyomo_model.obj = pyo.Objective(rule=_endogenous_objective_function, sense=pyo.minimize)
    else:
        pyomo_model.obj = pyo.Objective(rule=_exogenous_objective_function, sense=pyo.minimize) #TODO: sense and objective function itself should be built from config.yaml

    raise NotImplementedError('ORDO has not been implemented')


def _exogenous_objective_function(m):
    return sum(m.x[i, j] * m.D[i, j] for i in m.I for j in m.J)

def _endogenous_objective_function(m):
    raise NotImplementedError('ORDO with endogenous SoC proxy not yet implemented.')

_HPAT = re.compile(r"^(?P<base>.+)_h(?P<hour>\d{2})$")

def group_feature_columns(df_features: pd.DataFrame, preferred_features: list[str] | None = None):
    """
    Return:
      mode: 'daily' or 'hourly24'
      groups: dict[str, list[str]] mapping base feature -> columns (1 for daily, 24 for hourly)
    """
    cols = list(df_features.columns)
    # If preferred_features provided, filter to those + their hourly expansions
    if preferred_features:
        def keep(c):
            m = _HPAT.match(c)
            base = m.group("base") if m else c
            return base in preferred_features
        cols = [c for c in cols if keep(c)]

    # Try to detect hourly groups
    groups = {}
    for c in cols:
        m = _HPAT.match(c)
        if m:
            base = m.group("base")
            groups.setdefault(base, []).append(c)

    if groups and all(len(sorted(v)) >= 24 for v in groups.values()):
        # keep exactly the 24 hour columns per base
        groups = {k: sorted([c for c in v if _HPAT.match(c)])[:24] for k, v in groups.items()}
        return "hourly24", groups

    # Else: daily mode — each selected column is its own feature
    if preferred_features:
        groups = {f: [f] for f in preferred_features if f in df_features.columns}
    else:
        groups = {c: [c] for c in df_features.columns}

    return "daily", groups
 

def _minmax_signed(col: np.ndarray) -> np.ndarray:
    cmin = np.nanmin(col)
    cmax = np.nanmax(col)
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
        return np.zeros_like(col, dtype=float)
    z = (col - cmin) / (cmax - cmin)
    return 2.0 * z - 1.0

def _ordo_cost_matrix_from_features(
    df_features: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    preferred_features: list[str] | None = None,
    normalize: str = "minmax_signed",
) -> np.ndarray:
    """
    Build D_{ij} = sum_features w_f * ||X_f[i,:] - X_f[j,:]||^2
    - Works with 'daily' (1 col/feature) or 'hourly24' (24 cols/feature) formats.
    - Normalization applied per column.
    """
    mode, groups = group_feature_columns(df_features, preferred_features)
    N = len(df_features)
    if feature_weights is None:
        feature_weights = {}
    
    # Build a single big design matrix X: shape (N, sum_k d_k), where d_k is 1 (daily) or 24 (hourly)
    X_blocks = []
    W_blocks = []
    for base, cols in groups.items():
        block = df_features[cols].to_numpy(dtype=float, copy=False)
        # Normalize per column if requested
        if normalize == "minmax_signed":
            block = np.column_stack([_minmax_signed(block[:, j]) for j in range(block.shape[1])])
        elif normalize != "none":
            raise ValueError(f"Unknown normalize='{normalize}'")
        X_blocks.append(block)
        w = float(feature_weights.get(base, 1.0))
        # Weight is applied per block; implement as column scaling by sqrt(w)
        if w < 0:
            w = 0.0
        if w == 0:
            W_blocks.append(np.zeros(block.shape[1], dtype=float))
        else:
            W_blocks.append(np.sqrt(w) * np.ones(block.shape[1], dtype=float))
    
    if not X_blocks:
        raise ValueError("No features selected for cost matrix.")
    
    X = np.concatenate(X_blocks, axis=1)          # (N, Dtot)
    wcol = np.concatenate(W_blocks, axis=0)       # (Dtot,)
    # Apply weights by column scaling
    Xw = X * wcol[None, :]

    # Squared Euclidean distance matrix via (x - y)^2 = ||x||^2 + ||y||^2 - 2 x·y
    norms = np.sum(Xw * Xw, axis=1)               # (N,)
    G = Xw @ Xw.T                                 # Gram matrix
    D = norms[:, None] + norms[None, :] - 2.0 * G
    # numerical cleanup
    D[D < 0] = 0.0
    np.fill_diagonal(D, 0.0)
    return D

def solve_ordo_from_features(
    df_features: pd.DataFrame,
    k: int,
    feature_weights: dict[str, float] | None = None,
    preferred_features: list[str] | None = None,
    normalize: str = "minmax_signed",
    solver: str = "gurobi",
    mipgap: float = 0.01,
    verbose: bool = True,
):
    D = _ordo_cost_matrix_from_features(
        df_features=df_features,
        feature_weights=feature_weights,
        preferred_features=preferred_features,
        normalize=normalize,
    )
    return milp_tsa(D, k, solver=solver, mipgap=mipgap, verbose=verbose)