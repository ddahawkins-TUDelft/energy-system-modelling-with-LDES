import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing
import pyomo.environ as pyo
import json
import re

from utility_functions.helper_endogenous_biases import generate_endogenous_biases

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
    MIPGap: float = 0.01,
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
        print("[TSA] Building Pyomo model")

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
        print("[TSA] Solving with", solver)

    solver_obj = pyo.SolverFactory(solver)
    results = solver_obj.solve(
        model,
        tee=verbose,
        options={
            # 'TimeLimit': 1800, #20mins is often enough, see literature ref'd by Gonzato #TODO:
            'WorkLimit': 1500, #using worklimit as its more deterministic and transfers across machines better than timelimit which, for example, may exit earlier for 'slower' CPUs
            'MIPGap': MIPGap,
            'Threads': min(max_threads-4 if max_threads>4 else 1, 16), #give gurobi as many as it can take, 16 is where efficiency drops 
            'LogToConsole': int(verbose),
            # 'BarConvTol': 1e-8, #given that Branch and Bound is the real issue, we can relax tolerances on barrier a little. Default is 1e-10
            # 'method': 1
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

def milp_ordo_restricted_candidates(
    *,
    Dsub: np.ndarray,          # shape (|C|, N)  cost from candidate i (row aligned to C_ids) to day j (0..N-1)
    C_ids: list[int],          # GLOBAL day indices of candidates, len = |C|
    k: int,
    fix_reps: bool = False,    # if True: use all candidates as fixed reps (|C| must equal k)
    warm_start_rep_for_day: np.ndarray | None = None,  # length N, GLOBAL rep index per day
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 3600,
    verbose: bool = True,
    lp_method: str = "barrier",    # "barrier" or "dual"
):
    import numpy as np
    import pyomo.environ as pyo

    Ic_size, N = Dsub.shape
    assert Ic_size == len(C_ids), "Dsub rows and C_ids length mismatch"
    if fix_reps:
        assert k == Ic_size, "fix_reps=True requires k == len(C_ids)"

    # ---------------- Model indexed by GLOBAL candidate IDs ----------------
    m = pyo.ConcreteModel()
    # representatives (GLOBAL ids)
    m.I = pyo.Set(initialize=list(C_ids), ordered=True)
    # days (0..N-1)
    m.J = pyo.RangeSet(0, N - 1)

    # Build D as a param keyed by (GLOBAL_i, j) pulling from Dsub rows
    row_of_global = {g: r for r, g in enumerate(C_ids)}
    D_map = {(i, j): float(Dsub[row_of_global[int(i)], int(j)]) for i in m.I for j in m.J}
    m.D = pyo.Param(m.I, m.J, initialize=D_map, within=pyo.NonNegativeReals)

    # Variables
    m.y = pyo.Var(m.I, domain=pyo.Binary)                 # rep selection (GLOBAL keyed)
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)            # assignment (GLOBAL keyed)

    # Constraints
    def assign_once(_m, j):
        return sum(_m.x[i, j] for i in _m.I) == 1
    m.assign_once = pyo.Constraint(m.J, rule=assign_once)

    m.link = pyo.Constraint(m.I, m.J, rule=lambda _m, i, j: _m.x[i, j] <= _m.y[i])

    if fix_reps:
        for i in m.I:
            m.y[i].fix(1)
    else:
        m.num_reps = pyo.Constraint(expr=sum(m.y[i] for i in m.I) == k)

    # Objective (pure ORDO)
    m.obj = pyo.Objective(
        expr=sum(m.x[i, j] * m.D[i, j] for i in m.I for j in m.J),
        sense=pyo.minimize
    )

    # ---------------- Solve (with optional warm start) ----------------
    if solver == "gurobi":
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.set_instance(m)

        # Warm start from precluster assignment (GLOBAL ids)
        if warm_start_rep_for_day is not None:
            gvmap = opt._pyomo_var_to_solver_var_map
            C_set = set(C_ids)
            used = set()
            for j in range(N):
                g_rep = int(warm_start_rep_for_day[j])
                if g_rep in C_set:
                    # set chosen pair to 1 (and optionally 0 for others)
                    gvmap[m.x[g_rep, j]].Start = 1.0
                    used.add(g_rep)
            if not fix_reps:
                for i in m.I:
                    gvmap[m.y[i]].Start = 1.0 if int(i) in used else 0.0

        # Gurobi params
        opt.set_gurobi_param('MIPGap', MIPGap)
        opt.set_gurobi_param('Threads', int(threads) if threads else 12)
        opt.set_gurobi_param('LogToConsole', int(verbose))
        opt.set_gurobi_param('TimeLimit', int(timelimit))
        opt.set_gurobi_param('Presolve', 2)
        opt.set_gurobi_param('Aggregate', 2)
        if lp_method == "barrier":
            opt.set_gurobi_param('Method', 2)
            opt.set_gurobi_param('Crossover', 0)
        elif lp_method == "dual":
            opt.set_gurobi_param('Method', 1)

        res = opt.solve(tee=verbose)
    else:
        res = pyo.SolverFactory(solver).solve(m, tee=verbose, options={
            'MIPGap': MIPGap, 'TimeLimit': int(timelimit)
        })

    # ---------------- Extract in the SAME format as milp_tsa() ----------------
    # selected reps: list of GLOBAL day indices
    if fix_reps:
        selected_days = list(C_ids)
    else:
        selected_days = [int(i) for i in m.I if pyo.value(m.y[i]) > 0.5]

    # assignments: j -> GLOBAL representative day
    assignments = {}
    # numerical guard so we don't crash if all x[i,j] are ~0 due to tolerance
    for j in m.J:
        # primary: exact 1’s
        chosen = [int(i) for i in m.I if (pyo.value(m.x[i, j]) or 0.0) > 0.5]
        if chosen:
            assignments[int(j)] = chosen[0]
        else:
            # fallback: argmax over i
            best_i, best_val = None, -1.0
            for i in m.I:
                val = pyo.value(m.x[i, j]) or 0.0
                if val > best_val:
                    best_val, best_i = val, int(i)
            assignments[int(j)] = best_i  # still GLOBAL

    return {
        "selected_days": selected_days,   # GLOBAL ids
        "assignments": assignments,       # j -> GLOBAL id
        "model": m,
        "results": res,
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
    mapping_df = mapping_df.set_index('timesteps')
    mapping_df.to_csv(output_path)


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
    preferred_features: list[str],
    feature_weights: dict[str, float],
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
        w = float(feature_weights.get(base, 0.0)) #default to 0 if not called.
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

# === RR utilities (soft & hard) ===============================================

def build_feature_design_matrices(
    df_features: pd.DataFrame,
    candidates: list[int],
    preferred_features: list[str] | None = None,
    normalize: str = "minmax_signed",
    feature_weights: dict[str, float] | None = None,
):
    """
    Prepare matrices for RR:
      - Normalizes per column ([-1,1]) using the SAME logic as ORDO,
      - Builds design matrices for ALL days (X_norm) and CANDIDATE days (A_norm),
      - Applies sqrt(weights) per feature *only* for the FITTING copies (X_fit, A_fit),
      - Returns per-column (min,max) so we can invert later.

    Returns
    -------
    X_fit : (N, D)   # normalized + column-weighted for fitting
    A_fit : (C, D)   # normalized + column-weighted for fitting (rows are candidates)
    A_norm: (C, D)   # normalized (no weights) for reconstruction
    scale_params : list[tuple[float,float]]  # [(min,max) per column, in df_features units]
    column_order : list[str]                 # df_features columns used, in the order of the matrices
    """
    if feature_weights is None:
        feature_weights = {}

    # reuse your grouping logic so "daily" vs "hourly24" just works
    mode, groups = group_feature_columns(df_features, preferred_features)

    # Build one large design matrix X_norm (N x D) by concatenating blocks
    X_blocks = []
    A_blocks = []
    scale_params = []
    colnames = []

    def _scale_signed(col: np.ndarray):
        cmin = np.nanmin(col)
        cmax = np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            return np.zeros_like(col, dtype=float), (0.0, 1.0)  # degenerate → zeros
        z = (col - cmin) / (cmax - cmin)
        z = 2.0 * z - 1.0
        return z.astype(float), (cmin, cmax)

    for base, cols in groups.items():
        block = df_features[cols].to_numpy(dtype=float, copy=False)
        # normalize per column
        nb = block.shape[1]
        norm_cols = np.empty_like(block, dtype=float)
        mins_maxs = []
        for j in range(nb):
            norm_cols[:, j], mm = _scale_signed(block[:, j])
            mins_maxs.append(mm)

        X_blocks.append(norm_cols)            # all days
        A_blocks.append(norm_cols[candidates, :])  # candidate rows only
        scale_params.extend(mins_maxs)
        colnames.extend(cols)

    # concatenate blocks -> (N, D) and (C, D)
    X_norm = np.concatenate(X_blocks, axis=1)
    A_norm = np.concatenate(A_blocks, axis=1)

    # apply sqrt(weights) per FEATURE block (spread equally to its columns)
    # we recorded blocks in order; rebuild the same iteration to apply weights
    wcols = []
    for base, cols in groups.items():
        w = float(feature_weights.get(base, 1.0))
        if w < 0:
            w = 0.0
        factor = np.sqrt(w) if w > 0 else 0.0
        wcols.extend([factor] * len(cols))
    wcols = np.asarray(wcols, dtype=float)  # length D

    X_fit = X_norm * wcols[None, :]
    A_fit = A_norm * wcols[None, :]

    return X_fit, A_fit, A_norm, scale_params, colnames

# === RR (batched, TS + DC) =====================================================

def rebuild_from_assignments(df_features: pd.DataFrame, rep_for_day: np.ndarray) -> pd.DataFrame:
    """
    Build a synthetic df by copying the representative row chosen for each day.
    """
    assert len(rep_for_day) == len(df_features)
    df_hat = df_features.iloc[rep_for_day].reset_index(drop=True)
    return df_hat

def solve_ordo_from_features(
    df_features: pd.DataFrame,
    k: int,
    feature_weights: dict[str, float] | None = None,
    preferred_features: list[str] | None = None,
    normalize: str = "minmax_signed",
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    verbose: bool = True,
):
    D = _ordo_cost_matrix_from_features(
        df_features=df_features,
        feature_weights=feature_weights,
        preferred_features=preferred_features,
        normalize=normalize,
    )
    return milp_tsa(D, k, solver=solver, MIPGap=MIPGap, verbose=verbose)

def solve_ordo_from_features_restricted(
    *,
    df_features,                 # your daily feature DF
    k: int,
    feature_weights: dict[str, float],
    preferred_features: list[str] | None = None,
    normalize: str = "minmax_signed",
    candidates: list[int],       # global row indices (from precluster_to_row_indices)
    fix_reps: bool = False,      # True when |C| == k and you just want “reassign to fixed reps”
    warm_start_rep_for_day: np.ndarray | None = None,  # optional global rep-per-day from precluster
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 3600,
    verbose: bool = True,
    lp_method: str = "barrier",
):
    """
    Build the ORDO distance, restrict rows to `candidates`, and solve the MILP.
    """
    import numpy as np

    # 1) Build full NxN ORDO distance using your existing function
    D_full = _ordo_cost_matrix_from_features(
        df_features=df_features,
        feature_weights=feature_weights,
        preferred_features=preferred_features,
        normalize=normalize,
    )  # shape (N, N)

    # 2) Slice to candidate rows: shape (|C|, N)
    C_ids = list(map(int, candidates))
    Dsub = D_full[np.ix_(C_ids, np.arange(D_full.shape[0]))]

    # 3) Solve the restricted MILP (with optional warm start)
    return milp_ordo_restricted_candidates(
        Dsub=Dsub, C_ids=C_ids, k=k, fix_reps=fix_reps,
        warm_start_rep_for_day=warm_start_rep_for_day,
        solver=solver, MIPGap=MIPGap, threads=threads,
        timelimit=timelimit, verbose=verbose, lp_method=lp_method,
    )


# === MILP: ORDO + L1 SoC error ================================================
def milp_tsa_endogenous(
    *,
    D: np.ndarray,                 # (N x N) standard ORDO cost matrix
    k: int,
    soc_ref: np.ndarray,           # (T,)
    H: np.ndarray,                 # (T, N, N) from build_soc_coeff_tensor_for_assignments
    lambda_soc: float = 0.5,
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    verbose: bool = True
):
    """
    Minimize: (1-lambda_soc)*sum_{i,j} D[i,j] * x[i,j] + lambda_soc * sum_t z_t
    s.t.     z_t >=  soc_hat[t] - soc_ref[t]
             z_t >= -soc_hat[t] + soc_ref[t]
             (standard ORDO constraints)
    """
    import pyomo.environ as pyo

    N = D.shape[0]
    T = soc_ref.shape[0]

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.J = pyo.RangeSet(0, N - 1)
    m.T = pyo.RangeSet(0, T - 1)

    m.D = pyo.Param(m.I, m.J, initialize={(i,j): float(D[i,j]) for i in range(N) for j in range(N)}, within=pyo.NonNegativeReals)

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)

    # standard ORDO constraints
    def assign_once(_m, j): return sum(_m.x[i,j] for i in _m.I) == 1
    m.assign_once = pyo.Constraint(m.J, rule=assign_once)

    def link_x_y(_m, i, j): return _m.x[i,j] <= _m.y[i]
    m.link = pyo.Constraint(m.I, m.J, rule=link_x_y)

    def num_reps(_m): return sum(_m.y[i] for i in _m.I) == k
    m.num_reps = pyo.Constraint(rule=num_reps)

    # SoC pieces
    m.z = pyo.Var(m.T, domain=pyo.NonNegativeReals)  # L1 residuals
    m.socdiff = pyo.Var(m.T, domain=pyo.Reals)       # soc_hat[t] - soc_ref[t]

    # soc_hat[t] = sum_{i,j} H[t,i,j] * x[i,j]
    H_dict = {(t,i,j): float(H[t,i,j]) for t in range(T) for i in range(N) for j in range(N)}

    def soc_affine(_m, t):
        return _m.socdiff[t] == sum(H_dict[(t,i,j)] * _m.x[i,j] for i in _m.I for j in _m.J) - soc_ref[t]
    m.soc_def = pyo.Constraint(m.T, rule=soc_affine)

    # |socdiff[t]| <= z[t]
    def z_pos(_m, t): return  _m.z[t] >=  _m.socdiff[t]
    def z_neg(_m, t): return  _m.z[t] >= -_m.socdiff[t]
    m.z_pos = pyo.Constraint(m.T, rule=z_pos)
    m.z_neg = pyo.Constraint(m.T, rule=z_neg)

    # Good scaling baselines
    dist_scale = max(1e-12, np.mean(D) * D.shape[0])          # ~ N * mean(D_ij)
    T = len(soc_ref)
    prox_scale = max(1e-12, np.mean(np.abs(soc_ref)) * T)     # == sum(|soc_ref|)
    # or simply: prox_scale = max(1e-12, np.sum(np.abs(soc_ref)))

    def obj(_m):
        dist = (1 - lambda_soc) * (sum(_m.x[i,j]*_m.D[i,j] for i in _m.I for j in _m.J) / dist_scale)
        prox =      lambda_soc  * (sum(_m.z[t]             for t in _m.T) / prox_scale)
        return dist + prox
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)


    # Solve
    if verbose: print("[TSA] Solving with", solver)
    opt = pyo.SolverFactory(solver)
    res = opt.solve(m, tee=verbose, options={
        # 'TimeLimit': 1800, #20mins is often enough, see literature ref'd by Gonzato #TODO:
        'WorkLimit': 1500,
        'MIPGap': MIPGap,
        'Threads': min(max_threads-4 if max_threads>4 else 1, 16),
        'LogToConsole': int(verbose),
    })

    selected_days = [i for i in range(N) if pyo.value(m.y[i]) > 0.5]
    assignments = {j: next(i for i in range(N) if pyo.value(m.x[i,j]) > 0.5) for j in range(N)}
    return {"selected_days": selected_days, "assignments": assignments, "model": m, "results": res}

def milp_tsa_endogenous_basic(
    *,
    D: np.ndarray,                    # (N x N) ORDO cost matrix
    k: int,
    soc_ref: np.ndarray,              # (T,) reference SoC
    S_by_day: np.ndarray,             # (N x 24) hourly surplus per candidate day
    a: np.ndarray,                    # (T,) frozen gains: eta_ch or 1/eta_dis
    lambda_soc: float = 0.5,
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 1200,
    verbose: bool = True,
    root_lp: str = "barrier",         # "dual" | "barrier"
    endogenous_biases: list | np.ndarray | None = None,
    bias_minmax: tuple[float, float] = (0.2, 1.0),  # rescale weights to this closed interval
    bias_zero_threshold: float = 0.3,
):
    """
    Minimize: (1-λ)*sum_{i,j} D[i,j] x[i,j] + λ * sum_t b_t * |soc_hat[t] - soc_ref[t]|
    with endogenous SoC built from chosen representative-day sequence.

    Notes
    -----
    - 'endogenous_biases' are per-timestep weights b_t. They are rescaled to [bias_min, bias_max]
      and applied only to the proxy term. If None, b_t := 1 for all t.
    - We divide the proxy scaling by avg(b) so lambda_soc remains comparable across runs.
    """
    import numpy as np
    import pyomo.environ as pyo

    # ---------- Shapes & guards ----------
    print('[TSA] Constructing and testing inputs for endogenous MILP')
    N = D.shape[0]
    assert D.shape == (N, N), "D must be (N,N)"
    assert S_by_day.shape[0] == N, "S_by_day must be (N)"
    T = N
    assert soc_ref.shape[0] == T, f"soc_ref length {soc_ref.shape[0]} != {T}"
    assert a.shape[0] == T, f"a length {a.shape[0]} != {T}"

    # ---------- Bias handling ----------
    def _rescale_biases(b: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Rescale b to [lo, hi]. If constant or invalid, return ones * hi."""
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.size != T or not np.all(np.isfinite(b)):
            return np.ones(T, dtype=float) * hi
        bmin, bmax = float(np.min(b)), float(np.max(b))
        if bmax <= bmin + 1e-12:
            return np.ones(T, dtype=float) * hi
        return lo + (b - bmin) * (hi - lo) / (bmax - bmin)

    if endogenous_biases is None:
        b = np.ones(T, dtype=float) * bias_minmax[1]  # uniform at ceiling
    else:
        b = _rescale_biases(endogenous_biases, bias_minmax[0], bias_minmax[1])

    b_avg = float(np.mean(b))
    if not np.isfinite(b_avg) or b_avg <= 0:
        b_avg = 1.0

    # mapping t -> (day, hour) and cyc ramp r[t]
    day_of_t = list(range(T))

    # pack constants
    S_dict = {(i): float(S_by_day[i]) for i in range(N)}
    D_dict = {(i, j): float(D[i, j]) for i in range(N) for j in range(N)}
    a_vec  = [float(x) for x in a]
    b_vec  = [float(x) for x in b]

    print('[TSA] Constructing pyomo model')

    # ---------- Model ----------
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.J = pyo.RangeSet(0, N - 1)
    m.T = pyo.RangeSet(0, T - 1)

    m.D = pyo.Param(m.I, m.J, initialize=D_dict, within=pyo.NonNegativeReals)

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)

    # ORDO constraints
    m.assign_once = pyo.Constraint(m.J, rule=lambda _m, j: sum(_m.x[i, j] for i in _m.I) == 1)
    m.link        = pyo.Constraint(m.I, m.J, rule=lambda _m, i, j: _m.x[i, j] <= _m.y[i])
    m.num_reps    = pyo.Constraint(rule=lambda _m: sum(_m.y[i] for i in _m.I) == k)
    m.used_rep    = pyo.Constraint(m.I, rule=lambda _m, i: sum(_m.x[i, j] for j in _m.J) >= _m.y[i])

    # surplus_hat[t]
    def surplus_hat_rule(_m, t):
        j = day_of_t[t]
        return sum(S_dict[(i)] * _m.x[i, j] for i in _m.I)
    m.surplus_hat = pyo.Expression(m.T, rule=surplus_hat_rule)

    # SoC bounds (crude but finite)
    UB_soc_raw = 1.1 * max(1e-9, float(np.max(np.abs(soc_ref))))  # guard tiny signals

    # SoC dynamics
    m.soc_raw   = pyo.Var(m.T, bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_raw_0 = pyo.Constraint(expr=m.soc_raw[0] == a_vec[0] * m.surplus_hat[0])
    def soc_raw_dyn_rule(_m, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return _m.soc_raw[t] == _m.soc_raw[t - 1] + a_vec[t] * _m.surplus_hat[t]
    m.soc_raw_dyn = pyo.Constraint(m.T, rule=soc_raw_dyn_rule)

    m.soc_end     = pyo.Var(bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_end_def = pyo.Constraint(expr=m.soc_end == m.soc_raw[T - 1])

    # L1 residuals
    m.socdiff = pyo.Var(m.T)  # free (can be +/-)
    # sensible upper bound for z: UB on diff <= UB_soc_raw + |ref|
    def _z_bounds(_m, t):
        return (0.0, UB_soc_raw + abs(float(soc_ref[t])))
    m.z = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=_z_bounds)

    # socdiff definition and absolute-value modeling via z
    m.soc_def = pyo.Constraint(m.T, rule=lambda _m, t: _m.socdiff[t] == soc_ref[t] - _m.soc_raw[t])
    m.z_pos   = pyo.Constraint(m.T, rule=lambda _m, t: _m.z[t] >=  _m.socdiff[t])
    m.z_neg   = pyo.Constraint(m.T, rule=lambda _m, t: _m.z[t] >= -_m.socdiff[t])

    # Objective (scaled to O(1)).
    # Keep lambda_soc meaning stable by dividing proxy scale by avg(bias).
    dist_scale = max(1e-12, float(np.mean(D)))
    prox_scale = max(1e-12, float(np.mean(np.abs(soc_ref)))) * b_avg

    def obj(_m):
        dist = (1 - lambda_soc) * (sum(_m.x[i, j] * _m.D[i, j] for i in _m.I for j in _m.J) / dist_scale)
        prox =      lambda_soc  * (sum(b_vec[t] * _m.z[t]       for t in _m.T)                          / prox_scale)
        return dist + prox

    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # ---------- Solve ----------
    if solver == "gurobi":
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.set_instance(m)

        opt.set_gurobi_param('MIPGap', MIPGap)
        if threads is not None: opt.set_gurobi_param('Threads', int(threads))
        opt.set_gurobi_param('LogToConsole', int(verbose))
        opt.set_gurobi_param('TimeLimit', int(timelimit))
        opt.set_gurobi_param('Presolve', 2)
        opt.set_gurobi_param('Aggregate', 2)
        if root_lp == "barrier":
            opt.set_gurobi_param('Method', 2)
            opt.set_gurobi_param('Crossover', 0)
        elif root_lp == "dual":
            opt.set_gurobi_param('Method', 1)

        res = opt.solve(tee=verbose)
    else:
        res = pyo.SolverFactory(solver).solve(m, tee=verbose, options={
            'MIPGap': MIPGap, 'TimeLimit': int(timelimit)
        })

    # ---------- Extract ----------
    selected_days = [i for i in range(N) if pyo.value(m.y[i]) > 0.5]
    assignments   = {j: max(range(N), key=lambda i: pyo.value(m.x[i, j])) for j in range(N)}

    return {
        "selected_days": selected_days,
        "assignments": assignments,
        "model": m,
        "results": res,
        "biases_used": b,        # echo the effective biases for logging/debug
        "bias_avg": b_avg,
    }


def milp_tsa_endogenous_with_bias_handling(
    *,
    D: np.ndarray,                    # (N x N) ORDO cost matrix
    k: int,
    soc_ref: np.ndarray,              # (T,) reference SoC
    S_by_day: np.ndarray,             # (N x 24) hourly surplus per candidate day  (treated here as (N,))
    a: np.ndarray,                    # (T,) frozen gains: eta_ch or 1/eta_dis
    lambda_soc: float = 0.5,
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 2000,
    verbose: bool = True,
    root_lp: str = "barrier",         # "dual" | "barrier"
    endogenous_biases: list | np.ndarray | None = None,
    bias_minmax: tuple[float, float] = (0.0, 1.0),  # rescale weights to this interval
    bias_zero_threshold: float = 0.25,               # below/eq this => remove proxy residuals
):
    """
    Minimize: (1-λ)*Σ_{i,j} D[i,j] x[i,j]  +  λ * Σ_{t∈T+} b_t * |soc_hat[t] - soc_ref[t]|
    with endogenous SoC built from the chosen representative-day sequence.

    T+ = { t | bias(t) > bias_zero_threshold } ∪ {0, T-1, argmax(soc_ref), argmin(soc_ref)}

    Notes
    -----
    - 'endogenous_biases' are per-timestep weights b_t. They are rescaled to [bias_min, bias_max].
      If None, b_t := bias_max for all t (i.e., uniform).
    - Proxy residual variables/constraints are ONLY created for t ∈ T+ to reduce model size.
    - The proxy scaling divides by avg(b_t | t ∈ T+) so lambda_soc remains comparable.
    """
    import numpy as np
    import pyomo.environ as pyo

    # ---------- Shapes & guards ----------
    print('[TSA] Constructing and testing inputs for endogenous MILP (with bias handling)')
    N = D.shape[0]
    assert D.shape == (N, N), "D must be (N,N)"
    assert S_by_day.shape[0] == N, "S_by_day must be (N)"
    T = N
    assert soc_ref.shape[0] == T, f"soc_ref length {soc_ref.shape[0]} != {T}"
    assert a.shape[0] == T, f"a length {a.shape[0]} != {T}"

    # ---------- Bias handling ----------
    def _rescale_biases(b: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Rescale b to [lo, hi]. If invalid or constant, return ones * hi."""
        b = np.asarray(b, dtype=float).reshape(-1)
        if b.size != T or not np.all(np.isfinite(b)):
            return np.ones(T, dtype=float) * hi
        bmin, bmax = float(np.min(b)), float(np.max(b))
        if bmax <= bmin + 1e-12:
            return np.ones(T, dtype=float) * hi
        return lo + (b - bmin) * (hi - lo) / (bmax - bmin)

    if endogenous_biases is None:
        b_full = np.ones(T, dtype=float) * bias_minmax[1]  # uniform at ceiling
    else:
        b_full = _rescale_biases(endogenous_biases, bias_minmax[0], bias_minmax[1])

    # Active timesteps for proxy term (strictly greater than threshold)
    active_mask = b_full > float(bias_zero_threshold)
    T_pos = [t for t in range(T) if active_mask[t]]

    # Ensure critical anchors are always present in proxy term
    i_max = int(np.argmax(soc_ref))
    i_min = int(np.argmin(soc_ref))
    for t_force in (0, T - 1, i_max, i_min):
        if t_force not in T_pos:
            T_pos.append(t_force)
    T_pos = sorted(set(T_pos))

    # Average active bias to keep lambda_soc comparable in the objective scaling
    b_pos = np.array([float(b_full[t]) for t in T_pos], dtype=float)
    b_avg = float(np.mean(b_pos)) if b_pos.size else 1.0
    if not np.isfinite(b_avg) or b_avg <= 0:
        b_avg = 1.0

    # ---------- Convenience packs ----------
    day_of_t = list(range(T))
    S_dict = {(i): float(S_by_day[i]) for i in range(N)}
    D_dict = {(i, j): float(D[i, j]) for i in range(N) for j in range(N)}
    a_vec  = [float(x) for x in a]
    b_dict = {int(t): float(b_full[t]) for t in range(T)}  # for potential debugging

    print(f'[TSA] Proxy active timesteps: {len(T_pos)}/{T} '
          f'({100.0*len(T_pos)/max(1,T):.1f}%), bias_avg_active={b_avg:.3f}')

    # ---------- Model ----------
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.J = pyo.RangeSet(0, N - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.Tz = pyo.Set(initialize=T_pos, ordered=True)  # only these timesteps have proxy residuals

    m.D = pyo.Param(m.I, m.J, initialize=D_dict, within=pyo.NonNegativeReals)

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)

    # ORDO constraints
    m.assign_once = pyo.Constraint(m.J, rule=lambda _m, j: sum(_m.x[i, j] for i in _m.I) == 1)
    m.link        = pyo.Constraint(m.I, m.J, rule=lambda _m, i, j: _m.x[i, j] <= _m.y[i])
    m.num_reps    = pyo.Constraint(rule=lambda _m: sum(_m.y[i] for i in _m.I) == k)
    m.used_rep    = pyo.Constraint(m.I, rule=lambda _m, i: sum(_m.x[i, j] for j in _m.J) >= _m.y[i])

    # surplus_hat[t]
    def surplus_hat_rule(_m, t):
        j = day_of_t[t]
        return sum(S_dict[(i)] * _m.x[i, j] for i in _m.I)
    m.surplus_hat = pyo.Expression(m.T, rule=surplus_hat_rule)

    # SoC bounds (crude but finite)
    UB_soc_raw = 1.1 * max(1e-9, float(np.max(np.abs(soc_ref))))  # guard tiny signals

    # SoC dynamics (over all T to keep the trajectory continuous)
    m.soc_raw   = pyo.Var(m.T, bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_raw_0 = pyo.Constraint(expr=m.soc_raw[0] == a_vec[0] * m.surplus_hat[0])

    def soc_raw_dyn_rule(_m, t):
        if t == m.T.first():
            return pyo.Constraint.Skip
        return _m.soc_raw[t] == _m.soc_raw[t - 1] + a_vec[t] * _m.surplus_hat[t]
    m.soc_raw_dyn = pyo.Constraint(m.T, rule=soc_raw_dyn_rule)

    m.soc_end     = pyo.Var(bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_end_def = pyo.Constraint(expr=m.soc_end == m.soc_raw[T - 1])

    # L1 residuals ONLY on active set Tz
    def _z_bounds(_m, t):
        return (0.0, UB_soc_raw + abs(float(soc_ref[t])))

    m.socdiff = pyo.Var(m.Tz)  # free
    m.z       = pyo.Var(m.Tz, domain=pyo.NonNegativeReals, bounds=_z_bounds)

    m.soc_def = pyo.Constraint(m.Tz, rule=lambda _m, t: _m.socdiff[t] == soc_ref[t] - _m.soc_raw[t])
    m.z_pos   = pyo.Constraint(m.Tz, rule=lambda _m, t: _m.z[t] >=  _m.socdiff[t])
    m.z_neg   = pyo.Constraint(m.Tz, rule=lambda _m, t: _m.z[t] >= -_m.socdiff[t])

    # Objective (scaled to O(1)); proxy only over Tz, scaled by avg active bias
    dist_scale = max(1e-12, float(np.mean(D)))
    prox_scale = max(1e-12, float(np.mean(np.abs(soc_ref)))) * b_avg

    # Need bias values aligned to m.Tz iteration order
    b_for_Tz = {t: float(b_full[t]) for t in T_pos}

    def obj(_m):
        dist = (1 - lambda_soc) * (sum(_m.x[i, j] * _m.D[i, j] for i in _m.I for j in _m.J) / dist_scale)
        prox = 0.0
        if len(T_pos):
            prox = lambda_soc * (sum(b_for_Tz[int(t)] * _m.z[t] for t in _m.Tz) / prox_scale)
        return dist + prox

    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    # ---------- Solve ----------
    if solver == "gurobi":
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.set_instance(m)

        opt.set_gurobi_param('MIPGap', MIPGap)
        if threads is not None: opt.set_gurobi_param('Threads', int(threads))
        opt.set_gurobi_param('LogToConsole', int(verbose))
        opt.set_gurobi_param('TimeLimit', int(timelimit))
        opt.set_gurobi_param('Presolve', 2)
        opt.set_gurobi_param('Aggregate', 2)
        if root_lp == "barrier":
            opt.set_gurobi_param('Method', 2)
            opt.set_gurobi_param('Crossover', 0)
        elif root_lp == "dual":
            opt.set_gurobi_param('Method', 1)

        res = opt.solve(tee=verbose)
    else:
        res = pyo.SolverFactory(solver).solve(m, tee=verbose, options={
            'MIPGap': MIPGap, 'TimeLimit': int(timelimit)
        })

    # ---------- Extract ----------
    selected_days = [i for i in range(N) if pyo.value(m.y[i]) > 0.5]
    assignments   = {j: max(range(N), key=lambda i: pyo.value(m.x[i, j])) for j in range(N)}

    return {
        "selected_days": selected_days,
        "assignments": assignments,
        "model": m,
        "results": res,
        "biases_used": b_full,          # full vector after rescaling
        "bias_avg_active": b_avg,       # avg over active T+
        "proxy_active_timesteps": T_pos # list of indices used in proxy term
    }

def apply_linear_soc(s_hat: np.ndarray, a: np.ndarray, cyc: bool = True) -> np.ndarray:
    """Apply the linearised SoC operator in O(T) without building L."""
    y = np.cumsum(a * s_hat)  # raw cumulative
    if not cyc:
        return y
    soc_end = float(y[-1])
    r = np.linspace(0.0, 1.0, len(s_hat))
    return y - r * soc_end

def build_a_and_soc_ref(reference_surplus: np.ndarray, eta_ch: float, eta_dis: float, cyc: bool = True):
    """Frozen gains a_t from the reference sign pattern + the corresponding reference SoC."""
    # a = np.where(reference_surplus >= 0.0, eta_ch, 1.0 / eta_dis).astype(float)
    a = np.where(reference_surplus >= 0.0, 1, 1.0).astype(float)
    soc_ref = apply_linear_soc(reference_surplus, a, cyc=cyc)
    return a, soc_ref

def solve_ordo_with_endogenous_soc(
    *,
    df_features,
    k,
    feature_weights,
    preferred_features,
    eta_ch, eta_dis,
    lambda_soc,
    reference_surplus,
    normalize="minmax_signed",
    solver="gurobi",
    MIPGap=0.01,
    verbose=True,
    surplus_by_day: np.ndarray | None = None,   
    use_endogenous_biases: bool = False
):
    print('[TSA] Building cost matrix')
    # 1) D matrix...
    D = _ordo_cost_matrix_from_features(
        df_features=df_features,
        feature_weights=feature_weights,
        preferred_features=preferred_features,
        normalize=normalize,
    )

    # 2) Get N, T
    N = len(df_features) 
    T = N 

    

    # --- Sanity checks ---
    assert surplus_by_day.shape[0] == N, \
        f"surplus_hourly_by_day shape {surplus_by_day.shape[0]} != ({N})"
    assert reference_surplus.shape[0] == T, \
        f"reference_surplus length {reference_surplus.shape[0]} != {T}"
    
    if use_endogenous_biases:
        endo_params = dict(
        # detection / smoothing
        normalize_mode="minmax",
        smooth_window=1,
        extrema_window=121,
        min_prominence=0.05,
        # proximity shape
        tau_decay=12,              # tighter halo
        # dominance of key points
        weight_global_extreme_boost=True, coef_global_extreme=3.0, tau_global_extreme=5.0,
        weight_endpoints=True,     coef_endpoints=2.0,
        priority_floor_multiplier=1.5,
        # keep only these two structural terms
        weight_extremes=True,      coef_extremes=0.75,
        weight_span=True,          coef_span=1.25, span_alpha=2.0,
        # turn off dispersive terms
        weight_tail_softmax=False,
        weight_ramp=False,
        weight_curvature=False,
        weight_rarity=False,
        # viz
        plot=True
    )

        endogenous_biases = generate_endogenous_biases(surplus_by_day.cumsum(),params=endo_params)
    else:
        endogenous_biases = []


    # Scale surplus to per-unit of a robust peak (e.g., 99.5th percentile)
    s_scale = float(np.percentile(np.abs(surplus_by_day), 99.5))
    s_scale = max(s_scale, 1.0)  # avoid sub-1 scales

    S_by_day_scaled = surplus_by_day / s_scale
    reference_surplus_scaled = reference_surplus / s_scale

    # Build a, soc_ref on scaled surplus (O(T), no dense matrices)
    a, soc_ref = build_a_and_soc_ref(reference_surplus_scaled, eta_ch, eta_dis, cyc=True)

    

    return milp_tsa_endogenous_with_bias_handling(
        D=D,
        k=k,
        soc_ref=soc_ref,
        S_by_day=S_by_day_scaled,
        a=a,
        lambda_soc=lambda_soc,
        solver=solver,           # <- pass through
        MIPGap=MIPGap,           # <- pass through
        threads=min(max_threads-6 if max_threads>8 else 1, 16),
        verbose=verbose,
        endogenous_biases = endogenous_biases
    )


def milp_tsa_endogenous_restricted_candidates(
    *,
    Dsub: np.ndarray,                 # (|C|, N) ORDO cost: cand row vs all days
    C_ids: list[int],                 # GLOBAL row ids of candidates, len = |C|
    k: int,
    soc_ref: np.ndarray,              # (T,)
    S_by_cand: np.ndarray,            # (|C|, 24) hourly surplus for each candidate day (aligned to C_ids)
    a: np.ndarray,                    # (T,) frozen gains from reference sign / efficiencies
    lambda_soc: float = 0.5,
    hours_per_day: int = 24,
    fix_reps: bool = False,           # if True: fixes all y[i]=1 (|C| must equal k)
    warm_start_rep_for_day: np.ndarray | None = None,   # length N, GLOBAL rep id per day
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 3600,
    verbose: bool = True,
    lp_method: str = "barrier",       # "barrier" or "dual"
):
    import numpy as np
    import pyomo.environ as pyo

    Ic, N = Dsub.shape
    assert Ic == len(C_ids), "Dsub rows must align with C_ids"
    if fix_reps:
        assert k == Ic, "fix_reps=True requires k == len(C_ids)"
    assert S_by_cand.shape[0] == Ic, "S_by_cand must be (|C|, ..."

    # time maps
    T = int(N)
    assert soc_ref.shape[0] == T and a.shape[0] == T, "soc_ref and a must be length N"
    # day_of_t  = [t // hours_per_day for t in range(T)]
    # hour_of_t = [t %  hours_per_day for t in range(T)]
    day_of_t = list(range(T))  
    if soc_ref.shape == (N,hours_per_day):
        raise NotImplementedError('Capability for N*hours_per_day removed and required reimplementation')

    # Build params keyed by GLOBAL candidate id
    row_of_global = {g: r for r, g in enumerate(C_ids)}
    D_map = {(int(i), j): float(Dsub[row_of_global[int(i)], j])
             for i in C_ids for j in range(N)}
    # S_map = {(int(i), h): float(S_by_cand[row_of_global[int(i)], h])
    #          for i in C_ids for h in range(hours_per_day)}
    S_map = {int(i): float(S_by_cand[row_of_global[int(i)]]) for i in C_ids}

    # ---------- model ----------
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=list(map(int, C_ids)), ordered=True)  # GLOBAL ids
    m.J = pyo.RangeSet(0, N - 1)
    m.T = pyo.RangeSet(0, T - 1)

    m.D = pyo.Param(m.I, m.J, initialize=D_map, within=pyo.NonNegativeReals)

    m.y = pyo.Var(m.I, domain=pyo.Binary)             # select reps among candidates
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)        # assign each day to a candidate

    # assignment & linking
    m.assign_once = pyo.Constraint(m.J, rule=lambda _m, j: sum(_m.x[i, j] for i in _m.I) == 1)
    m.link        = pyo.Constraint(m.I, m.J, rule=lambda _m, i, j: _m.x[i, j] <= _m.y[i])

    if fix_reps:
        for i in m.I:
            m.y[i].fix(1)
    else:
        m.num_reps = pyo.Constraint(expr=sum(m.y[i] for i in m.I) == k)

    # surplus_hat[t] from assignments
    def surplus_hat_rule(_m, t):
        j = day_of_t[t]
        # h = hour_of_t[t]
        # sum over GLOBAL candidates
        return sum(S_map[int(i)] * _m.x[i, j] for i in _m.I)
    m.surplus_hat = pyo.Expression(m.T, rule=surplus_hat_rule)

    # SoC dynamics (same as basic endogenous form, just with restricted x)
    # Smax = float(np.max(np.abs(S_by_cand))) if Ic > 0 else 1.0
    # sum_abs_a = float(np.sum(np.abs(a)))
    # UB_soc_raw = max(1.0, Smax * sum_abs_a)
    UB_soc_raw = 1.1*max(np.abs(soc_ref)) #given the target of matching signals, we can set the max ref as the UB of the clustered signal

    m.soc_raw = pyo.Var(m.T, bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_raw_0 = pyo.Constraint(expr=m.soc_raw[0] == float(a[0]) * m.surplus_hat[0])
    def soc_raw_dyn(_m, t):
        if t == m.T.first(): return pyo.Constraint.Skip
        return _m.soc_raw[t] == _m.soc_raw[t-1] + float(a[t]) * _m.surplus_hat[t]
    m.soc_raw_dyn = pyo.Constraint(m.T, rule=soc_raw_dyn)

    m.soc_end = pyo.Var(bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_end_def = pyo.Constraint(expr=m.soc_end == m.soc_raw[T-1])

    # cyclical correction already baked into soc_ref (since you build a with cyc=True),
    # so no additional r[t]*soc_end term here.

    m.socdiff = pyo.Var(m.T)  # free
    m.z = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=lambda _m, t: (0.0, UB_soc_raw + abs(float(soc_ref[t]))))
    m.soc_def = pyo.Constraint(m.T, rule=lambda _m, t: _m.socdiff[t] == soc_ref[t] - _m.soc_raw[t]) #DANO, i flipped this, may need to flip back but shouldnt matter
    m.z_pos   = pyo.Constraint(m.T, rule=lambda _m, t: _m.z[t] >=  _m.socdiff[t])
    m.z_neg   = pyo.Constraint(m.T, rule=lambda _m, t: _m.z[t] >= -_m.socdiff[t])

    # objective (scaled)
    import numpy as _np
    dist_scale = max(1e-12, float(_np.mean(Dsub)))
    prox_scale = max(1e-12, float(_np.mean(_np.abs(soc_ref))))
    m.obj = pyo.Objective(
        expr=(1 - lambda_soc) * (sum(m.x[i, j] * m.D[i, j] for i in m.I for j in m.J) / dist_scale)
           +      lambda_soc  * (sum(m.z[t]                for t in m.T)              / prox_scale),
        sense=pyo.minimize
    )

    # ---------- solve (with optional warm start) ----------
    if solver == "gurobi":
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.set_instance(m)

        if warm_start_rep_for_day is not None:
            gv = opt._pyomo_var_to_solver_var_map
            Cset = set(int(i) for i in C_ids)
            used = set()
            for j in range(N):
                g = int(warm_start_rep_for_day[j])
                if g in Cset:
                    gv[m.x[g, j]].Start = 1.0
                    used.add(g)
            if not fix_reps:
                for i in m.I:
                    gv[m.y[i]].Start = 1.0 if int(i) in used else 0.0

        opt.set_gurobi_param('MIPGap', MIPGap)
        if threads is not None: opt.set_gurobi_param('Threads', int(threads))
        opt.set_gurobi_param('LogToConsole', int(verbose))
        opt.set_gurobi_param('TimeLimit', int(timelimit))
        opt.set_gurobi_param('Presolve', 2)
        opt.set_gurobi_param('Aggregate', 2)
        if lp_method == "barrier":
            opt.set_gurobi_param('Method', 2)
            opt.set_gurobi_param('Crossover', 0)
        elif lp_method == "dual":
            opt.set_gurobi_param('Method', 1)

        res = opt.solve(tee=verbose)
    else:
        res = pyo.SolverFactory(solver).solve(m, tee=verbose, options={
            'MIPGap': MIPGap, 'TimeLimit': int(timelimit)
        })

    # ---------- extract in GLOBAL coords ----------
    if fix_reps:
        selected_days = list(map(int, C_ids))
    else:
        selected_days = [int(i) for i in m.I if pyo.value(m.y[i]) > 0.5]

    assignments = {}
    for j in m.J:
        # prefer exact 1s, fall back to argmax
        chosen = [int(i) for i in m.I if (pyo.value(m.x[i, j]) or 0.0) > 0.5]
        if chosen:
            assignments[int(j)] = chosen[0]
        else:
            best_i, best_v = None, -1.0
            for i in m.I:
                v = pyo.value(m.x[i, j]) or 0.0
                if v > best_v:
                    best_v, best_i = v, int(i)
            assignments[int(j)] = best_i

    return {"selected_days": selected_days, "assignments": assignments, "model": m, "results": res}

def solve_ordo_with_endogenous_soc_restricted(
    *,
    df_features: pd.DataFrame,         # DAILY rows (N)
    k: int,
    feature_weights: dict[str, float],
    preferred_features: list[str] | None,
    candidates: list[int],             # GLOBAL daily row ids (from precluster_to_row_indices)
    fix_reps: bool,                    # True if |C| == k (pure reassignment to fixed reps)
    warm_start_rep_for_day: np.ndarray | None,  # length N, GLOBAL rep id per day
    eta_ch: float,
    eta_dis: float,
    lambda_soc: float,
    surplus_by_day: np.ndarray, # (N, 24) hourly surplus for EVERY day
    normalize: str = "minmax_signed",
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 3600,
    verbose: bool = True,
    lp_method: str = "barrier",
):
    import numpy as np

    # 1) ORDO distance over all days (daily / hourly24 features supported)
    D_full = _ordo_cost_matrix_from_features(
        df_features=df_features,
        feature_weights=feature_weights,
        preferred_features=preferred_features,
        normalize=normalize,
    )  # (N, N)

    N = D_full.shape[0]
    hours_per_day = 24

    # 2) Slice rows to candidate set
    C_ids = list(map(int, candidates))
    Dsub = D_full[np.ix_(C_ids, np.arange(N))]  # (|C|, N)

    # 3) Build SoC pieces (scale surplus for numerics like your earlier path)
    
    assert surplus_by_day.shape[0] == N, "surplus_hourly_by_day must be (N, ...)"
    s_scale = float(np.percentile(np.abs(surplus_by_day), 99.5))
    s_scale = max(s_scale, 1.0)
    S_day_scaled = surplus_by_day / s_scale
    if surplus_by_day.shape == (N, hours_per_day):
        S_by_cand = S_day_scaled[C_ids, :]
    else:
        S_by_cand = S_day_scaled[C_ids]                      # (|C|, ...)

    reference_surplus = surplus_by_day.reshape(-1)   # (T,)
    reference_surplus_scaled = reference_surplus / s_scale
    a, soc_ref = build_a_and_soc_ref(reference_surplus_scaled, eta_ch, eta_dis, cyc=True)

    # 4) Solve restricted endogenous MILP
    return milp_tsa_endogenous_restricted_candidates(
        Dsub=Dsub,
        C_ids=C_ids,
        k=k,
        soc_ref=soc_ref,
        S_by_cand=S_by_cand,
        a=a,
        lambda_soc=lambda_soc,
        hours_per_day=hours_per_day,
        fix_reps=fix_reps,
        warm_start_rep_for_day=warm_start_rep_for_day,
        solver=solver,
        MIPGap=MIPGap,
        threads=threads,
        timelimit=timelimit,
        verbose=verbose,
        lp_method=lp_method,
    )
