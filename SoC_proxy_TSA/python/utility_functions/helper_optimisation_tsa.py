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

def build_feature_design_matrices_for_rr(
    df_features: pd.DataFrame,
    candidates: list[int],
    preferred_features: list[str] | None = None,
    normalize: str = "minmax_signed",
    feature_weights: dict[str, float] | None = None,
):
    """
    Like build_feature_design_matrices, but also returns:
      - X_norm  : (N, D) normalized UNweighted (for DC histograms)
      - group_slices : {base: (start, stop)} slices into the concatenated columns
      - group_names  : list[str] in the concatenation order (parallel to group_slices)
    """
    if feature_weights is None:
        feature_weights = {}

    mode, groups = group_feature_columns(df_features, preferred_features)  # same as ORDO

    X_blocks = []
    A_blocks = []
    Xnorm_blocks = []        # unweighted, normalized
    scale_params = []
    colnames = []
    group_slices = {}
    group_names = []

    def _scale_signed(col: np.ndarray):
        cmin = np.nanmin(col); cmax = np.nanmax(col)
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax == cmin:
            return np.zeros_like(col, dtype=float), (0.0, 1.0)
        z = (col - cmin) / (cmax - cmin); z = 2.0 * z - 1.0
        return z.astype(float), (cmin, cmax)

    pos = 0
    for base, cols in groups.items():
        block = df_features[cols].to_numpy(dtype=float, copy=False)
        nb = block.shape[1]

        norm_cols = np.empty_like(block, dtype=float)
        mins_maxs = []
        for j in range(nb):
            norm_cols[:, j], mm = _scale_signed(block[:, j])
            mins_maxs.append(mm)

        Xnorm_blocks.append(norm_cols)           # (N, nb), unweighted
        X_blocks.append(norm_cols)               # we’ll weight later
        A_blocks.append(norm_cols[candidates, :])

        scale_params.extend(mins_maxs)
        colnames.extend(cols)

        group_slices[base] = (pos, pos + nb)
        group_names.append(base)
        pos += nb

    X_norm = np.concatenate(Xnorm_blocks, axis=1)     # (N, D)
    X_w    = np.concatenate(X_blocks, axis=1)         # (N, D)
    A_norm = np.concatenate(A_blocks, axis=1)         # (C, D)

    # column weights via √(feature weight)
    wcols = []
    for base, cols in groups.items():
        w = float(feature_weights.get(base, 1.0))
        if w < 0: w = 0.0
        factor = np.sqrt(w) if w > 0 else 0.0
        wcols.extend([factor] * len(cols))
    wcols = np.asarray(wcols, dtype=float)

    X_fit = X_w * wcols[None, :]
    A_fit = A_norm * wcols[None, :]

    return X_fit, A_fit, A_norm, X_norm, scale_params, colnames, group_slices, group_names

def build_duration_curve_quadratic_terms(
    *,
    X_norm: np.ndarray,            # (N, D) normalized (no weights)
    A_norm: np.ndarray,            # (C, D) normalized (no weights)
    group_slices: dict[str, tuple],
    group_weights: dict[str, float] | None = None,
    nbins: int = 16,
):
    """
    Construct DC penalty terms:
      R (C x C) and s (C),
    so that  DC_obj = (Σ_j v_j)^T R (Σ_j v_j) - 2 s^T (Σ_j v_j) + const
    where v_j ∈ R^C are the day-simplex weights and w = Σ_j v_j.

    We use simple per-group histograms over [-1,1] in `nbins` equal bins,
    built on normalized values. Counts are *raw* (not normalized to frequency);
    trade-off vs TS handled by a global scalar weight later.
    """
    C, D = A_norm.shape
    N    = X_norm.shape[0]
    edges = np.linspace(-1.0, 1.0, nbins + 1)

    if group_weights is None:
        group_weights = {g: 1.0 for g in group_slices.keys()}

    R = np.zeros((C, C), dtype=float)
    s = np.zeros(C, dtype=float)

    for g, (a, b) in group_slices.items():
        w_g = float(group_weights.get(g, 1.0))
        # original counts across all days for this group
        xg = X_norm[:, a:b].ravel()                         # length N * cols_in_group
        y_counts, _ = np.histogram(xg, bins=edges)          # (B,)

        # per-rep histogram for this group: H_g (B x C)
        cols_in_g = b - a
        H = np.zeros((nbins, C), dtype=float)
        for i in range(C):
            xi = A_norm[i, a:b]                             # (cols_in_g,)
            hi, _ = np.histogram(xi, bins=edges)
            # each representative day contributes once per column in group
            H[:, i] = hi

        # accumulate: R += w_g * H^T H ; s += w_g * H^T y
        R += w_g * (H.T @ H)                                # (C x C)
        s += w_g * (H.T @ y_counts)                         # (C,)
    return R, s

def rr_soft_fit_weights_batched(
    *,
    X_fit: np.ndarray,            # (N, D) normalized + √weights
    A_fit: np.ndarray,            # (C, D) normalized + √weights
    R_dc: np.ndarray,             # (C, C) from build_duration_curve_quadratic_terms
    s_dc: np.ndarray,             # (C,)   from build_duration_curve_quadratic_terms
    ts_weight: float = 1.0,
    dc_weight: float = 1.0,
    solver: str = "gurobi",
    verbose: bool = False,
):
    """
    One convex QP for all days:
      min_v  ts_weight * [Σ_j v_j^T Q v_j - 2 Σ_j c_j^T v_j]
             + dc_weight * [(Σ_j v_j)^T R (Σ_j v_j) - 2 s^T (Σ_j v_j)]
      s.t.   v_{ij} ≥ 0,  Σ_i v_{ij} = 1.

    where  Q = A_fit A_fit^T   (C x C)
           c_j = A_fit x_j     (C)

    Returns
    -------
    V : (C, N) with columns v_j ; also returns w := Σ_j v_j for convenience.
    """
    N, D = X_fit.shape
    C, D2 = A_fit.shape
    assert D == D2

    # Precompute TS matrices
    Q = A_fit @ A_fit.T            # (C x C), PSD
    Cmat = A_fit @ X_fit.T         # (C x N), columns c_j

    # Build QP
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, C - 1)
    m.J = pyo.RangeSet(0, N - 1)

    # Variables
    m.v = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)  # v[i,j]
    m.w = pyo.Var(m.I, domain=pyo.NonNegativeReals)       # w[i] = Σ_j v[i,j]

    # Constraints: simplex per day
    def sum_to_one(_m, j):
        return sum(_m.v[i, j] for i in _m.I) == 1.0
    m.simplex = pyo.Constraint(m.J, rule=sum_to_one)

    # Link w = Σ_j v[:,j]
    def link_w(_m, i):
        return _m.w[i] == sum(_m.v[i, j] for j in _m.J)
    m.link = pyo.Constraint(m.I, rule=link_w)

    # Objective pieces
    def obj_rule(_m):
        # TS part: Σ_j (v_j^T Q v_j - 2 c_j^T v_j)
        ts_quad = sum(_m.v[i, j] * Q[i, k] * _m.v[k, j]
                      for j in _m.J for i in _m.I for k in _m.I)
        ts_lin  = -2.0 * sum(Cmat[i, j] * _m.v[i, j] for j in _m.J for i in _m.I)

        # DC part: (w^T R w) - 2 s^T w
        dc_quad = sum(_m.w[i] * R_dc[i, k] * _m.w[k] for i in _m.I for k in _m.I)
        dc_lin  = -2.0 * sum(s_dc[i] * _m.w[i] for i in _m.I)

        return ts_weight * (ts_quad + ts_lin) + dc_weight * (dc_quad + dc_lin)

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    opt = pyo.SolverFactory(solver)
    opts = {}
    if solver.lower() == "gurobi":
        # Continuous QP → these parameters are fine
        opts = {"OutputFlag": int(verbose)}
    opt.solve(m, tee=verbose, options=opts)

    # Extract
    V = np.zeros((C, N), dtype=float)
    for j in range(N):
        for i in range(C):
            V[i, j] = pyo.value(m.v[i, j])
    w = np.array([pyo.value(m.w[i]) for i in range(C)], dtype=float)
    return V, w

def rr_soft_fit_weights(
    X_fit: np.ndarray,   # (N, D)  normalized + weighted
    A_fit: np.ndarray,   # (C, D)  normalized + weighted
    solver: str = "gurobi",
    mipgap: float = 0.01,
    verbose: bool = False,
):
    """
    Solve, for each day j,   min_{v >=0, 1^Tv=1} || A_fit^T v - x_fit ||_2^2
    Returns V of shape (C, N), columns are v_j.

    Notes
    -----
    - This builds a small convex QP per day. Simple and robust prototype.
    - You can later vectorize or warm-start if needed.
    """
    N, D = X_fit.shape
    C, D2 = A_fit.shape
    assert D == D2

    # Precompute shared Q = A A^T (C x C). It’s PSD → convex QP.
    Q = A_fit @ A_fit.T  # (C, C)

    V = np.zeros((C, N), dtype=float)

    for j in range(N):
        x = X_fit[j, :]                                  # (D,)
        c_vec = A_fit @ x                                # (C,)

        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(0, C - 1)
        m.v = pyo.Var(m.I, domain=pyo.NonNegativeReals)

        # sum v_i = 1
        m.sum_to_one = pyo.Constraint(expr=sum(m.v[i] for i in m.I) == 1.0)

        # objective: v^T Q v - 2 c^T v   (constant x^T x dropped)
        def obj_rule(_):
            quad = sum(m.v[i] * Q[i, k] * m.v[k] for i in m.I for k in m.I)
            lin  = -2.0 * sum(c_vec[i] * m.v[i] for i in m.I)
            return quad + lin

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        opt = pyo.SolverFactory(solver)
        opts = {}
        if solver.lower() == "gurobi":
            # quiet by default; toggle with verbose
            opts = {
                "MIPGap": mipgap,        # harmless here; QP is continuous
                "OutputFlag": int(verbose),
            }
        res = opt.solve(m, tee=verbose, options=opts)

        V[:, j] = [pyo.value(m.v[i]) for i in m.I]

    return V

def rr_soft_reconstruct(
    V: np.ndarray,           # (C, N)
    A_norm: np.ndarray,      # (C, D)  normalized (no weights)
    scale_params: list[tuple[float, float]],
    column_order: list[str],
) -> pd.DataFrame:
    """
    Reconstruct synthetic time series in ORIGINAL units, same columns as df_features subset.
    """
    # predicted in normalized space (no weights)
    Xhat_norm = V.T @ A_norm    # (N, D)

    # invert normalization per column
    def inv_col(z, cmin, cmax):
        # z in [-1,1] -> [cmin, cmax]
        return (0.5 * (z + 1.0)) * (cmax - cmin) + cmin

    cols = []
    for j, (cmin, cmax) in enumerate(scale_params):
        cols.append(inv_col(Xhat_norm[:, j], cmin, cmax))
    Xhat = np.column_stack(cols)

    df_hat = pd.DataFrame(Xhat, columns=column_order, index=None)
    return df_hat

def rr_hard_assign(
    D: np.ndarray,            # full (N x N) cost matrix in ORDO metric
    candidates: list[int],    # indices of candidate representative days
    solver: str = "gurobi",
    mipgap: float = 0.01,
    verbose: bool = False,
):
    """
    Restricted assignment MILP:
      min sum_{i in C, j in N} D[i,j] x_ij
      s.t. sum_{i in C} x_ij = 1,  x_ij ∈ {0,1}

    Returns
    -------
    rep_for_day : np.ndarray shape (N,), each value is the chosen candidate index for that day.
    """
    N = D.shape[0]
    Cset = sorted(set(int(i) for i in candidates))
    Ic = range(len(Cset))
    # map local index -> global candidate day id
    cand_idx = {ii: Cset[ii] for ii in Ic}

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(Cset) - 1)
    m.J = pyo.RangeSet(0, N - 1)
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)

    # each day assigned to exactly one candidate
    def one_rep_per_day(_m, j):
        return sum(_m.x[i, j] for i in _m.I) == 1
    m.assign = pyo.Constraint(m.J, rule=one_rep_per_day)

    # objective
    def obj_rule(_m):
        return sum(D[cand_idx[i], j] * _m.x[i, j] for i in _m.I for j in _m.J)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    opt = pyo.SolverFactory(solver)
    opts = {}
    if solver.lower() == "gurobi":
        opts = {"MIPGap": mipgap, "OutputFlag": int(verbose)}
    res = opt.solve(m, tee=verbose, options=opts)

    # extract assignment
    rep_for_day = np.zeros(N, dtype=int)
    for j in range(N):
        chosen = None
        for i in Ic:
            if pyo.value(m.x[i, j]) > 0.5:
                chosen = cand_idx[i]
                break
        rep_for_day[j] = chosen if chosen is not None else cand_idx[0]
    return rep_for_day

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

def build_linear_soc_operator_from_reference(
    surplus_ref: np.ndarray,              # shape (T,)
    eta_ch: float,
    eta_dis: float,
    enforce_cyclical: bool = True
):
    """
    Build a linear operator L (T x T) such that soc_lin = L @ surplus_hat
    when we *freeze the charge/discharge gain per time-step* from a reference surplus.

    We define gains a_t = eta_ch if surplus_ref[t] >= 0 else 1/eta_dis.
    Then soc_raw[t] = sum_{tau<=t} a_tau * surplus_hat[tau].

    If enforce_cyclical:
        we subtract a linear ramp so soc_lin[-1] == soc_lin[0] (0).
        That ramp is also linear in surplus_hat.
    """
    
    T = len(surplus_ref)

    raise Exception(f'This function builds several TxT matrices. Given that T = {T}, this will crash most PCs with memory requirements of >.1TB. Use the dynamic pyomo formulation instead.')

    # a = np.where(surplus_ref >= 0.0, eta_ch, 1.0/eta_dis).astype(float)

    # # Cumulative-sum operator with per-step gain: (lower-triangular)
    # # (soc_raw = C @ (a ⊙ surplus_hat))
    # C = np.tril(np.ones((T, T), dtype=float))
    # A = np.diag(a)
    # CA = C @ A  # shape (T, T)

    # if not enforce_cyclical:
    #     return CA

    # # Enforce cyclical by subtracting a ramp that removes final offset.
    # # soc_lin = CA @ s - r @ (e^T @ CA @ s), where
    # # e^T selects the final row, and r = linspace(0,1,T)
    # eT = np.zeros((1, T), dtype=float)
    # eT[0, -1] = 1.0
    # r  = np.linspace(0.0, 1.0, T).reshape(T, 1)
    # # For any s: soc_end = eT @ CA @ s  (scalar)
    # # soc_lin  = CA @ s - r * soc_end
    # # => L = CA - r @ (eT @ CA)
    # L = CA - r @ (eT @ CA)
    # return L

def build_soc_coeff_tensor_for_assignments(
    *,
    surplus_hourly_by_day: np.ndarray,  # shape (N_days, 24) in row order of df_daily
    L: np.ndarray,                      # (T x T) linear SoC operator from build_linear_soc_operator_from_reference
    hours_per_day: int = 24
):
    """
    Return:
        H  : np.ndarray with shape (T, N_days, N_days)
             soc_hat[t] = sum_{i,j} H[t, i, j] * x[i, j]
        T  = N_days * hours_per_day
    Logic:
        surplus_hat at hour (j,h) = sum_i surplus[i,h] * x[i,j]
        soc_hat = L @ surplus_hat  -> linear in x.
    """
    N = surplus_hourly_by_day.shape[0]
    T = N * hours_per_day
    assert L.shape == (T, T)

    # Flatten surplus (by day-hour) for each candidate day i into a T-vector basis B_i
    # where B_i[tau] = surplus[i, h(tau)] if tau belongs to some day j; BUT assignment x[i,j]
    # picks day j's hours from candidate i. So we need a per-(i,j) vector whose nonzeros are
    # the 24 hours of day j filled with the 24 hourly values from candidate i.
    H = np.zeros((T, N, N), dtype=float)

    for i in range(N):            # representative day index
        # template block for one day (24-length) from candidate i
        block = surplus_hourly_by_day[i, :]  # (24,)
        for j in range(N):        # assigned day index
            # Place this block into positions of day j in the full T-vector
            s_ij = np.zeros(T, dtype=float)
            pos0 = j * hours_per_day
            s_ij[pos0: pos0 + hours_per_day] = block
            # Map to SoC with L: contributes L @ s_ij scaled by x[i,j]
            H[:, i, j] = L @ s_ij

    return H

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
    hours_per_day: int = 24,
    enforce_cyclical: bool = True,
    solver: str = "gurobi",
    MIPGap: float = 0.01,
    threads: int | None = 12,
    timelimit: int = 1800,
    verbose: bool = True,
    root_lp: str = "auto",            # "auto" | "dual" | "barrier"
):
    """
    Minimize: (1-λ)*sum_{i,j} D[i,j] x[i,j] + λ * sum_t |soc_hat[t] - soc_ref[t]|
    with endogenous SoC built from chosen representative-day sequence.
    """
    import numpy as np
    import pyomo.environ as pyo

    print('[TSA] Constructing and testing inputs for endogenous MILP')
    # ---------- Shapes & guards ----------
    N = D.shape[0]
    assert D.shape == (N, N), "D must be (N,N)"
    assert S_by_day.shape == (N, hours_per_day), f"S_by_day must be (N,{hours_per_day})"
    T = N * hours_per_day
    assert soc_ref.shape[0] == T, f"soc_ref length {soc_ref.shape[0]} != {T}"
    assert a.shape[0] == T, f"a length {a.shape[0]} != {T}"

    # mapping t -> (day, hour) and cyc ramp r[t]
    day_of_t  = [t // hours_per_day for t in range(T)]
    hour_of_t = [t %  hours_per_day for t in range(T)]
    r = np.linspace(0.0, 1.0, T) if enforce_cyclical else np.zeros(T)

    # pack constants
    S_dict = {(i, h): float(S_by_day[i, h]) for i in range(N) for h in range(hours_per_day)}
    D_dict = {(i, j): float(D[i, j]) for i in range(N) for j in range(N)}
    a_vec  = [float(x) for x in a]
    soc_ref_vec = [float(x) for x in soc_ref]
    r_vec  = [float(x) for x in r]

    print('[TSA] Constructing pyomo model')

    # ---------- Model ----------
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.J = pyo.RangeSet(0, N - 1)
    m.T = pyo.RangeSet(0, T - 1)

    m.D = pyo.Param(m.I, m.J, initialize=D_dict, within=pyo.NonNegativeReals)

    m.y = pyo.Var(m.I, domain=pyo.Binary)
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)

    #branching prioritisation to help solutions converge
    m.branchpri = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    for i in m.I:
        m.branchpri[m.y[i]] = 100  # prioritize committing to representatives early

    # ORDO constraints
    m.assign_once = pyo.Constraint(m.J, rule=lambda _m, j: sum(_m.x[i, j] for i in _m.I) == 1)
    m.link        = pyo.Constraint(m.I, m.J, rule=lambda _m, i, j: _m.x[i, j] <= _m.y[i])
    m.num_reps    = pyo.Constraint(rule=lambda _m: sum(_m.y[i] for i in _m.I) == k)
    m.used_rep    = pyo.Constraint(m.I, rule=lambda _m, i: sum(_m.x[i, j] for j in _m.J) >= _m.y[i])

    # surplus_hat[t]
    def surplus_hat_rule(_m, t):
        j = day_of_t[t]
        h = hour_of_t[t]
        return sum(S_dict[(i, h)] * _m.x[i, j] for i in _m.I)
    m.surplus_hat = pyo.Expression(m.T, rule=surplus_hat_rule)

    # SoC bounds (crude but finite)
    Smax = float(np.max(np.abs(S_by_day)))
    sum_abs_a = float(np.sum(np.abs(a)))
    UB_soc_raw = max(1.0, Smax * sum_abs_a)   # much tighter than a_max * Smax * T
    UB_z = 2.0 * UB_soc_raw

    # SoC dynamics
    m.soc_raw = pyo.Var(m.T, bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_raw_0 = pyo.Constraint(expr=m.soc_raw[0] == a_vec[0] * m.surplus_hat[0])
    def soc_raw_dyn_rule(_m, t):
        if t == 0: return pyo.Constraint.Skip
        return _m.soc_raw[t] == _m.soc_raw[t - 1] + a_vec[t] * _m.surplus_hat[t]
    m.soc_raw_dyn = pyo.Constraint(m.T, rule=soc_raw_dyn_rule)

    m.soc_end = pyo.Var(bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_end_def = pyo.Constraint(expr=m.soc_end == m.soc_raw[T - 1])

    m.soc = pyo.Var(m.T, bounds=(-UB_soc_raw, UB_soc_raw))
    m.soc_def = pyo.Constraint(m.T, rule=lambda _m, t: _m.soc[t] == _m.soc_raw[t] - r_vec[t] * _m.soc_end)

    # L1 residuals
    m.socdiff = pyo.Var(m.T)  # free
    m.z       = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, UB_z))
    m.soc_def2 = pyo.Constraint(m.T, rule=lambda _m, t: _m.socdiff[t] == _m.soc[t] - soc_ref_vec[t])
    m.z_pos    = pyo.Constraint(m.T, rule=lambda _m, t: _m.z[t] >=  _m.socdiff[t])
    m.z_neg    = pyo.Constraint(m.T, rule=lambda _m, t: _m.z[t] >= -_m.socdiff[t])

    # Objective (scaled to O(1))
    dist_scale = max(1e-12, float(np.mean(D)))
    prox_scale = max(1e-12, float(np.mean(np.abs(soc_ref))))
    def obj(_m):
        dist = (1 - lambda_soc) * (sum(_m.x[i, j] * _m.D[i, j] for i in _m.I for j in _m.J) / dist_scale)
        prox =      lambda_soc  * (sum(_m.z[t]               for t in _m.T)               / prox_scale)
        return dist + prox
    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

    
    # ---------- Solve ----------
    opt = pyo.SolverFactory(solver)
    solve_opts = {
        'MIPGap': MIPGap,
        'LogToConsole': int(verbose),
        'TimeLimit': int(timelimit),
        'Presolve': 2,
        'Method': 3,
        'MIPFocus': 3,
        'Cuts': 2,
        'CutPasses': 2
    }
    if threads is not None:
        solve_opts['Threads'] = int(threads)

    # optional root LP strategy
    root_lp = (root_lp or "auto").lower()
    if root_lp == "dual":
        solve_opts['Method'] = 1                    # dual simplex at root
    elif root_lp == "barrier":
        solve_opts['Method'] = 2                    # barrier at root
        solve_opts['Crossover'] = 0                 # only when Method=2
        solve_opts['BarHomogeneous'] = 1            # mild stabilization

    print('[TSA] Solving with parameters {solve_opts}')

    res = opt.solve(m, tee=verbose, options=solve_opts)

    # ---------- Extract ----------
    selected_days = [i for i in range(N) if pyo.value(m.y[i]) > 0.5]
    assignments = {j: max(range(N), key=lambda i: pyo.value(m.x[i, j])) for j in range(N)}

    return {"selected_days": selected_days, "assignments": assignments, "model": m, "results": res}

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
    a = np.where(reference_surplus >= 0.0, eta_ch, 1.0 / eta_dis).astype(float)
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
    surplus_hourly_by_day: np.ndarray | None = None,   # NEW
    surplus_columns_24h: list[str] | None = None,      # legacy option
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
    hours_per_day = 24 
    T = N * hours_per_day

    # 3) Hourly surplus by day: use cache if provided, else column list
    if surplus_hourly_by_day is None:
        assert surplus_columns_24h is not None, "Provide either surplus_hourly_by_day or surplus_columns_24h."
        surplus_hourly_by_day = df_features[surplus_columns_24h].to_numpy(dtype=float)
    

    #it is possible to precompute the linear operators and a tensor. But given the scale of model, this will crash most computers and is not recommended. Memory (RAM) requirements scale cubically and exceed 1TB for a 5yr model.
    # print('[TSA] Building linear operators for endogenous soc proxy')
    # # 4) Linear operator & H tensor
    # L = build_linear_soc_operator_from_reference(reference_surplus, eta_ch, eta_dis, enforce_cyclical=True)
    # print('[TSA] Building tensor for endogenous soc proxy assignments')
    # H = build_soc_coeff_tensor_for_assignments(surplus_hourly_by_day=surplus_hourly_by_day, L=L)
    # return milp_tsa_endogenous(
    #     D=D, k=k,
    #     soc_ref=L @ reference_surplus,
    #     H=H, lambda_soc=lambda_soc,
    #     solver=solver, MIPGap=MIPGap, verbose=verbose
    # )

    # --- Sanity checks ---
    assert surplus_hourly_by_day.shape == (N, hours_per_day), \
        f"surplus_hourly_by_day shape {surplus_hourly_by_day.shape} != ({N}, {hours_per_day})"
    assert reference_surplus.shape[0] == T, \
        f"reference_surplus length {reference_surplus.shape[0]} != {T}"

    CYCLICAL_CONDITION = True 

    # Scale surplus to per-unit of a robust peak (e.g., 99.5th percentile)
    s_scale = float(np.percentile(np.abs(surplus_hourly_by_day), 99.5))
    s_scale = max(s_scale, 1.0)  # avoid sub-1 scales

    S_by_day_scaled = surplus_hourly_by_day / s_scale
    reference_surplus_scaled = reference_surplus / s_scale

    # Build a, soc_ref on scaled surplus (O(T), no dense matrices)
    a, soc_ref = build_a_and_soc_ref(reference_surplus_scaled, eta_ch, eta_dis, cyc=True)



    return milp_tsa_endogenous_basic(
        D=D,
        k=k,
        soc_ref=soc_ref,
        S_by_day=S_by_day_scaled,
        a=a,
        lambda_soc=lambda_soc,
        hours_per_day=hours_per_day,
        enforce_cyclical=CYCLICAL_CONDITION,
        solver=solver,           # <- pass through
        MIPGap=MIPGap,           # <- pass through
        threads=min(max_threads-6 if max_threads>8 else 1, 16),
        verbose=verbose,
    )

