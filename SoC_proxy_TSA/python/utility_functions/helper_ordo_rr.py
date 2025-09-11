
# utility_functions/helper_ordo_rr.py
from __future__ import annotations


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# Reuse existing utilities to avoid duplication.
from utility_functions.helper_optimisation_tsa import (
    distance_matrix as build_distance_matrix,
    milp_tsa,
    save_milp_result_to_cluster_map,
)

# If you prefer, import these only when needed to keep import times low.
from utility_functions.helper_timeseries_tools import calliope_ts_to_pandas
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy

"""
Skeleton implementation of ORDO / RR-style optimisation, designed to slot
alongside or replace the existing MILP TSA workflow.

Key ideas
---------
- Supports *pre-clustered* mode (candidate reps fixed by a cluster map) and
  *unclustered* mode (select reps endogenously, like milp_tsa).
- Supports three SoC proxy modes:
    - "none":       ignore SoC proxy entirely.
    - "exogenous":  include SoC-derived *features* (increments / aggregates) in
                    the distance matrix (akin to current milp_tsa usage).
    - "endogenous": add a *linearised* SoC proxy cost term directly to the
                    optimisation model objective (hook provided here as TODO).
- Structured to be flexible: config dataclass, a small dispatcher, and
  pluggable builders for features, distances, and Pyomo models.

This file intentionally contains *skeletons* / stubs for parts that depend on
your exact formulation (RR step, endogenous SoC linearisation, etc.).
Fill in the TODO sections when you implement the maths.
"""


# ----------------------------- Public API -------------------------------------

class SocProxyMode(str, Enum):
    NONE = "none"
    EXOGENOUS = "exogenous"
    ENDOGENOUS = "endogenous"


@dataclass
class OrdoRRConfig:
    # Core problem shape
    pre_clustered: bool = False                   # True => candidate reps are fixed by cluster map
    clustermap_path: Optional[str] = None         # Path to 'timesteps,PeriodNum' CSV when pre_clustered=True
    k: Optional[int] = None                       # number of representative periods to select (ignored if pre_clustered)
    hours_per_period: int = 24                    # used by some feature builders (e.g., hourly profiles)

    # SoC proxy handling
    soc_proxy_mode: SocProxyMode = SocProxyMode.NONE
    soc_proxy_inputs: List[str] = field(default_factory=list)     # e.g. ['soc_proxy_LDES'] or delta column names
    soc_proxy_window: Optional[Tuple[str, str]] = None            # optional (start, end) to focus proxy costs
    soc_decomposition: Dict = field(default_factory=dict)         # passed through to generate_soc_proxy
    dispatchable_techs: Dict = field(default_factory=dict)        # idem
    storage_process_losses: Dict = field(default_factory=dict)    # idem
    capacity_weights: Dict[str, float] = field(default_factory=dict)  # for generate_soc_proxy

    # Distance matrix settings
    matrix_weights: Dict[str, float] = field(default_factory=lambda: {"renewables": 1.0, "demand": 1.0, "proxy": 1.0})
    distance_metric: str = "euclidean"
    names_renewables: List[str] = field(default_factory=lambda: ["solar", "onshore_wind", "offshore_wind"])
    name_demand: List[str] = field(default_factory=lambda: ["demand_power"])

    # Solver settings
    solver: str = "gurobi"
    mipgap: float = 0.01
    timelimit_s: int = 3600
    verbose: bool = True

    # Data inputs
    reference_timeseries_path: Optional[str] = None   # Calliope-style CSV path
    reference_timeseries_df: Optional[pd.DataFrame] = None  # Or provide directly

    # Output
    out_cluster_map_path: Optional[str] = None

    # Misc
    resample_to_daily_resolution: bool = True         # collapse to daily features for MILP tractability


@dataclass
class OrdoRRResult:
    selected_days: List[int]                    # row indices (in daily index space) of representatives
    assignments: Dict[int, int]                 # map: day j -> representative i (both are integer positions)
    model: Optional[pyo.ConcreteModel] = None   # Pyomo model (if solved by MILP)
    solver_results: Optional[object] = None     # solver results (opaque)
    dates_index: Optional[pd.DatetimeIndex] = None  # mapping back to real dates


def ordo_rr_optimize(cfg: OrdoRRConfig) -> OrdoRRResult:
    """
    Main entry point. Builds features & distance matrix, then dispatches to either:
    - unclustered MILP (select reps & assign), or
    - pre-clustered assignment MILP (assign to fixed reps).
    Optionally writes a cluster map if cfg.out_cluster_map_path is given.
    """
    # 1) Load reference time series
    df_ts = _load_reference_timeseries(cfg)
    # 2) Build daily features
    feat_df, daily_index = _build_features(df_ts, cfg)
    # 3) Build distance matrix, with requested block weights and optional proxy window
    D = _build_distance(feat_df, cfg)

    if cfg.pre_clustered:
        reps_pos = _read_representative_positions(cfg.clustermap_path, daily_index)
        result = _solve_assignment_to_fixed_reps(D, reps_pos, cfg)
    else:
        if cfg.k is None:
            raise ValueError("cfg.k must be provided for unclustered optimisation.")
        result = _solve_unclustered_selection(D, cfg.k, cfg)

    result.dates_index = daily_index

    # 4) Optional RR refinement step (skeleton)
    #    Implement your Representative Refinement logic here if desired.
    #    Example: result = _refine_assignments_rr(feat_df, D, result, cfg)
    #    For now, we skip this.
    #    TODO: implement RR if/when you need it.

    # 5) Optional: write cluster map in Calliope format
    if cfg.out_cluster_map_path:
        save_milp_result_to_cluster_map(
            result={
                "selected_days": result.selected_days,
                "assignments": result.assignments,
                "model": result.model,
                "results": result.solver_results,
            },
            dates_index=result.dates_index,
            output_path=cfg.out_cluster_map_path,
        )

    return result


# ----------------------------- Build blocks -----------------------------------

def _load_reference_timeseries(cfg: OrdoRRConfig) -> pd.DataFrame:
    if cfg.reference_timeseries_df is not None:
        df = cfg.reference_timeseries_df.copy()
    elif cfg.reference_timeseries_path:
        df = calliope_ts_to_pandas(cfg.reference_timeseries_path)
    else:
        raise ValueError("Provide either reference_timeseries_df or reference_timeseries_path.")
    df = df.copy()
    df.set_index("timesteps", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns.name = None
    return df


def _build_features(df_ts: pd.DataFrame, cfg: OrdoRRConfig) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Build the feature matrix to feed the distance builder.
    - If EXOGENOUS or ENDOGENOUS: compute SoC proxy columns first and append requested inputs.
    - Then either resample to daily means (lightweight) or (optionally) construct daily 24h profiles.
    """
    # Ensure we only carry numeric columns for scaling
    working = df_ts.copy()

    # Optionally compute SoC proxy features
    proxy_cols: List[str] = []
    if cfg.soc_proxy_mode in (SocProxyMode.EXOGENOUS, SocProxyMode.ENDOGENOUS):
        # Expect that cfg.name_demand and cfg.capacity_weights / dispatchable / losses are configured
        # to let generate_soc_proxy build e.g. 'soc_proxy_LDES' or delta series.
        # NOTE: generate_soc_proxy returns a full df including original cols.
        working, _, _ = generate_soc_proxy(
            df=working,
            demand_field=cfg.name_demand[0],
            renewables_fields_and_weights=cfg.capacity_weights,
            dispatchable_techs=cfg.dispatchable_techs,
            storage_process_losses=cfg.storage_process_losses,
            soc_decomposition=cfg.soc_decomposition,
            timestamp_col=None,
        )
        # Only keep requested SoC columns if provided
        if cfg.soc_proxy_inputs:
            proxy_cols = [c for c in cfg.soc_proxy_inputs if c in working.columns]
        else:
            # Fall back: take any columns that look like SoC proxy outputs
            proxy_cols = [c for c in working.columns if "soc_" in c or "proxy" in c]

    # Select the core columns for renewables, demand, and proxy blocks
    cols_R = _cols_by_prefix(working, cfg.names_renewables)
    cols_D = _cols_by_prefix(working, cfg.name_demand)
    cols_P = proxy_cols

    # Subset working df
    keep_cols = sorted(set(cols_R + cols_D + cols_P))
    if not keep_cols:
        raise ValueError("No features found to build distance matrix. Check config names_renewables/name_demand/etc.")
    working = working[keep_cols]

    if cfg.resample_to_daily_resolution:
        # Simple daily means: shape (days x features)
        daily = working.resample("D").agg("mean")
        daily_index = daily.index
        feat_df = daily
    else:
        # Expand to daily 24h profiles: concatenate hourly vectors per feature (days x (features*24))
        daily_rows: List[np.ndarray] = []
        days: List[pd.Timestamp] = []
        for day, group in working.groupby(pd.Grouper(freq="D")):
            if len(group) != 24:
                continue  # skip incomplete days (DST edges, etc.)
            row = np.concatenate([group[c].to_numpy() for c in keep_cols])
            daily_rows.append(row)
            days.append(day.normalize())
        col_names = [f"{var}_h{h:02d}" for var in keep_cols for h in range(24)]
        feat_df = pd.DataFrame(daily_rows, columns=col_names, index=pd.DatetimeIndex(days, name="timesteps"))
        daily_index = feat_df.index

    return feat_df, pd.DatetimeIndex(daily_index, name="timesteps")


def _build_distance(feat_df: pd.DataFrame, cfg: OrdoRRConfig) -> np.ndarray:
    # If proxy mode is NONE, zero out proxy weight & prefixes; otherwise pass through.
    prefixes_proxy = cfg.soc_proxy_inputs if cfg.soc_proxy_mode != SocProxyMode.NONE else []
    weights = dict(cfg.matrix_weights or {})
    if cfg.soc_proxy_mode == SocProxyMode.NONE:
        weights["proxy"] = 0.0

    D = build_distance_matrix(
        feature_df=feat_df,
        matrix_weights=weights,
        metric=cfg.distance_metric,
        column_prefixes_renewables=cfg.names_renewables,
        column_prefixes_demand=cfg.name_demand,
        column_prefixes_proxy=prefixes_proxy,
        proxy_window=cfg.soc_proxy_window,
    )
    return D


def _cols_by_prefix(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    if not prefixes:
        return []
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


# ------------------------- Solve: unclustered vs fixed reps --------------------

def _solve_unclustered_selection(D: np.ndarray, k: int, cfg: OrdoRRConfig) -> OrdoRRResult:
    """
    Use existing milp_tsa to select k reps and assign all days.
    """
    res = milp_tsa(
        distance_matrix=D,
        k=k,
        solver=cfg.solver,
        mipgap=cfg.mipgap,
        verbose=cfg.verbose,
    )
    return OrdoRRResult(
        selected_days=list(res["selected_days"]),
        assignments=dict(res["assignments"]),
        model=res.get("model"),
        solver_results=res.get("results"),
    )


def _read_representative_positions(clustermap_path: Optional[str], daily_index: pd.DatetimeIndex) -> List[int]:
    """
    Read a pre-computed cluster map (timesteps,PeriodNum) and return the integer
    positions into `daily_index` corresponding to the unique representatives.
    """
    if not clustermap_path:
        raise ValueError("clustermap_path must be provided for pre_clustered mode.")
    df = pd.read_csv(clustermap_path, dtype={"timesteps": str, "PeriodNum": str})
    if list(df.columns[:2]) != ["timesteps", "PeriodNum"]:
        raise ValueError("Cluster map must have columns ['timesteps','PeriodNum'] in that order.")
    reps_unique = (
        pd.to_datetime(df["PeriodNum"], errors="coerce")
        .dropna()
        .drop_duplicates()
    )
    # Map rep timestamps to positions in the current daily index
    pos = []
    day_to_pos = {ts: i for i, ts in enumerate(pd.DatetimeIndex(daily_index))}
    for ts in pd.DatetimeIndex(reps_unique):
        i = day_to_pos.get(ts)
        if i is None:
            # If the rep date is missing in this horizon, skip or raise depending on needs
            raise ValueError(f"Representative date {ts.date()} not found in reference timeseries horizon.")
        pos.append(i)
    if len(pos) == 0:
        raise ValueError("No valid representatives found in clustermap.")
    return sorted(pos)


def _solve_assignment_to_fixed_reps(D: np.ndarray, reps_pos: List[int], cfg: OrdoRRConfig) -> OrdoRRResult:
    """
    Pre-clustered variant: choose among a *fixed* set of representatives.
    Here we build a pure assignment model (x[i,j]) with i restricted to reps_pos.
    You can extend this with additional constraints or an RR step.
    """
    n = D.shape[0]
    I = sorted(set(reps_pos))
    J = list(range(n))

    # Extract the submatrix only over candidate reps (rows I, all columns J)
    # NOTE: In the standard milp_tsa, rows are candidates, columns are days.
    D_sub = D[np.ix_(I, J)]

    # Build Pyomo model
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(len(I)))      # indices over candidate reps (local index)
    m.J = pyo.Set(initialize=J)                  # global day indices

    # Parameter: cost[i,j] = distance(rep=I[i], day=j)
    D_dict = {(ii, j): float(D_sub[ii, j]) for ii in m.I for j in m.J}
    m.D = pyo.Param(m.I, m.J, initialize=D_dict, within=pyo.NonNegativeReals)

    # Decision: assign each day to exactly one rep
    m.x = pyo.Var(m.I, m.J, domain=pyo.Binary)

    # Objective: minimise total assignment cost
    def obj_rule(m):
        return sum(m.x[ii, j] * m.D[ii, j] for ii in m.I for j in m.J)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Each day assigned once
    def assign_once_rule(m, j):
        return sum(m.x[ii, j] for ii in m.I) == 1
    m.assign_once = pyo.Constraint(m.J, rule=assign_once_rule)

    # (Optional) Endogenous SoC proxy: hook to add linearised terms to the objective
    if cfg.soc_proxy_mode == SocProxyMode.ENDOGENOUS:
        _attach_endogenous_soc_terms(m, I, J, cfg)
        # TODO: implement linearised SoC proxy penalty per assignment as needed.

    # Solve
    solver = pyo.SolverFactory(cfg.solver)
    results = solver.solve(
        m,
        tee=cfg.verbose,
        options={
            "TimeLimit": int(cfg.timelimit_s),
            "MipGap": float(cfg.mipgap),
        }
    )

    # Extract assignments
    assignments: Dict[int, int] = {}
    for j in J:
        # find unique ii with x[ii,j] == 1
        chosen_local = [ii for ii in m.I if pyo.value(m.x[ii, j]) > 0.5]
        if not chosen_local:
            # should not happen with correct solve, but be defensive
            # choose argmin as fallback
            chosen_local = [int(np.argmin([pyo.value(m.D[ii, j]) for ii in m.I]))]
        ii = chosen_local[0]
        rep_global = I[ii]
        assignments[j] = rep_global

    # Selected reps are exactly I
    return OrdoRRResult(
        selected_days=I,
        assignments=assignments,
        model=m,
        solver_results=results,
    )


def _attach_endogenous_soc_terms(m: pyo.ConcreteModel, I: List[int], J: List[int], cfg: OrdoRRConfig) -> None:
    """
    Hook point for adding a linearised SoC proxy term to the objective.
    This will depend on your chosen linearisation (e.g., cumulative ΔSoC with
    piecewise linear constraints, auxiliary variables, etc.).
    The function is intentionally left as a stub for now.
    """
    # Example sketch (pseudo-code):
    # - Introduce variables s[j] for cumulative SoC effect on day j
    # - Add constraints to build s[j] from assignments and precomputed daily increments
    # - Add weighted sum to objective: m.obj.expr += weight * sum(s[j] for j in J)
    # TODO: implement your linearisation here.
    pass


# ----------------------------- Optional RR step -------------------------------

def _refine_assignments_rr(feat_df: pd.DataFrame,
                           D: np.ndarray,
                           res: OrdoRRResult,
                           cfg: OrdoRRConfig) -> OrdoRRResult:
    """
    Placeholder for a Representative-Refinement (RR) step.
    Typical patterns include:
      - Local search: for each rep, test swapping it with a non-rep day if it lowers total cost.
      - K-medoids-like iteration: recompute medoids from current assignments, repeat until stable.
    Implement your preferred heuristic here if needed.
    """
    # TODO: implement if/when needed.
    return res
