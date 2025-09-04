# utility_functions/helper_post_cluster_opt.py
from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

from utility_functions.class_targets import TargetRegistry
from utility_functions.helper_cluster_tsa_with_extremes import ClusterResult

if TYPE_CHECKING:
    from utility_functions.class_tsa_model import tsa_model


def parse_objective_spec(spec_list: List[Dict]) -> Tuple[List[str], Dict[str, float]]:
    targets = []
    weights = {}
    for item in spec_list:
        t = item["target"]
        targets.append(t)
        weights[t] = float(item.get("weight", 1.0))
        # polarity/mode can be added later if you support non-"match" objectives
    return targets, weights

def build_target_cost_matrix(df_targets: pd.DataFrame,
                             rep_index: pd.Index,
                             metric: str = "mae",
                             weights: Dict[str, float] | None = None) -> np.ndarray:
    # days x targets
    X = df_targets.loc[df_targets.index].to_numpy()      # N x T
    C = df_targets.loc[rep_index].to_numpy()             # P x T
    diff = X[:, None, :] - C[None, :, :]                 # N x P x T
    TODO: Not sure this is accessing the correct target, we need to import a reference to serve as the soc target
    if metric == "mae":
        per_target = np.abs(diff)
    elif metric == "mse":
        per_target = diff**2
    else:
        raise ValueError("Unknown metric")

    if weights:
        w = np.array([weights.get(col, 1.0) for col in df_targets.columns])[None, None, :]
        per_target = per_target * w

    D = per_target.mean(axis=2)                          # N x P
    return D

def solve_reassignment_argmin(days_index: pd.Index,
                              reps_index: pd.Index,
                              D: np.ndarray) -> pd.Series:
    """For fixed reps, best assignment is argmin across reps for each day."""
    choice = np.argmin(D, axis=1)                        # N
    mapped = reps_index.values[choice]                   # dates
    return pd.Series(index=days_index, data=mapped, name="PeriodNum")

def save_updated_cluster_map(assignment: pd.Series, out_path: str) -> None:
    df = pd.DataFrame({"timesteps": assignment.index.normalize(), "PeriodNum": assignment.values})
    df.to_csv(out_path, index=False)

def apply_optimisation_on_cluster(cluster_result: ClusterResult, model: "tsa_model") -> ClusterResult:
    """Remap days to fixed reps using target-driven costs. Writes the updated CSV."""
    spec = (model.tsa.params.get("post_cluster_optimisation_params") or {}).get("objective", [])
    if not spec:
        return cluster_result

    targets, weights = parse_objective_spec(spec)

    # 1) Materialise targets (daily) just-in-time
    reg = TargetRegistry(model)
    df_targets = reg.daily_stacked(targets, how="mean")  # or "sum" reg.hourly(targets)

    # 2) Cost matrix over days × fixed reps
    D = build_target_cost_matrix(df_targets=df_targets,
                                 rep_index=cluster_result.representatives,
                                 metric="mae",
                                 weights=weights)

    # 3) Argmin reassignment
    new_assign = solve_reassignment_argmin(df_targets.index, cluster_result.representatives, D)

    # 4) Write updated CSV and return updated result
    save_updated_cluster_map(new_assign, cluster_result.cluster_map_path)

    return replace(cluster_result,
                   assignment=new_assign,
                   meta=cluster_result.meta | {"post_cluster_optimisation": spec})
