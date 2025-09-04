from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, List
import pandas as pd
from utility_functions.helper_timeseries_tools import calliope_ts_to_pandas, extrapolate_ts_from_cluster_map
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy

if TYPE_CHECKING:
    from utility_functions.class_tsa_model import tsa_model

@dataclass
class TargetRegistry:
    """Lazy, cached target builder so earlier pipeline stays clean."""
    model: "tsa_model"
    cache: Dict[str, pd.Series] = field(default_factory=dict)

    def _hourly_timeseries(self) -> pd.DataFrame:
        if self.model.tsa.type == "cluster":
            df, _ = extrapolate_ts_from_cluster_map(self.model.paths['cluster_map'],
                                                    self.model.paths['timeseries'])
        else:
            df = calliope_ts_to_pandas(
                source=self.model.paths['timeseries'],
                date_range_lower_bound=f"{self.model.calliope_model.params['date_range'][0]}-01-01",
                date_range_upper_bound=f"{self.model.calliope_model.params['date_range'][-1]}-12-31",
            )
        df = df.set_index('timesteps').sort_index()
        df.columns.name = None
        return df

    def hourly(self, name: str) -> pd.Series:
        if name in self.cache:
            return self.cache[name]
        df = self._hourly_timeseries()

        if name == "soc_proxy_ldes":
            p = self.model.soc_proxy.params
            demand = self.model.tsa.params['name_demand'][0]
            df_proxy, _, _ = generate_soc_proxy(
                df=df,
                demand_field=demand,
                renewables_fields_and_weights=p['capacity_weights'],
                dispatchable_techs=p['dispatchable_techs'],
                storage_process_losses=p['storage_process_losses'],
                soc_decomposition=p['soc_decomposition'],
                timestamp_col=None
            )
            s = df_proxy['soc_proxy_LDES']
        else:
            if name not in df.columns:
                raise KeyError(f"Target '{name}' not found.")
            s = df[name]

        self.cache[name] = s
        return s

    def daily(self, name: str, how: str = "mean") -> pd.Series:
        h = self.hourly(name)
        if how == "mean":
            return h.resample("D").mean()
        elif how == "sum":
            return h.resample("D").sum()
        else:
            raise ValueError(f"Unknown aggregation: {how}")

    def daily_stacked(self, names: List[str], how: str = "mean") -> pd.DataFrame:
        return pd.DataFrame({nm: self.daily(nm, how=how) for nm in names}).sort_index()
