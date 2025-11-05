
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import calliope
from netCDF4 import Dataset
import  yaml
import matplotlib.colors as mcolors
import json


import os



def read_clustered_netcdf(path):

    p = Path(path).resolve()
    if not p.is_file():
        raise Exception('File does not exist at: {path}')
    with Dataset(p, "a") as nc:  # append mode
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)  # remove lingering clustering
        g.setncattr("config", yaml.safe_dump(cfg))

    return calliope.read_netcdf(path)

def read_results(directory):

    _dir = os.listdir(directory)

    arr_id = []
    arr_runtime = []
    arr_horizon = []
    arr_clustering = []

    counter = 0
    lim = len(_dir)

    for file in _dir:
        filename = os.fsdecode(file)
        if filename.endswith(".nc"): 
            model_id = filename.split('.')[0]
            if 'shuffle' in model_id:
                lim -= 1
                continue
            else:
                counter += 1
                print(model_id, f'{counter}/{lim}')
                m = read_clustered_netcdf(f'{os.fsdecode(directory)}/{filename}')
                arr_id.append(model_id)
                arr_runtime.append(round(m.all_attrs.runtime.timings['solve_complete']-m.all_attrs.runtime.timings['build_start'],4))


                if 'standard' in model_id:
                    arr_clustering.append('ref')
                    arr_horizon.append(round(m.inputs.dims["timesteps"]/(24*365.25)))
                else:
                    param_path = f'SoC_proxy_TSA/data/parameters/{model_id}.json'
                    with open(param_path) as p:
                        d = json.load(p)
                        arr_horizon.append(round(d['calliope_params']['date_range'][1]-d['calliope_params']['date_range'][0]+1))
                        arr_clustering.append(d['tsa_params']['k_periods'])
                continue
        else:
            continue

    df = pd.DataFrame({
        "id": arr_id,
        "runtime": arr_runtime,
        "horizon": arr_horizon,
        "clustering": arr_clustering,
    }).set_index("id")


    return df

directory = os.fsencode('SoC_proxy_TSA/data/calliope_models')
df = read_results(directory)
df.to_csv('SoC_proxy_TSA/data/notes/runtime_log.csv')

df = pd.read_csv('SoC_proxy_TSA/data/notes/runtime_log.csv')

# 1) runtime → minutes
df["runtime_min"] = df["runtime"] / 60.0
df.loc[df["runtime_min"] <= 0, "runtime_min"] = np.nan

# 2) ordered horizons (2, 5, 10, ...)
horizons_num = sorted(df["horizon"].unique())
horizons = [str(h) for h in horizons_num]
h_map_num = {h: i for i, h in enumerate(horizons_num)}
df["x_num"] = df["horizon"].map(h_map_num).astype(float)

# 3) jitter
jitter_width = 0.12
rng = np.random.default_rng(42)
df["x_jitter"] = df["x_num"] + rng.uniform(-jitter_width, jitter_width, size=len(df))

fig, ax = plt.subplots(figsize=(6, 4))

# 4) ordered clustering categories
# split numeric vs non-numeric
# ---- ordered clustering categories ----
raw_clusters = df["clustering"].unique().tolist()

numeric_clusters = []
string_clusters = []

for c in raw_clusters:
    try:
        # try to interpret as number
        num = float(c)
        numeric_clusters.append((num, c))  # (numeric_value, original_label)
    except (ValueError, TypeError):
        string_clusters.append(c)

# sort numeric by numeric value
numeric_clusters_sorted = sorted(numeric_clusters, key=lambda x: x[0], reverse=True)

# final order: numeric (by value) then strings (e.g. 'ref')
clusters_ordered = [orig for _, orig in numeric_clusters_sorted] + sorted(string_clusters)

# put strings at the end (e.g. 'ref')
clusters = [*numeric_clusters, *sorted(string_clusters)]
n_clusters = len(clusters_ordered )

# 5) colours: sample plasma but stop before yellow, then make last grey
plasma = plt.get_cmap("plasma")
# go only up to 0.9 to avoid brightest yellow
plasma_colors = plasma(np.linspace(0, 0.9, max(n_clusters, 1)))
grey = "#808080"
if n_clusters > 0:
    plasma_colors[-1] = mcolors.to_rgba(grey)  # final category = grey

# 6) plot per cluster
for cluster, color in zip(clusters_ordered , plasma_colors):
    mask = df["clustering"] == cluster
    ax.scatter(
        df.loc[mask, "x_jitter"],
        df.loc[mask, "runtime_min"],
        label=str(cluster),
        color=color,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
        s=45,
    )

# x-axis
ax.set_xticks(range(len(horizons)))
ax.set_xticklabels(horizons)
ax.set_xlabel("Horizon (years)")

# y-axis log
ax.set_yscale("log")
ax.set_ylabel("Runtime (minutes)")

ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(title="Clustering", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()
fig.savefig("runtimes.pdf", dpi=600, bbox_inches="tight")
plt.show()