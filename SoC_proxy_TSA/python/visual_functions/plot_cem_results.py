
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import calliope
from netCDF4 import Dataset
import  yaml
import re
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "axes.unicode_minus": False,
})
mpl.rcParams["pgf.preamble"] = r""

colour_1 = '#0D0887'   
colour_2 = '#CC4778'
colour_3 = '#fdb42f'

def cem_results(df):

    
 
    list_power_cap_mean_errors = []
    list_ldes_cap_error = []

    for model in df.itertuples(index=True):
        

        # reference
        start_year, end_year = model.dates.split(",")
        ref_path = f'SoC_proxy_TSA/data/calliope_models/standard_{start_year}_{end_year}_reference.nc'

        model_reference = calliope.read_netcdf(ref_path)
        power_caps_reference, energy_caps_reference = get_capacities(model_reference)
        
        model_test = read_clustered_netcdf(f"SoC_proxy_TSA/data/calliope_models/{model.id}.nc")
        power_caps_test, energy_caps_test = get_capacities(model_test)

        #metrics
        _, e_mean_power = relative_error(power_caps_reference, power_caps_test)
        e_storage,_ = relative_error(energy_caps_reference, energy_caps_test)

        print(f'Extracted info for {model.id}, ldes_e={e_storage['h2_salt_cavern']}, macme={e_mean_power}')

        list_power_cap_mean_errors.append(e_mean_power)
        list_ldes_cap_error.append(e_storage['h2_salt_cavern'])

    df.insert(0,'macme',list_power_cap_mean_errors)
    df.insert(0,'ldes_error',list_ldes_cap_error)

    return df

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

def get_capacities(m: calliope.Model):

    df_power_caps = (
        m.results['flow_cap']
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame('capacity')
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
        )
    
    mask_drop = (
        df_power_caps["techs"].isin(["battery", "h2_salt_cavern", "demand"])
        | df_power_caps["techs"].str.startswith("demand")
        )
    df_power_caps = df_power_caps[~mask_drop & (
            ((df_power_caps["techs"] == "electrolyser") & (df_power_caps["carriers"] == "hydrogen")) |
            ((df_power_caps["techs"] != "electrolyser") & (df_power_caps["carriers"] == "power"))
        )]
    df_power_caps.set_index("techs", inplace=True)
    
    df_energy_caps = (
        m.results['storage_cap']
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame('capacity')
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
        )
    df_energy_caps = df_energy_caps[df_energy_caps['capacity']>0]
    df_energy_caps.set_index("techs", inplace=True)
    
    return df_power_caps['capacity'],df_energy_caps['capacity']

def relative_error(df_ref, df_test):

    e = (df_ref - df_test) / df_ref
    e_mean_abs = np.mean(np.abs(e))

    return e, e_mean_abs


config_src = pd.read_csv('SoC_proxy_TSA/data/notes/log.csv')
config_src['number_reps'] = (
    config_src["model_name"]
    .str.extract(r"reps\s*=\s*(\d+)", expand=False)
    .astype("Int64")   # nullable integer dtype
)
config_src["W_proxy"] = (
    config_src["model_name"]
    .str.extract(r"W_proxy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    .astype(float)
)
# tag = config_src["tvp"].str.extract(r"(?i)(shuffle[^/]*?)(?=\.csv\b)", expand=False)
# ref_tag = tag.fillna("standard_2010_2019_reference")
# config_src["reference_path"] = "SoC_proxy_TSA/data/calliope_models/" + ref_tag + ".nc"

df = config_src[['id','dates','W_proxy','number_reps']]
x_axis ='W_proxy'

#filter and control plot
# df=df[df['number_reps']==30]


df = cem_results(df)

df_plot = df[(df["W_proxy"] >= 0.5) & (df["number_reps"] >= 30)].copy()

# 1. get unique x values in order of appearance
x_vals = df_plot[x_axis].unique().tolist()
x_map = {val: i for i, val in enumerate(x_vals)}

# 2. make a numeric x column
df_plot["x_num"] = df_plot[x_axis].map(x_map)

# 3. jitter
jitter_width = 0.12
df_plot["_jitter"] = 0.0
for xnum in df_plot["x_num"].unique():
    mask = df_plot["x_num"] == xnum
    n = mask.sum()
    if n == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-jitter_width, jitter_width, n)
    df_plot.loc[mask, "_jitter"] = offsets

x_plot = df_plot["x_num"] + df_plot["_jitter"]

fig, ax = plt.subplots(figsize=(6, 4))



ax.scatter(
    x_plot,
    df_plot["macme"],
    color=colour_1,
    label=r'MACME, $\overline{\epsilon^C}$',
    alpha=0.8,
    edgecolor="white",
    linewidth=0.0,
)
ax.scatter(
    x_plot,
    df_plot["ldes_error"],
    color=colour_2,
    label=r'LDES Cap. Error, $\epsilon^C_{\mathrm{LDES}}$',
    alpha=0.8,
    edgecolor="white",
    linewidth=0.0,
)

mean_line_width = 0.2
grouped = df_plot.groupby(x_axis)
for label, g in grouped:
    xnum = x_map[label]
    macme_mean = g["macme"].mean()
    ldes_mean = g["ldes_error"].mean()

    ax.hlines(macme_mean, xnum - mean_line_width, xnum + mean_line_width,
              colors=colour_1, linewidth=2)
    ax.hlines(ldes_mean, xnum - mean_line_width, xnum + mean_line_width,
              colors=colour_2, linewidth=2)

# ticks from filtered x_vals
ax.set_xticks(range(len(x_vals)))
ax.set_xticklabels([rf'${v}$ Days' for v in x_vals])

# y limits based on filtered data
ymax = max(df_plot["macme"].max(), df_plot["ldes_error"].max())
ymin = min(df_plot["macme"].min(), df_plot["ldes_error"].min())
ymax_rounded = np.ceil(ymax * 10) / 10.0
ymin_rounded = np.floor(ymin * 10) / 10.0
ax.set_ylim(ymin_rounded, ymax_rounded)

ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

ax.set_xlabel('Proxy Weight' if x_axis == 'W_proxy' else 'Number of Representative Days')
ax.set_ylabel("Error")
ax.legend(frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.show()

# fig, ax = plt.subplots(figsize=(6, 4))

# # mask for W_proxy == 0
# mask_zero = df_plot["W_proxy"] == 0
# mask_nonzero = ~mask_zero

# # base coords
# x_all = x_plot
# y_macme = df_plot["macme"]
# y_ldes = df_plot["ldes_error"]

# # 1) MACME, nonzero W -> filled
# ax.scatter(
#     x_all[mask_nonzero],
#     y_macme[mask_nonzero],
#     color=colour_1,
#     alpha=0.8,
#     edgecolor="white",
#     linewidth=0.0,
# )

# # 2) MACME, W == 0 -> hollow, blue edge
# ax.scatter(
#     x_all[mask_zero],
#     y_macme[mask_zero],
#     facecolors="white",
#     edgecolors=colour_1,
#     linewidth=1.0,
# )

# # 3) LDES, nonzero W -> filled
# ax.scatter(
#     x_all[mask_nonzero],
#     y_ldes[mask_nonzero],
#     color=colour_2,
#     alpha=0.8,
#     edgecolor="white",
#     linewidth=0.0,
# )

# # 4) LDES, W == 0 -> hollow, magenta edge
# ax.scatter(
#     x_all[mask_zero],
#     y_ldes[mask_zero],
#     facecolors="white",
#     edgecolors=colour_2,
#     linewidth=1.0,
# )

# # means (same as before)
# mean_line_width = 0.2
# grouped = df_plot.groupby(x_axis)
# for label, g in grouped:
#     xnum = x_map[label]
#     macme_mean = g["macme"].mean()
#     ldes_mean = g["ldes_error"].mean()

#     ax.hlines(macme_mean, xnum - mean_line_width, xnum + mean_line_width,
#               colors=colour_1, linewidth=2)
#     ax.hlines(ldes_mean, xnum - mean_line_width, xnum + mean_line_width,
#               colors=colour_2, linewidth=2)

# # x-ticks
# ax.set_xticks(range(len(x_vals)))
# ax.set_xticklabels([rf'${v}$ Days' for v in x_vals])

# # y-lims
# ymax = max(df_plot["macme"].max(), df_plot["ldes_error"].max())
# ymin = min(df_plot["macme"].min(), df_plot["ldes_error"].min())
# ymax_rounded = np.ceil(ymax * 10) / 10.0
# ymin_rounded = np.floor(ymin * 10) / 10.0
# ax.set_ylim(ymin_rounded, ymax_rounded)

# ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

# ax.set_xlabel('Proxy Weight')
# ax.set_ylabel("Error")

# # ---- manual legend ----
# legend_handles = [
#     Line2D([0], [0], marker='o', color='none', markerfacecolor=colour_1,
#            markeredgecolor='white', label=r'MACME, $\overline{\epsilon^C}$'),
#     Line2D([0], [0], marker='o', color='none', markerfacecolor=colour_2,
#            markeredgecolor='white', label=r'LDES Cap. Error, $\epsilon^C_{\mathrm{LDES}}$'),
#     Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
#            markeredgecolor='grey', label=r'$W_x = 0$'),
# ]
# ax.legend(handles=legend_handles, frameon=False)

# ax.grid(axis="y", linestyle=":", alpha=0.6)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# fig.tight_layout()
# plt.show()

# fig.savefig('CEM_vs_K_masks.pdf', dpi=600, bbox_inches="tight")

# ---- HEATMAP OF LDES ERROR ----

# 1. absolute error
df["ldes_abs"] = df["ldes_error"].abs()

# 2. pivot: rows = number_reps, cols = W_proxy
# you can sort to keep it tidy
pivot = (
    df.pivot_table(
        index="number_reps",
        columns="W_proxy",
        values="ldes_abs",
        aggfunc="mean"
    )
    .sort_index(axis=0)   # sort number_reps
    .sort_index(axis=1)   # sort W_proxy
)

# pivot already defined above
data = pivot.values

# 1) pick bin step (5% = 0.05)
step = 0.05

# 2) find max and round up to nearest step
data_max = np.nanmax(data)
max_rounded = np.ceil(data_max / step) * step  # e.g. 0.49 -> 0.50

# 3) build boundaries: 0, 0.05, 0.10, ... max_rounded
boundaries = np.arange(0, max_rounded + step, step)

# 4) get plasma and sample as many colors as bins-1
plasma = plt.get_cmap("plasma")
n_colors = len(boundaries) - 1
colors = plasma(np.linspace(0, 1, n_colors))
cmap_discrete = mcolors.ListedColormap(colors)

# 5) norm that maps data into these bins
norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap_discrete.N)

fig_h, ax_h = plt.subplots(figsize=(6, 4))

im = ax_h.imshow(
    data,
    cmap=cmap_discrete,
    norm=norm,
    origin="lower",
    aspect="auto",
)

# ticks & labels
x_vals = pivot.columns.tolist()
y_vals = pivot.index.tolist()
ax_h.set_xticks(range(len(x_vals)))
ax_h.set_xticklabels([rf'$W_x={x}$' for x in x_vals])
ax_h.set_yticks(range(len(y_vals)))
ax_h.set_yticklabels([f"{y} days" for y in y_vals])

# hide tick lines
ax_h.tick_params(axis="x", length=0)
ax_h.tick_params(axis="y", length=0)

ax_h.set_xlabel("Proxy weight")
ax_h.set_ylabel("Number of Representative Days")

# 6) discrete colorbar
cbar = fig_h.colorbar(im, ax=ax_h, boundaries=boundaries, ticks=boundaries)
cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
cbar.set_label("Absolute LDES Capacity Error", rotation=90, labelpad=10)

fig_h.tight_layout()
plt.show()

fig_h.savefig("LDES_heatmap.pdf", dpi=600, bbox_inches="tight")
