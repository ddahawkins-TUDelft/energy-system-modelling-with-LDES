import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path
import calliope
from netCDF4 import Dataset
import yaml

# # If you want LaTeX:
# import matplotlib as mpl
# mpl.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "axes.unicode_minus": False,
# })
# mpl.rcParams["pgf.preamble"] = r""

# colours
colour_1 = '#0D0887'    # MACME
colour_2 = '#CC4778'    # LDES
colour_grey = '#666666'  # Wx=0 outline


def cem_results(df):
    """Add macme and ldes_error columns to df of models."""
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

        # metrics
        _, e_mean_power = relative_error(power_caps_reference, power_caps_test)
        e_storage, _ = relative_error(energy_caps_reference, energy_caps_test)

        print(
            f"Extracted info for {model.id}, "
            f"ldes_e={e_storage['h2_salt_cavern']}, macme={e_mean_power}"
        )

        list_power_cap_mean_errors.append(e_mean_power)
        list_ldes_cap_error.append(e_storage["h2_salt_cavern"])

    df.insert(0, "macme", list_power_cap_mean_errors)
    df.insert(0, "ldes_error", list_ldes_cap_error)

    return df


def read_clustered_netcdf(path):
    """Open model .nc and strip lingering clustering from attrs."""
    p = Path(path).resolve()
    if not p.is_file():
        raise Exception(f"File does not exist at: {path}")
    with Dataset(p, "a") as nc:  # append mode to edit attrs
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)
        g.setncattr("config", yaml.safe_dump(cfg))

    return calliope.read_netcdf(path)


def get_capacities(m: calliope.Model):
    """Return power_cap and storage_cap Series indexed by techs."""
    df_power_caps = (
        m.results["flow_cap"]
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame("capacity")
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )

    mask_drop = (
        df_power_caps["techs"].isin(["battery", "h2_salt_cavern", "demand"])
        | df_power_caps["techs"].str.startswith("demand")
    )
    df_power_caps = df_power_caps[
        ~mask_drop
        & (
            ((df_power_caps["techs"] == "electrolyser") & (df_power_caps["carriers"] == "hydrogen"))
            | ((df_power_caps["techs"] != "electrolyser") & (df_power_caps["carriers"] == "power"))
        )
    ]
    df_power_caps.set_index("techs", inplace=True)

    df_energy_caps = (
        m.results["storage_cap"]
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame("capacity")
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
    )
    df_energy_caps = df_energy_caps[df_energy_caps["capacity"] > 0]
    df_energy_caps.set_index("techs", inplace=True)

    return df_power_caps["capacity"], df_energy_caps["capacity"]


def relative_error(df_ref, df_test):
    e = (df_ref - df_test) / df_ref
    e_mean_abs = np.mean(np.abs(e))
    return e, e_mean_abs


def main():
    # load config
    config_src = pd.read_csv("SoC_proxy_TSA/data/notes/log_horizon.csv")

    config_src["number_reps"] = (
        config_src["model_name"]
        .str.extract(r"reps\s*=\s*(\d+)", expand=False)
        .astype("Int64")
    )
    config_src["W_proxy"] = (
        config_src["model_name"]
        .str.extract(r"W_proxy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
        .astype(float)
    )

    df = config_src[["id", "dates", "W_proxy", "number_reps", "date_range"]]

    # example filter
    df = df[(df["W_proxy"] != 0.75) & (df["number_reps"] >= 45)]

    # add cem metrics
    df = cem_results(df)

    # we will plot this
    df_plot = df.copy()

    # 1) sort horizons numerically (they are years as string/numbers)
    # convert to float for sorting
    horizons_sorted = sorted(df_plot["date_range"].unique(), key=lambda x: float(x))
    x_map = {val: i for i, val in enumerate(horizons_sorted)}
    df_plot["x_base"] = df_plot["date_range"].map(x_map).astype(float)

    # 2) define two channels per horizon
    offset = 0.15
    jitter_width = 0.05
    rng = np.random.default_rng(42)

    df_plot["x_macme"] = (
        df_plot["x_base"]
        - offset
        + rng.uniform(-jitter_width, jitter_width, size=len(df_plot))
    )
    df_plot["x_ldes"] = (
        df_plot["x_base"]
        + offset
        + rng.uniform(-jitter_width, jitter_width, size=len(df_plot))
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    # 4) plot LDES channel
    # Wx = 0 -> hollow grey outline, but still at LDES x
    mask_w0 = df_plot["W_proxy"] == 0.0
    mask_w_other = df_plot["W_proxy"] != 0.0

    # 3) plot MACME channel (always filled, colour_1)
    ax.scatter(
        df_plot.loc[mask_w_other,"x_macme"],
        df_plot.loc[mask_w_other,"macme"],
        color=colour_1,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.0,
        zorder=3,
    )

    # macme with W=0
    ax.scatter(
        df_plot.loc[mask_w0,"x_macme"],
        df_plot.loc[mask_w0,"macme"],
        facecolors="white",
        edgecolors=colour_1,
        linewidth=1.0,
        alpha=0.7,
        zorder=4,
    )

    

    # LDES with W>0
    ax.scatter(
        df_plot.loc[mask_w_other, "x_ldes"],
        df_plot.loc[mask_w_other, "ldes_error"],
        color=colour_2,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.0,
        zorder=3,
    )

    # LDES with W=0
    ax.scatter(
        df_plot.loc[mask_w0, "x_ldes"],
        df_plot.loc[mask_w0, "ldes_error"],
        facecolors="white",
        edgecolors=colour_2,
        linewidth=1.0,
        alpha=0.7,
        zorder=4,
    )

    # 5) means per horizon, per metric, and per W_x for LDES
    mean_halfwidth = 0.125
    wx_values = [0.0, 1.0]  # adjust if you have more

    for horizon in horizons_sorted:
        subshift = 0.025  # small sideways nudge per W

        
        g_h = df_plot[df_plot["date_range"] == horizon]
        x_b = x_map[horizon]


        # LDES: one mean per W_x on the right channel, slightly separated
        # so they don't sit exactly on top of each other
        for i, wx in enumerate(wx_values):
            g_hw = g_h[g_h["W_proxy"] == wx]
            if g_hw.empty:
                continue
            ldes_mean = g_hw["ldes_error"].mean()
            macme_mean = g_hw["macme"].mean()
            x_center_l = x_b + offset + (i - (len(wx_values)-1)/2) * subshift
            x_center_m = x_b - offset + (i - (len(wx_values)-1)/2) * subshift
            ax.hlines(
                ldes_mean,
                x_center_l - mean_halfwidth,
                x_center_l + mean_halfwidth,
                colors=colour_2,
                linewidth=1.5,
                linestyles='dashed' if float(wx)==0 else 'solid',
                zorder=5,
            )
            ax.hlines(
                macme_mean,
                x_center_m - mean_halfwidth,
                x_center_m + mean_halfwidth,
                colors=colour_1,
                linewidth=1.5,
                linestyles='dashed' if float(wx)==0 else 'solid',
                zorder=5,
            )

    # baseline
    ax.axhline(y=0, color="black", linewidth=1, zorder=1)

    # x-ticks
    ax.set_xticks(range(len(horizons_sorted)))
    ax.set_xticklabels([rf"${v}$ Years" for v in horizons_sorted])

    # y-lims
    # ymax = max(df_plot["macme"].max(), df_plot["ldes_error"].max())
    # ymin = min(df_plot["macme"].min(), df_plot["ldes_error"].min())
    ymax_rounded = 0.6 #np.ceil(ymax * 10) / 10.0
    ymin_rounded = -0.6 #np.floor(ymin * 10) / 10.0
    ax.set_ylim(ymin_rounded, ymax_rounded)

    # y axis as %
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Error")

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------- manual legend ----------
    legend_handles = [
        # MACME
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=colour_1,
            markeredgecolor="white",
            label=r"MACME, $\overline{\epsilon^C}$"
        ),
        # LDES
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=colour_2,
            markeredgecolor="white",
            label=r"LDES Cap. Error, $\epsilon^C_{\mathrm{LDES}}$"
        ),
        # Wx = 0
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=colour_grey,
            label=r"$W_x = 0$"
        ),
        # Mean bar
        Line2D(
            [0], [0],
            color=colour_grey,
            linewidth=2,
            label="Mean"
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False)

    fig.tight_layout()
    fig.savefig("CEM_vs_Horizon.pdf", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
