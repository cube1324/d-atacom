from contourpy import contour_generator

import wandb
import os
from mushroom_rl.utils.plot import get_mean_and_confidence
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# # Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
COLOR_PALETTE = ["#fd7f6f", "#bd7ebe", "#3293db", "#7cc202", "#04a777", "#ffb55a", "#bd7ebe", "#3f423e"]

# COLOR_PALETTE = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0",
#                  "#3f423e"]

title_dict = {"atacom_sac_0.5": "D-ATACOM@0.5", "wc_lag_sac_0.5": "WCSAC@0.5", "wc_lag_sac_0.9": "WCSAC@0.9",
              "wc_lag_sac_0.1": "WCSAC@0.1", "lag_sac": "LagSAC", "PPOLag": "PPOLag", "RCPO": "RCPO",
              "OnCRPO": "OnCRPO", "CPO": "CPO", "PCPO": "PCPO", "TRPOLag": "TRPOLag", "sac": "SAC",
              "atacom_sac_0.9": "D-ATACOM@0.9", "atacom_sac_0.1": "D-ATACOM@0.1",
              "baselineatacom_sac": "ATACOM + non-FI",
              "baselineatacom_sac_viability": "ATACOM + FI",
              "CAPPETS": "CAPPETS", "SafeLOOP": "SafeLOOP", "clbf_sac": "CBF-SAC", "safelayer_td3": "SafeLayerTD3",
              }

colour_dict = {"atacom_sac_0.9": 0, "atacom_sac_0.5": 0, "atacom_sac_0.1": 0, "wc_lag_sac_0.5": 1, "wc_lag_sac_0.9": 1,
               "wc_lag_sac_0.1": 1, "lag_sac": 2, "PPOLag": 3, "RCPO": 4, "OnCRPO": 5, "CPO": 6, "PCPO": 7,
               "TRPOLag": 8, "sac": 9, "baselineatacom_sac": 8, "baselineatacom_sac_viability": 7,
               "CAPPETS": 1, "SafeLOOP": 2, "clbf_sac": 3, "safelayer_td3": 4}


def plot_zero_level(ax, X, Y, Z, **kwargs):
    contour_gen = contour_generator(X, Y, Z)
    lines = contour_gen.lines(0)

    for line in lines:
        if len(line) > 20:
            ax.plot(*line.T, **kwargs)


def smooth(scalars, weight: float):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def plot_learning_curve(grouped_runs, metric_key, title, xlabel, ylabel, steps_per_epoch, save_dir=None, linewidth=8,
                        smooth_weight=None):
    # Spring Pastels from https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data

    plt.rcParams["font.size"] = 55
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['axes.linewidth'] = 2

    plt.figure(figsize=(16, 10))
    color_idx = 0

    # names = ["no decay", "decay=$0.97^{epoch}$", "decay=$0.98^{epoch}$"]

    for group_key in sorted(grouped_runs[metric_key].keys()):
        metric_df = grouped_runs[metric_key][group_key]

        mean, interval = get_mean_and_confidence(metric_df.transpose())

        mean = mean[:100]
        interval = interval[:100]

        if smooth_weight is not None:
            mean = np.array(smooth(mean, smooth_weight))
            interval = np.array(smooth(interval, smooth_weight))

        x = np.arange(mean.shape[-1]) * steps_per_epoch

        temp = {
            "atacom_sac_fixed_delta_mushroomv1_e4f9d1277326ef29d4fef6dc35144c20107c9365": "0.3",
            "atacom_sac_fixed_delta_mushroomv1_64116e77ce76c431e35ad2c047d8a149d64822e9": "0.1",
            "atacom_sac_fixed_delta_mushroomv1_b7ae2a47d34831205930b91c71a1127f211a8125": "0.5",
            "atacom_sac_fixed_delta_mushroomv1_3e5089655030c94bf6db8aa7523301fc41125d57": "1",
            "atacom_sac_fixed_delta_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "3",
            "atacom_sac_fill_up_1145e0587f424822aedbf9ed683748dec3297361": "learned"

        }

        # temp = {"atacom_sac_model_missmatch_0.8_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.8",
        #         "atacom_sac_model_missmatch_0.4_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.4",
        #         "atacom_sac_model_missmatch_0.2_mushroomv1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.2",
        #         "atacom_sac_model_miss_0.1_88a16eff42f0d896af418b577dc45234dd4fb21b": "0.1",
        #         "atacom_sac_fill_up_1145e0587f424822aedbf9ed683748dec3297361": "0"}

        plt.plot(x, mean, label=f"$\sigma$={temp[group_key]}", color=COLOR_PALETTE[color_idx], linewidth=linewidth)
        plt.fill_between(x, mean - interval, mean +
                         interval, alpha=0.2, color=COLOR_PALETTE[color_idx])
        color_idx += 1
    plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # leg = plt.legend(ncol=3)
    plt.title(title)
    # leg_lines = leg.get_lines()
    # plt.setp(leg_lines, linewidth=linewidth)
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params('both', length=20, width=4, which='major')
    ax.tick_params('both', length=10, width=2, which='minor')
    if metric_key == "sum_cost" or metric_key == "max_violation":
        axins = inset_axes(ax, 8, 4,
                           loc=1)  # , bbox_to_anchor=(0.2, 0.55), bbox_transform=ax.figure.transFigure)  # no zoom
        color_idx = 0
        for group_key in sorted(grouped_runs[metric_key].keys()):
            metric_df = grouped_runs[metric_key][group_key]

            mean, interval = get_mean_and_confidence(metric_df.transpose())

            axins.plot(x, mean, color=COLOR_PALETTE[color_idx], linewidth=linewidth)
            axins.fill_between(x, mean - interval, mean +
                               interval, alpha=0.2, color=COLOR_PALETTE[color_idx])
            color_idx += 1

        axins.set_xlim(0.5e6, 1e6)
        if metric_key == "sum_cost":
            axins.set_ylim(-0.1, 2)
        else:
            axins.set_ylim(0, 0.05)
        axins.xaxis.set_major_locator(ticker.NullLocator())
        box, c1, c2 = mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        plt.setp([box, c1, c2], linewidth=2)
    # plt.show()

    if save_dir is not None:
        plt.savefig(save_dir + f"/{ylabel}.pdf", dpi=1000)

    handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 3, 1, 2, 5]
    order = [0, 2, 5, 4, 1, 3]
    leg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=6, loc='center left',
                     bbox_to_anchor=(1, 0.5), frameon=False)

    # leg = plt.legend(ncol=6, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=linewidth + 2)

    export_legend(leg, filename=os.path.join(save_dir, "legend.pdf"))

    plt.show()


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def download_run_history(entity, project, save_path, samples, filters):
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}", filters=filters)

    for run in runs:
        run_hist = run.history(samples=samples)
        if "Metrics/EpJ" in run_hist.keys():
            run_hist["J"] = run_hist["Metrics/EpJ"]
            run_hist["R"] = run_hist["Metrics/EpRet"]
            run_hist["episode_length"] = run_hist["Metrics/EpLen"]
            run_hist["max_violation"] = run_hist["Metrics/EpMaxCost"]
            run_hist["sum_cost"] = run_hist["Metrics/EpCost"]
            # run_hist["success"] = run_hist["Metrics/Success"]

        if not "success" in run_hist.keys():
            run_hist["success"] = 0

        try:
            run_hist = run_hist[["J", "R", "episode_length", "max_violation", "sum_cost", "delta"]]

            run_hist.to_csv(f"{save_path}/{run.id}.csv", index=False)
        except:
            print(f"Failed to save {run.group} {run.id}")


def group_run_histories_by_key(entity, project, save_path, group_key, filters):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=filters)

    metrics = {}
    for run in runs:
        hist = pd.read_csv(f"{save_path}/{run.id}.csv")
        temp = run.config
        temp["group"] = run.group
        if "algo" in temp.keys():
            temp["alg"] = temp["algo"]
        for el in group_key:
            temp = temp.get(el)

        # group_key_val = float(temp)
        group_key_val = temp
        for key in hist.keys():
            if key not in metrics:
                metrics[key] = {}

            if group_key_val not in metrics[key]:
                metrics[key][group_key_val] = pd.DataFrame()

            # Only take 10 seeds for ablation studies
            if run.state == "finished" and metrics[key][group_key_val].shape[1] < 10:
                metrics[key][group_key_val][run.id] = hist[key]
    return metrics


def make_path(project, name):
    data_path = os.path.join("data", project, name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    plot_path = os.path.join("plots", project, name)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    return data_path, plot_path


def plot_air_hockey_constraint(agent):
    from air_hockey_challenge.utils.kinematics import inverse_kinematics
    grid_pos_x = np.linspace(-0.5, 0.5, 10)
    grid_pos_y = np.linspace(0.6, 1.2, 10)
    X, Y = np.meshgrid(grid_pos_x, grid_pos_y)
    joint_pos = []

    pass
