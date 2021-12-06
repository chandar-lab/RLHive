import argparse
import json
import logging
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

logging.basicConfig()


def find_single_run_data(run_folder):
    run_data_file = None
    for path, _, filenames in os.walk(run_folder):
        if "log_data.p" in filenames:
            run_data_file = os.path.join(path, "log_data.p")
            logging.info(f"Found run data at {run_data_file}")
            break
    if run_data_file is None:
        logging.info(f"Run data not found for {run_folder}")
        return

    with open(run_data_file, "rb") as f:
        run_data = pickle.load(f)

    return run_data


def find_all_runs_data(runs_folder):
    all_runs_data = defaultdict(lambda: [])
    for run_folder in os.listdir(runs_folder):
        run_data = find_single_run_data(os.path.join(runs_folder, run_folder))
        for key in run_data:
            all_runs_data[key].append(run_data[key])
    return all_runs_data


def find_all_experiments_data(experiments_folder, runs_folders):
    data = {}
    for runs_folder in runs_folders:
        full_runs_folderpath = os.path.join(experiments_folder, runs_folder)
        data[runs_folder] = find_all_runs_data(full_runs_folderpath)
    return data


def aggregate_data(
    experiment_data,
    x_key,
    y_key,
    num_sampled_points=1000,
    drop_last=True,
):
    if drop_last:
        y_data = [data[0][:-1] for data in experiment_data[y_key]]
        x_data = [
            [x_datas[x_key] for x_datas in data[1][:-1]]
            for data in experiment_data[y_key]
        ]
    else:
        y_data = [data[0] for data in experiment_data[y_key]]
        x_data = [
            [x_datas[x_key] for x_datas in data[1]] for data in experiment_data[y_key]
        ]
    min_x = min([min(xs) for xs in x_data])
    max_x = max([max(xs) for xs in x_data])
    full_xs = np.linspace(min_x, max_x, num_sampled_points)
    interpolated_ys = np.array(
        [np.interp(full_xs, xs, ys) for (xs, ys) in zip(x_data, y_data)]
    )

    return full_xs, interpolated_ys


def generate_lineplot(
    x_datas,
    y_datas,
    smoothing_fn=None,
    line_labels=None,
    xlabel=None,
    ylabel=None,
    cmap_name=None,
    output_file="output.png",
):
    plt.figure()
    if cmap_name is None:
        cmap_name = "tab10" if len(x_datas) <= 10 else "tab20"
    cmap = cm.get_cmap(cmap_name)
    if line_labels is None:
        line_labels = [None] * len(x_datas)
    for idx, (x_data, y_data, line_label) in enumerate(
        zip(x_datas, y_datas, line_labels)
    ):
        mean_ys = np.mean(y_data, axis=0)
        std_ys = np.std(y_data, axis=0)
        if smoothing_fn is not None:
            mean_ys = smoothing_fn(mean_ys)
            std_ys = smoothing_fn(std_ys)
        plt.plot(x_data, mean_ys, label=line_label, color=cmap(idx))
        plt.fill_between(
            x_data, mean_ys - std_ys, mean_ys + std_ys, color=cmap(idx), alpha=0.1
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def plot_results(
    experiments_folder,
    x_key,
    y_key,
    runs_folders=None,
    run_names=None,
    x_label=None,
    y_label=None,
    cmap_name=None,
    smoothing_fn=None,
    num_sampled_points=100,
    output_file="output.png",
):
    if runs_folders is None:
        runs_folders = os.listdir(experiments_folder)
    if run_names is None:
        run_names = runs_folders
    data = find_all_experiments_data(experiments_folder, runs_folders)

    aggregated_xs, aggregated_ys = list(
        zip(
            *[
                aggregate_data(
                    data[run_folder],
                    x_key,
                    y_key,
                    num_sampled_points=num_sampled_points,
                )
                for run_folder in runs_folders
            ]
        )
    )
    generate_lineplot(
        aggregated_xs,
        aggregated_ys,
        smoothing_fn=smoothing_fn,
        line_labels=run_names,
        xlabel=x_label,
        ylabel=y_label,
        cmap_name=cmap_name,
        output_file=output_file,
    )


def create_exponential_smoothing_fn(smoothing=0.1):
    def fn(values):
        values = np.array(values)
        return np.array(pd.DataFrame(values).ewm(alpha=1 - smoothing).mean()[0])

    return fn


def create_moving_average_smoothing_fn(running_average=10):
    def fn(values):
        return np.convolve(values, np.ones(running_average), "valid") / running_average

    return fn


def get_smoothing_fn(smoothing_fn, smoothing_fn_kwargs):
    if smoothing_fn == "exponential":
        return create_exponential_smoothing_fn(**smoothing_fn_kwargs)
    elif smoothing_fn == "moving_average":
        return create_moving_average_smoothing_fn(**smoothing_fn_kwargs)
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_folder", required=True)
    parser.add_argument("--x_key", required=True)
    parser.add_argument("--y_key", required=True)
    parser.add_argument("--runs_folders", nargs="+")
    parser.add_argument("--run_names", nargs="+")
    parser.add_argument("--x_label")
    parser.add_argument("--y_label")
    parser.add_argument("--cmap_name")
    parser.add_argument("--smoothing_fn", choices=["exponential", "moving_average"])
    parser.add_argument("--smoothing_fn_kwargs")
    parser.add_argument("--num_sampled_points", type=int, default=100)
    parser.add_argument("--output_file", default="output.png")

    args = parser.parse_args()
    if args.smoothing_fn is not None:
        if args.smoothing_fn_kwargs is not None:
            smoothing_fn_kwargs = json.loads(args.smoothing_fn_kwargs)
        smoothing_fn = get_smoothing_fn(args.smoothing_fn, smoothing_fn_kwargs)
    else:
        smoothing_fn = None
    plot_results(
        experiments_folder=args.experiments_folder,
        x_key=args.x_key,
        y_key=args.y_key,
        runs_folders=args.runs_folders,
        run_names=args.run_names,
        x_label=args.x_label,
        y_label=args.y_label,
        cmap_name=args.cmap_name,
        smoothing_fn=smoothing_fn,
        num_sampled_points=args.num_sampled_points,
        output_file=args.output_file,
    )
