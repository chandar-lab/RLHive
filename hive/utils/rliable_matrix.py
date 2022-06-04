import argparse
import os
import pickle as pkl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from rliable import library as rly
from rliable import metrics, plot_utils
from scipy.stats.stats import find_repeats


def score_normalization(score_, min_scores, max_scores):
    norm_scores = (score_.copy() - min_scores) / (max_scores - min_scores)
    return norm_scores


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


def load_and_read_rewards(
    agent_name, env_names, path, min_env_score, max_env_score, normalization
):
    score_dict = {}
    for env_index, env_name in enumerate(env_names):
        path_temp = os.path.join(path, env_name, agent_name)
        seed_s = [i for i in os.listdir(path_temp) if i[0] != "."]
        score_ = []
        for seed in seed_s:
            job_files = [
                i for i in os.listdir(os.path.join(path_temp, seed)) if i[0] != "."
            ]
            for job_file in job_files:
                pkl_file = open(
                    os.path.join(
                        path_temp, seed, job_file, "logger/logger_0/log_data.p"
                    ),
                    "rb",
                )
                data_file = pkl.load(pkl_file)
                pkl_file.close()
                score_.extend(data_file["test/agent_reward"][0])
        score_ = np.array(score_)
        if normalization:
            score_dict[env_name] = score_normalization(
                score_, min_env_score[env_index], max_env_score[env_index]
            )
    score_matrix = convert_to_matrix(score_dict)
    median, mean = metrics.aggregate_median(score_matrix), metrics.aggregate_mean(
        score_matrix
    )
    return score_matrix


def get_rliable_parameters(
    num_runs_aggregates,
    path,
    output_path,
    test_agent,
    normalization=True,
    min_env_score=None,
    max_env_score=None,
):
    """
    Args:
              num_runs_aggregates(int): Number of runs to be considered for aggregation.
              path: Path to the folder  where the rewards/scores are stored.
              output_path: Path to the folder where the matrices are stored.

            normalization (bool): Indicates whether the normalisation is applied or not.
            min_env_score: List of minimum value if normalisation is applied. Value must be provided for each environment.
                          If single float value is provided then it is taken as minimum value for all environment.
            max_env_score: List of maximum value if normalisation is applied. Value must be provided for each environment.
                          If single float value is provided then it is taken as maximum value for all environment
    """

    env_names = [i for i in os.listdir(path) if i[0] != "."]
    agent_names = [
        i for i in os.listdir(os.path.join(path, env_names[0])) if i[0] != "."
    ]
    if len(min_env_score) == 1:
        min_env_score = [float(min_env_score[0])] * len(env_names)
    if len(max_env_score) == 1:
        max_env_score = [float(max_env_score[0])] * len(env_names)
    score_data_dict = {}
    for agent_name in agent_names:
        score_data_dict[agent_name] = load_and_read_rewards(
            agent_name, env_names, path, min_env_score, max_env_score, normalization
        )

    # Aggregates
    aggregates_score_dict = {
        key: val[:num_runs_aggregates] for key, val in score_data_dict.items()
    }
    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        aggregates_score_dict, aggregate_func, reps=50000
    )

    # Calculate score distributions and average score distributions
    score_dict = {key: score_data_dict[key][:10] for key in agent_names}
    TAU = np.linspace(0.0, 2.0, 201)
    # Higher value of reps corresponds to more accurate estimates but are slower to computed. `reps` corresponds to number of bootstrap resamples.
    reps = 2000

    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, TAU, reps=reps
    )
    (
        avg_score_distributions,
        avg_score_distributions_cis,
    ) = rly.create_performance_profile(
        score_dict, TAU, use_score_distribution=False, reps=reps
    )

    # Compute Probability of Improvement for all comparisons
    all_pairs = {}
    for alg in agent_names:
        if alg == test_agent:
            continue
        pair_name = f"{test_agent}-{alg}"
        all_pairs[pair_name] = (
            aggregates_score_dict[test_agent],
            aggregates_score_dict[alg],
        )
    reps = 1000
    probabilities, probability_cis = rly.get_interval_estimates(
        all_pairs, metrics.probability_of_improvement, reps=reps
    )

    # print('Aggregate Scores {}'.format(aggregate_scores))
    # print('Aggregate Interval Estimates {}'.format(aggregate_interval_estimates))
    # print('Score Distributions {}'.format(score_distributions))
    # print('Average Score Distributions {}'.format(avg_score_distributions))
    # print('Probabilities {}'.format(probabilities))
    all_matrixs = {}
    all_matrixs["Aggregate_Scores"] = aggregate_scores
    all_matrixs["Aggregate_Interval_Estimates"] = aggregate_interval_estimates
    all_matrixs["Score_Distributions"] = score_distributions
    all_matrixs["Average_Score_Distributions"] = avg_score_distributions
    all_matrixs["Probabilities"] = probabilities

    output_path = output_path + "/matrixs.p"
    with open(output_path, "wb") as f:
        pkl.dump(all_matrixs, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_runs_aggregates", "--num_runs_aggregates", type=int)
    parser.add_argument("-path", "--path")
    parser.add_argument("-output_path", "--output_path")
    parser.add_argument("-test_agent", "--test_agent")
    parser.add_argument("-normalization", "--normalization", default=True)
    parser.add_argument(
        "-min_env_score", "--min_env_score", default=[-1.0, -1.0, 0.0], nargs="+"
    )
    parser.add_argument(
        "-max_env_score", "--max_env_score", default=[0.0, 1.0, 1.0], nargs="+"
    )

    args, _ = parser.parse_known_args()
    get_rliable_parameters(
        args.num_runs_aggregates,
        args.path,
        args.output_path,
        args.test_agent,
        args.normalization,
        args.min_env_score,
        args.max_env_score,
    )


if __name__ == "__main__":
    main()
