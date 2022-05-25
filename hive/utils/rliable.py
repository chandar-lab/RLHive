from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats.stats import find_repeats
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import argparse

#Aggregate metrics
IQM = lambda x: metrics.aggregate_iqm(x) # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0) # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)

def score_normalization(score_, min_scores, max_scores):
  norm_scores = (score_.copy() - float(min_scores))/(float(max_scores) - float(min_scores))
  return norm_scores

def save_fig(fig, name):
  file_name = '{}.pdf'.format(name)
  fig.savefig(file_name, format='pdf', bbox_inches='tight')

def convert_to_matrix(score_dict):
   keys = sorted(list(score_dict.keys()))
   return np.stack([score_dict[k] for k in keys], axis=1)

def load_and_read_rewards(env_names,path,min_env_score,max_env_score,Normalization):
  score_dict = {}
  for env_index,env_name in enumerate(env_names):
    score_ = np.load(path+'/'+env_name+'/Rewards.npy')
    if Normalization:
      score_dict[env_name] = score_normalization(score_, min_env_score[env_index], max_env_score[env_index])
  score_matrix = convert_to_matrix(score_dict)
  median, mean = MEDIAN(score_matrix), MEAN(score_matrix)
  print('{}: Median: {}, Mean: {}'.format(eval, median, mean))
  return score_matrix

def decorate_axis(ax, wrect=10, hrect=10, labelsize='large'):
  # Hide the right and top spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_linewidth(2)
  ax.spines['bottom'].set_linewidth(2)
  # Deal with ticks and the blank space at the origin
  ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
  # Pablos' comment
  ax.spines['left'].set_position(('outward', hrect))
  ax.spines['bottom'].set_position(('outward', wrect))

def plot_score_hist(agent_name,score_matrix,env_names,bins=20, figsize=(28, 14), fontsize='xx-large', N=6, extra_row=1):
  num_tasks = score_matrix.shape[1]
  N = min(N,num_tasks)
  N1 = (num_tasks // N) + extra_row
  fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
  for i in range(N):
    for j in range(N1):
      idx = j * N + i
      if idx < num_tasks:
        ax[j, i].set_title(env_names[idx], fontsize=fontsize)
        sns.histplot(score_matrix[:, idx], bins=bins, ax=ax[j,i], kde=True)
      else:
        ax[j, i].axis('off')
      decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
      ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
      if idx % N == 0:
        ax[j, i].set_ylabel(agent_name, size=fontsize)
      else:
        ax[j, i].yaxis.label.set_visible(False)
      ax[j, i].grid(axis='y', alpha=0.1)
  return fig

def get_rliable_parameters(num_eval_episodes,
                           num_runs_aggregates,
                           env_names,agent_names,
                           path,
                           test_agent,
                           Normalization=True,
                           min_env_score=None,
                           max_env_score=None,
                           plot=False):
  """
  Args:
            num_eval_episodes: No of Evaluation Episodes
            env_names: List of names of all the environments
            agent_names: List of names of all the agents/algorithms
            path: Path to the folder after where the rewards/scores are stored.
            
            The directory structure after above given path
            must be as follows:
          ├── Agent 1
          |  ├──Environment 1/
          |   ├──Rewards
          |  ├──Environment 2/
          |   ├──Rewards
          |  .
          |  .
          |  .
          |  .
          ├── Agent 2
          |  ├──Environment 1/
          |   ├──Rewards
          |  ├──Environment 2/
          |   ├──Rewards
          |  .
          |  .
          |  .
          |  .   
          ├── Agent 3
          |  ├──Environment 1/
          |   ├──Rewards
          |  ├──Environment 2/
          |   ├──Rewards
          |  .
          |  .
          |  .
          |  .            

          Here Agent 1,Agent 2,Agent 3 are the names of the algorithms provided in the agent_names parameters
          and Environment 1,Environment 2,Environment 3 are the names of the environments provided in the env_names parameters
          Rewards for all runs must be in form of a single numpy array.

          Normalization (bool): Indicates whether the normalisation is applied or not.
          min_env_score(float): Minimum value if normalisation is applied.
          max_env_score(float): Maximum value if normalisation is applied.
          num_runs_aggregates(int): Number of runs to be considered for aggregation. 
          plot(bool): Indicates whether the reliable parameters are to be plotted or not.
  """
  
  
  
  #Aggregates on environment (with default 10 runs)
  if num_runs_aggregates>num_eval_episodes:
    raise 'num_runs_aggregates should be less than or equal to num_eval_episodes'

  
  score_data_dict = {}
  for agent_name in agent_names:
    path_temp = path+'/'+agent_name
    score_data_dict[agent_name] = load_and_read_rewards(env_names,path_temp,min_env_score,max_env_score,Normalization)

  #Aggregates
  aggregates_score_dict = {key: val[:num_runs_aggregates] for key, val in score_data_dict.items()}
  aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
  aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(aggregates_score_dict, aggregate_func, reps=50000)

  if plot:
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,aggregate_interval_estimates,
        metric_names = ['Median', 'IQM', 'Mean', 'Optimality Gap'],
        algorithms=agent_names,
        xlabel_y_coordinate=-0.16,
        xlabel='Normalized Score')
    plt.show()
    save_fig(fig, 'Aggregate Result')

    #Scores Distribution plots
    for agent_name in agent_names: 
      fig_2 = plot_score_hist(agent_name,score_data_dict[agent_name],env_names,bins=20, N=6, figsize=(26, 11))
      fig_2.subplots_adjust(hspace=0.85, wspace=0.17)
      plt.show()
      save_fig(fig_2, 'Scores Distribution plots for {}'.format(agent_name))

  #Calculate score distributions and average score distributions
  score_dict = {key: score_data_dict[key][:10] for key in agent_names}
  TAU = np.linspace(0.0, 2.0, 201)
  # Higher value of reps corresponds to more accurate estimates but are slower
  # to computed. `reps` corresponds to number of bootstrap resamples.
  reps = 2000

  score_distributions, score_distributions_cis = rly.create_performance_profile(
      score_dict, TAU, reps=reps)
  avg_score_distributions, avg_score_distributions_cis = rly.create_performance_profile(
      score_dict, TAU, use_score_distribution=False, reps=reps)
  
  # Plot performance profiles (score distributions)
  if plot:
    fig_3, ax = plt.subplots(ncols=1, figsize=(7.25, 4.7))
    plot_utils.plot_performance_profiles(
    score_distributions, TAU,
    performance_profile_cis=score_distributions_cis,
    xlabel=r'Normalized Score $(\tau)$',
    labelsize='xx-large',
    ax=ax)
    
    ax.axhline(0.5, ls='--', color='k', alpha=0.4)
    fake_patches = [mpatches.Patch(color=np.random.random(3,),alpha=0.75) for alg in agent_names]
    legend = fig_3.legend(fake_patches, agent_names, loc='upper center', 
                      fancybox=True, ncol=len(agent_names), 
                      fontsize='x-large',
                      bbox_to_anchor=(0.6, 1.1))
    save_fig(fig_3, 'Performance profiles (score distributions)')

    #Plot performance profiles (score distributions) and contrast with average score distribution
    fig_4, axes = plt.subplots(ncols=2, figsize=(14.5, 4.7))

    plot_utils.plot_performance_profiles(
      score_distributions, TAU,
      performance_profile_cis=score_distributions_cis,
      xlabel=r'Normalized Score $(\tau)$',
      labelsize='xx-large',
      ax=axes[0])


    plot_utils.plot_performance_profiles(
      avg_score_distributions, TAU,
      performance_profile_cis=avg_score_distributions_cis,
      xlabel=r'Normalized Score $(\tau)$',
      ylabel=r'Fraction of tasks with score > $\tau$',
      labelsize='xx-large',
      ax=axes[1])

    axes[0].axhline(0.5, ls='--', color='k', alpha=0.4)
    axes[1].axhline(0.5, ls='--', color='k', alpha=0.4)

    fake_patches = [mpatches.Patch(color=np.random.random(3,), 
                                  alpha=0.75) for alg in agent_names]
    legend = fig_4.legend(fake_patches, agent_names, loc='upper center', 
                        fancybox=True, ncol=len(agent_names), 
                        fontsize='x-large',
                        bbox_to_anchor=(0.45, 1.1))
    fig_4.subplots_adjust(top=0.92, wspace=0.24)
    save_fig(fig_4, 'Performance profiles (score distributions) and contrast with average score distribution')

    # Performance profiles 
    fig_5, ax = plt.subplots(ncols=2, figsize=(7*2, 4.5))
    algorithms = agent_names
    plot_utils.plot_performance_profiles(
      score_distributions, TAU,
      performance_profile_cis=score_distributions_cis,
      xlabel=r'Normalized Score $(\tau)$',
      labelsize='xx-large',
      ax=ax[0])
    ax[0].set_title('Score Distributions ', size='x-large')

    xticks = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    plot_utils.plot_performance_profiles(
      score_distributions, TAU,
      performance_profile_cis=score_distributions_cis,
      xlabel=r'Normalized Score $(\tau)$',
      ylabel=r'Fraction of runs with score > $\tau$',
      labelsize='xx-large',
      use_non_linear_scaling=True,
      xticks=xticks,
      ax=ax[1])
    ax[1].set_title('Score Distributions with Non Linear Scaling', size='x-large')

    fake_patches = [mpatches.Patch(color=np.random.random(3,), 
                                  alpha=0.75) for alg in agent_names]
    legend = fig_5.legend(fake_patches, agent_names, loc='upper center', 
                        fancybox=True, ncol=len(agent_names), 
                        fontsize='x-large',
                        bbox_to_anchor=(0.45, 1.15))
    fig_5.subplots_adjust(wspace=0.24)
    save_fig(fig_5, 'Performance profiles')

  #Compute Probability of Improvement for all comparisons
  all_pairs =  {}
  for alg in (agent_names):
    if alg == test_agent:
      continue
    pair_name = f'{test_agent}-{alg}'
    all_pairs[pair_name] = (
        aggregates_score_dict[test_agent], aggregates_score_dict[alg]) 

  probabilities, probability_cis = {}, {}
  reps = 1000
  probabilities, probability_cis = rly.get_interval_estimates(all_pairs, metrics.probability_of_improvement, reps=reps)
  
  if plot:
    #Plot probabilities of improvement P(X > Y) with 95% CIs
    fig_6, ax = plt.subplots(figsize=(4, 3))
    h = 0.6
    algorithm_labels = []

    for i, (alg_pair, prob) in enumerate(probabilities.items()):
      _, alg1 = alg_pair.split('-')
      algorithm_labels.append(alg1)
      (l, u) = probability_cis[alg_pair]
      ax.barh(y=i, width=u-l, height=h, 
              left=l, color=np.random.random(3,), 
              alpha=0.75)
      ax.vlines(x=prob, ymin=i-7.5 * h/16, ymax=i+(6*h/16),
                color='k', alpha=0.85)
    ax.set_yticks(range(len(algorithm_labels)))
    ax.set_yticklabels(algorithm_labels)


    ax.set_title(fr'P({alg} > $Y$)', size='xx-large')
    plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
    ax.set_ylabel(r'Algorithm $Y$', size='xx-large')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    fig_6.subplots_adjust(wspace=0.25, hspace=0.45)
    save_fig(fig_6, 'Probabilities of improvement P(X > Y) with 95% CIs')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-num_eval_episodes", "--num_eval_episodes",type=int)
  parser.add_argument("-env_names", "--env_names",nargs="+",default=['a','b','c'])
  parser.add_argument("-agent_names", "--agent_names", nargs="+",default=['a','b','c'])
  parser.add_argument("-path", "--path")
  parser.add_argument("-test_agent", "--test_agent")
  parser.add_argument("-num_runs_aggregates", "--num_runs_aggregates",type=int)
  parser.add_argument("-Normalization", "--Normalization",default=True,type=bool)
  parser.add_argument("-min_env_score", "--min_env_score",default=[-1.0,-1.0,0.0],nargs="+",type=float)
  parser.add_argument("-max_env_score", "--max_env_score",default=[0.0,1.0,1.0],nargs="+",type=float)
  parser.add_argument("-plot", "--plot",default=False,type=bool)

  args, _ = parser.parse_known_args()
  get_rliable_parameters(
      args.num_eval_episodes,
      args.num_runs_aggregates,
      args.env_names,
      args.agent_names,
      args.path,
      args.test_agent,
      args.Normalization,
      args.min_env_score,args.max_env_score,args.plot)

if __name__ == "__main__":
    main()