#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:00:00

module load python/3.8
conda activate saihive38

export PYTHONPATH=$PYTHONPATH:/home/saikrish/RLHive

python /home/saikrish/RLHive/hive/runners/single_agent_loop.py --config /home/saikrish/RLHive/configs/atari_rainbow.yml
