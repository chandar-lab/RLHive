#!/usr/bin/env bash
userhome=~
vm_dir=~/python-vms
project_name=MABC
virtualhome=${vm_dir}/${project_name}

mkdir -p ${userhome}/scratch/${project_name}

MEMORY=2G
TIME=00:80:00
CPUS_PER_TASK=1
CC_ACCOUNT=def-adityam

output_dir=${virtualhome}/results
source_config=${virtualhome}/code/RLHive/configs/MABC/config.yml
temp_config=${virtualhome}/code/RLHive/configs/MABC/temp_config.yml

algos=(decentralized selfplay)
env_names=(MABC-v0)
stack_sizes=(2 5)
seeds=(0 1 2 3 4)

echo 'Submitting SBATCH jobs...'
for algo in ${algos[@]}
do
	for env_name in ${env_names[@]}
	do
		for stack_size in ${stack_sizes[@]}
		do
			for seed in ${seeds[@]}
			do
				echo "#!/bin/bash" >> temprun.sh
				echo "#SBATCH --account=${CC_ACCOUNT}" >> temprun.sh
				echo "#SBATCH --job-name=algo-${algo}_env-${env_name}_stacksize-${stack_size}_seed-${seed}" >> temprun.sh
				echo "#SBATCH --output=$userhome/scratch/${project_name}/%x-%j.out" >> temprun.sh
				echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> temprun.sh
				echo "#SBATCH --mem=${MEMORY}" >> temprun.sh
				echo "#SBATCH --time=${TIME}" >> temprun.sh

				echo "source $virtualhome/bin/activate" >> temprun.sh
				echo "cd $virtualhome/code/RLHive" >> temprun.sh
				echo "python -m mass_experiment_running.create_config --output-directory=${output_dir} --input-config=${source_config} --output-config=${temp_config} --env_name=${env_name} --algo=${algo} --exp_name=${project_name} --stack_size=${stack_size} --seed=${seed}" >> temprun.sh

				echo "python -m hive.runners.multi_agent_loop -c ${temp_config}" >> temprun.sh

				eval "sbatch temprun.sh"
				#added to submit job again if slurm error occurs (timeout error send/recv)
				while [ ! $? == 0 ]
				do
					eval "sbatch temprun.sh"
				done

				# sleep 1
				rm temprun.sh
				rm ${temp_config}
			done
		done
	done
done