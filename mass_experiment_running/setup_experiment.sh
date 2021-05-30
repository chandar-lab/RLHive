#!/usr/bin/env bash
my_repo=https://github.com/chandar-lab/RLHive
vm_dir=~/python-vms
project_name=MABC

module load python/3.7
mkdir ${vm_dir}
virtualenv --no-download ${vm_dir}/${project_name}

source ${vm_dir}/${project_name}/bin/activate

#setup code
mkdir ${vm_dir}/${project_name}/code
cd ${vm_dir}/${project_name}/code
git clone ${my_repo}
cd ${vm_dir}/${project_name}/code/RLHive
pip install -r requirements.txt
