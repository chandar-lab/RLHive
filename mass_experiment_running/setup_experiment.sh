#!/usr/bin/env bash
my_repo=https://github.com/chandar-lab/RLHive
vm_dir=~/python-vms
project_name=hive

module load python/3.7
mkdir ${vm_dir}
virtualenv --no-download ${vm_dir}/${project_name}

source ${vm_dir}/${project_name}/bin/activate

#setup code
mkdir ${vm_dir}/${project_name}/code
cd ${vm_dir}/${project_name}/code
git clone ${my_repo}
cd ${vm_dir}/${project_name}/code/RLHive
git checkout new-marlgrid-envs
pip install -r requirements.txt


#need to do this odd thing because marlgrid rendering gives errors on CC
#so remove the line which helps rendering
#this command puts a '#' at the start of line 2 in the given file argument
sed -i '2s/.*/#from pyglet.gl import */' ${vm_dir}/hive/lib/python3.7/site-packages/marlgrid/rendering.py