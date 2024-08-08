#!/bin/bash
#SBATCH --job-name ia_hwp
#SBATCH --partition L40
#SBATCH -N 1
#SBATCH -n 1 
#SBATCH --gres=gpu:l40:3
#SBATCH --cpus-per-task=21
#SBATCH --output=%j_train.out

source /share/home/tj90055/.bashrc
conda activate iaw
cd /share/home/tj90055/hzt/IA-HWP
# flag="--exp_name iaw_aux_new
#       --run-type eval
#       --exp-config eval_IAW.yaml

#       SIMULATOR_GPU_IDS [0]
#       TORCH_GPU_ID 0
#       TORCH_GPU_IDS [0]

#       IL.batch_size 2
#       IL.lr 1e-4
#       IL.way_lr 1e-5
#       IL.schedule_ratio 0.75
#       IL.max_traj_len 20
#       "
# python run.py $flag

export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nnodes=1 --nproc_per_node=3 run.py --exp_name iaw_train --run-type train --exp-config train_IAW.yaml