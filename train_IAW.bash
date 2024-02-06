
flag="--exp_name iaw_aux
      --run-type train
      --exp-config train_IAW.yaml

      SIMULATOR_GPU_IDS [1]
      TORCH_GPU_ID 1
      TORCH_GPU_IDS [1]

      IL.batch_size 2
      IL.lr 1e-4
      IL.way_lr 1e-5
      IL.schedule_ratio 0.75
      IL.max_traj_len 20
      "
python run.py $flag
