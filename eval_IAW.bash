
flag1="--exp_name iaw_train
      --run-type eval
      --exp-config eval_IAW.yaml
      EVAL.SPLIT val_seen
      "
CUDA_VISIBLE_DEVICES=1 torchrun --master_port 25958 --nnodes=1 --nproc_per_node=1 run.py $flag1
