flag1="--exp_name iaw_train_infer
      --run-type inference
      --exp-config inference_IAW.yaml
      INFERENCE.SPLIT test
      INFERENCE.CKPT_PATH logs/checkpoints/iaw_train/ckpt.8.pth
      INFERENCE.PREDICTIONS_FILE predictions_test.json
      "
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 25959 --nnodes=1 --nproc_per_node=1 run.py $flag1