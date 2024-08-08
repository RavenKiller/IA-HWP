

for i in 5
do
flag1="--exp_name single120
      --run-type eval
      --exp-config eval_IAW0.yaml
      NUM_ENVIRONMENTS 1
      EVAL.SPLIT single120
      EVAL_CKPT_PATH_DIR logs/checkpoints/iaw_train0/ckpt.${i}.pth
      EVAL.VISUALIZE True
      "
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 25907 --nnodes=1 --nproc_per_node=1 run.py $flag1
cp /share/home/tj90055/hzt/IA-HWP/logs/episode:120.png /share/home/tj90055/hzt/IA-HWP/logs/iaw_train0_ckpt${i}_ep120.png
cp /share/home/tj90055/hzt/IA-HWP/map1.npy /share/home/tj90055/hzt/IA-HWP/logs/iaw_train0_ckpt${i}_map1.npy
cp /share/home/tj90055/hzt/IA-HWP/map2.npy /share/home/tj90055/hzt/IA-HWP/logs/iaw_train0_ckpt${i}_map2.npy
done

