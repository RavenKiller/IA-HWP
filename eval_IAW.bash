
flag1="--exp_name iaw-aux-go
      --run-type eval
      --exp-config eval_IAW.yaml
      EVAL.SPLIT val_seen
      "


python run.py $flag1
