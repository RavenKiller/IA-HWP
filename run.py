#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import torch

# from habitat_sim import logger
import sys

# sys.path.append('/home/vlnce/habitat-lab-v0.2.1')
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config

# from vlnce_baselines.nonlearning_agents import (
#     evaluate_agent,
#     nonlearning_inference,
# )
import torch.distributed as dist

# Avoid too much logs
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"
os.environ["HABITAT_SIM_LOG"] = "quiet"
torch.autograd.set_detect_anomaly(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local gpu id")
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    exp_name: str, exp_config: str, run_type: str, opts=None, local_rank=None
) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    dist_train = True
    config = get_config(exp_config, opts)
    if dist_train:
        dist.init_process_group(backend="nccl")
    config.defrost()

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.LOG_FILE = exp_name + "_" + config.LOG_FILE

    config.local_rank = dist.get_rank()
    if dist_train:
        config.SIMULATOR_GPU_IDS = [dist.get_rank()]
        config.GPU_NUMBERS = dist.get_world_size()
    config.freeze()
    # logger.info(f"config: {config}")  # print out all configs
    # logger.add_filehandler('logs/running_log/'+config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # if run_type == "eval" and config.EVAL.EVAL_NONLEARNING:
    #     evaluate_agent(config)
    #     return

    # if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
    #     nonlearning_inference(config)
    #     return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
