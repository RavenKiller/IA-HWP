<div align="center">

<h1>Instruction-Aligned Hierarchical Waypoint Planner for Vision-and-Language Navigation in Continuous Environments</h1>

<div>
    <a href='https://link.springer.com/article/10.1007/s10044-024-01339-z' target='_blank'>[Paper]</a>
</div>
</div>

<br />
<div align="center">
    <img src="https://github.com/user-attachments/assets/3a2fb5af-1600-40ea-9d20-c3a9ea83641b", width="1000", alt="Model architecture">
</div>

# Preparation
1. Install [habitat-lab and VLN-CE](https://github.com/jacobkrantz/VLN-CE)
2. Install other requirements:
    ```shell
    pip install -r requirements.txt
    ```
3. Prepare datasets
    - R2R_VLNCE_v1-2
        - [R2R_VLNCE_v1-2.zip](https://drive.google.com/file/d/1rRdQtqWIpYDAIO7LXDEsB2J75Jl8HMdA/view?usp=sharing)
        - [R2R_VLNCE_v1-2_preprocessed.zip](https://drive.google.com/file/d/1j9sQ0w4wFYSafh42U8VCuKTwMrnrsV6z/view?usp=sharing)
    - [ddppo](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo)

    ```
    +-data
    |   +-checkpoints
    |   +-datasets        
    |   |   +-R2R_VLNCE_v1-2_preprocessed
    |   +- ddppo-models    
    ```

# Train, evaluate and test
1. `train_IAW.bash` is the script for training
2. `eval_IAW.bash` is the script for evaluating
3. `inference_IAW.bash` is the script for inference
4. The weight with the best performance on VLN-CE val unseen can be downloaded [here](https://www.jianguoyun.com/p/De-0swYQhY--CRiB49YFIAA) (Access code: `iahwp`).

# Citation
```
﻿@Article{He2024,
    author={He, Zongtao
        and Wang, Naijia
        and Wang, Liuyi
        and Liu, Chengju
        and Chen, Qijun},
    title={Instruction-aligned hierarchical waypoint planner for vision-and-language navigation in continuous environments},
    journal={Pattern Analysis and Applications},
    year={2024},
    month={Oct},
    day={03},
    volume={27},
    number={4},
    pages={132},
    issn={1433-755X},
    doi={10.1007/s10044-024-01339-z},
    url={https://doi.org/10.1007/s10044-024-01339-z}
}
```
