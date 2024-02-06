# Instruction-aligned waypoint model
# 准备
1. 安装[habitat-lab环境](https://github.com/jacobkrantz/VLN-CE)
2. 安装VLN-CE
3. 安装其他库
    ```shell
    python -m pip install -r requirements.txt
    ```
# 数据
准备数据集&预训练模型
- R2R_VLNCE_v1-2
    - [R2R_VLNCE_v1-2.zip](https://drive.google.com/file/d/1rRdQtqWIpYDAIO7LXDEsB2J75Jl8HMdA/view?usp=sharing)
    - [R2R_VLNCE_v1-2_preprocessed.zip](https://drive.google.com/file/d/1j9sQ0w4wFYSafh42U8VCuKTwMrnrsV6z/view?usp=sharing)
- [ddppo](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo)


目录结构
```
+-data
|   +-checkpoints
|   +-datasets        
|   |   +-R2R_VLNCE_v1-2_preprocessed
|   +- ddppo-models    
```

# 训练指令
1. 训练用了CMA的两阶段训练法，会先加载第一阶段训练中unseen表现最好的模型权重，然后再进行二阶段的训练。
2. 预加载模型地址：https://pan.baidu.com/s/1Pp4ly1hlcJEJ5YjQ6wowfA?pwd=h417 

3. 解压到logs/checkpoints/iaw-0.1/ 目录下
4. 然后执行
    ```shell
    sh train_IAW.bash
    ```
    或
    ```shell
    python -u run.py --exp_name iaw_train --run-type train --exp-config train_IAW.yaml 
    ```

5. 偷懒没有定义开关，如果想做消融：
    
    -   控制损失函数，修改ss_trainer_IAW.py
        ```python
        # 交叉熵损失
        current_loss = F.cross_entropy(logits, oracle_actions.squeeze(1), reduction="none")

        # focal-loss
        current_loss = self.focal_loss(logits, oracle_actions.squeeze(1))
        ```
    -   控制aux loss，修改ss_trainer_IAW.py 中的 dir_weight为全1
    -   控制 WA or GO，修改ss_trainer_IAW.py 中环境返回信息
        ```python
        # Goal-Oriented
        dist_k = self.envs.call_at(j, 
            "cand_dist_to_goal", {
                "angle": angle_k, "forward": forward_k,
            })

        # Waypoint-Aligned
        way_pos = observations[j]['vln_law_action_sensor']
        dist_k = self.envs.call_at(j, 
            "cand_dist_to_law", {
                "angle": angle_k, "forward": forward_k, "pos": way_pos,
            })
        ```



# 验证
1. 模型地址：https://pan.baidu.com/s/1VjtQfq99oiMqBQ6PLzk8HQ?pwd=h417

2. 解压到logs/checkpoints/iaw_aux/ 目录下

3. 执行
    ```shell
    sh eval_IAW.bash
    ```
    或
    ```shell
    python -u run.py --exp_name iaw_eval --run-type eval --exp-config eval_IAW.yaml 
    ```
4. 指定eval的验证集
    ```shell
    python -u run.py --exp_name iaw_eval --run-type eval --exp-config eval_IAW.yaml EVAL.SPLIT val_unseen
    ```
# 推理
    
1. 将在r2r/test数据集上测试

    ```shell
    sh inference_IAW.bash
    ```