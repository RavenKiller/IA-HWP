import gc
import os
import random
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations
from .utils import (
    length2mask, dir_angle_feature,
)
from .Focal_Loss import focal_loss
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy

@baseline_registry.register_trainer(name="schedulesampler-IAW")
class SSTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.focal_loss = focal_loss(alpha=[1,1,1,1,1,1,1,1,1,1,1,1,1], num_classes=13,reduce=False) # 对最后一个类别(stop), 施加0.75权重
        # 辅助损失权重 dir weight，stop与其他非停止的类别角度相差pi/2
        self.dir_weight = np.array([0,0.06699,0.25,0.5,0.75,0.933,1,0.933,0.75,0.5,0.25,0.06699,0.5])
        self.loss_weight = 2

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        # os.makedirs(self.lmdb_features_dir, exist_ok=True)
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def save_checkpoint(self, epoch: int, step_id: int) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "waypoint_model_state_dict": self.waypoint_model.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                # "way_optim_state": self.way_optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"),
        )

    def allocate_allowed_episode_by_scene(self):
        ''' discrete waypoints coordinates directly projected from MP3D '''
        with gzip.open(
            self.config.TASK_CONFIG.DATASET.DATA_PATH.format(
                split=self.split)
        ) as f:
            data = json.load(f) # dict_keys(['episodes', 'instruction_vocab'])

        ''' continuous waypoints coordinates by shortest paths in Habitat '''
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=self.split)
        ) as f:
            gt_data = json.load(f)

        data = data['episodes']
        # long_episode_ids = [int(k) for k,v in gt_data.items() if len(v['actions']) > self.config.IL.max_traj_len]
        long_episode_ids = []
        average_length = (len(data) - len(long_episode_ids))//self.world_size

        episodes_by_scene = {}
        for ep in data:
            scan = ep['scene_id'].split('/')[1]
            if scan not in episodes_by_scene.keys():
                episodes_by_scene[scan] = []
            if ep['episode_id'] not in long_episode_ids:
                episodes_by_scene[scan].append(ep['episode_id'])
            else:
                continue

        ''' split data in each environments evenly to different GPUs ''' # averaging number set problem
        values_to_scenes = {}
        values = []
        for k,v in episodes_by_scene.items():
            values.append(len(v))
            if len(v) not in values_to_scenes.keys():
                values_to_scenes[len(v)] = []
            values_to_scenes[len(v)].append(k)

        groups = self.world_size
        values.sort(reverse=True)
        last_scene_episodes = episodes_by_scene[values_to_scenes[values[0]].pop()]
        values = values[1:]

        load_balance_groups = [[] for grp in range(groups)]
        scenes_groups = [[] for grp in range(groups)]

        for v in values:
            current_total = [sum(grp) for grp in load_balance_groups]
            min_index = np.argmin(current_total)
            load_balance_groups[min_index].append(v)
            scenes_groups[min_index] += episodes_by_scene[values_to_scenes[v].pop()]

        for grp in scenes_groups:
            add_number = average_length - len(grp)
            grp += last_scene_episodes[:add_number]
            last_scene_episodes = last_scene_episodes[add_number:]

        # episode_ids = [ep['episode_id'] for ep in data if
        #                ep['episode_id'] not in long_episode_ids]
        # scenes_groups[self.local_rank] = episode_ids[
        #                 self.local_rank:self.world_size * average_length:self.world_size]
        return scenes_groups[self.local_rank]

    def train_ml(self, in_train=True, train_tf=False, train_rl=False):
        self.envs.resume_all()
        observations = self.envs.reset()

        shift_index = 0
        for i, ep in enumerate(self.envs.current_episodes()):
            if ep.episode_id in self.trained_episodes:
                i = i - shift_index
                observations.pop(i)
                self.envs.pause_at(i)
                shift_index += 1
                if self.envs.num_envs == 0:
                    break
            else:
                self.trained_episodes.append(ep.episode_id)

        if self.envs.num_envs == 0:
            return -1

        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, dtype=torch.bool, device=self.device
        )

        ml_loss = 0.
        angle_reward = 0.
        total_weight = 0.
        not_done_index = list(range(self.envs.num_envs))
        init_num_envs = self.envs.num_envs

        # initialize previous predicted step distance encoding
        prev_way_dists = [0.0] * self.envs.num_envs
        way_hidden_states = []
        way_logp_dict = []
        way_rewards_dict = []
        way_masks_dict = []
        # initialize view state (single state -- no decoupling)
        view_states = torch.zeros(
            self.envs.num_envs,
            self.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        # # initialize waypoint states and languages
        # view model languages
        all_view_lang_feats, all_view_lang_masks = self.policy.net(
            mode = "language",
            observations = batch,
        )

        for stepk in range(self.max_len):
            view_language_features = all_view_lang_feats[not_done_index]
            view_lang_mask = all_view_lang_masks[not_done_index]

            # agent's current position and heading
            positions = []; prev_headings = []
            for ob_i in range(len(observations)):
                agent_state_i = self.envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                prev_headings.append(agent_state_i['heading'])

            # encoding views
            cand_rgb, cand_depth, cand_direction, \
            cand_mask, candidate_lengths, \
            batch_angles = self.policy.net(
                mode = "view",
                observations = batch,
                in_train = in_train,
            )

            # view selection action logits
            logits, view_states = self.policy.net(
                mode = 'selection',
                observations = batch,
                instruction = view_language_features,
                lang_mask = view_lang_mask,
                view_states = view_states,
                # way_states = way_states,
                prev_headings = prev_headings,
                cand_rgb = cand_rgb, 
                cand_depth = cand_depth,
                cand_direction = cand_direction,
                cand_mask = cand_mask,
                masks = not_done_masks,
            )
            logits = logits.masked_fill_(cand_mask, -float('inf'))
            total_weight += len(candidate_lengths)

            # get resulting distances by executing candidate actions
            # use the resulting distances as a pseudo-ground-truth
            # the last value in each list is the current distance
            batch_angles = batch_angles * cand_rgb.size(0) # [0,11/6*pi]
            # 辅助损失 direction aware loss
            dir_weight = torch.from_numpy(self.dir_weight).float().to(cand_rgb.device)
            dir_weight = torch.reshape(dir_weight, (1, 13)).expand(cand_rgb.size(0),13)
            last_dist = np.zeros(len(batch_angles), np.float32)
            # cand_dists_to_law = [[] for _ in range(len(batch_angles))]
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            oracle_stop = []
            # 生成伪标签, 使用固定的前进距离0.25作为伪标签
            # way_pos = 
            for j in range(len(batch_angles)): # batch 维
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    # test 0.25 meters forward to get a pseudo-ground-truth
                    forward_k = 0.25
                    dist_k = self.envs.call_at(j, 
                        "cand_dist_to_goal", {
                            "angle": angle_k, "forward": forward_k,
                        })
                    cand_dists_to_goal[j].append(dist_k)
                # 每次向12个方向探索, (尝试0.25m看哪个距离终点更近)
                curr_dist_to_goal = self.envs.call_at(
                    j, "current_dist_to_goal")
                last_dist[j] = curr_dist_to_goal
                # if within target range (metrics def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                    oracle_stop.append(True)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
                    oracle_stop.append(False)

            # for j in range(len(batch_angles)):
            #     way_pos = observations[j]['vln_law_action_sensor'] # [B, 1, 3]
            #     for k in range(len(batch_angles[j])): # 12个角度
            #         angle_k = batch_angles[j][k]
            #         # test 0.25 meters forward to get a pseudo-ground-truth
            #         forward_k = 0.25
            #         dist_k = self.envs.call_at(j, 
            #             "cand_dist_to_law", {
            #                 "angle": angle_k, "forward": forward_k, "pos": way_pos,
            #             })
            #         cand_dists_to_goal[j].append(dist_k)

            if train_rl:
                None
            elif train_tf:  # training
                oracle_actions = torch.tensor(oracle_cand_idx, device=self.device).unsqueeze(1)
                actions = logits.argmax(dim=-1, keepdim=True)
                actions = torch.where(
                        torch.rand_like(actions, dtype=torch.float) <= self.ratio,
                        oracle_actions, actions)
                # 交叉熵损失
                # current_loss = F.cross_entropy(logits, oracle_actions.squeeze(1), reduction="none")
                current_loss = self.focal_loss(logits, oracle_actions.squeeze(1)) #[1.5501, 1.7599]
                # focal loss
                ml_loss += torch.sum(current_loss)
                # 计算辅助损失
                logit_score = F.softmax(logits, dim=1)
                new_score = logit_score * dir_weight
                new_score = new_score.sum(axis=1)
                angle_reward += torch.sum(new_score)
            else:  # inference
                actions = logits.argmax(dim=-1, keepdim=True)

            # waypoint prediction
            # way_states, \
            angle_offsets, dist_offsets, way_log_probs = self.waypoint_model(
                mode = 'way_actor',
                instruction = view_language_features,
                lang_mask = view_lang_mask,
                # view_states = view_states,
                way_states = view_states,
                prev_headings = prev_headings,
                prev_way_dists = prev_way_dists,
                batch_angles = batch_angles,
                actions = actions,
                cand_rgb = cand_rgb, 
                cand_depth = cand_depth,
                cand_direction = cand_direction,
                # cand_mask = cand_mask,
                masks = not_done_masks,
            )
            way_step_states = torch.zeros(init_num_envs, 768, device=self.device)
            way_step_states[not_done_index] = view_states.squeeze(1)
            way_hidden_states.append(way_step_states)

            # move the agent
            env_actions = []
            prev_way_dists = []  # reset predicted step dists (for not done agents)
            pred_stop = []
            for j in range(logits.size(0)):
                action_j = actions[j].item()
                if train_rl and (action_j == candidate_lengths[j]-1 or stepk == self.max_len-1):
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                    pred_stop.append(True)
                elif action_j == candidate_lengths[j] - 1:
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                    pred_stop.append(True)
                else:
                    env_actions.append({'action':
                        {'action': 4,  # HIGHTOLOW ACTION
                        'action_args':{
                            'angle': angle_offsets[j].item(),  # [action_j]
                            'distance': dist_offsets[j].item(),  # [action_j]
                        }}})
                    pred_stop.append(False)
                    prev_way_dists.append(dist_offsets[j].item())  # [action_j]

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in
                                             zip(*outputs)]

            # reward shaping waypoint predictor
            way_step_mask = np.zeros(init_num_envs, np.float32)
            way_step_reward = np.zeros(init_num_envs, np.float32)
            way_step_logp = torch.zeros(init_num_envs, requires_grad=True).to(self.device)
            for j in range(logits.size(0)):
                perm_index = not_done_index[j]
                curr_dist = self.envs.call_at(j, "current_dist_to_goal")

                if not pred_stop[j]:
                    way_step_mask[perm_index] = 1.0
                    way_step_logp[perm_index] = way_log_probs[j]

                    # distance reward
                    # if curr_dist < last_dist[j]:
                    way_step_reward[perm_index] = last_dist[j] - curr_dist
                    # else:
                    #     way_step_reward[perm_index] = -1.0

                    # collision penalty
                    if observations[j]['collision']:
                        way_step_reward[perm_index] -= 1.0 # 0.50
                    # print(j, curr_dist, last_dist[j], 
                    #     observations[j]['collision'], way_step_reward[j])

                    # scaled slack reward
                    way_step_reward[perm_index] -= 0.05 * dist_offsets[j] / 0.25

            way_masks_dict.append(way_step_mask)
            way_rewards_dict.append(way_step_reward)
            way_logp_dict.append(way_step_logp)

            if sum(dones) > 0:
                view_states = view_states[np.array(dones)==False]

                shift_index = 0
                for i in range(self.envs.num_envs):
                    if dones[i]:
                        i = i - shift_index
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        if self.envs.num_envs == 0:
                            break
                        observations.pop(i)
                        infos.pop(i)
                        shift_index += 1

            if self.envs.num_envs == 0:
                break
            not_done_masks = torch.ones(
                self.envs.num_envs, 1, dtype=torch.bool, device=self.device
            )

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )

            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # A2C waypoint model
        way_rl_length = len(way_rewards_dict) - 1  # 走了多少步, -1 because zero at the stopping step
        way_discount_reward = np.zeros(init_num_envs, np.float32)
        loss_way_actor = 0.0
        loss_way_critic = 0.0
        way_rl_total = 0.0
        for t in range(way_rl_length-1, -1, -1):
            way_discount_reward = way_discount_reward * 0.90 + way_rewards_dict[t]  # If it ended, the reward will be 0
            way_mask_ = Variable(torch.from_numpy(way_masks_dict[t]), 
                requires_grad=False).to(self.device)
            way_clip_reward = way_discount_reward.copy()
            r_ = Variable(torch.from_numpy(way_clip_reward), requires_grad=False).to(self.device)
            v_ = self.waypoint_model(
                mode = 'way_critic', 
                way_states = way_hidden_states[t], ).squeeze()
            a_ = (r_ - v_).detach()
            loss_way_actor += (-way_logp_dict[t] * a_ * way_mask_).sum()
            loss_way_critic += (((r_ - v_) ** 2) * way_mask_).sum() * 0.5
            way_rl_total += np.sum(way_masks_dict[t])
        if way_rl_total != 0.0:
            loss_way_actor /= way_rl_total
            loss_way_critic /= way_rl_total
        else:
            # loss_way_actor = torch.FloatTensor([0.0])
            # loss_way_critic = torch.FloatTensor([0.0])
            loss_way_actor = torch.zeros_like(ml_loss, requires_grad=True)
            loss_way_critic = torch.zeros_like(ml_loss, requires_grad=True)
        if train_rl:
            None
        elif train_tf:
            loss_view = ml_loss / total_weight
            loss_dir = angle_reward / total_weight

        return loss_view, loss_way_actor, loss_way_critic, loss_dir

    def train(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT

        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.config.IL.max_traj_len
        if (
            self.config.IL.DAGGER.expert_policy_sensor
            not in self.config.TASK_CONFIG.TASK.SENSORS
        ):
            self.config.TASK_CONFIG.TASK.SENSORS.append(
                self.config.IL.DAGGER.expert_policy_sensor
            )
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.NUM_ENVIRONMENTS = self.config.IL.batch_size // len(
            self.config.SIMULATOR_GPU_IDS)
        self.config.use_pbar = not is_slurm_batch_job()

        # view selection
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(sectors=self.config.MODEL.num_cameras)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS

        self.config.freeze()
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            # print(self.local_rank,self.device)

        self.split = split
        episode_ids = self.allocate_allowed_episode_by_scene()

        # self.temp_envs = get_env_class(self.config.ENV_NAME)(self.config)
        # self.temp_envs.episodes contains all 10819 GT samples
        # episodes_allowed is slightly smaller -- 10783 valid episodes
        # check the usage of self.temp_envs._env.sim.is_navigable([0,0,0])

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME),
            episodes_allowed=episode_ids,
            auto_reset_done=False
        )
        num_epoches_per_ratio = int(np.ceil(self.config.IL.epochs/self.config.IL.decay_time))
        print('\nFinished constructing environments')

        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]

        self.obs_transforms = get_active_obs_transforms(self.config)

        # disable the default CenterCropperPerSensor()
        self.obs_transforms = []

        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        # self.inflection_weight = torch.tensor([1.0,
        #             self.config.IL.inflection_weight_coef], device=self.device)

        print('\nInitializing policy network ...')
        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        print('\nTraining starts ...')
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR,
            flush_secs=self.flush_secs,
            purge_step=0,
        ) as writer:
            AuxLosses.activate()
            batches_per_epoch = int(np.ceil(dataset_length/self.batch_size))

            for epoch in range(self.start_epoch, self.config.IL.epochs):
                epoch_str = f"{epoch + 1}/{self.config.IL.epochs}"

                t_ = (
                    tqdm.trange(
                        batches_per_epoch, leave=False, dynamic_ncols=True
                    )
                    if self.config.use_pbar & (self.local_rank < 1)
                    else range(batches_per_epoch)
                )
                self.ratio = np.power(self.config.IL.schedule_ratio, epoch//num_epoches_per_ratio + 1)

                self.trained_episodes = []
                # reconstruct env for every epoch to ensure load same data
                if epoch != self.start_epoch:
                    self.envs = None
                    self.envs = construct_envs(
                        self.config, get_env_class(self.config.ENV_NAME),
                        episodes_allowed=episode_ids,
                        auto_reset_done=False
                    )

                for batch_idx in t_:
                    # 添加辅助损失 
                    loss_view, loss_way_actor, loss_way_critic, aux_loss = self.train_ml(
                        in_train=True, 
                        train_tf=True, train_rl=False)
                    loss = loss_view + loss_way_actor + loss_way_critic + aux_loss * self.loss_weight

                    if loss == -1:
                        break
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(
                        self.waypoint_model.parameters(), 40.)
                    self.optimizer.step()
                    losses = [loss_view, loss_way_actor, loss_way_critic, aux_loss]

                    if self.world_size > 1:
                        for i in range(len(losses)):
                            reduce_loss(losses[i], self.local_rank, self.world_size)
                            losses[i] = losses[i].item()
                    else:
                        for i in range(len(losses)):
                            losses[i] = losses[i].item()
                    # loss = losses[0]
                    loss_view = losses[0]
                    loss_actor = losses[1]
                    loss_critic = losses[2]
                    loss_dir = losses[3]
                    if self.config.use_pbar:
                        if self.local_rank < 1:  # seems can be removed
                            t_.set_postfix(
                                {
                                    "epoch": epoch_str,
                                    "loss_view": round(loss_view, 4),
                                    "loss_actor": round(loss_actor, 4),
                                    "loss_critic": round(loss_critic, 4),
                                    "loss_dir": round(loss_dir, 4),
                                }
                            )
                            writer.add_scalar("loss_view", loss_view, self.step_id)
                            writer.add_scalar("loss_actor", loss_actor, self.step_id)
                            writer.add_scalar("loss_critic", loss_critic, self.step_id)
                            writer.add_scalar("loss_dir", loss_dir, self.step_id)
                    self.step_id += 1  # noqa: SIM113

                if self.local_rank < 1:  # and epoch % 3 == 0:
                    self.save_checkpoint(epoch, self.step_id)

                AuxLosses.deactivate()
