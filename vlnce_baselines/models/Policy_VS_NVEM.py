#!/usr/bin/env python
# coding=UTF-8
'''
Author: Wang Naijia
Date: 2022-04-04 17:03:13
LastEditors: Wang Naijia
LastEditTime: 2024-02-07 00:50:06
Descripttion: 
'''
from typing import Dict, Optional, Tuple

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from habitat_baselines.utils.common import CustomFixedCategorical

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder
)
# 处理视觉
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.utils import (
    length2mask, angle_feature, dir_angle_feature)

from vlnce_baselines.models.policy import ILPolicy
import math
import torch.distributed as dist
          
@baseline_registry.register_policy
class PolicyViewSelectionNVEM(ILPolicy):
    def __init__(self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
        ):
        
        super().__init__(
            NvemNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )
    
    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

    def get_state_value(self, h_t):
        """
        构建critic state value function，与网络共享特征提取层！
        state value function is trainable!
        """
        return self.net.critic(h_t)
    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)
        
class NvemNet(Net):
    """
    预测的结果是对动作分布的预测（FWD,L,R,STOP）
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ) -> None:
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        device = (
            torch.device("cuda", dist.get_rank())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet152", "TorchVisionResNet50"
        ], "RGB_ENCODER.cnn_type must be TorchVisionResNet152 or TorchVisionResNet50"
        if model_config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
            self.rgb_encoder = TorchVisionResNet50(
                observation_space,
                model_config.RGB_ENCODER.output_size,
                device,
                spatial_output=model_config.spatial_output,
            )

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        self.batch_angles = [np.arange(0, 2*math.pi, 2*math.pi/self.model_config.num_cameras)]
        self.angle_features = dir_angle_feature(self.batch_angles).to(self.device)

        # merging visual inputs
        self.rgb_linear = nn.Sequential(
            nn.Linear(
                2048,
                model_config.RGB_ENCODER.output_size,          # 256
            ),
            nn.ReLU(True),
        )

        self.depth_linear = nn.Sequential(
            nn.Linear(
                128,        # 128
                model_config.DEPTH_ENCODER.output_size,    # 128
            ),
            nn.ReLU(True),
        )

        # visual = depth 128 + rgb 1024 + dir 64  -> dim(vis_h) 512
        self.vismerge_linear = nn.Sequential(
            nn.Linear(
                model_config.DEPTH_ENCODER.output_size + model_config.RGB_ENCODER.output_size + model_config.VISUAL_DIM.directional,
                model_config.VISUAL_DIM.vis_hidden,
            ),
            nn.ReLU(True),
        )

        # a_{t-1}
        self.enc_prev_act = nn.Sequential(
            nn.Linear(model_config.VISUAL_DIM.directional, model_config.VISUAL_DIM.directional),
            nn.Tanh(),
        )

        # action feat 64 -> 512
        self.value_action = nn.Sequential(
            nn.Linear(model_config.VISUAL_DIM.directional, model_config.VISUAL_DIM.vis_hidden), 
            nn.Tanh(),
        )

        # Init the RNN state decoder 512 + 64 -> 768
        self.state_encoder = build_rnn_state_encoder(
            input_size=model_config.VISUAL_DIM.vis_hidden + model_config.VISUAL_DIM.directional,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )
        # attention0
        self.prev_state_vis_attn = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.VISUAL_DIM.vis_hidden,
            output_tilde=False
        )
        # attention1
        self.text_vis_attn = SoftDotAttention(
            self.instruction_encoder.output_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.VISUAL_DIM.vis_hidden,
            output_tilde=False
        )
        self.action_vis_attn = SoftDotAttention(
            self.instruction_encoder.output_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.VISUAL_DIM.vis_hidden,
            output_tilde=False
        )

        # attention2, 为了得到带权重的语义向量,将output_tilde设为true
        self.state_text_attn = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size,
            self.instruction_encoder.output_size,
            self.instruction_encoder.output_size,
            # output_tilde=True 
            output_tilde=False
        )
        self.state_action_attn = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size,
            self.instruction_encoder.output_size,
            self.instruction_encoder.output_size,
            # output_tilde=True
            output_tilde=False
        )

        # fusing
        self.fuse_vis = nn.Linear(self.instruction_encoder.output_size, 1)
        self.fuse_act = nn.Linear(self.instruction_encoder.output_size, 1)

        # attention3 768 + 512 + 384 --- 512 ---> 768
        self.state_vis_logits = SoftDotAttention(
            model_config.STATE_ENCODER.hidden_size + model_config.VISUAL_DIM.vis_hidden + self.instruction_encoder.output_size,
            model_config.VISUAL_DIM.vis_hidden,
            model_config.STATE_ENCODER.hidden_size,
            output_tilde=False
        )

        # attention4 text + action -> logits  768 + 256 --- 512 ---> 768
        self.text_action_logits = SoftDotAttention(
            model_config.VISUAL_DIM.vis_hidden  + self.instruction_encoder.output_size, # q
            model_config.VISUAL_DIM.vis_hidden, # kv
            model_config.STATE_ENCODER.hidden_size, # hidden_size
            output_tilde=False
        )
        # attention4 text_action -> logits  768 --- 512 ---> 768
        # self.text_action_logits = SoftDotAttention(
        #     self.instruction_encoder.output_size, # q
        #     model_config.VISUAL_DIM.vis_hidden, # kv 512
        #     model_config.STATE_ENCODER.hidden_size, # hidden_size 
        #     output_tilde=False
        # )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        # 空间池化
        self.space_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten(start_dim=2),)
                
        # 实例化progress_monitor,输入为state vector,单层网络,输出一个实数
        self.progress_monitor = nn.Linear(
            model_config.STATE_ENCODER.hidden_size, 1
        )
        
        # 初始化线性层
        self._init_layers()
        
        # 设置为可训练        
        self.train()

    def _init_layers(self):
        nn.init.kaiming_normal_(
            self.progress_monitor.weight, nonlinearity="tanh"
        )
        nn.init.constant_(self.progress_monitor.bias, 0)
        
    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, mode=None,
                observations=None, 
                instruction=None, lang_mask=None,
                view_states=None,
                way_states=None,
                cand_rgb=None, cand_depth=None,
                cand_direction=None, cand_mask=None,
                prev_headings=None, masks=None,
                in_train=True):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embed: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embed: [batch_size x RGB_ENCODER.output_size]
        """
        if mode == 'language':
            ctx, all_lang_masks = self.instruction_encoder(observations)
            return ctx, all_lang_masks
        elif mode == 'view':
            batch_size = observations['instruction'].size(0)

            # encoding rgb/depth at all directions
            NUM_IMGS = self.model_config.num_cameras
            depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
            rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)
            # order of input images is counter-clockwise
            a_count = 0
            
            # a temporarily easy-get list
            depth_sensor_ids = ['depth', 'depth_30.0', 'depth_60.0',
                'depth_90.0', 'depth_120.0', 'depth_150.0',
                'depth_180.0', 'depth_210.0', 'depth_240.0',
                'depth_270.0', 'depth_300.0', 'depth_330.0']
            for i, k in enumerate(depth_sensor_ids):
                for bi in range(observations[k].size(0)): # observations是个字典包数组的, bi是batch idx 
                    depth_batch[a_count+bi*NUM_IMGS] = observations[k][bi]
                    rgb_batch[a_count+bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                a_count += 1    

            obs_views = {}
            obs_views['rgb'] = rgb_batch
            obs_views['depth'] = depth_batch

            rgb_embed_spatial = self.rgb_encoder(obs_views)      # torch.Size([bs*#camera, 2048, 7, 7])
            depth_embed_spatial = self.depth_encoder(obs_views)  # torch.Size([bs*#camera, 128, 4, 4])

            rgb_embed_rs = rgb_embed_spatial.reshape(
                batch_size, NUM_IMGS, *rgb_embed_spatial.shape[1:])
            depth_embed_rs = depth_embed_spatial.reshape(
                batch_size, NUM_IMGS, *depth_embed_spatial.shape[1:])

            candidate_lengths = [NUM_IMGS + 1] * batch_size  # including stop
            max_candidate = max(candidate_lengths)
            cand_mask = length2mask(candidate_lengths, device=self.device)

            cand_rgb = torch.cat([
                rgb_embed_rs, 
                torch.zeros(
                    (batch_size, 1, *rgb_embed_spatial.shape[1:]),
                    dtype=torch.float32, device=self.device)], 
                dim=1)
            cand_depth = torch.cat([
                depth_embed_rs, 
                torch.zeros(
                    (batch_size, 1, *depth_embed_spatial.shape[1:]),
                    dtype=torch.float32, device=self.device)], 
                dim=1)

            cand_direction = self.angle_features.repeat(batch_size, 1, 1)

            return cand_rgb, cand_depth, cand_direction, cand_mask, candidate_lengths, self.batch_angles

        elif mode == 'selection':
            vis_mask = observations["vis_mask"]
            act_mask = observations["act_mask"]
            # pool and merge visual features (no spatial encoding)
            rgb_in = self.rgb_linear(self.space_pool(cand_rgb))
            depth_in = self.depth_linear(self.space_pool(cand_depth))
            vis_in = self.vismerge_linear(
                torch.cat((rgb_in, depth_in, cand_direction), dim=2),) # 在dim2的维度上拼接,512+256+64

            # aggregate visual features by agent's previous state
            prev_state = view_states[:, 
                0:self.state_encoder.num_recurrent_layers].squeeze(1)
            vis_prev_state, _ = self.prev_state_vis_attn(
                prev_state, vis_in, cand_mask) #[B,512]
            
            # update state with new visual observation and past decision
            prev_actions = angle_feature(prev_headings, device=self.device)
            prev_actions = self.enc_prev_act(prev_actions)
            state_in = torch.cat([vis_prev_state, prev_actions], dim=1) #[B,576] h_{t-1}
            view_states_out = view_states.detach().clone()
            # state = h_t
            (
                state,
                view_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
            ) = self.state_encoder(
                state_in,
                view_states[:, 0 : self.state_encoder.num_recurrent_layers],
                masks,
            )

            # language attention using state
            L = lang_mask.shape[1]
            text_state, text_vis_weight = self.state_text_attn(
                state, instruction, torch.logical_or(vis_mask[:, :L], lang_mask))
            action_state, text_act_weight = self.state_action_attn(
                state, instruction, torch.logical_or(act_mask[:, :L], lang_mask))
            # text_state, text_u, _ = self.state_text_attn(
            #     state, instruction, lang_mask)
            # action_state, action_u, _ = self.state_action_attn(
            #     state, instruction, lang_mask)

            # 学习融合权重
            fusion_weight = torch.cat([self.fuse_vis(text_state), self.fuse_act(action_state)], dim=-1)
            # fusion_weight = torch.cat([self.fuse_vis(text_u), self.fuse_act(action_u)], dim=-1)
            fusion_weight = F.softmax(fusion_weight, dim=-1)

            # visual attention using attended language
            vis_text_feats, _ = self.text_vis_attn(
                text_state, vis_in, cand_mask)

            # action value [2,13,512]
            action_in = self.value_action(cand_direction)
            # 从12个dir中注意一个
            action_feat, _ = self.action_vis_attn(
                action_state, action_in, cand_mask)

            # view selection probability 
            # 768 + 512 + 768
            x_vis = torch.cat((state, vis_text_feats, text_state), dim=1)
            # 512 的动作特征 + 768 的状态向量
            x_act = torch.cat((action_feat, action_state), dim=1)

            # _, logits = self.state_vis_logits(
            #     x_vis, vis_in, cand_mask, output_prob=False)
            # 可视化注意力权重
            _, logits_vis = self.state_vis_logits(
                x_vis, vis_in, cand_mask, output_prob=False)
            _, logits_act = self.text_action_logits(
                x_act, action_in, cand_mask, output_prob=False)

            logits = torch.cat([logits_vis.unsqueeze(2), logits_act.unsqueeze(2)], dim=-1)
            logits = torch.matmul(logits, fusion_weight.unsqueeze(2)).squeeze(2)

            # return logits, view_states_out
            # 为了可视化权重
            return logits, view_states_out, text_vis_weight, text_act_weight


class SoftDotAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, output_tilde=False):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_q = nn.Linear(q_dim, hidden_dim, bias=True)
        self.linear_kv = nn.Linear(kv_dim, hidden_dim, bias=True)
        self.sm = nn.Softmax(dim=1)

        self.output_tilde = output_tilde
        if output_tilde:
            self.linear_out = nn.Linear(q_dim + hidden_dim, hidden_dim, bias=False)
            self.tanh = nn.Tanh()


    def forward(self, q, kv, mask=None, output_prob=True):
        '''Propagate h through the network.
        q: (query) batch x dim
        kv: (keys and values) batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''

        # try:
        x_q = self.linear_q(q).unsqueeze(2)  # batch x dim x 1
        x_kv = self.linear_kv(kv)
        # except:
        #     import pdb; pdb.set_trace()

        # Get attention
        attn = torch.bmm(x_kv, x_q).squeeze(2)  # batch x seq_len

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn_prob = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn_prob = attn_prob.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_x_kv = torch.bmm(attn_prob, x_kv).squeeze(1)  # batch x dim
        if output_prob:
            attn = attn_prob.squeeze(1)
        if self.output_tilde:
            h_tilde = torch.cat((weighted_x_kv, q), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_x_kv, attn