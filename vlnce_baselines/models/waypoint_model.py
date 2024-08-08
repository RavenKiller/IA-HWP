import torch
import torch.nn as nn
import numpy as np

from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from vlnce_baselines.models.utils import (
    angle_distance_feature,
)

from vlnce_baselines.models.vlnbert.init_vlnbert import get_vlnbert_models


class Waypoint_Model(nn.Module):
    def __init__(self, model_config=None, device=None):
        super(Waypoint_Model, self).__init__()
        self.model_config = model_config
        self.device = device
        # !!! add dropout
        self.main_dropout = nn.Dropout(0.0)

        self.space_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.spatial_embeddings = nn.Embedding(
            4 * 4, model_config.WAY_MODEL.directional
        )

        self.rgb_Wa = nn.Sequential(
            nn.Linear(2048, model_config.RGB_ENCODER.output_size),
            nn.ReLU(True),
        )
        self.depth_Wa = nn.Sequential(
            nn.Linear(128, model_config.DEPTH_ENCODER.output_size),
            nn.ReLU(True),
        )
        self.vismerge_linear = nn.Sequential(
            nn.Linear(
                model_config.RGB_ENCODER.output_size
                + model_config.DEPTH_ENCODER.output_size
                + model_config.VISUAL_DIM.directional,
                model_config.WAY_MODEL.hidden_size,
            ),
            nn.ReLU(True),
        )

        # spatial attention
        self.prev_state_spatial_attn = SoftDotAttention(
            model_config.WAY_MODEL.hidden_size,
            model_config.WAY_MODEL.hidden_size,
            model_config.WAY_MODEL.hidden_size,
            activate_inputs=True,
            output_tilde=False,
        )
        self.state_text_attn = SoftDotAttention(
            model_config.WAY_MODEL.hidden_size,
            model_config.WAY_MODEL.hidden_size,
            model_config.WAY_MODEL.hidden_size,
            activate_inputs=True,
            output_tilde=False,
        )
        self.text_state_spatial_attn = SoftDotAttention(
            model_config.WAY_MODEL.hidden_size,
            model_config.WAY_MODEL.hidden_size,
            model_config.WAY_MODEL.hidden_size,
            activate_inputs=True,
            output_tilde=False,
        )

        # waypoint angle and distance predictors
        self.way_feats_linear = nn.Sequential(
            nn.Linear(
                model_config.WAY_MODEL.hidden_size * 2,
                model_config.WAY_MODEL.hidden_size,
            ),
            nn.ReLU(True),
            nn.Linear(model_config.WAY_MODEL.hidden_size, 10 * 8),
            nn.Softmax(dim=1),
        )

        # critic for A2C
        # self.waypoint_critic = nn.Sequential(
        #     nn.Linear(model_config.STATE_ENCODER.hidden_size,
        #         model_config.STATE_ENCODER.hidden_size//4),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(model_config.STATE_ENCODER.hidden_size//4, 1),
        # )

    def forward(
        self,
        mode=None,
        lang_idx_tokens=None,
        lang_mask=None,
        instruction=None,
        # view_states=None,
        way_states=None,
        prev_headings=None,
        prev_way_dists=None,
        batch_angles=None,
        actions=None,
        cand_rgb=None,
        cand_depth=None,
        cand_direction=None,
        masks=None,
    ):

        if mode == "language":
            language_features = self.vlnbert(
                "language",
                instruction,
                attention_mask=lang_mask,
                lang_mask=lang_mask,
            )
            return language_features

        elif mode == "way_actor":
            # extract visual features at the selected view
            batch_size = instruction.size(0)

            rgb_feats = cand_rgb[
                range(batch_size), actions.squeeze(), :
            ]  # [B, 2048, 7, 7]
            depth_feats = cand_depth[range(batch_size), actions.squeeze(), :]
            # direction_feats = cand_direction[range(batch_size), actions.squeeze(), :]
            cand_angles = torch.zeros(
                batch_size, dtype=torch.float32, device=self.device
            )
            for i in range(batch_size):
                act_idx = actions[i]
                if act_idx[0] != len(batch_angles[0]):
                    cand_angles[i] = batch_angles[0][act_idx]

            # project visual features (space_pool to 4x4 to match depth maps)
            rgb_x = self.rgb_Wa(
                self.space_pool(rgb_feats)
                .reshape(rgb_feats.size(0), rgb_feats.size(1), -1)
                .permute(0, 2, 1),
            )  # [1,16,512]
            depth_x = self.depth_Wa(
                depth_feats.reshape(
                    depth_feats.size(0), depth_feats.size(1), -1
                ).permute(0, 2, 1),
            )  # [1,16,256]
            spatial_feats = self.spatial_embeddings(
                torch.arange(
                    0,
                    self.spatial_embeddings.num_embeddings,
                    device=self.device,
                    dtype=torch.long,
                )  # [1,16,64]
            ).repeat(batch_size, 1, 1)
            vis_in = self.vismerge_linear(
                torch.cat((rgb_x, depth_x, spatial_feats), dim=2)
            )

            # language attention using waypoint state
            text_state, _ = self.state_text_attn(
                way_states.squeeze(1), self.main_dropout(instruction), lang_mask
            )

            # current-text-state-visual spatial attention
            vis_tilde, _ = self.text_state_spatial_attn(
                text_state, self.main_dropout(vis_in)
            )

            # predict waypoint angle and distance
            prob_x = self.way_feats_linear(
                torch.cat((way_states.squeeze(1), vis_tilde), dim=1)
            )
            # 创建以参数prob_x为标准的类别分布
            probs_c = torch.distributions.Categorical(prob_x)
            # 按照prob_x的概率进行采样,概率高的索引返回的可能性更高
            way_act = probs_c.sample().detach()
            angle_offset = cand_angles + ((way_act % 10 + 1) - 5) * 3.0 / 180.0 * np.pi
            dist_offset = (way_act // 10 + 1) * 0.25
            way_log_prob = probs_c.log_prob(way_act)

            return angle_offset, dist_offset, way_log_prob  # way_states_out

        elif mode == "way_critic":
            return self.waypoint_critic(way_states)

        else:
            raise NotImplementedError


class Waypoint_Model_Critic(nn.Module):
    def __init__(self, model_config=None, device=None):
        super(Waypoint_Model_Critic, self).__init__()
        self.model_config = model_config
        self.device = device

        # critic for A2C
        self.waypoint_critic = nn.Sequential(
            nn.Linear(
                model_config.STATE_ENCODER.hidden_size,
                model_config.STATE_ENCODER.hidden_size // 4,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(model_config.STATE_ENCODER.hidden_size // 4, 1),
        )

    def forward(
        self,
        mode=None,
        lang_idx_tokens=None,
        lang_mask=None,
        instruction=None,
        # view_states=None,
        way_states=None,
        prev_headings=None,
        prev_way_dists=None,
        batch_angles=None,
        actions=None,
        cand_rgb=None,
        cand_depth=None,
        cand_direction=None,
        masks=None,
    ):

        if mode == "way_critic":
            return self.waypoint_critic(way_states)

        else:
            raise NotImplementedError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SoftDotAttention(nn.Module):
    def __init__(
        self, q_dim, kv_dim, hidden_dim, activate_inputs=False, output_tilde=False
    ):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        if not activate_inputs:
            self.linear_q = nn.Linear(q_dim, hidden_dim, bias=True)
            self.linear_kv = nn.Linear(kv_dim, hidden_dim, bias=True)
        else:
            self.linear_q = nn.Sequential(
                nn.Linear(q_dim, hidden_dim, bias=True),
                nn.ReLU(True),
            )
            self.linear_kv = nn.Sequential(
                nn.Linear(kv_dim, hidden_dim, bias=True),
                nn.ReLU(True),
            )
        self.sm = nn.Softmax(dim=1)

        self.output_tilde = output_tilde
        if output_tilde:
            self.linear_out = nn.Linear(q_dim + hidden_dim, hidden_dim, bias=False)
            self.tanh = nn.Tanh()

    def forward(self, q, kv, mask=None, output_prob=True):
        """Propagate h through the network.
        q: (query) batch x dim
        kv: (keys and values) batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        x_q = self.linear_q(q).unsqueeze(2)  # batch x dim x 1
        x_kv = self.linear_kv(kv)

        # Get attention
        attn = torch.bmm(x_kv, x_q).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float("inf"))
        attn = self.sm(
            attn
        )  # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_x_kv = torch.bmm(attn3, x_kv).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if self.output_tilde:
            h_tilde = torch.cat((weighted_x_kv, q), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_x_kv, attn
