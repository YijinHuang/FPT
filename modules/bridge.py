import math
import torch
import torch.nn as nn


class FineGrainedPromptTuning(torch.nn.Module):
    def __init__(self, side_encoder, fusion_module):
        super().__init__()
        self.side_encoder = side_encoder
        self.fusion_module = fusion_module

    def forward(self, x_coarse, key_states, value_states):
        fine_grained_states = self.fusion_module(key_states, value_states)
        out = self.side_encoder(x_coarse, fine_grained_states, interpolate_pos_encoding=True)
        return out.logits


class FusionModule(torch.nn.Module):
    def __init__(self, num_layer, in_dim, out_dim, num_heads, num_prompts, prompts_dim, p_dropout=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_prompts = num_prompts
        self.prompts_dim = prompts_dim
        self.head_size = in_dim // num_heads

        self.prompts = nn.Parameter(torch.zeros(num_layer, 1, num_prompts, prompts_dim))
        self.in_proj = torch.nn.Linear(prompts_dim, in_dim)
        self.out_proj = torch.nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p_dropout)

        nn.init.normal_(self.prompts, std=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, key_layer, value_layer):
        prompts = self.prompts.expand(-1, key_layer.shape[1], -1, -1)
        query_layer = self.transpose_for_scores(self.in_proj(prompts))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.in_dim,)
        context_layer = context_layer.view(new_context_layer_shape)
        context_layer = self.out_proj(context_layer)
        context_layer = self.dropout(context_layer)
        context_layer = context_layer + prompts
        return context_layer
