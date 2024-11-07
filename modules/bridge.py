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
    def __init__(self, num_layers, in_dim, out_dim, num_heads, num_prompts, 
                 prompt_dim=None, prompt_norm=True, prompt_proj=False, p_dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_prompts = num_prompts
        self.head_size = in_dim // num_heads

        self.prompt_dim = in_dim if prompt_dim is None else prompt_dim
        assert prompt_proj or self.prompt_dim == in_dim, 'Prompt projection is required when prompt dimension is different from input dimension.'

        self.prompts = nn.Parameter(torch.zeros(num_layers, 1, num_prompts, self.prompt_dim))
        self.layer_norm = nn.LayerNorm(self.prompt_dim) if prompt_norm else nn.Identity()
        self.prompt_proj = nn.Linear(self.prompt_dim, in_dim) if prompt_proj else nn.Identity()

        self.out_proj = torch.nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p_dropout)

        nn.init.normal_(self.prompts, std=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, key_layer, value_layer):
        assert key_layer.shape[0] == self.num_layers, 'Key layer should have the same number of layers as the fusion module.'
        assert value_layer.shape[0] == self.num_layers, 'Value layer should have the same number of layers as the fusion module.'
        prompts = self.prompts.expand(-1, key_layer.shape[1], -1, -1)
        query_layer = self.prompt_proj(self.layer_norm(prompts))
        query_layer = self.transpose_for_scores(query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.in_dim,)
        context_layer = context_layer.view(new_context_layer_shape)

        if self.prompt_dim == self.in_dim:
            context_layer = context_layer + prompts
            context_layer = self.out_proj(context_layer)
        else:
            context_layer = self.out_proj(context_layer)
            context_layer = context_layer + prompts
        context_layer = self.dropout(context_layer)
        return context_layer
