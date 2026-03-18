import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionUnit(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_source, key_value_source, attn_mask=None):
        q = self.q_proj(query_source)
        k = self.k_proj(key_value_source)
        v = self.v_proj(key_value_source)

        scale = q.size(-1) ** 0.5
        score = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attn_mask is not None:
            score = score + attn_mask

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out), attn
