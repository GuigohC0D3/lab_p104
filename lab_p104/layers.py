import torch.nn as nn

try:
    from lab_p104.attention import AttentionUnit
    from lab_p104.components import FeedForwardBlock, ResidualNorm
except ModuleNotFoundError:
    from attention import AttentionUnit
    from components import FeedForwardBlock, ResidualNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.self_attn = AttentionUnit(d_model)
        self.res1 = ResidualNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, d_ff)
        self.res2 = ResidualNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x)
        x = self.res1(x, attn_out)
        ffn_out = self.ffn(x)
        x = self.res2(x, ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.masked_attn = AttentionUnit(d_model)
        self.res1 = ResidualNorm(d_model)
        self.cross_attn = AttentionUnit(d_model)
        self.res2 = ResidualNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, d_ff)
        self.res3 = ResidualNorm(d_model)

    def forward(self, y, memory, causal_mask):
        masked_out, _ = self.masked_attn(y, y, causal_mask)
        y = self.res1(y, masked_out)
        cross_out, _ = self.cross_attn(y, memory)
        y = self.res2(y, cross_out)
        ffn_out = self.ffn(y)
        y = self.res3(y, ffn_out)
        return y


class EncoderStack(nn.Module):
    def __init__(self, depth, d_model, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderStack(nn.Module):
    def __init__(self, depth, d_model, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff) for _ in range(depth)])

    def forward(self, y, memory, causal_mask):
        for layer in self.layers:
            y = layer(y, memory, causal_mask)
        return y
