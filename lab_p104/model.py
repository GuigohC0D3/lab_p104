import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lab_p104.components import SinusoidalPosition
    from lab_p104.layers import DecoderStack, EncoderStack
except ModuleNotFoundError:
    from components import SinusoidalPosition
    from layers import DecoderStack, EncoderStack


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, depth=6, d_ff=2048, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = SinusoidalPosition(d_model, max_len=max_len)
        self.encoder = EncoderStack(depth, d_model, d_ff)
        self.decoder = DecoderStack(depth, d_model, d_ff)
        self.output_head = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def encode(self, encoder_tokens):
        x = self.embedding(encoder_tokens)
        x = self.position(x)
        return self.encoder(x)

    def decode(self, decoder_tokens, memory):
        y = self.embedding(decoder_tokens)
        y = self.position(y)
        mask = self.make_causal_mask(y.size(1), y.device)
        return self.decoder(y, memory, mask)

    def forward(self, encoder_tokens, decoder_tokens):
        memory = self.encode(encoder_tokens)
        decoded = self.decode(decoder_tokens, memory)
        logits = self.output_head(decoded)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
