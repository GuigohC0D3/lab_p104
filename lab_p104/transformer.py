import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from lab_p104.attention import AttentionUnit
    from lab_p104.components import FeedForwardBlock, ResidualNorm, SinusoidalPosition
except ModuleNotFoundError:
    from attention import AttentionUnit
    from components import FeedForwardBlock, ResidualNorm, SinusoidalPosition


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


@torch.no_grad()
def run_generation(model, encoder_tokens, start_idx, eos_idx, vocab, max_steps=20):
    generated = [start_idx]
    original_training_state = model.training
    model.eval()

    encoder_tokens = encoder_tokens.to(model.embedding.weight.device)

    print("Starting generation:", [vocab[i] for i in generated])

    try:
        while len(generated) < max_steps:
            decoder_tokens = torch.tensor(
                [generated],
                dtype=torch.long,
                device=encoder_tokens.device,
            )
            _, probs = model(encoder_tokens, decoder_tokens)
            next_id = torch.argmax(probs[:, -1, :], dim=-1).item()
            generated.append(next_id)
            print(f"Step {len(generated) - 1}: '{vocab[next_id]}'")
            if next_id == eos_idx:
                break
    finally:
        model.train(original_training_state)

    return generated


if __name__ == "__main__":
    torch.manual_seed(7)

    d_model = 512
    d_ff = 2048
    depth = 6
    vocab_size = 100
    max_steps = 20

    vocab = [f"token_{i}" for i in range(vocab_size - 2)] + ["<START>", "<EOS>"]
    start_idx = vocab_size - 2
    eos_idx = vocab_size - 1

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        depth=depth,
        d_ff=d_ff,
        max_len=64,
    )

    encoder_tokens = torch.tensor([[3, 17]], dtype=torch.long)

    encoder_memory = model.encode(encoder_tokens)
    print("Encoder input shape:", model.embedding(encoder_tokens).shape)
    print("Encoder output Z shape:", encoder_memory.shape)

    result_ids = run_generation(
        model=model,
        encoder_tokens=encoder_tokens,
        start_idx=start_idx,
        eos_idx=eos_idx,
        vocab=vocab,
        max_steps=max_steps,
    )

    print("\nGenerated sequence:", " ".join(vocab[i] for i in result_ids))
