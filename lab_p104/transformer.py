import torch

try:
    from lab_p104.generation import run_generation
    from lab_p104.model import SimpleTransformer
except ModuleNotFoundError:
    from generation import run_generation
    from model import SimpleTransformer


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
