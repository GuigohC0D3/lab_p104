import torch


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
