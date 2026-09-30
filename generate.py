import torch

from model import RoshanLM
from tokenizer import BPETokenizer


# -------------------------
# Configuration
# -------------------------

CHECKPOINT_PATH = "checkpoints/roshanlm.pt"
PROMPT = "Python is "

MAX_NEW_TOKENS = 80

TEMPERATURE = 0.7

TOP_K = 10


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# Load checkpoint
# -------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False
)


# -------------------------
# Restore tokenizer
# -------------------------

tokenizer = BPETokenizer(
    vocab_size=checkpoint["target_vocab_size"]
)

tokenizer.token_to_id = checkpoint["token_to_id"]

tokenizer.id_to_token = checkpoint["id_to_token"]

tokenizer.merges = checkpoint["merges"]


# -------------------------
# Restore model
# -------------------------

model = RoshanLM(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    dropout=0.0
).to(device)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -------------------------
# Generate
# -------------------------

def generate_text(
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=20
):

    token_ids = tokenizer.encode(prompt)

    tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )


    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the model's context window
            context = tokens[
                :, -checkpoint["context_length"]:
            ]

            # Model prediction
            logits = model(context)

            # Last token prediction
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature


            # Top-K sampling
            if top_k is not None:

                values, indices = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                filtered_logits = torch.full_like(
                    logits,
                    float("-inf")
                )

                filtered_logits.scatter_(
                    1,
                    indices,
                    values
                )

                logits = filtered_logits


            # Convert logits to probabilities
            probabilities = torch.softmax(
                logits,
                dim=-1
            )


            # Sample next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )


            # Add token to sequence
            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )


    return tokenizer.decode(
        tokens[0].tolist()
    )


# -------------------------
# Run generation
# -------------------------

print("\nPrompt:")
print(PROMPT)

generated = generate_text(
    PROMPT,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K
)

print("\nGenerated text:")
print(generated)