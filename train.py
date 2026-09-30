from dataset import create_dataloaders
from model import RoshanLM
from config import (
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS
)

import torch
import torch.nn as nn
import math
import copy
from tqdm import tqdm


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# Dataset
# -------------------------

train_loader, validation_loader, tokenizer = create_dataloaders()


# -------------------------
# Model
# -------------------------

model = RoshanLM(
    tokenizer.vocab_size(),
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)


# -------------------------
# Loss
# -------------------------

loss_function = nn.CrossEntropyLoss()


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.1
)


# -------------------------
# Early Stopping
# -------------------------

best_validation_loss = float("inf")

patience = 5

patience_counter = 0

best_checkpoint = None


# -------------------------
# Training
# -------------------------

for epoch in range(EPOCHS):

    # =========================
    # Training
    # =========================

    model.train()

    total_train_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for x, y in progress_bar:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = loss_function(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # =========================
    # Validation
    # =========================

    model.eval()

    total_validation_loss = 0.0

    with torch.no_grad():

        for x, y in validation_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_validation_loss += loss.item()


    validation_loss = (
        total_validation_loss /
        len(validation_loader)
    )


    perplexity = math.exp(
        min(validation_loss, 20)
    )


    # =========================
    # Metrics
    # =========================

    print(f"\nEpoch {epoch + 1}")

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {validation_loss:.4f}"
    )

    print(
        f"Perplexity: {perplexity:.2f}"
    )


    # =========================
    # Best Model
    # =========================

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        patience_counter = 0

        # IMPORTANT:
        # Deep copy the weights so they cannot
        # change during future training epochs.

        best_checkpoint = {
            "model_state_dict": copy.deepcopy(
                model.state_dict()
            ),

            "vocab_size":
                tokenizer.vocab_size(),

            "context_length":
                CONTEXT_LENGTH,

            "embedding_dim":
                EMBEDDING_DIM,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "dropout":
                DROPOUT,

            "token_to_id":
                tokenizer.token_to_id,

            "id_to_token":
                tokenizer.id_to_token,

            "merges":
                tokenizer.merges,

            "target_vocab_size":
                tokenizer.target_vocab_size
        }

        print(
            "✓ New best model!"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{patience})"
        )

        if patience_counter >= patience:

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"Best Validation Loss: "
                f"{best_validation_loss:.4f}"
            )

            break


# -------------------------
# Save BEST checkpoint
# -------------------------

if best_checkpoint is not None:

    torch.save(
        best_checkpoint,
        "checkpoints/roshanlm.pt"
    )

    print(
        "\nBest model checkpoint saved to:"
    )

    print(
        "checkpoints/roshanlm.pt"
    )

else:

    print(
        "\nNo checkpoint was saved."
    )