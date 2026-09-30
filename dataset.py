import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tokenizer import BPETokenizer
from config import CONTEXT_LENGTH, BATCH_SIZE


class LanguageModelDataset(Dataset):

    def __init__(self, token_ids, context_length):
        self.tokens = token_ids
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length

    def __getitem__(self, index):

        x = self.tokens[
            index:index + self.context_length
        ]

        y = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


def create_dataloaders():

    # Load separate training and validation corpora
    train_text = Path(
        "data/train_corpus.txt"
    ).read_text(encoding="utf-8")

    validation_text = Path(
        "data/validation_corpus.txt"
    ).read_text(encoding="utf-8")

    # Train tokenizer ONLY on training data
    tokenizer = BPETokenizer(vocab_size=200)

    tokenizer.train(train_text)

    # Encode both datasets using the same tokenizer
    train_tokens = tokenizer.encode(train_text)

    validation_tokens = tokenizer.encode(validation_text)

    print("Training tokens:", len(train_tokens))
    print("Validation tokens:", len(validation_tokens))

    if len(train_tokens) <= CONTEXT_LENGTH:
        raise ValueError(
            "Training corpus is too small for the selected context length."
        )

    if len(validation_tokens) <= CONTEXT_LENGTH:
        raise ValueError(
            "Validation corpus is too small for the selected context length."
        )

    # Create datasets
    train_dataset = LanguageModelDataset(
        train_tokens,
        CONTEXT_LENGTH
    )

    validation_dataset = LanguageModelDataset(
        validation_tokens,
        CONTEXT_LENGTH
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, validation_loader, tokenizer


if __name__ == "__main__":

    train_loader, validation_loader, tokenizer = create_dataloaders()

    print("\nVocabulary size:", tokenizer.vocab_size())

    print(
        "Training samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(validation_loader.dataset)
    )

    x, y = next(iter(train_loader))

    print(
        "\nTraining input shape:",
        x.shape
    )

    print(
        "Training target shape:",
        y.shape
    )

    x, y = next(iter(validation_loader))

    print(
        "\nValidation input shape:",
        x.shape
    )

    print(
        "Validation target shape:",
        y.shape
    )