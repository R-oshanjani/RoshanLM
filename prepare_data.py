from pathlib import Path
import random
import re


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/train.txt"

TRAIN_FILE = "data/train_corpus.txt"

VALIDATION_FILE = "data/validation_corpus.txt"

VALIDATION_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# Clean text
# ============================================================

def clean_text(text):

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# Load corpus
# ============================================================

def load_corpus():

    path = Path(
        INPUT_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    return clean_text(text)


# ============================================================
# Split corpus
# ============================================================

def split_corpus(text):

    # Each paragraph becomes one document

    documents = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    print(
        "Total documents:",
        len(documents)
    )

    # Shuffle reproducibly

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        documents
    )

    validation_count = max(
        1,
        int(
            len(documents)
            * VALIDATION_RATIO
        )
    )

    validation_documents = documents[
        :validation_count
    ]

    train_documents = documents[
        validation_count:
    ]

    return (
        train_documents,
        validation_documents
    )


# ============================================================
# Save datasets
# ============================================================

def save_dataset(
    documents,
    filename
):

    Path(
        filename
    ).write_text(
        "\n\n".join(documents),
        encoding="utf-8"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading corpus..."
    )

    text = load_corpus()

    print(
        "Characters:",
        len(text)
    )

    print(
        "Words:",
        len(text.split())
    )

    print(
        "\nSplitting dataset..."
    )

    (
        train_documents,
        validation_documents
    ) = split_corpus(
        text
    )

    print(
        "Training documents:",
        len(train_documents)
    )

    print(
        "Validation documents:",
        len(validation_documents)
    )

    # Save

    save_dataset(
        train_documents,
        TRAIN_FILE
    )

    save_dataset(
        validation_documents,
        VALIDATION_FILE
    )

    print(
        "\nSaved:"
    )

    print(
        TRAIN_FILE
    )

    print(
        VALIDATION_FILE
    )


if __name__ == "__main__":

    main()