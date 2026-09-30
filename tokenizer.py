from pathlib import Path
from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size=500):

        self.target_vocab_size = vocab_size

        self.token_to_id = {}
        self.id_to_token = {}

        self.merges = {}

    # ========================================================
    # Train tokenizer
    # ========================================================

    def train(self, text):

        # Character-level initial representation

        words = text.split()

        word_counts = Counter(words)

        vocabulary = {}

        for word, count in word_counts.items():

            # Add end-of-word marker
            symbols = list(word) + ["</w>"]

            vocabulary[
                tuple(symbols)
            ] = count

        # Initial vocabulary

        tokens = set()

        for word in vocabulary:

            for symbol in word:

                tokens.add(symbol)

        # Reserve special tokens

        special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        self.token_to_id = {}

        for token in special_tokens:

            self.token_to_id[token] = len(
                self.token_to_id
            )

        # Add initial symbols

        for token in sorted(tokens):

            if token not in self.token_to_id:

                self.token_to_id[token] = len(
                    self.token_to_id
                )

        # ----------------------------------------------------
        # BPE merge loop
        # ----------------------------------------------------

        while len(self.token_to_id) < self.target_vocab_size:

            pair_counts = Counter()

            # Count adjacent symbol pairs

            for symbols, count in vocabulary.items():

                for i in range(
                    len(symbols) - 1
                ):

                    pair = (
                        symbols[i],
                        symbols[i + 1]
                    )

                    pair_counts[pair] += count

            if not pair_counts:

                break

            # Most frequent pair

            best_pair, _ = pair_counts.most_common(1)[0]

            new_token = (
                best_pair[0] +
                best_pair[1]
            )

            # Avoid duplicate token

            if new_token in self.token_to_id:

                break

            # Add token

            self.token_to_id[
                new_token
            ] = len(self.token_to_id)

            # Save merge

            self.merges[
                best_pair
            ] = new_token

            # Apply merge

            new_vocabulary = {}

            for symbols, count in vocabulary.items():

                merged = []

                i = 0

                while i < len(symbols):

                    if (
                        i < len(symbols) - 1
                        and
                        (
                            symbols[i],
                            symbols[i + 1]
                        ) == best_pair
                    ):

                        merged.append(
                            new_token
                        )

                        i += 2

                    else:

                        merged.append(
                            symbols[i]
                        )

                        i += 1

                new_vocabulary[
                    tuple(merged)
                ] = count

            vocabulary = new_vocabulary

        # Reverse vocabulary

        self.id_to_token = {
            idx: token
            for token, idx
            in self.token_to_id.items()
        }

    # ========================================================
    # Encode
    # ========================================================

    def encode(self, text):

        token_ids = []

        words = text.split()

        for word in words:

            symbols = list(word) + ["</w>"]

            # Apply learned merges repeatedly

            changed = True

            while changed:

                changed = False

                for pair, merged_token in self.merges.items():

                    new_symbols = []

                    i = 0

                    while i < len(symbols):

                        if (
                            i < len(symbols) - 1
                            and
                            (
                                symbols[i],
                                symbols[i + 1]
                            ) == pair
                        ):

                            new_symbols.append(
                                merged_token
                            )

                            i += 2

                            changed = True

                        else:

                            new_symbols.append(
                                symbols[i]
                            )

                            i += 1

                    symbols = new_symbols

            for symbol in symbols:

                token_id = self.token_to_id.get(
                    symbol,
                    self.token_to_id["<UNK>"]
                )

                token_ids.append(token_id)

        return token_ids

    # ========================================================
    # Decode
    # ========================================================

    def decode(self, token_ids):

        tokens = []

        for token_id in token_ids:

            token = self.id_to_token.get(
                token_id,
                "<UNK>"
            )

            tokens.append(token)

        text = "".join(tokens)

        text = text.replace(
            "</w>",
            " "
        )

        return text.strip()

    # ========================================================
    # Vocabulary size
    # ========================================================

    def vocab_size(self):

        return len(self.token_to_id)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    text = Path(
        "data/train.txt"
    ).read_text(
        encoding="utf-8"
    )

    tokenizer = BPETokenizer(
        vocab_size=200
    )

    tokenizer.train(text)

    print(
        "Vocabulary size:",
        tokenizer.vocab_size()
    )

    sample = "Python is programming"

    encoded = tokenizer.encode(
        sample
    )

    print("\nOriginal:")
    print(sample)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(
        tokenizer.decode(encoded)
    )