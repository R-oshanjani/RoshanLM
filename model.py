import torch
import torch.nn as nn
import math


# ============================================================
# 1. Multi-Head Self-Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        # ----------------------------------------------------
        # Create Query, Key, Value
        # ----------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # ----------------------------------------------------
        # Split into multiple attention heads
        # ----------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Calculate attention scores
        # ----------------------------------------------------

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        # ----------------------------------------------------
        # Causal attention mask
        # Prevents looking at future tokens
        # ----------------------------------------------------

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # ----------------------------------------------------
        # Attention probabilities
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # ----------------------------------------------------
        # Weighted sum of Values
        # ----------------------------------------------------

        output = torch.matmul(
            attention_weights,
            V
        )

        # ----------------------------------------------------
        # Combine attention heads
        # ----------------------------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        output = self.output_projection(
            output
        )

        return output


# ============================================================
# 2. Feed Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            nn.GELU(),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 3. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.feed_forward = FeedForward(
            embedding_dim
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Self-Attention + Residual Connection
        # ----------------------------------------------------

        attention_output = self.attention(
            self.norm1(x)
        )

        x = x + self.dropout(
            attention_output
        )

        # ----------------------------------------------------
        # Feed Forward + Residual Connection
        # ----------------------------------------------------

        feed_forward_output = self.feed_forward(
            self.norm2(x)
        )

        x = x + self.dropout(
            feed_forward_output
        )

        return x


# ============================================================
# 4. RoshanLM
# ============================================================

class RoshanLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length=64,
        embedding_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.3
    ):

        super().__init__()

        self.context_length = context_length
        self.embedding_dim = embedding_dim

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Positional Embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        self.transformer_blocks = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )

                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------
        # Final Layer Normalization
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # Language Model Head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        # ----------------------------------------------------
        # Check context length
        # ----------------------------------------------------

        if sequence_length > self.context_length:

            raise ValueError(
                "Sequence length exceeds context length"
            )

        # ----------------------------------------------------
        # Token Embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(
            token_ids
        )

        # ----------------------------------------------------
        # Position Embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_embeddings = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine Token + Position Embeddings
        # ----------------------------------------------------

        x = token_embeddings + position_embeddings

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        for block in self.transformer_blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final Normalization
        # ----------------------------------------------------

        x = self.final_norm(x)

        # ----------------------------------------------------
        # Vocabulary Logits
        # ----------------------------------------------------

        logits = self.lm_head(x)

        return logits


# ============================================================
# 5. Test RoshanLM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test Configuration
    # --------------------------------------------------------

    vocab_size = 300

    batch_size = 2

    sequence_length = 8

    context_length = 64

    embedding_dim = 64

    num_heads = 2

    num_layers = 2

    dropout = 0.3

    # --------------------------------------------------------
    # Create Model
    # --------------------------------------------------------

    model = RoshanLM(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )

    # --------------------------------------------------------
    # Generate Fake Token IDs
    # --------------------------------------------------------

    token_ids = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length
        )
    )

    # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------

    logits = model(token_ids)

    # --------------------------------------------------------
    # Print Shapes
    # --------------------------------------------------------

    print("Input shape:")
    print(token_ids.shape)

    print("\nOutput shape:")
    print(logits.shape)

    # --------------------------------------------------------
    # Count Parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nTotal parameters:")
    print(total_parameters)