import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts CNN feature maps into patch embeddings (tokens)
    """
    def __init__(self, in_channels, embed_dim, patch_size=1):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)            # (B, D, H, W)
        B, D, H, W = x.shape

        x = x.flatten(2)           # (B, D, N)
        x = x.transpose(1, 2)      # (B, N, D)

        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings
    """
    def __init__(self, num_patches, embed_dim):
        super().__init__()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Feed-forward
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    ViT operating on CNN feature maps
    """
    def __init__(
        self,
        in_channels=256,
        embed_dim=128,
        depth=2,
        num_heads=4,
        patch_size=1,
        feature_map_size=(28, 28)
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            in_channels,
            embed_dim,
            patch_size
        )

        # Number of tokens
        num_patches = (feature_map_size[0] // patch_size) * \
                      (feature_map_size[1] // patch_size)

        # Positional encoding
        self.pos_embed = PositionalEncoding(num_patches, embed_dim)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, 256, 28, 28)
        returns: (B, 784, 128)
        """

        # Convert to tokens
        x = self.patch_embed(x)
        print("After patch embedding:", x.shape)

        # Add positional encoding
        x = self.pos_embed(x)

        # Transformer layers
        for block in self.encoder:
            x = block(x)

        # Normalize
        x = self.norm(x)

        print("ViT output:", x.shape)

        return x


# Quick test
if __name__ == "__main__":
    model = VisionTransformer()
    dummy = torch.randn(1, 256, 28, 28)
    out = model(dummy)
    print("Final Output:", out.shape)
