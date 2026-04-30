import torch
import torch.nn as nn

from encoder import MultiScaleConvEncoder


class PhysicsNet(nn.Module):
    """
    Small MLP that processes hand-crafted physicochemical features.

    The normalization layer at the start makes this robust to unnormalized
    input — useful during inference on new data.
    """
    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinFunctionPredictor(nn.Module):
    """
    Hybrid CNN + physics feature model for GO term prediction.

    Architecture
    ------------
    Sequence (one-hot B×20×L)  →  MultiScaleConvEncoder  →  (B, 512)
                                                                        ↘
                                                                  concat (B, 576)  →  MLP  →  (B, num_go_terms)
                                                                        ↗
    Physics features (B, 426)  →  PhysicsNet             →  (B, 64)

    The sequence branch learns motif-level biology from scratch.
    The physics branch gives the model explicit biochemical priors
    (composition, charge, hydrophobicity, pI) as a fast shortcut,
    especially useful when training data is limited.

    Loss
    ----
    Use nn.BCEWithLogitsLoss — this is multi-label classification.
    The final layer outputs raw logits; sigmoid is applied at inference.
    """

    def __init__(self,
                 physics_dim: int,
                 num_go_terms: int,
                 cnn_channels: int = 64,
                 cnn_depth: int = 3,
                 physics_out: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        self.sequence_encoder = MultiScaleConvEncoder(
            in_channels=20,
            branch_channels=cnn_channels,
            depth=cnn_depth,
            dropout=dropout,
        )

        self.physics_encoder = PhysicsNet(
            input_dim=physics_dim,
            output_dim=physics_out,
            dropout=dropout,
        )

        combined_dim = self.sequence_encoder.output_dim + self.physics_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_go_terms),
            # No sigmoid — BCEWithLogitsLoss expects raw logits
        )

    def forward(self,
                sequence_onehot: torch.Tensor,
                physics_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_onehot:  (B, 20, L) float32 — one-hot encoded sequence
            physics_features: (B, physics_dim) float32 — extracted features

        Returns:
            logits: (B, num_go_terms) — raw scores, apply sigmoid for probabilities
        """
        seq_emb  = self.sequence_encoder(sequence_onehot)    # (B, 512)
        phys_emb = self.physics_encoder(physics_features)    # (B, 64)
        combined = torch.cat([seq_emb, phys_emb], dim=1)     # (B, 576)
        return self.classifier(combined)                      # (B, num_go_terms)

    def predict_proba(self,
                      sequence_onehot: torch.Tensor,
                      physics_features: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities"""
        return torch.sigmoid(self.forward(sequence_onehot, physics_features))
