import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConv(nn.Module):
    """
    Process:
        1. -> Input (Batch, C_in, Length)
        2. -> Conv1d (C_in → C_out, kernel_size, padding)
        3. -> BatchNorm1d
        4. -> GELU
        5. -> Dropout
        6. -> Conv1d (C_out → C_out, kernel_size, padding)
        7. -> BatchNorm1d
        8. -> Add input (residual connection)
        9. -> GELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.block(x)

        # The residual part is to prevent vanishing gradients.
        return F.gelu(x + residual)
    
    
class ConvBranch(nn.Module):
    """
    Encoder will use multiple ConvBranches (each separately) with different kernel sizes
    to capture more filters/patterns at different scales.

    Kernel sizes:
        3  — dipeptide / tripeptide motifs
        5  — short functional motifs
        9  — medium structural patterns (e.g. alpha-helix turns ~3.6 aa)
        15 — longer domain-level patterns
    
    Each branch:
        1) projects input to branch_channels (e.g. 64)
        2) stacks several residual conv blocks (e.g. 3)
        3) global max pool + mean pool → 2*branch_channels output features
    """

    def __init__(self, in_channels, out_channels, kernel_size, depth, dropout):
        super().__init__()
        padding = kernel_size // 2

        self.project = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        self.residuals = nn.Sequential(
            *[ResidualConv(out_channels, out_channels, kernel_size, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.project(x)
        x = self.residuals(x) # (Batch, out_channels, Length)

        # Max pool "is this pattern present anywhere? (1 if present, 0 if not)"
        # Mean pool "how prevalent is this pattern across the sequence? (0 to 1)

        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1) # (Batch, out_channels)
        mean_pool = x.mean(dim=-1) # (Batch, out_channels)

        return torch.cat([max_pool, mean_pool], dim=1) # (Batch, 2*out_channels)
    
class ProteinEncoder(nn.Module):
    """
    MultiBranch encoder.
        - Input: (B, 20, L) one-hot encoded sequence
        - Output: (B, 2 * branch_channels * num_branches) concatenated features from all branches
    
        The output dimension is determined by the number of branches and the branch_channels:
            For example, with 3 branches and 64 channels each → 2 * 64 * 3 = 384 features.
    """
    def __init__(self, in_channels:int = 20, branch_channels:int = 64, kernel_sizes: list = [3, 5, 9], depth:int = 3, dropout:float = 0.1):
        super().__init__()
        
        self.branches = nn.ModuleList([
            ConvBranch(in_channels, branch_channels, k, depth, dropout) for k in kernel_sizes
        ])

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1) # Cat will concate features from all branches as rows.



