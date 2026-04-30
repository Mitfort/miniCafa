import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from src.parser import parse_gzip, filter_proteins
from src.feature_extraction_1 import FeatureExtractor

# One hot 

STANDARD_AA = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
]

AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

def one_hot_encode(sequence: str, max_length: int) -> torch.Tensor:
    """
    Encode a protein sequence as a (20, max_length) float32 tensor.

    Sequences shorter than max_length are zero-padded on the right.
    Unknown amino acids are silently skipped (left as zero — a neutral
    "unknown" token that the CNN can learn to ignore).
    """
    tensor = torch.zeros(20, max_length, dtype=torch.float32)
    for i, aa in enumerate(sequence[:max_length]):
        idx = AA_TO_IDX.get(aa)
        if idx is not None:
            tensor[idx, i] = 1.0
    return tensor

# Dataset 

class ProteinDataset(Dataset):
    """
    Each item returns:
        sequence_onehot : (20, max_length) float32
        physics         : (physics_dim)   float32
        label           : (num_go_terms)  float32  (multi-hot)
    """

    def __init__(self,
                 proteins: list,
                 physics_features: np.ndarray,
                 labels: np.ndarray,
                 max_length: int = 1000):
        self.proteins = proteins
        self.physics = torch.tensor(physics_features, dtype=torch.float32)
        self.labels = torch.tensor(labels,dtype=torch.float32)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx):
        seq_onehot = one_hot_encode(self.proteins[idx]['sequence'], self.max_length)
        return seq_onehot, self.physics[idx], self.labels[idx]

def build_dataloaders(
    file_path: str,
    max_sequence_length: int = 1000,
    min_go_term_count: int = 50,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = -1,
):
    """
    Parse → filter → encode → normalize → split → DataLoaders.

    Args:
        file_path:            Path to a .dat.gz UniProt flat-file.
        max_sequence_length:  Sequences longer than this are discarded.
        min_go_term_count:    GO terms appearing in fewer than N proteins
                              are dropped. Avoids training on near-unique labels.
        batch_size:           Samples per batch.
        val_split:            Fraction of data for validation.
        test_split:           Fraction of data for test.
        num_workers:          DataLoader workers.

    Returns:
        train_loader, val_loader, test_loader,
        mlb        (MultiLabelBinarizer — needed for label decoding),
        norm_stats (dict with 'mean' and 'std' — needed for inference)
    """

    print("Parsing proteins...")
    proteins = list(parse_gzip(file_path))
    proteins = filter_proteins(proteins, max_sequence_length)
    print(f"{len(proteins):,} proteins after length/annotation filter")

    # 2. Drop rare GO terms
    all_go = [go for p in proteins for go in p['go_terms']]
    go_freq = Counter(all_go)
    valid_go = {go for go, n in go_freq.items() if n >= min_go_term_count}

    for p in proteins:
        p['go_terms'] = [go for go in p['go_terms'] if go in valid_go]
    proteins = [p for p in proteins if p['go_terms']]

    print(f"{len(proteins):,} proteins | {len(valid_go):,} GO terms "
          f"(min_count={min_go_term_count})")

    # 3. Binarize GO term labels  →  multi-hot matrix
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform([p['go_terms'] for p in proteins])  # (N, num_go_terms)

    # 4. Extract physics features → (N, physics_dim)
    extractor = FeatureExtractor(proteins)
    physics_features = extractor.extract_features()# (N, 426)

    mean = physics_features.mean(axis=0)
    std = physics_features.std(axis=0) + 1e-8
    physics_features = (physics_features - mean) / std
    norm_stats = {'mean': mean, 'std': std, 'physics_dim': mean.shape[0], 'num_go_terms': labels.shape[1]}

    indices = np.arange(len(proteins))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42
    )
    adj_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=adj_val, random_state=42
    )

    print(f"Split: train={len(train_idx):,} | val={len(val_idx):,} | test={len(test_idx):,}")

    def make_dataset(idx):
        return ProteinDataset(
            proteins=[proteins[i] for i in idx],
            physics_features=physics_features[idx],
            labels=labels[idx],
            max_length=max_sequence_length,
        )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = make_loader(make_dataset(train_idx),shuffle=True)
    val_loader = make_loader(make_dataset(val_idx), shuffle=False)
    test_loader = make_loader(make_dataset(test_idx),shuffle=False)

    return train_loader, val_loader, test_loader, mlb, norm_stats
