import numpy as np
import torch
import random
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
    max_samples: int = None,
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
    
    if max_samples is not None and len(proteins) > max_samples:
        proteins = random.sample(proteins, max_samples)
        print(f"Resized to max_samples={max_samples}")

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
        indices, test_size=test_split
    )
    adj_val = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=adj_val
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


def load_dataloaders_from_preprocessed(
    data_dir: str = '../data',
    batch_size: int = 64,
    max_length: int = 1000,
    num_workers: int = -1,
):
    """
    Wczytaj DataLoadery z preprocessowanych danych (output z data_analysis_2.ipynb).
    
    Oczekuje następujących plików w data_dir:
      - mini_cafa.csv (zawiera accession, sequence, seq_len, labels)
      - mini_cafa_features_train.csv, mini_cafa_features_val.csv, mini_cafa_features_test.csv
      - sample_weights_train.npy
      - train_indices.npy, val_indices.npy, test_indices.npy
    
    Returns:
        train_loader, val_loader, test_loader, mlb, norm_stats
    """
    import pandas as pd
    
    print("Wczytywanie preprocessowanych danych...")
    
    # 1. Wczytaj mini_cafa.csv i indeksy
    df = pd.read_csv(f'{data_dir}/mini_cafa.csv')
    train_idx = np.load(f'{data_dir}/train_indices.npy')
    val_idx = np.load(f'{data_dir}/val_indices.npy')
    test_idx = np.load(f'{data_dir}/test_indices.npy')
    
    # 2. Wczytaj features (preprocessowane)
    features_train = pd.read_csv(f'{data_dir}/mini_cafa_features_train.csv').values.astype(np.float32)
    features_val = pd.read_csv(f'{data_dir}/mini_cafa_features_val.csv').values.astype(np.float32)
    features_test = pd.read_csv(f'{data_dir}/mini_cafa_features_test.csv').values.astype(np.float32)
    
    # 3. Przygotuj protein objects wg indeksów
    proteins = df.to_dict(orient='records')
    proteins_train = [proteins[i] for i in train_idx]
    proteins_val = [proteins[i] for i in val_idx]
    proteins_test = [proteins[i] for i in test_idx]
    
    # 4. Binaryzuj labels (GO terms) - muszą być w tej samej kolejności co train_idx, val_idx, test_idx
    train_data = df.iloc[train_idx].reset_index(drop=True)
    val_data = df.iloc[val_idx].reset_index(drop=True)
    test_data = df.iloc[test_idx].reset_index(drop=True)
    
    # Parsuj labels z stringów "label1;label2;..."
    def parse_labels_column(data):
        return [
            [lbl.strip() for lbl in row.split(';') if lbl.strip()]
            for row in data['labels']
        ]
    
    mlb = MultiLabelBinarizer()
    labels_train_list = parse_labels_column(train_data)
    labels_all_train = mlb.fit_transform(labels_train_list)
    
    labels_val_list = parse_labels_column(val_data)
    labels_val = mlb.transform(labels_val_list)
    
    labels_test_list = parse_labels_column(test_data)
    labels_test = mlb.transform(labels_test_list)
    
    # 5. Utwórz DataLoaders
    def make_loader(proteins, features, labels, shuffle):
        dataset = ProteinDataset(
            proteins=proteins,
            physics_features=features,
            labels=labels,
            max_length=max_length
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    train_loader = make_loader(proteins_train, features_train, labels_all_train, shuffle=True)
    val_loader = make_loader(proteins_val, features_val, labels_val, shuffle=False)
    test_loader = make_loader(proteins_test, features_test, labels_test, shuffle=False)
    
    norm_stats = {
        'physics_dim': features_train.shape[1],
        'num_go_terms': labels_all_train.shape[1]
    }
    
    print(f"DataLoaders załadowane:")
    print(f"  Train: {len(train_loader.dataset)} (batches: {len(train_loader)})")
    print(f"  Val:   {len(val_loader.dataset)} (batches: {len(val_loader)})")
    print(f"  Test:  {len(test_loader.dataset)} (batches: {len(test_loader)})")
    print(f"  GO Terms: {norm_stats['num_go_terms']}")
    
    return train_loader, val_loader, test_loader, mlb, norm_stats
