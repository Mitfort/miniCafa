import numpy as np
from collections import Counter
from itertools import product

class FeatureExtractor:
    STANDARD_AA = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    ]

    AA_WEIGHTS = {
        'A': 89,  'C': 121, 'D': 133, 'E': 147, 'F': 165, 'G': 75,
        'H': 155, 'I': 131, 'K': 146, 'L': 131, 'M': 149, 'N': 132,
        'P': 115, 'Q': 146, 'R': 174, 'S': 105, 'T': 119, 'V': 117,
        'W': 204, 'Y': 181,
    }

    HYDROPHOBIC = set(['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'])
    POSITIVE = set(['K', 'R', 'H'])
    NEGATIVE = set(['D', 'E'])
    AROMATIC = set(['F', 'W', 'Y'])

    def __init__(self, proteins):
        self.proteins = proteins
        self._dimers = [''.join(p) for p in product(self.STANDARD_AA, repeat=2)]

    @classmethod
    def feature_names(cls):
        freq_cols  = [f"freq_{aa}" for aa in cls.STANDARD_AA]
        dimer_cols = [f"dimer_{d}" for d in (''.join(p) for p in product(cls.STANDARD_AA, repeat=2))]
        
        global_cols = [
            "avg_aa_mass", "charge", "hydrophobicity",
            "aromaticity", "log_length",
        ]
        return freq_cols + dimer_cols + global_cols  # 20 + 400 + 5 = 425

    def extract_features(self) -> np.ndarray:
        all_features = []

        for p in self.proteins:
            seq = p.get("sequence") or ""
            if not isinstance(seq, str):
                seq = str(seq)
            length = len(seq)
            counts = Counter(seq)

            if length == 0:
                all_features.append(np.zeros(len(self.feature_names()), dtype=np.float32))
                continue

            # --- 1. Amino acid frequencies (20)
            aa_freq = np.array([counts[aa] / length for aa in self.STANDARD_AA],dtype=np.float32)

            # --- 2. Dipeptide frequencies (400)
            dimer_counts = Counter(seq[i:i + 2] for i in range(length - 1))
            n_dimers = max(length-1,1)
            dimer_freq = np.array([dimer_counts[d] / n_dimers for d in self._dimers],dtype=np.float32)

            # --- 3. Global physicochemical features (5)
            avg_mass = sum(self.AA_WEIGHTS.get(aa, 0) for aa in seq) / length
            charge = sum(
                1 if aa in self.POSITIVE else -1 if aa in self.NEGATIVE else 0
                for aa in seq
            ) / length
            hydrophobicity = sum(1 for aa in seq if aa in self.HYDROPHOBIC) / length
            aromaticity = sum(1 for aa in seq if aa in self.AROMATIC) / length
            log_length = np.log1p(length)

            global_feats = np.array([avg_mass, charge, hydrophobicity, aromaticity, log_length],dtype=np.float32)

            all_features.append(np.concatenate([aa_freq, dimer_freq, global_feats]))

        return np.array(all_features, dtype=np.float32)
