import numpy as np

class FeatureExtractor:
    def __init__(self, proteins):
        self.proteins = proteins

    def extract_features(self):
        for p in self.proteins:
            length = len(p["sequence"])

            standard_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            seq = [1 if aa in p["sequence"] else 0 for aa in standard_aa]

            aa_frequency = np.array([p["sequence"].count(aa) for aa in standard_aa]) / length
            
            aa_weights = {
                'A': 89, 'C': 121, 'D': 133, 'E': 147, 'F': 165, 'G': 75, 'H': 155, 'I': 131,
                'K': 146, 'L': 131, 'M': 149, 'N': 132, 'P': 115, 'Q': 146, 'R': 174, 'S': 105,
                'T': 119, 'V': 117, 'W': 204, 'Y': 181
            }

            sequence_weight = sum(aa_weights[aa] for aa in p["sequence"] if aa in aa_weights) / length

            charge = sum(1 if aa in ['K','R','H'] else -1 if aa in ['D', 'E'] else 0 for aa in p["sequence"]) / length

            hydrofobic = sum(1 if aa in ['A', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'] else 0 for aa in p["sequence"]) / length

            feature_vector = np.concatenate([seq, aa_frequency, sequence_weight, charge, hydrofobic, length])

        return feature_vector