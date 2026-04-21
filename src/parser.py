import gzip
import re

def parse_gzip(file_path):
    with gzip.open(file_path, 'rt') as f:
        entry = []
        for line in f:
            if line.startswith('//'):
                yield process_entry(entry)
                entry = []
            else:
                entry.append(line.strip())

def filter_proteins(proteins, max_sequence_length=1000):
    filtered = []
    for p in proteins:
        if len(p['sequence']) <= max_sequence_length and p['go_terms']:
            filtered.append(p)
    return filtered

def process_entry(lines):
    accession = None
    sequence = []
    go_terms = []
    reading_seq = False

    for line in lines:
        if line.startswith('AC') and accession is None:
            accession = line[5:].split(';')[0].strip()
        elif line.startswith('DR') and 'GO;' in line:
            go_term = line.split('GO;')[1].split(';')[0].strip()
            go_terms.append(go_term)

        elif line.startswith('SQ'):
            reading_seq = True
        elif reading_seq:
            seq = re.sub(r'[^A-Z]', '', line)
            sequence.append(seq)

    return {
        'accession': accession,
        'sequence': ''.join(sequence),
        'go_terms': go_terms
    }