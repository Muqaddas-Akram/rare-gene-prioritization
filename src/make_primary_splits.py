from pathlib import Path
import argparse
import hashlib
import json
import random
import pandas as pd


def read(path, columns):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(columns) <= set(df.columns):
        raise ValueError(f'{path.name}: missing required columns {columns}')
    for col in columns:
        df[col] = df[col].str.strip()
    if df[columns].eq('').any().any():
        raise ValueError(f'{path.name}: empty required values')
    return df


def pairs(df):
    return set(zip(df.disease_id, df.gene_id))


def build(root, seed, ratio):
    source = root / 'data' / 'primary_dataset'
    output = root / 'data' / f'primary_splits_seed{seed}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; archive/rename it first.')
    dg = read(source / 'edges_disease_gene.csv', ['disease_id', 'gene_id'])
    genes = read(source / 'nodes_gene.csv', ['gene_id'])
    blocked_df = read(source / 'negative_block_pairs.csv', ['disease_id', 'gene_id'])
    if dg.empty or dg.duplicated(['disease_id', 'gene_id']).any():
        raise ValueError('Empty or duplicated positive target pairs')
    if genes.gene_id.duplicated().any():
        raise ValueError('Duplicate gene node IDs')
    gene_ids = sorted(genes.gene_id.tolist())
    if not set(dg.gene_id) <= set(gene_ids):
        raise ValueError('Positive target gene missing from nodes_gene.csv')
    blocked = pairs(blocked_df)
    if not pairs(dg) <= blocked:
        raise ValueError('Block list does not contain all positive targets')
    rng = random.Random(seed)
    groups = {d: g.sort_values('gene_id').reset_index(drop=True) for d, g in dg.groupby('disease_id', sort=True)}
    if any(len(g) < 2 for g in groups.values()):
        raise ValueError('Each disease must have >=2 target genes')
    two_gene_diseases = sorted(d for d, g in groups.items() if len(g) == 2)
    rng.shuffle(two_gene_diseases)
    two_gene_val = set(two_gene_diseases[:len(two_gene_diseases)//2])
    allocations = {s: [] for s in ['train', 'val', 'test']}
    audit = []
    for disease, group in groups.items():
        order = list(range(len(group)))
        rng.shuffle(order)
        n = len(group)
        if n == 2:
            n_val = int(disease in two_gene_val)
            n_test = 1 - n_val
        else:
            n_val = max(1, int(n * 0.10))
            n_test = max(1, int(n * 0.20))
            if n_val + n_test >= n:
                raise ValueError(f'No training edge would remain for {disease}')
        n_train = n - n_val - n_test
        chunks = {'train': order[:n_train], 'val': order[n_train:n_train+n_val], 'test': order[n_train+n_val:]}
        for split, idx in chunks.items():
            allocations[split].extend(group.iloc[idx].to_dict('records'))
        audit.append({'disease_id': disease, 'total_positive_genes': n, 'train': n_train, 'val': n_val, 'test': n_test})
    positives = {s: pd.DataFrame(rows, columns=dg.columns) for s, rows in allocations.items()}
    positive_sets = {s: pairs(df) for s, df in positives.items()}
    if any(df.empty for df in positives.values()):
        raise ValueError('A split is empty; inspect cohort size and rules.')
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        assert not positive_sets[a] & positive_sets[b]
    assert set.union(*positive_sets.values()) == pairs(dg)
    train_diseases = set(positives['train'].disease_id)
    assert train_diseases == set(groups)

    by_disease_blocked = {}
    for d, g in blocked:
        by_disease_blocked.setdefault(d, set()).add(g)
    negative_rows = {s: [] for s in positives}
    for disease in sorted(groups):
        candidates = [g for g in gene_ids if g not in by_disease_blocked.get(disease, set())]
        counts = {s: ratio * int(df.disease_id.eq(disease).sum()) for s, df in positives.items()}
        needed = sum(counts.values())
        if needed > len(candidates):
            raise ValueError(f'{disease}: need {needed} unique negatives, only {len(candidates)} allowed. No replacement performed.')
        selected = rng.sample(candidates, needed)
        offset = 0
        for split, count in counts.items():
            negative_rows[split].extend({'disease_id': disease, 'gene_id': gene} for gene in selected[offset:offset+count])
            offset += count
    negatives = {s: pd.DataFrame(rows, columns=['disease_id','gene_id']) for s, rows in negative_rows.items()}
    negative_sets = {s: pairs(df) for s, df in negatives.items()}
    for s, df in negatives.items():
        assert len(df) == len(negative_sets[s])
        assert not negative_sets[s] & blocked
        assert len(df) == ratio * len(positives[s])
    for a, b in [('train','val'), ('train','test'), ('val','test')]:
        assert not negative_sets[a] & negative_sets[b]
    train_genes = set(positives['train'].gene_id)
    summary = []
    samples = {}
    for split, df in positives.items():
        pos = df[['disease_id','gene_id']].copy()
        pos['label'] = 1
        neg = negatives[split].copy()
        neg['label'] = 0
        combined = pd.concat([pos, neg], ignore_index=True)
        order = list(range(len(combined)))
        rng.shuffle(order)
        combined = combined.iloc[order].reset_index(drop=True)
        combined['split'] = split
        assert not combined.duplicated(['disease_id','gene_id']).any()
        samples[split] = combined
        unseen = ~df.gene_id.isin(train_genes)
        summary.append({'split': split, 'positive_edges': len(df), 'positive_fraction': len(df)/len(dg),
                        'diseases': df.disease_id.nunique(), 'negative_pairs': len(neg),
                        'duplicate_sample_pairs': 0, 'positive_edges_gene_absent_from_training_DG': int(unseen.sum()),
                        'unique_genes_absent_from_training_DG': df.loc[unseen,'gene_id'].nunique()})
    output.mkdir(parents=True, exist_ok=True)
    for split in positives:
        positives[split].to_csv(output / f'{split}_edges.csv', index=False)
        samples[split].to_csv(output / f'{split}_classification_samples.csv', index=False)
    pd.DataFrame(audit).to_csv(output / 'disease_split_allocation.csv', index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output / 'split_statistics.csv', index=False)
    source_files = ['edges_disease_gene.csv','negative_block_pairs.csv','nodes_gene.csv']
    config = {'split_seed': seed, 'negative_to_positive_ratio': ratio,
              'two_gene_diseases': len(two_gene_diseases), 'two_gene_diseases_val': len(two_gene_val),
              'two_gene_diseases_test': len(two_gene_diseases)-len(two_gene_val),
              'rules': ['n=2: one train positive; one held-out positive assigned to val or test, balanced across diseases',
                        'n>=3: val=max(1,floor(0.1*n)); test=max(1,floor(0.2*n)); train=remainder',
                        'each disease has >=1 training target; no guarantee all genes have a training target',
                        'negatives sampled without replacement per disease then partitioned across splits',
                        'all known mapped association types excluded from negative candidates'],
              'message_passing': 'not built; later use train_edges.csv only for disease-gene edges',
              'evaluation_scope': 'known diseases, retained candidate genes only; report genes with no training target separately',
              'source_sha256': {f: hashlib.sha256((source/f).read_bytes()).hexdigest() for f in source_files}}
    (output / 'split_manifest.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('\n===== CORRECTED SPLIT STATISTICS =====')
    print(summary_df.to_string(index=False))
    print(f'\nTwo-gene diseases: {len(two_gene_diseases)}; held-out allocation val={len(two_gene_val)}, test={len(two_gene_diseases)-len(two_gene_val)}')
    print('PASS: positive split overlaps = 0')
    print('PASS: negative split overlaps = 0')
    print('PASS: duplicate classification pairs = 0')
    print('PASS: sampled negatives intersect known blocked associations = 0')
    print('PASS: every disease retains at least one training positive')
    print('Saved to:', output)
    print('STOP: old training scripts still reference the old graph/split folders. Do not run them yet.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--negative-ratio', type=int, default=3)
    args = parser.parse_args()
    if args.negative_ratio < 1:
        raise ValueError('--negative-ratio must be >=1')
    build(args.root.resolve(), args.seed, args.negative_ratio)