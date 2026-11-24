from pathlib import Path
import argparse
import hashlib
import json
import random
import pandas as pd


def read(path, required):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(required).issubset(frame.columns):
        raise ValueError(f'{path.name}: missing required columns {required}')
    for col in required:
        frame[col] = frame[col].str.strip()
    if frame[required].eq('').any().any():
        raise ValueError(f'{path.name}: empty required values')
    return frame


def pairs(frame):
    return set(zip(frame.disease_id, frame.gene_id))


def build(root, split_seed, negative_seed, ratio):
    source = root / 'data' / 'primary_dataset'
    original_splits = root / 'data' / f'primary_splits_seed{split_seed}'
    output = root / 'data' / f'warm_connected_seed{split_seed}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} already has files. Archive/rename it before rerunning.')
    original = {s: read(original_splits / f'{s}_edges.csv', ['disease_id','gene_id']) for s in ['train','val','test']}
    all_targets = read(source / 'edges_disease_gene.csv', ['disease_id','gene_id'])
    blocks = read(source / 'negative_block_pairs.csv', ['disease_id','gene_id']).drop_duplicates()
    nodes = {t: read(source / f'nodes_{t}.csv', [c]) for t,c in [('disease','disease_id'),('gene','gene_id'),('phenotype','phenotype_id')]}
    dp = read(source / 'edges_disease_phenotype.csv', ['disease_id','hpo_id']).drop_duplicates(['disease_id','hpo_id'])
    for s, frame in original.items():
        if frame.empty or frame.duplicated(['disease_id','gene_id']).any():
            raise ValueError(f'{s}: empty or duplicate positive pairs')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if pairs(original[a]) & pairs(original[b]):
            raise ValueError(f'Positive split overlap {a}/{b}')
    if set.union(*(pairs(f) for f in original.values())) != pairs(all_targets):
        raise ValueError('Source positive splits do not cover primary target list exactly')
    connected = set(original['train'].gene_id)
    training_diseases = set(original['train'].disease_id)
    if not connected <= set(nodes['gene'].gene_id):
        raise ValueError('Missing gene nodes')
    selected = {'train': original['train'].copy()}
    excluded = []
    for s in ['val','test']:
        keep = original[s].gene_id.isin(connected)
        selected[s] = original[s][keep].copy()
        removed = original[s][~keep].copy()
        removed['original_split'] = s
        removed['exclusion_reason'] = 'gene_has_no_training_disease_gene_edge'
        excluded.append(removed)
        if not set(selected[s].disease_id) <= training_diseases:
            raise ValueError('Held-out disease is absent from training')
        if selected[s].empty:
            raise ValueError(f'{s}: no connected-gene targets remain')
    candidate_ids = sorted(connected)
    blocked_pairs = pairs(blocks)
    if not pairs(all_targets) <= blocked_pairs:
        raise ValueError('Negative block list missing known causal pairs')
    blocks_by_disease = {}
    for disease,gene in blocked_pairs:
        blocks_by_disease.setdefault(disease,set()).add(gene)
    rng = random.Random(negative_seed)
    negative_rows = {s: [] for s in selected}
    for disease in sorted(training_diseases):
        counts = {s: ratio * int(frame.disease_id.eq(disease).sum()) for s,frame in selected.items()}
        candidates = [g for g in candidate_ids if g not in blocks_by_disease.get(disease,set())]
        need = sum(counts.values())
        if need > len(candidates):
            raise ValueError(f'{disease}: {need} unique negatives needed, only {len(candidates)} available; no replacement allowed')
        sampled = rng.sample(candidates,need)
        offset = 0
        for s,count in counts.items():
            negative_rows[s].extend({'disease_id': disease,'gene_id':g} for g in sampled[offset:offset+count])
            offset += count
    negatives = {s:pd.DataFrame(rows,columns=['disease_id','gene_id']) for s,rows in negative_rows.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not pairs(negatives[a]) & pairs(negatives[b])
    samples,summary = {},[]
    for s, positive in selected.items():
        negative = negatives[s]
        assert len(negative) == ratio * len(positive)
        assert len(negative) == len(pairs(negative))
        assert not pairs(negative) & blocked_pairs
        combined = pd.concat([positive[['disease_id','gene_id']].assign(label=1),negative.assign(label=0)],ignore_index=True)
        combined = combined.sample(frac=1,random_state=negative_seed).reset_index(drop=True)
        combined['split'] = s
        assert not combined.duplicated(['disease_id','gene_id']).any()
        assert set(combined.gene_id) <= connected
        samples[s] = combined
        summary.append({'split':s,'original_positive_pairs':len(original[s]),'retained_positive_pairs':len(positive),
                        'excluded_isolated_gene_positive_pairs':len(original[s])-len(positive),
                        'retained_diseases':positive.disease_id.nunique(),'sampled_unobserved_negative_pairs':len(negative),
                        'candidate_genes':len(candidate_ids),'duplicate_classification_pairs':0})
    retained_targets = pd.concat(list(selected.values()),ignore_index=True)
    genes = nodes['gene'][nodes['gene'].gene_id.isin(connected)].sort_values('gene_id').copy()
    diseases = nodes['disease'][nodes['disease'].disease_id.isin(training_diseases)].copy()
    # Preserve old full-cohort counts explicitly, then report this benchmark's counts.
    for col in ['gene_count','phenotype_count']:
        if col in diseases.columns:
            diseases = diseases.rename(columns={col:'source_'+col})
    dp = dp[dp.disease_id.isin(training_diseases)].copy()
    diseases['training_gene_count'] = diseases.disease_id.map(original['train'].groupby('disease_id').gene_id.nunique()).fillna(0).astype(int)
    diseases['retained_target_gene_count'] = diseases.disease_id.map(retained_targets.groupby('disease_id').gene_id.nunique()).fillna(0).astype(int)
    diseases['phenotype_count'] = diseases.disease_id.map(dp.groupby('disease_id').hpo_id.nunique()).fillna(0).astype(int)
    hp = nodes['phenotype'][nodes['phenotype'].phenotype_id.isin(set(dp.hpo_id))].copy()
    if set(genes.gene_id) != connected or set(diseases.disease_id) != training_diseases:
        raise ValueError('Node coverage mismatch')
    if not set(dp.hpo_id) <= set(hp.phenotype_id):
        raise ValueError('Phenotype nodes missing')
    output.mkdir(parents=True,exist_ok=True)
    for s in selected:
        selected[s].to_csv(output / f'{s}_edges.csv',index=False)
        samples[s].to_csv(output / f'{s}_classification_samples.csv',index=False)
    genes.to_csv(output / 'nodes_gene.csv',index=False)
    genes[['gene_id']].to_csv(output / 'candidate_genes.csv',index=False)
    diseases.to_csv(output / 'nodes_disease.csv',index=False)
    hp.to_csv(output / 'nodes_phenotype.csv',index=False)
    dp.to_csv(output / 'edges_disease_phenotype.csv',index=False)
    # All known causal positives within this candidate universe, for filtered ranking only.
    known_causal = all_targets[all_targets.gene_id.isin(connected) & all_targets.disease_id.isin(training_diseases)]
    known_causal[['disease_id','gene_id']].drop_duplicates().to_csv(output / 'known_causal_pairs_for_filtering.csv',index=False)
    blocks[blocks.gene_id.isin(connected) & blocks.disease_id.isin(training_diseases)].to_csv(output / 'negative_block_pairs.csv',index=False)
    pd.concat(excluded,ignore_index=True).to_csv(output / 'excluded_isolated_gene_targets.csv',index=False)
    stats = pd.DataFrame(summary)
    stats.to_csv(output / 'split_statistics.csv',index=False)
    source_paths = [original_splits / f'{s}_edges.csv' for s in original] + [source / 'negative_block_pairs.csv']
    manifest = {'scope':'known-disease link completion restricted to genes with >=1 training target association',
                'split_seed':split_seed,'negative_seed':negative_seed,'negative_ratio':ratio,
                'candidate_selection':'training positive gene IDs only; no validation/test performance used',
                'negative_selection':'only training-connected candidates, excluding all mapped known association types',
                'excluded_targets':'preserved in excluded_isolated_gene_targets.csv; not evaluated by this baseline',
                'message_passing_rule':'use train_edges.csv and reverse edges ONLY for disease-gene adjacency',
                'known_causal_pairs_for_filtering':'evaluation filter only; NEVER use as message-passing edges',
                'features':'not generated','gene_phenotype_edges':'excluded','hierarchy':'excluded',
                'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    (output / 'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n===== TRAINING-CONNECTED BENCHMARK =====')
    print(stats.to_string(index=False))
    print('\nTraining positives unchanged:', pairs(selected['train']) == pairs(original['train']))
    print('PASS: candidates selected from training positive links only')
    print('PASS: all classification samples use the same connected-gene candidate universe')
    print('PASS: no duplicate or cross-split overlapping negative pairs')
    print('PASS: no blocked known associations sampled as negatives')
    print('Excluded isolated-gene targets preserved:',sum(len(f) for f in excluded))
    print('Saved to:',output)
    print('No training performed. Old training scripts do not yet read this folder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--negative-seed',type=int,default=4201)
    parser.add_argument('--negative-ratio',type=int,default=3)
    args = parser.parse_args()
    if args.negative_ratio < 1:
        raise ValueError('negative ratio must be positive')
    build(args.root.resolve(),args.split_seed,args.negative_seed,args.negative_ratio)