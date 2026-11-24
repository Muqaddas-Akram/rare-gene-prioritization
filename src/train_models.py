from pathlib import Path
import argparse
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GraphConv, SAGEConv, GATConv
import torch_geometric.transforms as T

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
INPUT_DIM = 32
HIDDEN = 64
LR = 0.005
WEIGHT_DECAY = 5e-4
PATIENCE = 15


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_csv(path, required, optional=False):
    if optional and not path.exists():
        return pd.DataFrame(columns=required)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        if optional:
            return pd.DataFrame(columns=required)
        raise ValueError(f"Empty CSV: {path}")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for col in required:
        df[col] = df[col].str.strip()
    if not df.empty and df[required].eq("").any().any():
        raise ValueError(f"{path.name}: empty required values")
    return df


def pairs(df):
    return set(zip(df['disease_id'], df['gene_id']))


def load_inputs():
    mappings = {}
    for typ, col in [('disease', 'disease_id'), ('gene', 'gene_id'), ('phenotype', 'phenotype_id')]:
        df = read_csv(PROCESSED / f'nodes_{typ}.csv', [col])
        if df.empty or df[col].duplicated().any():
            raise ValueError(f'nodes_{typ}.csv is empty or has duplicate IDs')
        mappings[typ] = {v: i for i, v in enumerate(df[col])}
    known = read_csv(PROCESSED / 'edges_disease_gene.csv', ['disease_id', 'gene_id'])
    positives, samples = {}, {}
    for split in ['train', 'val', 'test']:
        df = read_csv(SPLITS / f'{split}_edges.csv', ['disease_id', 'gene_id'])
        if df.empty or len(pairs(df)) != len(df):
            raise ValueError(f'{split}_edges.csv: empty or duplicated pairs')
        positives[split] = df
        sf = read_csv(SPLITS / f'{split}_classification_samples.csv', ['disease_id', 'gene_id', 'label'])
        sf['label'] = pd.to_numeric(sf['label'], errors='raise')
        if set(sf['label'].unique()) != {0, 1}:
            raise ValueError(f'{split}: both labels 0 and 1 are required')
        for frame in [df, sf]:
            if not frame.disease_id.isin(mappings['disease']).all() or not frame.gene_id.isin(mappings['gene']).all():
                raise ValueError(f'{split}: unknown disease/gene IDs. Fix upstream files.')
        if pairs(sf[sf.label == 1]) != pairs(df):
            raise ValueError(f'{split}: positive sample pairs do not match split edges')
        if pairs(sf[sf.label == 0]) & pairs(known):
            raise ValueError(f'{split}: known positives were sampled as negatives')
        samples[split] = sf
    sets = {s: pairs(df) for s, df in positives.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if sets[a] & sets[b]:
            raise ValueError(f'Positive split overlap: {a}/{b}')
    if set.union(*sets.values()) != pairs(known):
        raise ValueError('Split union does not match all known disease-gene edges')
    train_diseases = set(positives['train'].disease_id)
    if (set(positives['val'].disease_id) | set(positives['test'].disease_id)) - train_diseases:
        raise ValueError('Held-out diseases without a training disease-gene link')
    for split, sf in samples.items():
        n = int(sf.duplicated(['disease_id', 'gene_id']).sum())
        if n:
            print(f'WARNING: {split} classification samples contain {n} duplicate pairs.')
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = pairs(samples[a][samples[a].label == 0]) & pairs(samples[b][samples[b].label == 0])
        if overlap:
            print(f'WARNING: {a}/{b} share {len(overlap)} sampled negative pairs; audit sampling before publication.')
    print('PASS: positive splits are disjoint; all target message-passing edges will be TRAIN ONLY.')
    print('NOTE: gene-ID harmonization and phenotype provenance are NOT validated by this script.')
    return mappings, positives, samples, known


def make_graph(mappings, train, seed, setting='full'):
    set_seed(seed)
    data = HeteroData()
    features = {typ: torch.randn(len(mp), INPUT_DIM) for typ, mp in mappings.items()}
    active = ['disease', 'gene'] if setting == 'gene_disease_only' else list(mappings)
    for typ in active:
        data[typ].num_nodes = len(mappings[typ])
        data[typ].x = features[typ]

    def add(frame, source, relation, target, source_col, target_col):
        mask = frame[source_col].isin(mappings[source]) & frame[target_col].isin(mappings[target])
        skipped = int((~mask).sum())
        if skipped:
            print(f'WARNING: {relation}: excluded {skipped} rows with unmapped endpoints')
        frame = frame.loc[mask, [source_col, target_col]].drop_duplicates()
        if frame.empty:
            print(f'WARNING: {relation}: no usable edges')
            return
        index = torch.tensor([
            frame[source_col].map(mappings[source]).tolist(),
            frame[target_col].map(mappings[target]).tolist()
        ], dtype=torch.long)
        data[source, relation, target].edge_index = index

    add(train, 'disease', 'associated_with', 'gene', 'disease_id', 'gene_id')
    if setting in ['full', 'no_gene_phenotype']:
        dp = read_csv(PROCESSED / 'edges_disease_phenotype.csv', ['disease_id', 'hpo_id'])
        add(dp, 'disease', 'has_phenotype', 'phenotype', 'disease_id', 'hpo_id')
    if setting in ['full', 'no_disease_phenotype']:
        gp = read_csv(PROCESSED / 'edges_gene_phenotype.csv', ['gene_id', 'hpo_id'])
        add(gp, 'gene', 'associated_with_phenotype', 'phenotype', 'gene_id', 'hpo_id')
    print('Hierarchy excluded in this correction stage; fix ontology parsing before including it.')
    data = T.ToUndirected()(data)
    if not data.validate(raise_on_error=True):
        raise ValueError('Invalid graph')
    return data


def verify_target_graph(data, mappings, positives):
    reverse_d = {i: v for v, i in mappings['disease'].items()}
    reverse_g = {i: v for v, i in mappings['gene'].items()}
    ei = data['disease', 'associated_with', 'gene'].edge_index
    actual = {(reverse_d[d], reverse_g[g]) for d, g in ei.t().tolist()}
    rev = data['gene', 'rev_associated_with', 'disease'].edge_index
    actual_rev = {(reverse_d[d], reverse_g[g]) for g, d in rev.t().tolist()}
    expected = pairs(positives['train'])
    assert actual == expected == actual_rev, 'Target graph differs from training edges'
    for split in ['val', 'test']:
        assert not (actual & pairs(positives[split])), f'{split} targets leaked'
    print(f'PASS: {len(actual)} forward and reverse training target edges; zero held-out target links.')


def encode(df, mappings):
    return (
        torch.tensor(df.disease_id.map(mappings['disease']).tolist(), dtype=torch.long),
        torch.tensor(df.gene_id.map(mappings['gene']).tolist(), dtype=torch.long),
        torch.tensor(df.label.to_numpy(dtype=np.float32))
    )


class PairModel(torch.nn.Module):
    def __init__(self, data, kind):
        super().__init__()
        self.kind = kind
        self.projections = torch.nn.ModuleDict({t: torch.nn.Linear(INPUT_DIM, HIDDEN) for t in data.node_types})
        self.layers = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        if kind != 'mlp':
            for i in range(2):
                channels = INPUT_DIM if i == 0 else HIDDEN
                relations = {}
                for edge_type in data.edge_types:
                    if kind == 'graphconv':
                        layer = GraphConv((channels, channels), HIDDEN)
                    elif kind == 'sage':
                        layer = SAGEConv((channels, channels), HIDDEN)
                    elif kind == 'gat':
                        layer = GATConv((channels, channels), HIDDEN, heads=2, concat=False, add_self_loops=False)
                    else:
                        raise ValueError(kind)
                    relations[edge_type] = layer
                self.layers.append(HeteroConv(relations, aggr='sum'))
                self.skips.append(torch.nn.ModuleDict({t: torch.nn.Linear(channels, HIDDEN) for t in data.node_types}))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(2 * HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))

    def embeddings(self, data):
        if self.kind == 'mlp':
            return {t: F.relu(self.projections[t](x)) for t, x in data.x_dict.items()}
        x = data.x_dict
        for conv, skip in zip(self.layers, self.skips):
            messages = conv(x, data.edge_index_dict)
            x = {t: F.relu(skip[t](v) + messages[t]) if t in messages else F.relu(skip[t](v)) for t, v in x.items()}
        return x

    def score(self, z, d, g):
        return self.decoder(torch.cat([z['disease'][d], z['gene'][g]], dim=-1)).squeeze(-1)

    def forward(self, data, d, g):
        return self.score(self.embeddings(data), d, g)


def metrics(y, logits):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    labels = y.cpu().numpy()
    prediction = (probabilities >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(labels, prediction),
        'precision': precision_score(labels, prediction, zero_division=0),
        'recall': recall_score(labels, prediction, zero_division=0),
        'f1': f1_score(labels, prediction, zero_division=0),
        'auroc': roc_auc_score(labels, probabilities),
        'auprc': average_precision_score(labels, probabilities)
    }


def fit(kind, data, samples, mappings, seed, epochs):
    set_seed(seed)
    model = PairModel(data, kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    td, tg, ty = encode(samples['train'], mappings)
    vd, vg, vy = encode(samples['val'], mappings)
    weight = (ty == 0).sum().float() / (ty == 1).sum().float()
    best, best_state, patience = -1.0, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data, td, tg)
        loss = F.binary_cross_entropy_with_logits(logits, ty, pos_weight=weight)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            ap = metrics(vy, model(data, vd, vg))['auprc']
        history.append({'epoch': epoch, 'loss': loss.item(), 'val_auprc': ap})
        if ap > best:
            best, best_state, patience = ap, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if epoch % 10 == 0:
            print(f'{kind} seed={seed} epoch={epoch} loss={loss.item():.4f} val_AP={ap:.4f}')
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


@torch.no_grad()
def ranking(model, data, test_edges, known, mappings):
    model.eval()
    z = model.embeddings(data)
    genes = torch.arange(len(mappings['gene']), dtype=torch.long)
    known_by_disease = {}
    for d, g in pairs(known):
        if g in mappings['gene']:
            known_by_disease.setdefault(d, set()).add(mappings['gene'][g])
    ranks = []
    for disease, group in test_edges.groupby('disease_id'):
        d = torch.full((len(genes),), mappings['disease'][disease], dtype=torch.long)
        scores = model.score(z, d, genes)
        for true_gene in group.gene_id:
            target = mappings['gene'][true_gene]
            allowed = torch.ones(len(genes), dtype=torch.bool)
            other_known = known_by_disease.get(disease, set()) - {target}
            if other_known:
                allowed[list(other_known)] = False
            candidate_scores = scores[allowed]
            target_score = scores[target]
            rank = 1.0 + (candidate_scores > target_score).sum().item() + 0.5 * ((candidate_scores == target_score).sum().item() - 1)
            ranks.append(rank)
    if not ranks:
        raise ValueError('No ranking queries')
    ranks = np.asarray(ranks, dtype=float)
    return {'hits@10': float(np.mean(ranks <= 10)), 'hits@20': float(np.mean(ranks <= 20)),
            'mrr': float(np.mean(1 / ranks)), 'mean_rank': float(ranks.mean()), 'num_test_edges': len(ranks)}


def summarize(df, grouping):
    columns = ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc', 'hits@10', 'hits@20', 'mrr', 'mean_rank']
    summary = df.groupby(grouping, sort=False)[columns].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('--epochs must be positive')
    output = ROOT / 'results' / OUTPUT_FOLDER
    output.mkdir(parents=True, exist_ok=True)
    mappings, positives, samples, known = load_inputs()
    audit_graph = make_graph(mappings, positives['train'], args.seeds[0])
    verify_target_graph(audit_graph, mappings, positives)
    print(audit_graph)
    if args.audit_only:
        print('Audit finished. No model trained. Biological identifier mapping still needs verification.')
        return
    metadata = {'stage': 'development; gene mapping/provenance not yet audited', 'seeds': args.seeds,
                'epochs': args.epochs, 'feature_seed': 'same as run seed', 'target_graph': 'train_edges.csv only',
                'hierarchy': 'excluded', 'features': 'fixed Gaussian random features per seed',
                'architecture': 'two relation-specific layers with learned skip projections',
                'ranking': 'all candidate genes; filtered other known positives; average ties',
                'split_sizes': {s: len(df) for s, df in positives.items()},
                'torch_version': str(torch.__version__)}
    (output / 'run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []
    for setting, kind in EXPERIMENTS:
        for seed in args.seeds:
            print(f'\nSETTING={setting} MODEL={kind} SEED={seed}')
            data = make_graph(mappings, positives['train'], seed, setting)
            verify_target_graph(data, mappings, positives)
            model, history = fit(kind, data, samples, mappings, seed, args.epochs)
            d, g, labels = encode(samples['test'], mappings)
            with torch.no_grad():
                result = metrics(labels, model(data, d, g))
            result.update(ranking(model, data, positives['test'], known, mappings))
            result.update(model=kind.upper(), ablation=setting, seed=seed)
            rows.append(result)
            stem = f'{setting}_{kind}_{seed}'
            history.to_csv(output / f'{stem}_history.csv', index=False)
            torch.save(model.state_dict(), output / f'{stem}.pt')
            df = pd.DataFrame(rows)
            df.to_csv(output / 'seed_results.csv', index=False)
            summarize(df, GROUPING).to_csv(output / SUMMARY_NAME, index=False)
            print(result)
    print('\nFINAL DEVELOPMENT SUMMARY')
    print(summarize(pd.DataFrame(rows), GROUPING).to_string(index=False))
    print('Saved to:', output)
    print('Do NOT replace manuscript results until gene IDs and phenotype provenance are audited.')


DEFAULT_SEEDS = [42]
EXPERIMENTS = [('full', k) for k in ['mlp', 'graphconv', 'sage', 'gat']]
OUTPUT_FOLDER = 'step1_train_only_single'
GROUPING = 'model'
SUMMARY_NAME = 'baseline_model_results.csv'

if __name__ == '__main__':
    main()