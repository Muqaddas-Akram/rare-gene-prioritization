from pathlib import Path
import json
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'primary_dataset'
SPLITS = ROOT / 'data' / 'primary_splits_seed42'
OUTPUT = ROOT / 'data' / 'primary_graph_audit_seed42'


def read(path, columns):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(columns) <= set(df.columns):
        raise ValueError(f'{path.name}: missing required columns')
    for col in columns:
        df[col] = df[col].str.strip()
    if df[columns].eq('').any().any():
        raise ValueError(f'{path.name}: empty required values')
    return df


def pairs(frame):
    return set(zip(frame.disease_id, frame.gene_id))


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f'{OUTPUT} already has outputs; archive/rename it first.')
    node_frames, maps = {}, {}
    for typ, col in [('disease','disease_id'), ('gene','gene_id'), ('phenotype','phenotype_id')]:
        frame = read(SOURCE / f'nodes_{typ}.csv', [col])
        if frame[col].duplicated().any():
            raise ValueError(f'Duplicate {typ} node IDs')
        frame = frame.sort_values(col).reset_index(drop=True)
        node_frames[typ] = frame
        maps[typ] = {v: i for i, v in enumerate(frame[col])}
    all_targets = read(SOURCE / 'edges_disease_gene.csv', ['disease_id','gene_id'])
    positives = {s: read(SPLITS / f'{s}_edges.csv', ['disease_id','gene_id']) for s in ['train','val','test']}
    for s, frame in positives.items():
        if frame.empty or frame.duplicated(['disease_id','gene_id']).any():
            raise ValueError(f'{s}: empty or duplicate targets')
        if not frame.disease_id.isin(maps['disease']).all() or not frame.gene_id.isin(maps['gene']).all():
            raise ValueError(f'{s}: invalid endpoints')
    sets = {s: pairs(f) for s, f in positives.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if sets[a] & sets[b]:
            raise ValueError(f'Positive split overlap: {a}/{b}')
    if set.union(*sets.values()) != pairs(all_targets):
        raise ValueError('Positive split union differs from primary targets')
    dp = read(SOURCE / 'edges_disease_phenotype.csv', ['disease_id','hpo_id']).drop_duplicates(['disease_id','hpo_id'])
    if not dp.disease_id.isin(maps['disease']).all() or not dp.hpo_id.isin(maps['phenotype']).all():
        raise ValueError('Disease-phenotype edge has invalid endpoint')
    training = positives['train']
    train_genes = set(training.gene_id)
    train_diseases = set(training.disease_id)
    if train_diseases != set(maps['disease']):
        raise ValueError('Some disease nodes have no training target edge')
    graphs, graph_stats, split_stats = {}, [], []
    for setting in ['dg_only','dg_plus_dp']:
        graph = HeteroData()
        types = ['disease','gene'] if setting == 'dg_only' else ['disease','gene','phenotype']
        for typ in types:
            graph[typ].num_nodes = len(maps[typ])
            graph[typ].node_id = torch.arange(len(maps[typ]), dtype=torch.long)
        target_index = torch.tensor([
            training.disease_id.map(maps['disease']).tolist(),
            training.gene_id.map(maps['gene']).tolist()
        ], dtype=torch.long)
        graph['disease','associated_with','gene'].edge_index = target_index
        if setting == 'dg_plus_dp':
            graph['disease','has_phenotype','phenotype'].edge_index = torch.tensor([
                dp.disease_id.map(maps['disease']).tolist(),
                dp.hpo_id.map(maps['phenotype']).tolist()
            ], dtype=torch.long)
        graph = T.ToUndirected()(graph)
        graph.validate(raise_on_error=True)
        forward = graph['disease','associated_with','gene'].edge_index
        backward = graph['gene','rev_associated_with','disease'].edge_index
        if not torch.equal(forward, backward.flip(0)):
            raise ValueError('Reverse target edges differ from forward training edges')
        inv_d = {i:d for d,i in maps['disease'].items()}
        inv_g = {i:g for g,i in maps['gene'].items()}
        actual = {(inv_d[d],inv_g[g]) for d,g in forward.t().tolist()}
        if actual != sets['train'] or actual & (sets['val'] | sets['test']):
            raise ValueError('Held-out target edge found or training graph mismatch')
        gene_has_edge = torch.zeros(len(maps['gene']), dtype=torch.bool)
        gene_has_edge[forward[1]] = True
        isolated = int((~gene_has_edge).sum())
        for s in ['val','test']:
            frame = positives[s]
            unseen = ~frame.gene_id.isin(train_genes)
            split_stats.append({'setting': setting, 'split': s, 'positive_queries': len(frame),
                                'queries_with_gene_isolated_in_graph': int(unseen.sum()),
                                'isolated_gene_query_fraction': float(unseen.mean()),
                                'unique_isolated_target_genes': frame.loc[unseen,'gene_id'].nunique()})
        graph_stats.append({'setting':setting, 'diseases':len(maps['disease']), 'genes':len(maps['gene']),
                            'phenotypes':len(maps['phenotype']) if setting=='dg_plus_dp' else 0,
                            'train_dg_edges_forward':forward.shape[1],
                            'train_dg_edges_reverse':backward.shape[1],
                            'dp_edges_forward':len(dp) if setting=='dg_plus_dp' else 0,
                            'isolated_gene_nodes':isolated})
        graphs[setting] = graph
        print('\nSETTING:', setting)
        print(graph)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for setting,graph in graphs.items():
        torch.save(graph, OUTPUT / f'{setting}.pt')
    for typ,frame in node_frames.items():
        frame.to_csv(OUTPUT / f'node_order_{typ}.csv', index=False)
    (OUTPUT / 'node_mappings.json').write_text(json.dumps(maps,indent=2),encoding='utf-8')
    pd.DataFrame(graph_stats).to_csv(OUTPUT / 'graph_statistics.csv',index=False)
    pd.DataFrame(split_stats).to_csv(OUTPUT / 'isolated_target_queries.csv',index=False)
    (OUTPUT / 'audit_notes.json').write_text(json.dumps({
        'target_message_edges':'training only; held-out positives absent in both directions',
        'gene_phenotype_edges':'excluded', 'hierarchy':'excluded', 'features':'not generated',
        'training':'not performed',
        'warning':'isolated genes have no message-passing evidence; feature and evaluation strategy must be decided before interpreting a benchmark',
        'scope':'known-disease link completion, not clinical diagnostic validation'
    },indent=2),encoding='utf-8')
    print('\n===== GRAPH AUDIT =====')
    print(pd.DataFrame(graph_stats).to_string(index=False))
    print('\n===== ISOLATED TARGET QUERIES =====')
    print(pd.DataFrame(split_stats).to_string(index=False))
    print('\nPASS: training-only disease-gene edges, including reverse direction.')
    print('PASS: no gene-phenotype or hierarchy edges included.')
    print('Saved to:', OUTPUT)
    print('No node features generated; no training performed. Review isolated targets before choosing features.')


if __name__ == '__main__':
    main()