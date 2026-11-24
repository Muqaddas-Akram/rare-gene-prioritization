from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data'/'warm_connected_seed42'
OUTPUT=ROOT/'results'/'warm_connected'/'structural_baselines_validation'


def read(name,cols):
    frame=pd.read_csv(SOURCE/name,dtype=str,keep_default_na=False)
    if not set(cols)<=set(frame.columns):
        raise ValueError(f'{name}: required columns missing')
    for col in cols:
        frame[col]=frame[col].str.strip()
    if frame[cols].eq('').any().any():
        raise ValueError(f'{name}: empty required values')
    return frame


def pairset(frame):
    return set(zip(frame.disease_id,frame.gene_id))


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f'{OUTPUT} already has files; archive/rename it first.')
    used={}
    for typ,col in [('disease','disease_id'),('gene','gene_id'),('phenotype','phenotype_id')]:
        frame=read(f'nodes_{typ}.csv',[col])
        if frame[col].duplicated().any():
            raise ValueError(f'Duplicate {typ} IDs')
        used[typ]={v:i for i,v in enumerate(sorted(frame[col]))}
    train=read('train_edges.csv',['disease_id','gene_id']).drop_duplicates(['disease_id','gene_id'])
    val=read('val_edges.csv',['disease_id','gene_id']).drop_duplicates(['disease_id','gene_id'])
    samples=read('val_classification_samples.csv',['disease_id','gene_id','label'])
    known=read('known_causal_pairs_for_filtering.csv',['disease_id','gene_id'])
    blocked=read('negative_block_pairs.csv',['disease_id','gene_id'])
    dp=read('edges_disease_phenotype.csv',['disease_id','hpo_id']).drop_duplicates(['disease_id','hpo_id'])
    if pairset(train)&pairset(val):
        raise ValueError('Train/validation positive overlap')
    samples['label']=pd.to_numeric(samples.label,errors='raise')
    if set(samples.label.unique())!={0,1} or samples.duplicated(['disease_id','gene_id']).any():
        raise ValueError('Invalid validation classification samples')
    if pairset(samples[samples.label==1])!=pairset(val):
        raise ValueError('Validation sample positives do not match targets')
    if pairset(samples[samples.label==0])&pairset(blocked):
        raise ValueError('Known association sampled as a negative')
    for frame in [train,val,samples,known]:
        if not frame.disease_id.isin(used['disease']).all() or not frame.gene_id.isin(used['gene']).all():
            raise ValueError('Unknown disease/gene endpoint')
    if not dp.disease_id.isin(used['disease']).all() or not dp.hpo_id.isin(used['phenotype']).all():
        raise ValueError('Unknown phenotype endpoint')
    nd,ng,nh=[len(used[t]) for t in ['disease','gene','phenotype']]
    A=np.zeros((nd,ng),dtype=np.float64)
    A[train.disease_id.map(used['disease']),train.gene_id.map(used['gene'])]=1.
    B=np.zeros((nd,nh),dtype=np.float64)
    B[dp.disease_id.map(used['disease']),dp.hpo_id.map(used['phenotype'])]=1.
    disease_degree=A.sum(axis=1)
    gene_degree=A.sum(axis=0)
    if (disease_degree==0).any() or (gene_degree==0).any():
        raise ValueError('This baseline expects training-connected diseases and genes')
    Pdg=A/disease_degree[:,None]
    Pgd=A.T/gene_degree[:,None]
    walk3=Pdg@Pgd@Pdg
    lengths=np.linalg.norm(B,axis=1)
    normalized=B/np.maximum(lengths[:,None],1.)
    similarity=normalized@normalized.T
    np.fill_diagonal(similarity,0.)
    similarity=np.maximum(similarity,0.)
    phenotype_transfer=similarity@A
    phenotype_transfer/=np.maximum(similarity.sum(axis=1,keepdims=True),1e-12)
    scores_by_method={
        'gene_degree':np.broadcast_to(gene_degree,(nd,ng)),
        'dg_three_step_walk':walk3,
        'dp_cosine_transfer':phenotype_transfer
    }
    known_by_d={}
    for d,g in pairset(known):
        known_by_d.setdefault(d,set()).add(used['gene'][g])
    d_idx=samples.disease_id.map(used['disease']).to_numpy()
    g_idx=samples.gene_id.map(used['gene']).to_numpy()
    labels=samples.label.to_numpy(dtype=int)
    rows=[]
    artifacts={}
    for method,scores in scores_by_method.items():
        if not np.isfinite(scores).all():
            raise ValueError(f'{method}: invalid scores')
        pair_scores=scores[d_idx,g_idx]
        ranking_rows=[]
        for _,row in val.iterrows():
            d,g=row.disease_id,row.gene_id
            gi=used['gene'][g]
            full_scores=scores[used['disease'][d]]
            allowed=np.ones(ng,dtype=bool)
            other_known=known_by_d.get(d,set())-{gi}
            if other_known:
                allowed[list(other_known)]=False
            target=full_scores[gi]
            candidate_scores=full_scores[allowed]
            n_greater=int((candidate_scores>target).sum())
            n_equal=int((candidate_scores==target).sum())
            rank=1+n_greater+0.5*(n_equal-1)
            ranking_rows.append({'disease_id':d,'gene_id':g,'rank':rank,'tied_candidates':n_equal,
                                 'target_score':target,'filtered_candidate_count':int(allowed.sum())})
        ranking=pd.DataFrame(ranking_rows)
        r=ranking['rank'].to_numpy()
        rows.append({'setting': 'dg_plus_dp' if method=='dp_cosine_transfer' else 'dg_only',
                     'model':method,'split':'val','auroc':roc_auc_score(labels,pair_scores),
                     'average_precision':average_precision_score(labels,pair_scores),
                     'hits@10':float(np.mean(r<=10)),'hits@20':float(np.mean(r<=20)),
                     'mrr':float(np.mean(1/r)),'mean_rank':float(r.mean()),'num_ranking_queries':len(r),
                     'mean_tied_candidates':float(ranking.tied_candidates.mean()),
                     'zero_score_target_fraction':float((ranking.target_score==0).mean())})
        predictions=samples.copy()
        predictions['raw_score']=pair_scores
        artifacts[method]=(predictions,ranking)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(OUTPUT/'baseline_metrics.csv',index=False)
    for method,(pred,rank) in artifacts.items():
        pred.to_csv(OUTPUT/f'{method}_val_predictions.csv',index=False)
        rank.to_csv(OUTPUT/f'{method}_val_ranks.csv',index=False)
    config={'scope':'validation-only controls; no test scoring; deterministic, no seed SD',
            'gene_degree':'number of training diseases associated with gene; ignores query disease',
            'dg_three_step_walk':'(row_normalize(A)) @ (row_normalize(A.T)) @ (row_normalize(A))',
            'dp_cosine_transfer':'cosine similarity of direct disease HPO profiles, diagonal zero; weighted sum of training disease-gene links divided by similarity sum',
            'not_reimplementations':'these simple controls are not PRINCE, RWR-with-restart or Phen2Disease',
            'ranking':'filter other known causal pairs; average-tie rank, same convention as current neural script',
            'tie_warning':'Hits@K/MRR from average ranks are not expected random-tie-breaking Hits@K/MRR; sparse-score baselines can have many ties',
            'scores':'uncalibrated raw ranking scores; no threshold-based classification metrics',
            'source_hashes':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['train_edges.csv','val_edges.csv','val_classification_samples.csv','edges_disease_phenotype.csv','known_causal_pairs_for_filtering.csv']}}
    (OUTPUT/'baseline_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('\n===== STRUCTURAL BASELINES: VALIDATION ONLY =====')
    print(result.to_string(index=False))
    print('\nPASS: scores built from training disease-gene links only; test scores NOT computed.')
    print('These are deterministic controls: no repeated-seed SD should be invented.')
    print('Many ties are possible. Average-tie ranks are used consistently; inspect tie diagnostics before drawing ranking conclusions.')
    print('Saved to:',OUTPUT)


if __name__=='__main__':
    main()