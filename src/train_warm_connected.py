from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score)
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GraphConv, SAGEConv, GATConv
import torch_geometric.transforms as T
import torch_geometric

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'warm_connected_seed42'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read(name, cols):
    df = pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
    if not set(cols) <= set(df.columns):
        raise ValueError(f'{name}: missing required columns')
    for col in cols:
        df[col] = df[col].str.strip()
    if df[cols].eq('').any().any():
        raise ValueError(f'{name}: blank required fields')
    return df


def pairset(df):
    return set(zip(df.disease_id, df.gene_id))


def load_data():
    maps = {}
    for typ,col in [('disease','disease_id'), ('gene','gene_id'), ('phenotype','phenotype_id')]:
        df = read(f'nodes_{typ}.csv', [col]).sort_values(col)
        if df.empty or df[col].duplicated().any():
            raise ValueError(f'Invalid {typ} node table')
        maps[typ] = {v:i for i,v in enumerate(df[col])}
    candidates = read('candidate_genes.csv', ['gene_id'])
    if set(candidates.gene_id) != set(maps['gene']):
        raise ValueError('Candidate universe differs from gene nodes')
    positives = {s:read(f'{s}_edges.csv', ['disease_id','gene_id']) for s in ['train','val','test']}
    samples = {s:read(f'{s}_classification_samples.csv', ['disease_id','gene_id','label']) for s in positives}
    blocked = pairset(read('negative_block_pairs.csv',['disease_id','gene_id']))
    for s,df in samples.items():
        df['label'] = pd.to_numeric(df.label,errors='raise')
        if set(df.label.unique()) != {0,1} or df.duplicated(['disease_id','gene_id']).any():
            raise ValueError(f'{s}: invalid labels or duplicate sample pairs')
        if not df.disease_id.isin(maps['disease']).all() or not df.gene_id.isin(maps['gene']).all():
            raise ValueError(f'{s}: sample endpoint missing from node table')
        if pairset(df[df.label==1]) != pairset(positives[s]):
            raise ValueError(f'{s}: classification positives differ from split targets')
        if pairset(df[df.label==0]) & blocked:
            raise ValueError(f'{s}: sampled negatives include known associations')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if pairset(samples[a]) & pairset(samples[b]):
            raise ValueError(f'Classification pair overlap {a}/{b}')
    if set(positives['train'].gene_id) != set(maps['gene']):
        raise ValueError('Candidate gene without a training target link')
    dp = read('edges_disease_phenotype.csv',['disease_id','hpo_id'])
    if not dp.disease_id.isin(maps['disease']).all() or not dp.hpo_id.isin(maps['phenotype']).all():
        raise ValueError('Invalid phenotype edge endpoint')
    known = read('known_causal_pairs_for_filtering.csv',['disease_id','gene_id'])
    if not set.union(*(pairset(f) for f in positives.values())) <= pairset(known):
        raise ValueError('Ranking known-positive list incomplete')
    return maps,positives,samples,dp,known


def graph_for(setting,maps,positives,dp):
    data = HeteroData()
    active = ['disease','gene'] + (['phenotype'] if setting=='dg_plus_dp' else [])
    for typ in active:
        data[typ].num_nodes = len(maps[typ])
    train = positives['train']
    ei = torch.tensor([train.disease_id.map(maps['disease']).tolist(),train.gene_id.map(maps['gene']).tolist()],dtype=torch.long)
    data['disease','associated_with','gene'].edge_index = ei
    if setting=='dg_plus_dp':
        data['disease','has_phenotype','phenotype'].edge_index = torch.tensor([
            dp.disease_id.map(maps['disease']).tolist(),dp.hpo_id.map(maps['phenotype']).tolist()],dtype=torch.long)
    data = T.ToUndirected()(data)
    data.validate(raise_on_error=True)
    invd = {i:d for d,i in maps['disease'].items()}
    invg = {i:g for g,i in maps['gene'].items()}
    actual = {(invd[d],invg[g]) for d,g in data['disease','associated_with','gene'].edge_index.t().tolist()}
    reverse = data['gene','rev_associated_with','disease'].edge_index
    if not torch.equal(ei,reverse.flip(0)) or actual != pairset(train):
        raise ValueError('Training graph edge mismatch')
    if actual & (pairset(positives['val']) | pairset(positives['test'])):
        raise ValueError('Held-out positive leaked into training adjacency')
    return data


def encode(df,maps):
    return (torch.tensor(df.disease_id.map(maps['disease']).tolist(),dtype=torch.long),
            torch.tensor(df.gene_id.map(maps['gene']).tolist(),dtype=torch.long),
            torch.tensor(df.label.to_numpy(dtype=np.float32)))


class Model(torch.nn.Module):
    def __init__(self,data,kind,width=64):
        super().__init__()
        self.kind = kind
        self.ids = torch.nn.ModuleDict({t:torch.nn.Embedding(data[t].num_nodes,32) for t in data.node_types})
        self.project = torch.nn.ModuleDict({t:torch.nn.Linear(32,width) for t in data.node_types})
        self.layers = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        if kind!='id_mlp':
            for i in range(2):
                cin = 32 if i==0 else width
                relations = {}
                for edge in data.edge_types:
                    if kind=='graphconv':
                        layer = GraphConv((cin,cin),width)
                    elif kind=='sage':
                        layer = SAGEConv((cin,cin),width)
                    elif kind=='gat':
                        layer = GATConv((cin,cin),width,heads=2,concat=False,add_self_loops=False)
                    else:
                        raise ValueError(kind)
                    relations[edge]=layer
                self.layers.append(HeteroConv(relations,aggr='sum'))
                self.skips.append(torch.nn.ModuleDict({t:torch.nn.Linear(cin,width) for t in data.node_types}))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(2*width,width),torch.nn.ReLU(),torch.nn.Linear(width,1))

    def embed(self,data):
        x = {t:self.ids[t].weight for t in data.node_types}
        if self.kind=='id_mlp':
            return {t:F.relu(self.project[t](v)) for t,v in x.items()}
        for conv,skip in zip(self.layers,self.skips):
            message = conv(x,data.edge_index_dict)
            x = {t:F.relu(skip[t](v)+message[t]) if t in message else F.relu(skip[t](v)) for t,v in x.items()}
        return x

    def score(self,z,d,g):
        return self.decoder(torch.cat([z['disease'][d],z['gene'][g]],dim=-1)).squeeze(-1)

    def forward(self,data,d,g):
        return self.score(self.embed(data),d,g)


def classification(y,logits):
    p = torch.sigmoid(logits).detach().numpy()
    labels = y.numpy()
    pred = (p>=0.5).astype(int)
    return {'accuracy':float(accuracy_score(labels,pred)), 'precision':float(precision_score(labels,pred,zero_division=0)),
            'recall':float(recall_score(labels,pred,zero_division=0)), 'f1':float(f1_score(labels,pred,zero_division=0)),
            'auroc':float(roc_auc_score(labels,p)), 'average_precision':float(average_precision_score(labels,p))}


def train(kind,data,samples,maps,seed,args):
    seed_all(seed)
    model = Model(data,kind,args.hidden)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=5e-4)
    td,tg,ty = encode(samples['train'],maps)
    vd,vg,vy = encode(samples['val'],maps)
    pw = (ty==0).sum().float()/(ty==1).sum().float()
    best,best_state,stale,best_epoch = -1.,None,0,-1
    history=[]
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data,td,tg)
        loss = F.binary_cross_entropy_with_logits(logits,ty,pos_weight=pw)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = classification(vy,model(data,vd,vg))
        ap = val['average_precision']
        history.append({'epoch':epoch,'train_loss':loss.item(),'val_average_precision':ap})
        if ap>best:
            best,best_state,stale,best_epoch = ap,copy.deepcopy(model.state_dict()),0,epoch
        else:
            stale+=1
        if epoch%10==0 or epoch==args.epochs-1:
            print(f'  epoch={epoch:03d} loss={loss.item():.4f} val_AP={ap:.4f}',flush=True)
        if stale>=15:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model,pd.DataFrame(history),best_epoch


@torch.no_grad()
def evaluate(model,data,sample,positive,known,maps):
    d,g,y = encode(sample,maps)
    z = model.embed(data)
    logits = model.score(z,d,g)
    result = classification(y,logits)
    pred = sample.copy()
    pred['logit']=logits.numpy()
    pred['probability']=torch.sigmoid(logits).numpy()
    known_by_d={}
    for disease,gene in pairset(known):
        if gene in maps['gene']:
            known_by_d.setdefault(disease,set()).add(maps['gene'][gene])
    gene_index=torch.arange(len(maps['gene']))
    rank_rows=[]
    for disease,group in positive.groupby('disease_id'):
        ds=torch.full((len(gene_index),),maps['disease'][disease],dtype=torch.long)
        scores=model.score(z,ds,gene_index)
        for gene in group.gene_id:
            target=maps['gene'][gene]
            allowed=torch.ones(len(gene_index),dtype=torch.bool)
            others=known_by_d.get(disease,set())-{target}
            if others:
                allowed[list(others)]=False
            values=scores[allowed]
            score=scores[target]
            rank=1+(values>score).sum().item()+0.5*((values==score).sum().item()-1)
            rank_rows.append({'disease_id':disease,'gene_id':gene,'rank':rank,'filtered_candidate_count':int(allowed.sum())})
    ranks=pd.DataFrame(rank_rows)
    if ranks.empty:
        raise ValueError('No ranking queries')
    r=ranks['rank'].to_numpy()
    result.update({'hits@10':float(np.mean(r<=10)),'hits@20':float(np.mean(r<=20)),
                   'mrr':float(np.mean(1/r)),'mean_rank':float(r.mean()),'num_ranking_queries':len(r)})
    return result,pred,ranks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--hidden',type=int,default=64)
    parser.add_argument('--lr',type=float,default=0.005)
    parser.add_argument('--run-name',default='smoke_validation_only')
    parser.add_argument('--evaluate-test',action='store_true')
    parser.add_argument('--models',nargs='+',choices=['id_mlp','graphconv','sage','gat'],default=['id_mlp','graphconv','sage','gat'])
    args=parser.parse_args()
    if args.epochs<1 or args.hidden<1 or args.lr<=0:
        raise ValueError('epochs, hidden and lr must be positive')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        raise ValueError('Use letters, numbers, underscores or hyphens for run-name')
    output=ROOT/'results'/'warm_connected'/args.run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} has outputs. Choose a new --run-name.')
    maps,positives,samples,dp,known=load_data()
    graphs={s:graph_for(s,maps,positives,dp) for s in ['dg_only','dg_plus_dp']}
    print('PASS: target message-passing edges are training-only in both directions.')
    print(f'Candidate genes: {len(maps["gene"])}; test scoring enabled: {args.evaluate_test}')
    output.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    config.update({'device':'cpu','torch':str(torch.__version__),'pyg':torch_geometric.__version__,
                   'features':'learned ID embeddings, 32 dimensions; no independent biological features',
                   'decoder':'concatenation MLP; fixed 0.5 classification threshold',
                   'ranking':'all training-connected candidate genes; filter other known causal pairs; average ties',
                   'scope':'restricted warm-connected development benchmark; not clinical validation',
                   'models':'relation-specific 2-layer GNNs with learned skips; ID-MLP is not the old fixed-random-feature MLP',
                   'weight_decay':5e-4,'patience':15,'gradient_clip':5.,
                   'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv'))}})
    (output/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (output/'node_mappings.json').write_text(json.dumps(maps),encoding='utf-8')
    rows=[]
    for setting,data in graphs.items():
        for kind in args.models:
            # ID-MLP has no graph dependence: do not duplicate it across graph settings.
            if setting=='dg_plus_dp' and kind=='id_mlp':
                continue
            for seed in args.seeds:
                print(f'\nSETTING={setting} MODEL={kind} SEED={seed}',flush=True)
                model,history,best_epoch=train(kind,data,samples,maps,seed,args)
                stem=f'{setting}_{kind}_seed{seed}'
                history.to_csv(output/f'{stem}_history.csv',index=False)
                torch.save(model.state_dict(),output/f'{stem}_weights.pt')
                for split in (['val','test'] if args.evaluate_test else ['val']):
                    result,pred,ranks=evaluate(model,data,samples[split],positives[split],known,maps)
                    result.update(setting=setting,model=kind,seed=seed,split=split,best_epoch=best_epoch)
                    rows.append(result)
                    pred.to_csv(output/f'{stem}_{split}_predictions.csv',index=False)
                    ranks.to_csv(output/f'{stem}_{split}_ranks.csv',index=False)
                    pd.DataFrame(rows).to_csv(output/'metrics_per_seed.csv',index=False)
                    print(f'  {split}: AUROC={result["auroc"]:.4f}, AP={result["average_precision"]:.4f}, MRR={result["mrr"]:.4f}',flush=True)
    result=pd.DataFrame(rows)
    cols=['auroc','average_precision','f1','hits@10','hits@20','mrr','mean_rank']
    summary=result.groupby(['setting','model','split'],sort=False)[cols].agg(['mean','std'])
    summary.columns=[f'{metric}_{stat}' for metric,stat in summary.columns]
    summary.reset_index().to_csv(output/'summary.csv',index=False)
    print('\n===== RUN COMPLETE =====')
    print(result[['setting','model','seed','split','auroc','average_precision','hits@10','mrr']].to_string(index=False))
    print('Saved to:',output)
    if not args.evaluate_test:
        print('Validation-only run: test scores were NOT computed. Smoke results are NOT manuscript findings.')
    print('One seed has undefined between-seed SD (blank/NaN), not zero.')


if __name__=='__main__':
    main()