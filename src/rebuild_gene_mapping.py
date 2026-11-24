from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import pandas as pd


def text(element, path):
    return (element.findtext(path) or '').strip()


def ncbi(value):
    value = str(value).strip()
    value = re.sub(r'^(NCBIGene:|NCBI Gene:|ENTREZ:)', '', value, flags=re.I)
    return 'NCBIGene:' + str(int(value)) if value.isdigit() and int(value) > 0 else ''


def hgnc(value):
    value = re.sub(r'^HGNC:', '', str(value).strip(), flags=re.I)
    return 'HGNC:' + str(int(value)) if value.isdigit() else ''


def canonical_disease(value):
    value = value.strip()
    match = re.fullmatch(r'(?:ORPHA|Orphanet):(\d+)', value, flags=re.I)
    return 'ORPHA:' + str(int(match.group(1))) if match else value


def norm_header(value):
    return re.sub(r'[^a-z0-9]', '', value.lower().lstrip('#').strip())


def read_annotation(path, kind):
    header = None
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for line in handle:
            fields = next(csv.reader([line.rstrip('\r\n')], delimiter='\t'))
            keys = [norm_header(v) for v in fields]
            if kind == 'disease':
                is_header = 'hpoid' in keys and any(k in keys for k in ['databaseid', 'diseaseid'])
            else:
                is_header = 'hpoid' in keys and any(k in keys for k in ['ncbigeneid', 'entrezgeneid'])
            if is_header:
                header = keys
                break
        if header is None:
            raise ValueError(f'{path.name}: supported header not found. Do not guess column positions; inspect the raw header.')
        for line_no, fields in enumerate(csv.reader(handle, delimiter='\t'), start=1):
            if not fields or fields[0].startswith('#'):
                continue
            if len(fields) != len(header):
                raise ValueError(f'{path.name}: malformed data row {line_no}, {len(fields)} fields versus {len(header)} header columns')
            yield dict(zip(header, fields))


def first(row, keys):
    return next((row[k].strip() for k in keys if row.get(k, '').strip()), '')


def save(rows, columns, path, unique=None):
    df = pd.DataFrame(rows, columns=columns)
    if unique:
        df = df.drop_duplicates(unique)
    df.to_csv(path, index=False)
    return df


def build(root, min_genes, min_phenotypes):
    raw = root / 'data' / 'raw'
    output = root / 'data' / 'processed_gene_ids_fixed'
    files = {name: raw / name for name in ['hgnc_complete_set.txt', 'en_product6.xml', 'phenotype.hpoa', 'genes_to_phenotype.txt']}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty. Rename/archive it before rerunning; existing outputs will not be overwritten.')
    hg = pd.read_csv(files['hgnc_complete_set.txt'], sep='\t', dtype=str, keep_default_na=False)
    required = {'hgnc_id', 'symbol', 'entrez_id', 'status'}
    if not required.issubset(hg.columns):
        raise ValueError(f'HGNC missing columns: {required - set(hg.columns)}')
    approved = hg[hg.status.eq('Approved')].copy()
    approved['gene_id'] = approved.entrez_id.map(ncbi)
    approved = approved[approved.gene_id.ne('')]
    ambiguous_hg = set(approved.loc[approved.duplicated('hgnc_id', keep=False), 'hgnc_id'])
    ambiguous_ncbi = set(approved.loc[approved.duplicated('gene_id', keep=False), 'gene_id'])
    approved = approved[~approved.hgnc_id.isin(ambiguous_hg) & ~approved.gene_id.isin(ambiguous_ncbi)]
    hg_map = approved.set_index('hgnc_id').to_dict('index')
    gene_map = approved.set_index('gene_id').to_dict('index')
    print('Unambiguous approved HGNC records with NCBI mapping:', len(approved))

    tree = ET.parse(files['en_product6.xml'])
    for node in tree.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    diseases, associations, rejects, discrepancies = [], [], [], []
    for disorder in tree.getroot().iter('Disorder'):
        code = text(disorder, 'OrphaCode')
        if not code.isdigit():
            continue
        did = 'ORPHA:' + str(int(code))
        diseases.append({'disease_id': did, 'disease_name': text(disorder, 'Name'),
                         'disorder_type': text(disorder, 'DisorderType/Name'),
                         'disorder_group': text(disorder, 'DisorderGroup/Name')})
        for association in disorder.findall('./DisorderGeneAssociationList/DisorderGeneAssociation'):
            gene = association.find('Gene')
            if gene is None:
                continue
            symbol = text(gene, 'Symbol')
            refs = {hgnc(text(ref, 'Reference')) for ref in gene.findall('./ExternalReferenceList/ExternalReference') if text(ref, 'Source').upper() == 'HGNC'} - {''}
            if len(refs) != 1:
                rejects.append([did, symbol, '|'.join(sorted(refs)), 'missing_or_multiple_HGNC_IDs'])
                continue
            hid = next(iter(refs))
            mapped = hg_map.get(hid)
            if mapped is None:
                rejects.append([did, symbol, hid, 'no_unambiguous_approved_HGNC_to_NCBI_mapping'])
                continue
            if symbol and symbol != mapped['symbol']:
                discrepancies.append(['Orphanet', symbol, mapped['symbol'], mapped['gene_id']])
            associations.append({'disease_id': did, 'gene_id': mapped['gene_id'], 'gene_symbol': mapped['symbol'],
                                 'hgnc_id': hid, 'source_gene_symbol': symbol,
                                 'association_type': text(association, 'DisorderGeneAssociationType/Name'),
                                 'association_status': text(association, 'DisorderGeneAssociationStatus/Name'),
                                 'edge_type': 'DISEASE_ASSOCIATED_WITH_GENE'})
    if not associations:
        raise ValueError('No Orphanet gene associations mapped. Inspect XML and mapping reports.')
    all_diseases = pd.DataFrame(diseases).drop_duplicates('disease_id')
    dg = pd.DataFrame(associations).drop_duplicates(['disease_id', 'gene_id', 'association_type', 'association_status'])

    dp_rows, source_counts = [], {}
    for row in read_annotation(files['phenotype.hpoa'], 'disease'):
        did = canonical_disease(first(row, ['databaseid', 'diseaseid']))
        prefix = did.split(':')[0]
        source_counts[prefix] = source_counts.get(prefix, 0) + 1
        if not did.startswith('ORPHA:'):
            continue
        if 'NOT' in row.get('qualifier', '').upper().split('|'):
            continue
        aspect = row.get('aspect', '').strip()
        if aspect and aspect.upper() != 'P':
            continue
        hp = first(row, ['hpoid'])
        if re.fullmatch(r'HP:\d+', hp):
            dp_rows.append({'disease_id': did, 'hpo_id': hp, 'edge_type': 'DISEASE_HAS_PHENOTYPE'})
    dp = pd.DataFrame(dp_rows, columns=['disease_id', 'hpo_id', 'edge_type']).drop_duplicates(['disease_id', 'hpo_id'])
    if dp.empty:
        raise ValueError('No direct ORPHA phenotype annotations. OMIM-to-ORPHA inference is deliberately not guessed.')
    dg_unique = dg.drop_duplicates(['disease_id', 'gene_id'])
    gene_counts = dg_unique.groupby('disease_id').size()
    phenotype_counts = dp.groupby('disease_id').size()
    all_diseases['gene_count'] = all_diseases.disease_id.map(gene_counts).fillna(0).astype(int)
    all_diseases['phenotype_count'] = all_diseases.disease_id.map(phenotype_counts).fillna(0).astype(int)
    selected = all_diseases[(all_diseases.gene_count >= min_genes) & (all_diseases.phenotype_count >= min_phenotypes)].copy()
    if selected.empty:
        raise ValueError('No diseases meet current criteria; inspect source overlap before changing thresholds.')
    ids = set(selected.disease_id)
    dg = dg[dg.disease_id.isin(ids)].copy()
    dg_unique = dg.drop_duplicates(['disease_id', 'gene_id']).copy()
    dp = dp[dp.disease_id.isin(ids)].copy()
    used_genes, used_hp = set(dg.gene_id), set(dp.hpo_id)

    gp_rows, gp_provenance, excluded_gp, hp_names = [], [], [], {}
    raw_gp_count = 0
    for row in read_annotation(files['genes_to_phenotype.txt'], 'gene'):
        raw_gp_count += 1
        gid = ncbi(first(row, ['ncbigeneid', 'entrezgeneid']))
        hp = first(row, ['hpoid'])
        source_symbol = first(row, ['genesymbol', 'entrezgenesymbol'])
        if gid not in gene_map:
            excluded_gp.append([gid, source_symbol, 'no_unambiguous_approved_HGNC_record'])
            continue
        if gid not in used_genes or hp not in used_hp:
            continue
        symbol = gene_map[gid]['symbol']
        if source_symbol and source_symbol != symbol:
            discrepancies.append(['HPO', source_symbol, symbol, gid])
        gp_rows.append({'gene_id': gid, 'gene_symbol': symbol, 'hpo_id': hp,
                        'edge_type': 'GENE_ASSOCIATED_WITH_PHENOTYPE'})
        source_disease = first(row, ['diseaseid', 'databaseid'])
        gp_provenance.append({'gene_id': gid, 'hpo_id': hp, 'source_disease_id': canonical_disease(source_disease),
                              'raw_gene_symbol': source_symbol, 'provenance_available': bool(source_disease)})
        name = first(row, ['hponame', 'hpotermname'])
        if name:
            hp_names[hp] = name
    gp = pd.DataFrame(gp_rows, columns=['gene_id', 'gene_symbol', 'hpo_id', 'edge_type']).drop_duplicates(['gene_id', 'hpo_id'])
    if gp.empty:
        raise ValueError('No gene-phenotype edges survived mapping and cohort filtering. Inspect raw HPO header.')
    nodes_gene = pd.DataFrame([{'gene_id': gid, 'gene_symbol': gene_map[gid]['symbol'],
                               'hgnc_id': gene_map[gid]['hgnc_id'], 'source': 'HGNC_mapped_Orphanet'} for gid in sorted(used_genes)])
    nodes_hp = pd.DataFrame([{'phenotype_id': hp, 'phenotype_name': hp_names.get(hp, '')} for hp in sorted(used_hp)])
    checks = nodes_gene[['gene_id', 'gene_symbol']].merge(gp[['gene_id', 'gene_symbol']].drop_duplicates(), on='gene_id', suffixes=('_node', '_gp'))
    assert checks.gene_symbol_node.eq(checks.gene_symbol_gp).all()
    assert nodes_gene.gene_id.str.match(r'^NCBIGene:\d+$').all()
    assert set(gp.gene_id) <= set(nodes_gene.gene_id)
    assert set(dg_unique.disease_id) <= set(selected.disease_id)

    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / 'nodes_disease.csv', index=False)
    nodes_gene.to_csv(output / 'nodes_gene.csv', index=False)
    nodes_hp.to_csv(output / 'nodes_phenotype.csv', index=False)
    dg_unique.to_csv(output / 'edges_disease_gene.csv', index=False)
    dg.to_csv(output / 'disease_gene_association_evidence.csv', index=False)
    dp.to_csv(output / 'edges_disease_phenotype.csv', index=False)
    gp.to_csv(output / 'edges_gene_phenotype.csv', index=False)
    all_diseases.to_csv(output / 'disease_selection_audit.csv', index=False)
    save(rejects, ['disease_id', 'raw_gene_symbol', 'hgnc_id', 'reason'], output / 'unmapped_orphanet_records.csv')
    save(excluded_gp, ['gene_id', 'raw_gene_symbol', 'reason'], output / 'unmapped_hpo_records.csv', ['gene_id', 'raw_gene_symbol', 'reason'])
    save(discrepancies, ['source', 'raw_gene_symbol', 'approved_gene_symbol', 'gene_id'], output / 'symbol_changes.csv', ['source', 'raw_gene_symbol', 'approved_gene_symbol', 'gene_id'])
    save(gp_provenance, ['gene_id', 'hpo_id', 'source_disease_id', 'raw_gene_symbol', 'provenance_available'], output / 'gene_phenotype_provenance.csv')
    stats = {'source_diseases': len(all_diseases), 'selected_diseases': len(selected),
             'genes': len(nodes_gene), 'phenotypes': len(nodes_hp), 'disease_gene_edges': len(dg_unique),
             'disease_phenotype_edges': len(dp), 'gene_phenotype_edges': len(gp),
             'genes_with_phenotype_edges': gp.gene_id.nunique(), 'unmapped_orphanet_association_rows': len(rejects),
             'raw_hpo_gene_annotation_rows': raw_gp_count,
             'average_genes_per_selected_disease': float(selected.gene_count.mean()),
             'average_phenotypes_per_selected_disease': float(selected.phenotype_count.mean())}
    pd.DataFrame([stats]).to_csv(output / 'graph_statistics.csv', index=False)
    dg.groupby(['association_type', 'association_status'], dropna=False).size().reset_index(name='evidence_rows').to_csv(output / 'association_types.csv', index=False)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'canonical_gene_namespace': 'NCBIGene',
                'mapping': 'unambiguous approved HGNC IDs to NCBI; no symbol-only fallback',
                'min_genes': min_genes, 'min_phenotypes': min_phenotypes,
                'disease_phenotype_mapping': 'direct ORPHA/Orphanet IDs only',
                'association_filter': 'all mapped association types retained; NOT all are necessarily causal',
                'hierarchy': 'not built', 'splits': 'not built',
                'provenance_audit': 'gene-phenotype annotations may be disease-derived; needs leakage review before training',
                'hpo_disease_source_counts': source_counts,
                'files': {name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in files.items()}}
    (output / 'build_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\n===== CORRECTED GRAPH STATISTICS =====')
    print(pd.Series(stats).to_string())
    print('\n===== ASSOCIATION TYPES (review before claiming causal genes) =====')
    print(pd.read_csv(output / 'association_types.csv').to_string(index=False))
    print('\nPASS: all retained gene IDs use NCBIGene:<ID>.')
    print('PASS: gene-phenotype symbols match canonical gene nodes.')
    print('Saved to:', output)
    print('STOP HERE: do not use old splits or train yet. Association types and phenotype provenance require review.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--min-genes', type=int, default=2)
    parser.add_argument('--min-phenotypes', type=int, default=5)
    args = parser.parse_args()
    if args.min_genes < 1 or args.min_phenotypes < 1:
        raise ValueError('Thresholds must be positive')
    build(args.root.resolve(), args.min_genes, args.min_phenotypes)