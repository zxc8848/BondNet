from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import json, csv

def split(idx):
    h=2166136261
    for b in f'{idx}|42'.encode(): h=((h^b)*16777619)&0xffffffff
    u=h/2**32
    return 'test' if u<.1 else 'val' if u<.2 else 'train'

def batches(path, n=1000):
    batch=[]; block=[]
    with open(path,'r') as f:
        for line in f:
            block.append(line)
            if line.strip()=='$$$$':
                batch.append(''.join(block));block=[]
                if len(batch)==n: yield batch;batch=[]
    if batch: yield batch

def work(batch):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    rows=[]
    for block in batch:
        s=Chem.SDMolSupplier();s.SetData(block,sanitize=False,removeHs=False,strictParsing=False)
        m=s[0]; idx=int(m.GetProp('geom_mol_idx'))
        Chem.SanitizeMol(m);Chem.SetAromaticity(m,Chem.AromaticityModel.AROMATICITY_MDL);Chem.SanitizeMol(m)
        mh=Chem.RemoveHs(m)
        rows.append((idx,split(idx),Chem.MolToSmiles(mh,isomericSmiles=False),Chem.MolToSmiles(mh,isomericSmiles=True)))
    return rows

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description='Build canonical identity map from GEOM source records.')
    ap.add_argument('--source',type=Path,default=Path('data/geom_drugs_all_random1_rel10.sdf'))
    ap.add_argument('--output',type=Path,default=Path('results/revision_v5_identity_overlap'))
    args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(work,batches(args.source)):
            rows.extend(batch)
            if len(rows)%50000==0: print('Scanned',len(rows),flush=True)
    result={'n':len(rows),'split_counts':dict(Counter(r[1] for r in rows))}
    for col,name in [(2,'nonisomeric_smiles'),(3,'isomeric_smiles')]:
        groups=defaultdict(list)
        for r in rows: groups[r[col]].append(r[:2])
        overlap=[]
        for smi,g in groups.items():
            if len(set(s for _,s in g))>1:
                overlap.append({'smiles':smi,'records':g})
        test_train=[idx for g in overlap if any(s=='train' for _,s in g['records']) for idx,s in g['records'] if s=='test']
        result[name]={'unique':len(groups),'cross_split_groups':len(overlap),'test_records_with_train_identity':len(test_train),'test_ids':test_train,'examples':overlap[:10]}
        (out/f'overlap_{name}.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
    (out/'identity_overlap_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (out/'geom_identity_map.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['geom_mol_idx','split','nonisomeric_smiles','isomeric_smiles']);w.writerows(rows)
    print(json.dumps({k:v if not isinstance(v,dict) or 'test_ids' not in v else {a:b for a,b in v.items() if a not in ['test_ids','examples']} for k,v in result.items()},indent=2))
