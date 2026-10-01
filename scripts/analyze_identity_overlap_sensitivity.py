"""Post-hoc GEOM identity exclusions. Original splits, weights and primary results remain unchanged."""
from pathlib import Path
import argparse,csv,json,hashlib,sys,statistics
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import p0c_yuelbond_compare as sc
from scripts.audit_rdkit_configurations import predict,quiet_native_stdout
OUT=ROOT/'results/revision_v5_identity_overlap'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def f1(tp,fp,fn):
 d=2*tp+fp+fn
 return np.divide(2*tp,d,out=np.zeros_like(d,dtype=float),where=d>0).mean(axis=-1)
def stat(true,pred,returned=True):
 tp=np.zeros(4,dtype=np.int64);fp=tp.copy();fn=tp.copy()
 for pair,y in true.items():
  if pred.get(pair)==y: tp[y]+=1
  else: fn[y]+=1
 for pair,y in pred.items():
  if true.get(pair)!=y: fp[y]+=1
 return tp,fp,fn,int(returned and true==pred)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--identity-map',type=Path,default=ROOT/'tmp/final_submission_audit/geom_identity_map.csv')
 ap.add_argument('--bootstrap',type=int,default=2000)
 a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 from rdkit import RDLogger
 RDLogger.DisableLog('rdApp.*')
 rows=list(csv.DictReader(a.identity_map.open(encoding='utf-8-sig')))
 assert len(rows)==269739 and len({r['geom_mol_idx'] for r in rows})==len(rows)
 from bondnet.data.dataset import _assign_split_label
 assert all(r['split']==_assign_split_label({'geom_mol_idx':int(r['geom_mol_idx'])},0) for r in rows)
 test={int(r['geom_mol_idx']):r for r in rows if r['split']=='test'}
 excludes={'all':set()};info={};matches=[]
 for field in ['isomeric_smiles','nonisomeric_smiles']:
  for scope in ['train','train_val']:
   pools=defaultdict(list)
   for r in rows:
    if r['split']=='train' or scope=='train_val' and r['split']=='val': pools[r[field]].append(r)
   key=field+'_'+scope
   excludes[key]={i for i,r in test.items() if r[field] in pools}
   info[key]={'excluded':len(excludes[key]),'retained':len(test)-len(excludes[key])}
   for i in sorted(excludes[key]):
    for match in pools[test[i][field]]:
     matches.append([key,i,match['geom_mol_idx'],match['split'],test[i][field]])
 assert [info[k]['excluded'] for k in list(info)]==[113,131,382,439],info
 with (OUT/'identity_matches.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['exclusion','test_geom_mol_idx','matched_geom_mol_idx','matched_split','canonical_smiles']);w.writerows(matches)
 write(OUT/'exclusions.json',{k:sorted(v) for k,v in excludes.items()})
 write(OUT/'protocol.json',{'status':'post-hoc descriptive sensitivity; no new independent test set',
 'identity_map_sha256':sha(a.identity_map),'source_sdf_sha256':sha(ROOT/'data/geom_drugs_all_random1_rel10.sdf'),
 'rdkit_version':__import__('rdkit').__version__,'subsets':info,'bootstrap_replicates':a.bootstrap,
 'bootstrap_seed':20261001,'bootstrap_comparison':'joint minus heavy, paired molecules, fixed fitted seeds',
 'rule_hueckel':'Full-cohort archived sufficient counts minus rerun counts for excluded records; frozen configuration unchanged.'})
 # Publish the map so sensitivity analysis is reproducible without scratch files.
 import shutil
 if a.identity_map.resolve() != (OUT/'geom_identity_map.csv').resolve():
  shutil.copy2(a.identity_map,OUT/'geom_identity_map.csv')
 summary=[];perseed=[];intervals=[]
 for tag in ['000','010','020']:
  sigma=int(tag)/100; data={}
  for model in ['joint','heavy','staged','oracle']:
   for seed in [42,43,44]:
    p=ROOT/(f'results/revision_v4_heavy_hcount/geom/seed{seed}/sigma_{tag}.npz' if model=='oracle' else f'results/revision_v3_undirected_audit/{model}/seed{seed}/sigma_{tag}.npz')
    z=np.load(p);data[model,seed]={k:z[k] for k in ['mol_idx','u_pipe_tp','u_pipe_fp','u_pipe_fn','u_hh_graph_exact']}
  ids=data['joint',42]['mol_idx'];assert set(ids)==set(test)
  assert all(np.array_equal(v['mol_idx'],ids) for v in data.values())
  sdf=ROOT/f'data/robustness_27240_keyed/sigma_{tag}.sdf'
  cache=OUT/f'rules_{tag}.npz'
  if not cache.exists():
   print('Loading rule references',tag,flush=True)
   mols=sc._read_valid_molecules(str(sdf))
   assert [int(m.GetProp('geom_mol_idx')) for m in mols]==ids.tolist()
   rules={}
   for method,path in [
     ('rdkit',ROOT/f'results/revision_v2_rule_baselines/predictions/rdkit_sigma_{tag}_preds.json'),
     ('openbabel',ROOT/f'results/revision_v4_openbabel_fixed/geom/predictions/openbabel_sigma_{tag}_preds.json')]:
    pred=json.loads(path.read_text());meta=pred.pop('__meta__')
    assert meta['sdf_sha256']==sha(sdf) and meta['n_molecules']==len(ids)
    vals=[]
    for pos,m in enumerate(mols):
     pp={tuple(sorted((int(i),int(j)))):sc._pred_label(o) for i,j,o in pred.get(str(pos),[])}
     vals.append(stat(sc._true_bonds(m),pp,str(pos) in pred))
    for j,k in enumerate(['tp','fp','fn','exact']):rules[method+'_'+k]=np.array([v[j] for v in vals])
   # Only excluded records need Hückel re-inference; all other records use archived totals.
   needed=excludes['nonisomeric_smiles_train_val'];vals=[]
   with quiet_native_stdout():
    for pos,m in enumerate(mols):
     if int(ids[pos]) not in needed:continue
     try: pp=predict(m,{'useHueckel':True});ok=True
     except Exception:pp={};ok=False
     vals.append((pos,*stat(sc._true_bonds(m),pp,ok)))
   rules['hueckel_positions']=np.array([v[0] for v in vals])
   for j,k in enumerate(['tp','fp','fn','exact']):rules['hueckel_'+k]=np.array([v[j+1] for v in vals])
   np.savez_compressed(cache,**rules)
   print('Rule count cache complete',tag,flush=True)
  rules=np.load(cache)
  hp=ROOT/('results/revision_v4_rdkit_configs_f1/geom_test_sigma_020_hueckel.json' if tag=='020' else f'results/revision_v4_rdkit_configs_f1/geom_test_sigma_{tag}.json')
  hrow=next(r for r in json.loads(hp.read_text())['rows'] if r['configuration']=='use_hueckel')
  for subset,exc in excludes.items():
   keep=np.array([int(i) not in exc for i in ids]); n=int(keep.sum())
   for model in ['joint','heavy','staged','oracle']:
    vals=[]
    for seed in [42,43,44]:
     d=data[model,seed];v=float(f1(*(d['u_pipe_'+k][keep].sum(0) for k in ['tp','fp','fn'])));e=float(d['u_hh_graph_exact'][keep].mean()*100)
     vals.append((v,e));perseed.append(dict(subset=subset,model=model,seed=seed,sigma=sigma,n=n,f1=v,hh_exact=e))
    summary.append(dict(subset=subset,model=model,sigma=sigma,n=n,f1=statistics.mean(v[0] for v in vals),f1_sd=statistics.stdev(v[0] for v in vals),hh_exact=statistics.mean(v[1] for v in vals),hh_sd=statistics.stdev(v[1] for v in vals)))
   for method in ['rdkit','openbabel','hueckel']:
    if method=='hueckel':
     drop=np.array([int(ids[pos]) in exc for pos in rules['hueckel_positions']])
     counts=[np.array(hrow['pipeline_'+k])-rules['hueckel_'+k][drop].sum(0) for k in ['tp','fp','fn']]
     exact=int(hrow['counts']['exact'])-int(rules['hueckel_exact'][drop].sum())
    else:
     counts=[rules[method+'_'+k][keep].sum(0) for k in ['tp','fp','fn']]
     exact=int(rules[method+'_exact'][keep].sum())
    summary.append(dict(subset=subset,model=method,sigma=sigma,n=n,f1=float(f1(*counts)),f1_sd=None,hh_exact=exact/n*100,hh_sd=None))
   # Recompute conditional molecule-bootstrap intervals for the H comparison on every exclusion.
   if subset!='all' and a.bootstrap:
    rng=np.random.default_rng(20261001+int(tag))
    vfs=[];ves=[]
    packed={}
    for model in ['joint','heavy']:
     for seed in [42,43,44]:
      d=data[model,seed]
      packed[model,seed]=np.concatenate([d['u_pipe_'+k][keep] for k in ['tp','fp','fn']]+[d['u_hh_graph_exact'][keep,None]],axis=1)
    for start in range(0,a.bootstrap,10):
     ix=rng.integers(0,n,size=(min(10,a.bootstrap-start),n),dtype=np.int32)
     seedf=[];seede=[]
     for seed in [42,43,44]:
      js=packed['joint',seed][ix].sum(1);hs=packed['heavy',seed][ix].sum(1)
      seedf.append(f1(js[:,:4],js[:,4:8],js[:,8:12])-f1(hs[:,:4],hs[:,4:8],hs[:,8:12]))
      seede.append((js[:,12]-hs[:,12])/n*100)
     vfs.extend(np.mean(seedf,axis=0));ves.extend(np.mean(seede,axis=0))
    pv=[next(r for r in perseed if r['subset']==subset and r['model']=='joint' and r['seed']==seed and r['sigma']==sigma) for seed in [42,43,44]]
    hv=[next(r for r in perseed if r['subset']==subset and r['model']=='heavy' and r['seed']==seed and r['sigma']==sigma) for seed in [42,43,44]]
    item={'subset':subset,'sigma':sigma,'n':n}
    for metric,boot in [('f1',vfs),('hh_exact',ves)]:
     delta=[p[metric]-h[metric] for p,h in zip(pv,hv)];mean=statistics.mean(delta);half=4.302652729911275*statistics.stdev(delta)/3**.5
     item[metric]={'delta':mean,'seed_t95':[mean-half,mean+half],'molecule_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    intervals.append(item)
    print('Bootstrap complete',tag,subset,flush=True)
  write(OUT/'summary.json',summary);write(OUT/'per_seed.json',perseed);write(OUT/'paired_intervals.json',intervals)
 with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print('DONE',OUT,flush=True)
if __name__=='__main__':main()
