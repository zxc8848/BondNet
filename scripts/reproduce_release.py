"""Verify and replay released inference checkpoints without altering the frozen manifest."""
from pathlib import Path,PurePosixPath
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,zipfile
ROOT=Path(__file__).resolve().parents[1]
INTEGRITY='release_integrity.json'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(root,rel):
 p=PurePosixPath(rel.replace('\\','/'))
 if p.is_absolute() or '..' in p.parts or ':' in rel:raise ValueError(f'Unsafe release path: {rel}')
 dest=(root/Path(*p.parts)).resolve()
 if not dest.is_relative_to(root.resolve()):raise ValueError('Path escapes release root')
 return dest
def checked(root,rel,expected):
 p=safe(root,rel)
 if not p.is_file() or digest(p)!=expected:raise ValueError(f'Integrity check failed: {rel}')
 return p
def tensor_hash(ck):
 import torch
 keys=sorted(k for k in ck if k.endswith('state_dict') and k not in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict'))
 if not keys:raise ValueError('No model state dictionary')
 h=hashlib.sha256()
 for wk in keys:
  h.update(wk.encode())
  for k in sorted(ck[wk]):
   t=ck[wk][k].detach().cpu().contiguous()
   h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
   h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def mappings(path):
 """SOURCE_PATHS permits individual files and directory-prefix mappings."""
 out={}
 for line in path.read_text(encoding='utf-8-sig').splitlines():
  if '<-' not in line:continue
  public,original=map(str.strip,line.split('<-',1))
  out[original]=public
 return out
def resolve_mapping(original,map_):
 if original in map_:return map_[original]
 for src,dst in map_.items():
  if src.endswith('/') and original.startswith(src):return dst+original[len(src):]
 raise ValueError(f'No released path for {original}')
def verify(root=ROOT):
 import torch
 proof=json.loads((root/INTEGRITY).read_text(encoding='utf-8'))
 manifest_path=checked(root,'data/external_v3/manifest.json',proof['frozen_manifest_sha256'])
 m=json.loads(manifest_path.read_text(encoding='utf-8'))
 recorded=(root/'data/external_v3/manifest.sha256').read_text().split()[0]
 if recorded!=proof['frozen_manifest_sha256']:raise ValueError('Frozen manifest digest record changed')
 for rel,h in proof['code_sha256'].items():checked(root,rel,h)
 for rel,h in m['frozen_files_sha256'].items():checked(root,rel,h)
 for rel,h in m['protocol_code_sha256'].items():
  frozen=checked(root,'frozen_protocol/'+rel,h)
  active=safe(root,rel)
  if active.read_bytes().replace(b'\r\n',b'\n')!=frozen.read_bytes().replace(b'\r\n',b'\n'):
   raise ValueError(f'Protocol differs beyond line endings: {rel}')
 checked(root,'checkpoints/CHECKPOINT_HASHES.tsv',proof['checkpoint_table_sha256'])
 checked(root,'checkpoints/SOURCE_PATHS.txt',proof['source_paths_sha256'])
 mp=mappings(root/'checkpoints/SOURCE_PATHS.txt')
 rows=list(csv.DictReader((root/'checkpoints/CHECKPOINT_HASHES.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
 by_public={r['archive_path']:r for r in rows}
 for r in rows:
  ckpath=checked(root,r['archive_path'],r['weights_only_sha256'])
  if ckpath.stat().st_size!=int(r['weights_only_bytes']):raise ValueError('Checkpoint size differs')
  ck=torch.load(ckpath,map_location='cpu',weights_only=False)
  if tensor_hash(ck)!=r['state_dict_tensor_sha256']:raise ValueError(f"Tensor mismatch: {ckpath}")
  if any(k in ck for k in ('optimizer_state_dict','scheduler_state_dict','scaler_state_dict')):raise ValueError('Unexpected optimizer state in inference archive')
  evidence=proof['checkpoint_original_evidence'][r['archive_path']]
  if evidence['original_sha256']!=r['original_sha256'] or evidence['tensor_sha256']!=r['state_dict_tensor_sha256']:
   raise ValueError('Original-to-release checkpoint evidence differs')
 for name,entry in m['evaluation_plan']['models'].items():
  for key in ('model_file','stage2_file'):
   if key not in entry:continue
   r=by_public[resolve_mapping(entry[key]['path'],mp)]
   if r['original_sha256']!=entry[key]['sha256']:raise ValueError(f'Frozen checkpoint link broken: {name}/{key}')
 print(f"[verify-release] OK: {len(rows)} inference checkpoints, {len(m['frozen_files_sha256'])} frozen data files, code and original tensor links",flush=True)
 return m,mp,proof
def install(args):
 proof=json.loads((ROOT/INTEGRITY).read_text())
 archives=['checkpoints_v4_weights.zip','external_pubchem3d_cohort_v4.zip']
 if args.with_results:archives+=['results_v4.zip']
 if args.with_geom:archives+=['geom_fixed_split_and_test_inputs.zip','geom_random1_re10_source_sdf.zip']
 for name in archives:
  archive=args.archives/name
  if digest(archive)!=proof['published_archives_sha256'][name]:raise ValueError(f'Archive checksum differs: {name}')
  with zipfile.ZipFile(archive) as z:
   for member in z.infolist():
    if member.is_dir():continue
    rel=member.filename
    # The published GEOM archives are rooted at geom/, unlike the experiment paths.
    if name.startswith('geom_'):
     if rel=='geom/geom_drugs_all_random1_rel10.sdf':rel='data/geom_drugs_all_random1_rel10.sdf'
     elif rel.startswith('geom/fixed_test_keyed_noise/'):rel='data/robustness_27240_keyed/'+rel.split('/')[-1]
     elif rel.startswith('geom/'):rel='data/'+rel[len('geom/'):]
    target=safe(ROOT,rel);target.parent.mkdir(parents=True,exist_ok=True)
    # Never overwrite a different local file.
    if target.exists():
     with z.open(member) as f:
      h=hashlib.sha256()
      for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
     if digest(target)!=h.hexdigest():raise ValueError(f'Existing file differs: {target}')
    else:
     with z.open(member) as fin,open(target,'wb') as fout:shutil.copyfileobj(fin,fout,1024*1024)
  print('[install]',name,flush=True)
 verify()
def run(args):
 m,mp,proof=verify()
 keys=list(m['evaluation_plan']['models']) if args.models==['all'] else args.models
 if any(k not in m['evaluation_plan']['models'] for k in keys):raise ValueError('Unknown model name')
 if args.max_mols is not None and args.max_mols<=0:raise ValueError('max-mols must be positive')
 dest=safe(ROOT,args.output)
 if dest.exists() and any(dest.iterdir()):raise ValueError('Output directory must be empty; use a new directory for each replay')
 dest.mkdir(parents=True,exist_ok=True)
 record={'kind':'post-publication replay; not the original frozen experiment',
 'frozen_manifest_sha256':proof['frozen_manifest_sha256'],'models':keys,'sigmas':args.sigmas,'max_mols':args.max_mols,
 'python':sys.executable,'device':args.device,'release_integrity_sha256':digest(ROOT/INTEGRITY)}
 (dest/'replay_protocol.json').write_text(json.dumps(record,indent=2))
 for name in keys:
  entry=m['evaluation_plan']['models'][name]
  cmd=[sys.executable,'scripts/export_per_molecule_stats.py','--checkpoint',resolve_mapping(entry['model_file']['path'],mp),
       '--cache',entry['cache'],'--output_dir',str(dest/name),'--noise_levels',*map(str,args.sigmas),
       '--eval_noise_seed','20260921','--batch_size',str(args.batch_size),'--cutoff','2.5','--h_cutoff','2.5',
       '--conn_threshold','0.5','--device',args.device]
  if 'stage2_file' in entry:cmd+=['--stage2_ckpt',resolve_mapping(entry['stage2_file']['path'],mp)]
  if args.max_mols is not None:cmd+=['--max_mols',str(args.max_mols)]
  subprocess.run(cmd,cwd=ROOT,check=True)
 (dest/'COMPLETED.json').write_text(json.dumps(record,indent=2))
def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
 p=sub.add_parser('install');p.add_argument('--archives',type=Path,required=True);p.add_argument('--with-geom',action='store_true');p.add_argument('--with-results',action='store_true')
 sub.add_parser('verify')
 p=sub.add_parser('evaluate');p.add_argument('--models',nargs='+',default=['all']);p.add_argument('--sigmas',nargs='+',type=float,choices=[0,.1,.2],default=[0,.1,.2]);p.add_argument('--max-mols',type=int)
 p.add_argument('--output',default='results/release_replay');p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=128)
 args=ap.parse_args()
 if args.command=='install':install(args)
 elif args.command=='verify':verify()
 else:run(args)
if __name__=='__main__':main()
