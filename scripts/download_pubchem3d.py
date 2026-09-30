"""
Download 3D conformers from PubChem REST API and save as a single SDF.

Sequential mode:   download CIDs start_cid, start_cid+1, ...
Random mode:       randomly sample CIDs from [cid_min, cid_max]

PubChem has ~110M CIDs; ~26M have 3D conformers (mostly CID < 50M).
In random mode, expect ~20-30% hit rate (many CIDs have no 3D conformer).

Usage:
    # Sequential
    python scripts/download_pubchem3d.py --output data/pubchem3d_100k.sdf --n_mols 100000

    # Random (recommended for diversity)
    python scripts/download_pubchem3d.py --output data/pubchem3d_100k.sdf --n_mols 100000 ^
        --random --cid_min 1 --cid_max 5000000 --seed 42
"""
import argparse
import random
import time
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--n_mols', type=int, default=100000)
    p.add_argument('--random', action='store_true',
                   help='Randomly sample CIDs instead of sequential')
    p.add_argument('--cid_min', type=int, default=1)
    p.add_argument('--cid_max', type=int, default=5000000,
                   help='Upper CID bound for random sampling. '
                        'CID 1-5M covers most drug-like organic compounds.')
    p.add_argument('--start_cid', type=int, default=1,
                   help='Starting CID for sequential mode')
    p.add_argument('--batch_size', type=int, default=100,
                   help='CIDs per request (PubChem max=100)')
    p.add_argument('--sleep', type=float, default=0.3,
                   help='Seconds between requests (rate limit)')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def fetch_batch(cids: list, session) -> str:
    url = (
        'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
        f'/compound/cid/{",".join(map(str, cids))}'
        '/record/SDF?record_type=3d'
    )
    resp = session.get(url, timeout=30)
    if resp.status_code == 200:
        return resp.text
    return ''


def count_mols(text: str) -> int:
    return text.count('$$$$')


def cid_generator_random(cid_min, cid_max, batch_size, seed):
    """Yield shuffled batches of CIDs from [cid_min, cid_max]."""
    rng = random.Random(seed)
    all_cids = list(range(cid_min, cid_max + 1))
    rng.shuffle(all_cids)
    for i in range(0, len(all_cids), batch_size):
        yield all_cids[i:i + batch_size]


def cid_generator_sequential(start_cid, batch_size):
    """Yield sequential batches of CIDs."""
    cid = start_cid
    while True:
        yield list(range(cid, cid + batch_size))
        cid += batch_size


def main():
    args = parse_args()
    try:
        import requests
    except ImportError:
        print('ERROR: pip install requests')
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers['User-Agent'] = 'BondNet-research/1.0 (academic)'

    if args.random:
        gen = cid_generator_random(args.cid_min, args.cid_max, args.batch_size, args.seed)
        print(f'Random mode: sampling CIDs from [{args.cid_min}, {args.cid_max}]')
        print(f'Expected hit rate ~25%, need ~{args.n_mols * 4} CID attempts')
    else:
        gen = cid_generator_sequential(args.start_cid, args.batch_size)
        print(f'Sequential mode: starting from CID {args.start_cid}')

    n_written = 0
    n_attempts = 0
    t0 = time.time()
    print(f'Target: {args.n_mols} molecules → {args.output}')

    with open(out_path, 'w') as fout:
        for batch in gen:
            if n_written >= args.n_mols:
                break

            n_attempts += len(batch)
            try:
                sdf_text = fetch_batch(batch, session)
            except Exception as e:
                print(f'  Request error: {e}')
                time.sleep(args.sleep * 5)
                continue

            if sdf_text:
                fout.write(sdf_text)
                n_written += count_mols(sdf_text)

            if n_written > 0 and n_written % 5000 < args.batch_size:
                elapsed = time.time() - t0
                hit_rate = n_written / max(n_attempts, 1) * 100
                rate = n_written / max(elapsed, 1)
                eta = (args.n_mols - n_written) / max(rate, 1e-9)
                print(f'  {n_written}/{args.n_mols} | '
                      f'hit_rate={hit_rate:.0f}% | '
                      f'{rate:.0f} mol/s | ETA {eta:.0f}s')

            time.sleep(args.sleep)

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    hit_rate = n_written / max(n_attempts, 1) * 100
    print(f'\nDone: {n_written} molecules, hit_rate={hit_rate:.1f}%, '
          f'{elapsed:.0f}s → {out_path} ({size_mb:.0f} MB)')


if __name__ == '__main__':
    main()
