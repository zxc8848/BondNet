"""
Extract SMILES from drugs_featurized.msgpack and generate ETKDG 3D conformers.

Uses multiprocessing for speed. ~50k mol/hour per core.

Usage:
    python scripts/geom_smiles_to_sdf.py \
        --input data/drugs_featurized.msgpack.tar.gz \
        --output data/geom_drugs.sdf \
        --max_mols 200000 \
        --workers 8
"""

import argparse, sys, time, tarfile, msgpack
from pathlib import Path
from multiprocessing import Pool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max_mols', type=int, default=200000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def smiles_to_molblock(args_tuple):
    smiles, seed = args_tuple
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def extract_smiles(input_path: str, max_mols: int):
    """Stream SMILES from msgpack without loading all into memory."""
    smiles_list = []
    with tarfile.open(input_path, 'r:gz') as tar:
        f = tar.extractfile(tar.getmembers()[0])
        unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False, max_buffer_size=0)
        for chunk in unpacker:
            if isinstance(chunk, dict):
                for smi in chunk.keys():
                    smiles_list.append(smi)
                    if len(smiles_list) >= max_mols:
                        return smiles_list
    return smiles_list


def main():
    args = parse_args()

    print(f"Extracting SMILES from {args.input} ...")
    t0 = time.time()
    smiles_list = extract_smiles(args.input, args.max_mols)
    print(f"  Got {len(smiles_list)} SMILES in {time.time()-t0:.1f}s")

    tasks = [(smi, args.seed + i) for i, smi in enumerate(smiles_list)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating 3D conformers with {args.workers} workers ...")
    t1 = time.time()
    n_written = n_failed = 0

    with open(str(out_path), 'w') as fout:
        with Pool(processes=args.workers) as pool:
            for i, molblock in enumerate(pool.imap(smiles_to_molblock, tasks, chunksize=50)):
                if i % 5000 == 0 and i > 0:
                    elapsed = time.time() - t1
                    rate = i / elapsed
                    eta = (len(tasks) - i) / max(rate, 1e-9)
                    print(f"  {i}/{len(tasks)}  written={n_written}  "
                          f"{rate:.0f} mol/s  ETA {eta:.0f}s")

                if molblock is None:
                    n_failed += 1
                    continue

                fout.write(molblock)
                fout.write('\n$$$$\n')
                n_written += 1

    elapsed = time.time() - t1
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone: {n_written} written, {n_failed} failed in {elapsed:.0f}s")
    print(f"  Rate: {n_written/elapsed:.0f} mol/s")
    print(f"  Output: {out_path} ({size_mb:.0f} MB)")


if __name__ == '__main__':
    main()
