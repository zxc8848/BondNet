"""Count molecules and sample the structure of drugs_crude.msgpack.tar.gz."""
import tarfile, msgpack, time, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'data/drugs_crude.msgpack.tar.gz'
t0 = time.time()
n_mols = 0
n_chunks = 0
sample_printed = False

with tarfile.open(path, 'r:gz') as tar:
    member = tar.getmembers()[0]
    print(f'Member: {member.name}  size={member.size/1e9:.1f} GB', flush=True)
    f = tar.extractfile(member)
    unpacker = msgpack.Unpacker(f, raw=False, strict_map_key=False, max_buffer_size=0)
    for chunk in unpacker:
        n_chunks += 1
        if isinstance(chunk, dict):
            n_mols += len(chunk)
            if not sample_printed:
                sample_printed = True
                smiles, mol_data = next(iter(chunk.items()))
                print(f'\nSample SMILES: {smiles}')
                print(f'Mol data keys: {list(mol_data.keys()) if isinstance(mol_data, dict) else type(mol_data)}')
                if isinstance(mol_data, dict):
                    confs = mol_data.get('conformers', [])
                    print(f'Num conformers: {len(confs)}')
                    if confs:
                        print(f'Conformer keys: {list(confs[0].keys())}')
                        atoms = confs[0].get('atoms', [])
                        if atoms:
                            print(f'Atom keys: {list(atoms[0].keys())}')
                print(flush=True)
        elif isinstance(chunk, list):
            n_mols += len(chunk)
        if n_mols % 50000 == 0 and n_mols > 0:
            print(f'  {n_mols} molecules... ({time.time()-t0:.0f}s)', flush=True)

print(f'\nTotal: {n_mols} molecules in {n_chunks} chunks  ({time.time()-t0:.1f}s)')
