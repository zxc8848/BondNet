#!/usr/bin/env python3
"""Assemble release_v4/ (GitHub code snapshot + Zenodo archives) for the revision.

Nothing in the project is modified; files are copied or zipped into release_v4/.
release_v3/ is only read (unchanged GEOM archives and the checkpoint set are reused).

Output
  release_v4/github/   code snapshot to push (README.md is written separately)
  release_v4/zenodo/   checkpoints_v4.zip, external_pubchem3d_cohort_v4.zip, results_v4.zip,
                       geom_fixed_split_and_test_inputs.zip, geom_random1_re10_source_sdf.zip,
                       BondNet_code_v4.zip, README_zenodo.md (written separately), SHA256SUMS.txt
Progress is appended to release_v4/build.log.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "release_v3" / "zenodo"
OUT = ROOT / "release_v4"
GH = OUT / "github"
ZN = OUT / "zenodo"
LOG = OUT / "build.log"

CODE_DIRS = ["bondnet", "scripts", "tests"]
CODE_FILES = ["train.py", "train_stage2.py", "evaluate.py", "setup.py", "requirements.txt", "LICENSE", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

NEW_RESULT_DIRS = [
    "results/revision_v4_rebuilt_graph",
    "results/revision_v4_identity_v2",
    "results/revision_v4_openbabel_fixed",
    "results/revision_v4_openbabel_datadir_check",
    "results/revision_v4_heavy_hcount",
    "results/revision_v4_runtime",
]
EXTRA_RESULT_FILES = [
    "paper_revision/revision_v3_figure_values.csv",
    "paper_revision/revision_v4_experiment_log.md",
    "paper_revision/CHANGELOG.md",
]


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def skip(path: Path) -> bool:
    return any(p in SKIP_PARTS for p in path.parts) or path.suffix == ".pyc"


def method_for(name: str):
    return zipfile.ZIP_STORED if name.endswith((".pt", ".npz", ".zip", ".gz")) else zipfile.ZIP_DEFLATED


def add_tree(zf, rel_dir: str):
    base = ROOT / rel_dir
    if not base.exists():
        raise FileNotFoundError(base)
    n = 0
    for p in sorted(base.rglob("*")):
        if p.is_file() and not skip(p.relative_to(ROOT)):
            arc = p.relative_to(ROOT).as_posix()
            zf.write(p, arc, compress_type=method_for(arc))
            n += 1
    return n


def copy_zip_entries(src: Path, zf, rename=None, exclude=()):
    with zipfile.ZipFile(src) as zin:
        for info in zin.infolist():
            if info.filename in exclude:
                continue
            arc = rename(info.filename) if rename else info.filename
            zi = zipfile.ZipInfo(arc, info.date_time)
            zi.compress_type = method_for(arc)
            with zin.open(info) as fin, zf.open(zi, "w", force_zip64=True) as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)


def step_github():
    if GH.exists():
        log("[SKIP] github/ exists")
        return
    GH.mkdir(parents=True)
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not skip(p.relative_to(ROOT)):
                dst = GH / p.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    for f in CODE_FILES:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, GH / f)
    log(f"github/: {sum(1 for p in GH.rglob('*') if p.is_file())} files")


def step_checkpoints():
    out = ZN / "checkpoints_v4.zip"
    if out.exists():
        log("[SKIP] checkpoints_v4.zip")
        return
    tmp = out.with_suffix(".part")
    src_paths = ""
    with zipfile.ZipFile(V3 / "checkpoints_v3.zip") as zin:
        src_paths = zin.read("checkpoints/SOURCE_PATHS.txt").decode("utf-8")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        copy_zip_entries(V3 / "checkpoints_v3.zip", zf, exclude={"checkpoints/SOURCE_PATHS.txt"})
        extra = ["", "# Oracle hydrogen-count control (post-hoc, revision round 2)"]
        for s in (42, 43, 44):
            srcdir = ROOT / f"checkpoints/revision_v4_heavy_hcount_oracle_seed{s}"
            for f in ("best_e2e.pt", "train.log"):
                arc = f"checkpoints/oracle_hcount_heavy/seed{s}/{f}"
                zf.write(srcdir / f, arc, compress_type=method_for(arc))
            extra.append(f"checkpoints/oracle_hcount_heavy/seed{s}/  <-  checkpoints/revision_v4_heavy_hcount_oracle_seed{s}/")
        zf.writestr("checkpoints/SOURCE_PATHS.txt", src_paths.rstrip("\n") + "\n" + "\n".join(extra) + "\n",
                    compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(out)
    log(f"checkpoints_v4.zip {out.stat().st_size/1e6:.0f} MB")


def step_external():
    out = ZN / "external_pubchem3d_cohort_v4.zip"
    if out.exists():
        log("[SKIP] external_pubchem3d_cohort_v4.zip")
        return
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        copy_zip_entries(V3 / "external_pubchem3d_cohort_v3.zip", zf)
        zf.write(ROOT / "data/external_v3/DEVIATION_002.md", "data/external_v3/DEVIATION_002.md")
    tmp.replace(out)
    log(f"external_pubchem3d_cohort_v4.zip {out.stat().st_size/1e6:.0f} MB")


def step_results():
    out = ZN / "results_v4.zip"
    if out.exists():
        log("[SKIP] results_v4.zip")
        return
    with zipfile.ZipFile(V3 / "results_v3.zip") as zin:
        v3_dirs = sorted({"/".join(n.split("/")[:2]) for n in zin.namelist() if n.startswith("results/")})
    dirs = sorted(set(v3_dirs) | set(NEW_RESULT_DIRS))
    tmp = out.with_suffix(".part")
    total = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for d in dirs:
            p = ROOT / d
            if p.is_file():
                zf.write(p, d); total += 1
            elif p.is_dir():
                total += add_tree(zf, d)
            else:
                log(f"[WARN] missing {d}")
        for f in EXTRA_RESULT_FILES:
            if (ROOT / f).exists():
                zf.write(ROOT / f, f); total += 1
    tmp.replace(out)
    log(f"results_v4.zip {total} files, {out.stat().st_size/1e6:.0f} MB")


def step_geom():
    for name in ("geom_fixed_split_and_test_inputs.zip", "geom_random1_re10_source_sdf.zip"):
        dst = ZN / name
        if dst.exists():
            log(f"[SKIP] {name}")
            continue
        shutil.copy2(V3 / name, dst.with_suffix(".part"))
        dst.with_suffix(".part").replace(dst)
        log(f"{name} copied (unchanged from v3)")


def step_code():
    out = ZN / "BondNet_code_v4.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(GH.rglob("*")):
            if p.is_file():
                zf.write(p, "BondNet/" + p.relative_to(GH).as_posix())
    log(f"BondNet_code_v4.zip {out.stat().st_size/1e6:.1f} MB")


def step_sums():
    lines = []
    for p in sorted(ZN.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt" and not p.name.endswith(".part"):
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {p.name}")
    (ZN / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("SHA256SUMS.txt written")


STEPS = {"github": step_github, "checkpoints": step_checkpoints, "external": step_external,
         "results": step_results, "geom": step_geom, "code": step_code, "sums": step_sums}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ZN.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or list(STEPS)
    for name in wanted:
        log(f"[RUN] {name}")
        STEPS[name]()
    log("[DONE] " + " ".join(wanted))
