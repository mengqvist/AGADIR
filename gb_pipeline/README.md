# GB pipeline

Everything that produces the Generalized Born (GB) tables read by the opt-in electrostatic law `dipole_law="gbx"`
(`pyagadir/electrostatics/gb.py`): the builder of the helix and coil structures and their GB energies, the transfer of
the Table 1 values, the build environment, and the intermediates with their checksums. The default point-charge law
(`pyagadir/electrostatics/point.py`) uses none of it.

`verify.sh` checks that this folder reproduces the package tables (section "Verification").

## Layout

| Path | Contents |
| --- | --- |
| `helix_builds/build_gb_dipole_table.py` | Builds ideal helices (L = 4-40, residue/acetyl/amide caps) and PPII/beta coil chains with every charged side-chain form, and computes their GB energies with the backbone (`--raw`); averages them into the dipole table (`--derive`). |
| `transfer/build_gb_transfer.py` | The Table 1 transfer offsets of gbx. It runs on the package itself (project environment). |
| `data/` | Raw and intermediate outputs (section "Data"). |
| `env/build.txt` | The exact package versions of the build environment. |
| `MANIFEST.sha256` | SHA-256 of the intermediates in `data/` that the package tables are derived from. |

## Environment

The builder needs PeptideBuilder and pdb2pqr (AMBER ff14SB charges and radii), which are not package dependencies, so it
has its own environment, created at the repository root:

    uv venv --python 3.11.2 .venv-build && uv pip install --python .venv-build/bin/python -r gb_pipeline/env/build.txt

The transfer script and `verify.sh`'s transfer step use the project environment (`uv run`). APBS was used only to
check the GB table, not to make it.

## How each table is made

Commands run from the repository root. `B` stands for `.venv-build/bin/python`; set `OMP_NUM_THREADS=1` for every `B`
step.

**`sidechain_dipole_gb.tsv`**

1. Structures and energies, about 6 h on 20 workers:
   `B gb_pipeline/helix_builds/build_gb_dipole_table.py --raw gb_pipeline/data/gb_raw_v2.tsv.gz --workers 20`
2. The table, 1 min (the strict-rule supplement is computed if missing, in minutes):
   `B gb_pipeline/helix_builds/build_gb_dipole_table.py --derive gb_pipeline/data/gb_raw_v2.tsv.gz --supplement gb_pipeline/data/gb_raw_v2_strict_supplement.tsv.gz --out pyagadir/params/sidechain_dipole_gb.tsv`

**`sidechain_dipole_gb_transfer.tsv`** (gbx): `uv run python gb_pipeline/transfer/build_gb_transfer.py`

### Parameter overrides

`gbx_overrides.tsv` was assembled by hand from refits of individual cells against peptide data (sources and values in
`pyagadir/params/README.md`, "gbx"). The refit runs used the development notes' runtime harness and are not part of
this folder.

## Data

`data/` holds the raw and intermediate outputs that the dipole table is derived from: `gb_raw_v2.tsv.gz` and its
strict-rule supplement. Git tracks the code and the environment file (`.gitignore`); the compressed tables stay out of
git. `MANIFEST.sha256` lists them with their checksums, so a copy restored from an archive can be checked
(`sha256sum -c MANIFEST.sha256`, run in this folder). For publication, deposit `data/` in a data archive and cite it
here.

## Verification

    BUILD_PY=.venv-build/bin/python gb_pipeline/verify.sh

It checks the intermediates against `MANIFEST.sha256`, re-derives `sidechain_dipole_gb` and
`sidechain_dipole_gb_transfer` from them, and compares each table's data rows with `pyagadir/params`. Header comments are
not compared, because some package headers were edited by hand. Both tables come out identical. The expensive step
(structures and their GB energies) is checked only through its output's checksum.

## Retired with gbp (2026-10-22)

The owner kept one GB law, gbx, and retired gbp. What made gbp's tables was removed from this folder and can be
recovered with `git show 5a6e3f8:<path>`. The intermediates that git did not track are still on disk, if this copy has
them: `data/gb_pair_raw_v3forms.*`, `data/n181_*`, the trajectories and checkpoints under `md/`.
- `helix_builds/build_gb_pair_table.py`: side chain pairs X(i), Y(i+k) on built helices and coils, every protonation
  form (`--raw --forms`), and the pair table on a (T, kappa) grid (`--derive-grid`);
  `sidechain_dipole_gb_neutral.tsv` came from `build_gb_dipole_table.py --derive --neutral`.
- `postprocessing/`: side chains threaded onto MD or ideal backbones, with their GB pair energies and desolvation as the
  builders compute them (`n181_coil_gb.py`, `n181_coil_run.py`, `n181_helix_run.py`); GB desolvation per position over
  nine structure orientations (`n181_dipole_desolv.py`, `n181_desolv_table.py`); a validation of the coil distances
  against Lacroix 1998 Table VI (`n181_coil_validate.py`).
- `md/`: the coil MD of Ac-A16-NH2, the coil reference of the pair table. Eight replicas of 50 ns in CHARMM36m with
  CHARMM TIP3P (OpenMM 8.6.1), with setup, runner, queue, frame extraction and status scripts; the full protocol is in
  `pyagadir/params/README.md` ("Retired: gbp"). Also `env/md.txt`, the MD environment.
- `transfer/build_gbp_transfer.py`: gbp's Table 1 transfer, a fixed point.
- Their sections of this file ("How each table is made", "Coil MD", "Data", "Verification") are in the same commit.

## Maintenance

When a GB table in `pyagadir/params` changes, add the scripts and intermediates that made it here and update this file,
`MANIFEST.sha256` and `verify.sh`. The copies here are the ones that made the shipped tables, with their paths made relative.
