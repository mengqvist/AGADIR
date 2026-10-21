# GB pipeline

Everything that produces the Generalized Born (GB) tables read by the opt-in electrostatic laws `dipole_law="gb"`,
`"gbx"` and `"gbp"` (`pyagadir/electrostatics/gb.py`): the builders of the helix and coil structures and their GB
energies, the coil MD of Ac-A16-NH2, the post-processing that turns MD frames and built helices into GB energies, the
transfer of the Table 1 values, the two software environments, and the intermediates with their checksums. The
default point-charge law (`pyagadir/electrostatics/point.py`) uses none of it.

`verify.sh` checks that this folder reproduces the package tables (section "Verification").

## Layout

| Path | Contents |
| --- | --- |
| `helix_builds/build_gb_dipole_table.py` | Builds ideal helices (L = 4-40, residue/acetyl/amide caps) and PPII/beta coil chains with every charged and neutral side-chain form, and computes their GB energies with the backbone (`--raw`); averages them into the dipole tables (`--derive`, `--derive --neutral`). |
| `helix_builds/build_gb_pair_table.py` | The same for two side chains X(i), Y(i+k) on built helices and coils, every protonation form (`--raw --forms`); the pair table on a (T, kappa) grid (`--derive-grid`). |
| `postprocessing/n181_coil_gb.py` | Threads side chains onto MD (or ideal) backbones and computes their GB pair energies and desolvation exactly as the builders do. Library for the three scripts below. |
| `postprocessing/n181_coil_run.py` | GB pair energies on the MD coil frames, one output per replica. |
| `postprocessing/n181_helix_run.py` | The helix rows of the pair table, computed with the same frame pipeline on built helices. |
| `postprocessing/n181_dipole_desolv.py`, `n181_desolv_table.py` | A side chain's GB desolvation, helix minus coil, per position; averaged over nine structure orientations into the desolvation table. |
| `postprocessing/n181_coil_validate.py` | Validation only: coil charge-charge distances of the MD ensemble against Lacroix 1998 Table VI. Produces no table. |
| `md/scripts/` | MD setup (`build.py`, `n181_prepare.py`), runner (`s1b_run.py`), GPU queue (`s1b_queue.py`), frame extraction (`n181_frames.py`), `benchmark.py`, `status.sh`. |
| `md/runs/n181/cA16_coil1..8/` | The eight coil replicas: `job.json` (every setting and seed), `start.pdb`, `peptide.pdb`, `system.xml.gz` (the full OpenMM system), checkpoints `state_*.xml`, trajectory chunks `traj/chunk_*.xtc` (peptide only, 20 ps), `progress.log`. |
| `md/results/` | `n181_build.txt` (build report), `n181_frames_cA16_coil1..8.npz` (the frames used). |
| `transfer/build_gb_transfer.py`, `build_gbp_transfer.py` | Table 1 transfer offsets for gbx and gbp. They run on the package itself (project environment). |
| `data/` | Raw and intermediate outputs (next section). |
| `env/build.txt`, `env/md.txt` | The exact package versions of the two environments. |
| `MANIFEST.sha256` | SHA-256 of every file in `data/` and `md/`. |

## Environments

The builders and post-processing need PeptideBuilder and pdb2pqr (AMBER ff14SB charges and radii); the MD needs OpenMM.
Neither is a package dependency, so each has its own environment, created at the repository root:

    uv venv --python 3.11.2 .venv-build && uv pip install --python .venv-build/bin/python -r gb_pipeline/env/build.txt
    uv venv --python 3.12.13 .venv-md && uv pip install --python .venv-md/bin/python -r gb_pipeline/env/md.txt

`env/md.txt` pins OpenMM 8.6.1 with its CUDA 12 platform, openmmforcefields 0.15.1 (CHARMM36m, `charmm36_2024.xml`),
mdtraj 1.11.2 and PeptideBuilder 1.1.0. The transfer scripts and `verify.sh`'s transfer steps use the project
environment (`uv run`). APBS was used only to check the GB tables, not to make them.

## How each table is made

Commands run from the repository root unless a step says otherwise. `B` stands for `.venv-build/bin/python`; set
`OMP_NUM_THREADS=1` for every `B` step. "Inferred" marks a flag that the run's log and outputs imply but no record states.

**`sidechain_dipole_gb.tsv`** (charged forms) and **`sidechain_dipole_gb_neutral.tsv`** (neutral forms)

1. Structures and energies, about 6 h on 20 workers:
   `B gb_pipeline/helix_builds/build_gb_dipole_table.py --raw gb_pipeline/data/gb_raw_v2.tsv.gz --workers 20`
2. The tables, 1 min each (the strict-rule supplement is computed if missing, in minutes):
   `B gb_pipeline/helix_builds/build_gb_dipole_table.py --derive gb_pipeline/data/gb_raw_v2.tsv.gz --supplement gb_pipeline/data/gb_raw_v2_strict_supplement.tsv.gz --out pyagadir/params/sidechain_dipole_gb.tsv`,
   and the same with `--neutral --out pyagadir/params/sidechain_dipole_gb_neutral.tsv`.

**`sidechain_pair_gb.tsv`** (gbp)

1. Pair rows on built structures, every protonation form (1,500 tasks; log `data/gb_pair_raw_v3forms.log`):
   `B gb_pipeline/helix_builds/build_gb_pair_table.py --raw gb_pipeline/data/gb_pair_raw_v3forms.tsv.gz --forms --workers 16`
2. Helix rows with the frame pipeline (900 tasks, chi1 wells only; log `data/n181_helix_chi1.log`):
   `B gb_pipeline/postprocessing/n181_helix_run.py gb_pipeline/data/n181_helix_chi1.npz --chi2-kmax 0` (`--chi2-kmax 0` inferred from the log).
3. Coil MD, eight replicas of Ac-A16-NH2 (next section).
4. Frames, from `gb_pipeline/md/scripts` with `.venv-md`: for N in 1..8,
   `python n181_frames.py cA16_coilN --every-ps 2000 --discard-ns 10 cA16_coilN` (frames every 2 ns after the first 10 ns).
5. GB on the frames, per replica:
   `B gb_pipeline/postprocessing/n181_coil_run.py gb_pipeline/md/results/n181_frames_cA16_coilN.npz gb_pipeline/data/n181_coil_cA16_coilN.npz --workers 6`
   (seed 181, the default). Run time was not recorded.
6. The table, minutes:
   `B gb_pipeline/helix_builds/build_gb_pair_table.py --derive-grid gb_pipeline/data/gb_pair_raw_v3forms.tsv.gz --helix-md gb_pipeline/data/n181_helix_chi1.npz --coil-md gb_pipeline/data/n181_coil_cA16_coil1.npz,...,gb_pipeline/data/n181_coil_cA16_coil8.npz --eps-in 16 --out pyagadir/params/sidechain_pair_gb.tsv`

**`sidechain_dipole_gb_desolv.tsv`** (gbp)

1. Nine orientations of the L = 24 helix and coil structures (inferred from the output names and logs): for k = 0..8,
   `N181_DESOLV_L=24 N181_DESOLV_TABLE=1 N181_DESOLV_ROT=k N181_DESOLV_OUT=n181_dipole_desolv_L24_rk.txt B gb_pipeline/postprocessing/n181_dipole_desolv.py`,
   where orientation 0 writes `n181_dipole_desolv_L24.txt` (no suffix). Outputs and logs are in `data/`. Run time was not recorded.
2. The average, seconds: `B gb_pipeline/postprocessing/n181_desolv_table.py` (interior from 5 positions, the default)
   writes `data/n181_dipole_desolv_avg.tsv`, whose data rows are the package table.

**`sidechain_dipole_gb_transfer.tsv`** (gbx): `uv run python gb_pipeline/transfer/build_gb_transfer.py`

**`sidechain_gbp_transfer.tsv`** (gbp): `uv run python gb_pipeline/transfer/build_gbp_transfer.py`. It starts from the
current file's values and iterates to convergence; it needs the four other gbp tables in place first.

### Parameter overrides

`gbx_overrides.tsv` and `gbp_overrides.tsv` were assembled by hand from refits of individual cells against peptide data
(their sources and values are in `pyagadir/params/README.md`, "gbx" and "gbp"). The refit runs used the development
notes' runtime harness and are not part of this folder.

## Coil MD

Ac-A16-NH2 in CHARMM36m with CHARMM TIP3P (OpenMM 8.6.1): acetyl and amide caps merged into residues 1 and 16; coil
starts drawn from PPII/beta (backbone span <= 5.0 nm, clash-free; the draws are in each `job.json`); rhombic
dodecahedron, d = 6.5 nm, no ions (neutral peptide), about 19,040 atoms; PME 1.2 nm (switch 1.0 nm), HBonds
constraints, hydrogen mass 3 amu, 4 fs, Langevin middle integrator 1/ps, 278 K, Monte Carlo barostat 1 bar after 1 ns
NVT. 50 ns per replica in 2 ns chunks; the first 10 ns are discarded. About 5 h per replica at 246 ns/day on an
RTX 2080 Ti.

From `gb_pipeline/md/scripts` with `.venv-md`:

    python n181_prepare.py -j 8        # writes ../runs/n181/<job>/, ../runs/n181/jobs.tsv, ../results/n181_build.txt
    python s1b_queue.py --root ../runs/n181 --jobs cA16_coil1,cA16_coil2,cA16_coil3,cA16_coil4,cA16_coil5,cA16_coil6,cA16_coil7,cA16_coil8 --gpus 0,1 --stop-at-uk 21:00
    ./status.sh                        # progress of every run

`n181_prepare.py` skips any job whose `job.json` exists, so move `runs/n181` aside to rebuild from scratch.
`s1b_queue.py` needs a stop time (`--stop-at-uk HH:MM`) or `--max-hours`; a job pauses at a chunk boundary
and the next call resumes it.

**Status:** replicas 1-6 are complete (50 ns). Replica 7 paused at 14 ns and replica 8 at 20 ns, so they give 2 and 5
of the 127 frames. Leaving them out changes the coil rows of the pair table by at most 0.002 kcal/mol. Both can be resumed with the queue
command above.

## Data

`data/` holds the raw and intermediate outputs that the tables are derived from: `gb_raw_v2.tsv.gz` and its strict-rule
supplement (dipole tables), `gb_pair_raw_v3forms.tsv.gz`, `n181_helix_chi1.npz` and `n181_coil_cA16_coil1..8.npz`
(pair table), and `n181_dipole_desolv_L24*.txt` (desolvation table), with their logs. Together with `md/` this is about
690 MB.

Git tracks the code, the environment files, the job records, the logs and the small text intermediates (`.gitignore`).
The compressed tables, `.npz` files, structures, checkpoints and trajectories stay out of git. `MANIFEST.sha256` lists
every file under `data/` and `md/` with its checksum, so a copy restored from an archive can be checked
(`sha256sum -c MANIFEST.sha256`, run in this folder). For publication, deposit `data/` and `md/` in a data archive and
cite it here.

## Verification

    BUILD_PY=.venv-build/bin/python gb_pipeline/verify.sh

It checks the intermediates against `MANIFEST.sha256`. It then re-derives `sidechain_dipole_gb`, `_neutral`,
`sidechain_pair_gb`, `sidechain_dipole_gb_desolv`, `sidechain_dipole_gb_transfer` and `sidechain_gbp_transfer` from
them and compares each table's data rows with `pyagadir/params`. Header comments are not compared, because some
package headers were edited by hand. The steps run in parallel and take under 2 minutes. The first five tables come
out identical. `sidechain_gbp_transfer` is a fixed point written to four decimals: a re-run from the package values
moves the His N-cap dphi by one unit in the last digit (-1.9420 against the package's -1.9421; the code as committed
does the same), so that table is compared within 1e-4. The expensive steps (structures, MD, GB on MD frames) are
checked only through their outputs' checksums.

## Maintenance

When a GB table in `pyagadir/params` changes, add the scripts and intermediates that made it here and update this file,
`MANIFEST.sha256` and `verify.sh`. Development copies of the post-processing and MD scripts also exist in the
development notes. The copies here are the ones that made the shipped tables, with their paths made relative.
