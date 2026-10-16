"""Build pyagadir/params/sidechain_dipole_gb.tsv: the Generalized Born field of the helix backbone on a charged side
chain, for the side chain-macrodipole law selected by dipole_law="gb".

Physics (continuum electrostatics, nothing fitted)
--------------------------------------------------
A charged residue X (Lys, Arg, His+, Asp, Glu, Cys-, Tyr-) sits at position k of an ideal poly-Ala helix with N-cap
(k = 0), L helical residues (k = 1..L) and C-cap (k = L + 1). Its energy with the backbone is the Generalized Born energy
(Still et al. 1990, J. Am. Chem. Soc. 112, 6127) between its charges and the backbone atoms N, H, CA, HA, C, O of
every residue from the N-cap to the C-cap except its own:

    E = sum 332.06 q_s q_a [ (1/r - 1/f) / eps_in  +  exp(-kappa f) / (eps_w f) ],
    f = sqrt(r^2 + R_s R_a exp(-r^2 / (4 R_s R_a))),

with salt as Srinivasan et al. (1999, Theor. Chem. Acc. 101, 426). The source charges are X's AMBER charges minus
Ala's backbone charges on X's backbone atoms (the charge the Ala -> X substitution adds). The Born radii R come from
each atom's burial in the built structure: the r^-6 volume integral (Grycuk 2003, J. Chem. Phys. 119, 4817) over
the solvent outside the solvent-excluded surface (1.4 A probe), on a 0.3 A grid. eps_in = 4. The same quantity in
the all-PPII and all-beta chain is the coil reference; it does not depend on position, and the table keeps its mean.

Inputs: PeptideBuilder ideal geometry (helix -60/-45; N-cap -167/147; C-cap -60/150; one extended residue before the
N-cap and one PPII residue after the C-cap, uncharged in the sum, as stand-ins for the caps' neighbours), AMBER charges
and radii from pdb2pqr; Cys- is AMBER's thiolate (CYM). AMBER has no tyrosinate: Tyr- is AMBER's Tyr without its
hydroxyl proton, the remaining -0.6008 spread over the ring and oxygen in the pattern of PARSE's tyrosinate (Sitkoff,
Sharp & Honig 1994; TY- minus TYR), scaled so that the residue carries exactly -1. The energy is averaged over the side
chain's chi1 rotamers (-60, 180, 60; other dihedrals 180, Arg's guanidinium 0/180, Tyr's ring chi2 90) that have no
heavy-atom contact closer than 2.8 A. Where all three clash (Asp, His+ and Tyr- at the N-cap), chi2 is tried at
90, -90, 60, -60 and 0 for each chi1 and the first clash-free one is used. If none is (Tyr- at the N-cap), the three
clashing structures are averaged and the row has n_rot = 0; the model does not use such rows.

Against Poisson-Boltzmann (APBS, same structures, same charges; Lys, Arg, His+, Asp, Glu): the first and last turns
within about 0.1-0.2 kcal/mol at 0.01 and 2.5 M; residues five to seven positions apart about 60 % of APBS; salt
screening of the far field stronger than APBS's. Cys- and Tyr- have not been compared with APBS.

Output: per (residue, state, L, k) the rotamer mean of
    E_in = sum 332.06 q_s q_a (1/r - 1/f) / eps_in            (kcal/mol; no salt or temperature dependence)
    W(kappa) = sum 332.06 q_s q_a exp(-kappa f) / f           (kcal/mol x eps_w; on a kappa grid in 1/A)
so that E(kappa, T) = E_in + W(kappa) / eps_w(T). Helix rows: L = 4..40, k = 0..L+1. Coil rows: state "coil".
The table holds the charged form's energy; the model multiplies it by the helix-state ionisation, as the default law
does with its point charge.

Requirements (not package dependencies): PeptideBuilder 1.1, biopython, pdb2pqr 3.7, numpy, scipy.
Run:  python tools/build_gb_dipole_table.py [--lmax 40] [--workers 16]   (about 80 minutes on 24 cores)
      --residues D H+ --ncap-only --merge   recomputes only those rows and merges them into the existing table.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import PeptideBuilder
from Bio.PDB import PDBIO
from PeptideBuilder import Geometry
from scipy.ndimage import distance_transform_edt

OUT = Path(__file__).resolve().parents[1] / "pyagadir/params/sidechain_dipole_gb.tsv"
PDB2PQR = str(Path(sys.executable).parent / "pdb2pqr")
BACKBONE = {"N", "H", "CA", "HA", "C", "O"}
PDIE, PROBE, RB, H_GRID, CLASH = 4.0, 1.4, 10.0, 0.3, 2.8
KAPPA = np.round(np.concatenate([np.arange(0.0, 0.2, 0.01), np.arange(0.2, 0.8001, 0.025)]), 4)  # 1/A
NCHI = {"K": 4, "E": 3, "D": 2, "R": 4, "H+": 2, "C": 1, "Y": 2}
RENAME = {"H+": ("HIS", "HIP"), "C": ("CYS", "CYM")}  # pdb2pqr residue names of the charged forms
CHI2_TRY = (90.0, -90.0, 60.0, -60.0, 0.0)
# Tyr-: AMBER Tyr minus HH, plus PARSE's deprotonation pattern (TY- minus TYR) scaled to make the residue -1 in total
_PARSE_DEPROT = {"CG": -0.070, "CD1": -0.070, "CE1": -0.070, "CZ": -0.125, "OH": -0.090, "CE2": -0.070, "CD2": -0.070}
CHI1 = (-60.0, 180.0, 60.0)
COIL_L, COIL_K = 16, range(4, 14)


def rotamer(aa, chi1, chi2=None):
    tail = [180.0] * (NCHI[aa] - 1)
    if aa == "Y":
        tail[0] = 90.0
    if chi2 is not None and tail:
        tail[0] = chi2
    return (chi1,) + tuple(tail) + ((0.0, 180.0) if aa == "R" else ())


def layout(L, aa, idx, conf):
    """Residue letters and (phi, psi). Index 0 = flank, 1 = N-cap, 2..L+1 = helix, L+2 = C-cap, L+3 = flank."""
    n = L + 4
    seq = ["A"] * n
    seq[idx] = aa
    if conf == "ppii":
        return seq, [(-75.0, 145.0)] * n
    if conf == "beta":
        return seq, [(-120.0, 120.0)] * n
    ang = [(-60.0, -45.0)] * n
    ang[0], ang[1], ang[L + 2], ang[n - 1] = (-120.0, 120.0), (-167.0, 147.0), (-60.0, 150.0), (-75.0, 145.0)
    return seq, ang


def build_pqr(seq, angles, idx, rot, aa, tmp):
    geos = []
    for r, letter in enumerate(seq):
        g = Geometry.geometry(letter)
        if r > 0:
            g.phi, g.psi_im1 = angles[r][0], angles[r - 1][1]
        if r == idx:
            g.inputRotamers(list(rot))
        geos.append(g)
    io = PDBIO()
    io.set_structure(PeptideBuilder.make_structure_from_geos(geos))
    pdb, pqr = os.path.join(tmp, "p.pdb"), os.path.join(tmp, "p.pqr")
    io.save(pdb)
    if aa in RENAME:  # the charged form of the source residue: HIP (both ring N protonated), CYM (thiolate)
        old, new = RENAME[aa]
        lines = open(pdb).read().splitlines(keepends=True)
        open(pdb, "w").write("".join(ln[:17] + new + ln[20:] if ln.startswith(("ATOM", "HETATM")) and ln[17:20] == old
                                     and int(ln[22:26]) == idx + 1 else ln for ln in lines))
    subprocess.run([PDB2PQR, "--ff", "AMBER", "--nodebump", "--noopt", "--keep-chain", "--whitespace", pdb, pqr],
                   check=True, capture_output=True)
    atoms = []
    for line in open(pqr):
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            atoms.append((f[2], f[3], int(f[-6]), np.array([float(t) for t in f[-5:-2]]), float(f[-2]), float(f[-1])))
    if aa == "Y":
        src = sorted({a[2] for a in atoms})[idx]
        hh = [a for a in atoms if a[2] == src and a[0] == "HH"]
        assert len(hh) == 1, "Tyr HH not found"
        scale = (1.0 - hh[0][4]) / -sum(_PARSE_DEPROT.values())
        atoms = [(a[0], a[1], a[2], a[3], a[4] + scale * _PARSE_DEPROT.get(a[0], 0.0) if a[2] == src else a[4], a[5])
                 for a in atoms if not (a[2] == src and a[0] == "HH")]
        assert abs(sum(a[4] for a in atoms if a[2] == src) - (-1.0)) < 0.01, "Tyr- net charge"
    return atoms


def min_contact(atoms, src):
    """Closest heavy-atom contact: pairs at least two residues apart (backbone O-N pairs excluded), plus X's side chain
    beyond C-beta against its neighbours."""
    heavy = [a for a in atoms if not a[0].startswith("H")]
    xyz = np.array([a[3] for a in heavy])
    res = np.array([a[2] for a in heavy])
    name = np.array([a[0] for a in heavy])
    sc = (res == src) & ~np.isin(name, list(BACKBONE | {"CB"}))
    gap = np.abs(res[:, None] - res[None, :])
    on = ((name[:, None] == "O") & (name[None, :] == "N")) | ((name[:, None] == "N") & (name[None, :] == "O"))
    ok = ((gap > 1) | ((gap == 1) & (sc[:, None] | sc[None, :]))) & ~on
    ok &= np.triu(np.ones_like(ok), 1).astype(bool)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return float(d[ok].min())


def born_radii(xyz, rad, need):
    pad = RB + 1.0
    lo = xyz.min(0) - pad
    n = np.ceil((xyz.max(0) + pad - lo) / H_GRID).astype(int) + 1
    sas = np.zeros(n, bool)
    for c, r in zip(xyz, rad):
        rr = r + PROBE
        i0 = np.maximum(np.floor((c - rr - lo) / H_GRID).astype(int), 0)
        i1 = np.minimum(np.ceil((c + rr - lo) / H_GRID).astype(int) + 1, n)
        gx, gy, gz = (lo[d] + H_GRID * np.arange(i0[d], i1[d]) for d in range(3))
        d2 = (gx[:, None, None] - c[0]) ** 2 + (gy[None, :, None] - c[1]) ** 2 + (gz[None, None, :] - c[2]) ** 2
        sas[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= d2 < rr * rr
    ses = sas & (distance_transform_edt(sas, sampling=H_GRID) > PROBE)  # solvent-excluded interior
    w = int(math.ceil(RB / H_GRID))
    solute = lo + H_GRID * np.argwhere(ses)
    out = {}
    for i in need:
        c = xyz[i]
        ic = np.round((c - lo) / H_GRID).astype(int)
        i0, i1 = np.maximum(ic - w, 0), np.minimum(ic + w + 1, n)
        sub = ses[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        gx, gy, gz = (lo[d] + H_GRID * np.arange(i0[d], i1[d]) - c[d] for d in range(3))
        d2 = gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2
        near = (~sub) & (d2 < RB * RB) & (d2 > 1e-9)
        s_in = float(np.sum(H_GRID ** 3 / d2[near] ** 3))
        dd = np.sum((solute - c) ** 2, axis=1)
        s_far = float(np.sum(H_GRID ** 3 / dd[dd >= RB * RB] ** 3))
        inv = 3.0 / (4.0 * math.pi) * (s_in + 4.0 * math.pi / (3.0 * RB ** 3) - s_far)
        out[i] = max(min(inv ** (-1.0 / 3.0), 50.0) if inv > 0 else 50.0, rad[i])
    return out


def energy_terms(atoms, L, idx):
    """(E_in, W on KAPPA) between X's source charges and the backbone of the N-cap..C-cap, X excluded."""
    resseqs = sorted({a[2] for a in atoms})
    src = resseqs[idx]
    ala = {}
    for a in atoms:
        if a[1] == "ALA" and a[2] != src and a[0] in BACKBONE:
            ala.setdefault(a[2], {})[a[0]] = a[4]
    full = sorted(r for r, d in ala.items() if set(d) == BACKBONE)
    ala_bb = ala[full[len(full) // 2]]
    xyz = np.array([a[3] for a in atoms])
    rad = np.array([a[5] for a in atoms])
    s_i, s_q = [], []
    for i, a in enumerate(atoms):
        if a[2] == src:
            q = a[4] - ala_bb[a[0]] if a[0] in BACKBONE else a[4]
            if q != 0.0:
                s_i.append(i)
                s_q.append(q)
    b_i = [i for i, a in enumerate(atoms) if a[0] in BACKBONE and resseqs[1] <= a[2] <= resseqs[L + 2] and a[2] != src]
    R = born_radii(xyz, rad, sorted(set(s_i) | set(b_i)))
    qs, qb = np.array(s_q), np.array([atoms[i][4] for i in b_i])
    r = np.linalg.norm(xyz[s_i][:, None, :] - xyz[b_i][None, :, :], axis=-1)
    rr = np.array([R[i] for i in s_i])[:, None] * np.array([R[i] for i in b_i])[None, :]
    f = np.sqrt(r * r + rr * np.exp(-r * r / (4.0 * rr)))
    qq = 332.06 * qs[:, None] * qb[None, :]
    e_in = float(np.sum(qq * (1.0 / r - 1.0 / f)) / PDIE)
    w = np.array([float(np.sum(qq * np.exp(-k * f) / f)) for k in KAPPA])
    return e_in, w


def task(job):
    aa, state, conf, L, k = job
    idx = 1 + k
    seq, ang = layout(L, "H" if aa == "H+" else aa, idx, conf)
    vals, clashed = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for chi1 in CHI1:
            atoms = build_pqr(seq, ang, idx, rotamer(aa, chi1), aa, tmp)
            src = sorted({a[2] for a in atoms})[idx]
            terms = energy_terms(atoms, L, idx)
            (clashed if min_contact(atoms, src) < CLASH else vals).append(terms)
        if not vals and NCHI[aa] > 1:  # no clash-free rotamer: try chi2 for each chi1, first clash-free one
            for chi1 in CHI1:
                for chi2 in CHI2_TRY:
                    atoms = build_pqr(seq, ang, idx, rotamer(aa, chi1, chi2), aa, tmp)
                    if min_contact(atoms, sorted({a[2] for a in atoms})[idx]) >= CLASH:
                        vals.append(energy_terms(atoms, L, idx))
                        break
    use = vals or clashed
    e_in = float(np.mean([t[0] for t in use]))
    w = np.mean([t[1] for t in use], axis=0)
    return aa, state, conf, L, k, len(vals), e_in, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, default=40)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--residues", nargs="+", default=list(NCHI), help="subset (with --merge)")
    ap.add_argument("--ncap-only", action="store_true", help="only the N-cap rows of --residues (with --merge)")
    ap.add_argument("--merge", action="store_true", help="replace these rows in the existing table, keep the rest")
    a = ap.parse_args()
    res = [x for x in NCHI if x in a.residues]
    jobs = [(aa, "helix", "helix", L, k) for aa in res for L in range(4, a.lmax + 1) for k in range(0, L + 2)
            if not a.ncap_only or k == 0]
    if not a.ncap_only:
        jobs += [(aa, "coil", conf, COIL_L, k) for aa in res for conf in ("ppii", "beta") for k in COIL_K]
    jobs.sort(key=lambda j: -j[3])  # longest first
    rows, coil = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (aa, state, conf, L, k, nrot, e_in, w) in enumerate(ex.map(task, jobs, chunksize=1)):
            if state == "helix":
                rows.append((aa, L, k, nrot, e_in, w))
            else:
                coil.setdefault(aa, []).append((e_in, w))
            if n % 500 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    head = ["aa", "state", "L", "k", "n_rot", "E_in"] + [f"W_{x:.4f}" for x in KAPPA]
    if a.merge:  # keep every existing row that was not recomputed
        done = {(r[0], r[1], r[2]) for r in rows}
        for line in open(a.out):
            if line.startswith("#") or line.startswith("aa\t"):
                continue
            f = line.rstrip("\n").split("\t")
            aa, state, L, k = f[0], f[1], int(f[2]), int(f[3])
            if state == "coil" and aa not in coil:
                coil[aa] = [(float(f[5]), np.array([float(x) for x in f[6:]]))] * int(f[4])
            elif state == "helix" and (aa, L, k) not in done:
                rows.append((aa, L, k, int(f[4]), float(f[5]), np.array([float(x) for x in f[6:]])))
    with open(a.out, "w") as fh:
        fh.write("# Generalized Born field of the helix backbone on a charged side chain (tools/build_gb_dipole_table.py).\n")
        fh.write("# E(kappa, T) = E_in + W(kappa) / eps_w(T), kcal/mol, for the charged form; W columns at kappa in 1/A.\n")
        fh.write("# helix rows: L helical residues, k = 0 (N-cap) .. L + 1 (C-cap); coil rows: all-PPII/beta mean.\n")
        fh.write("\t".join(head) + "\n")
        for aa in NCHI:
            ew = coil[aa]
            fh.write("\t".join([aa, "coil", "0", "0", str(len(ew)), f"{np.mean([t[0] for t in ew]):.5f}"]
                               + [f"{v:.5f}" for v in np.mean([t[1] for t in ew], axis=0)]) + "\n")
        for aa, L, k, nrot, e_in, w in sorted(rows, key=lambda r: (list(NCHI).index(r[0]), r[1], r[2])):
            fh.write("\t".join([aa, "helix", str(L), str(k), str(nrot), f"{e_in:.5f}"] + [f"{v:.5f}" for v in w]) + "\n")
    print("wrote", a.out, len(rows), "helix rows")


if __name__ == "__main__":
    main()
