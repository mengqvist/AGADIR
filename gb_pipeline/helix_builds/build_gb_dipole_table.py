"""Build pyagadir/params/sidechain_dipole_gb.tsv: the Generalized Born field of the helix backbone on a charged side
chain, on which the side chain-macrodipole laws dipole_law="gbx" and "gbp" are built (the field alone was the former
law "gb", removed on 2026-10-22: it counts first-turn and capping electrostatics twice without the Table 1 transfer).

Stages:
  --raw OUT.tsv.gz   the expensive part: build every structure, compute its energies, write one row per structure
                     (about six hours on 20 workers of a 12-core machine; all forms, charged and neutral)
  --derive RAW --supplement SUP [--rule strict|hbond]
                     the averaging rule: read the raw rows and write the package table (seconds). With the default
                     rule "strict", SUP holds the strict-rule members of the few wells (Asp only) whose raw member is
                     allowed only through the hydrogen-bond allowance; it is computed (minutes) if missing. Changing
                     the rule or the weights needs only this stage.

Physics (continuum electrostatics, nothing fitted)
--------------------------------------------------
A residue X sits at position k of an ideal poly-Ala helix with N-cap (k = 0), L helical residues (k = 1..L) and C-cap
(k = L + 1). Its energy with the backbone is the Generalized Born energy (Still et al. 1990, J. Am. Chem. Soc. 112,
6127) between its charges and the backbone atoms N, H, CA, HA, C, O of every residue from the N-cap to the C-cap except
its own (and every atom of an acetyl or amide cap):

    E = sum 332.06 q_s q_a [ (1/r - 1/f) / eps_in  +  exp(-kappa f) / (eps_w f) ],
    f = sqrt(r^2 + R_s R_a exp(-r^2 / (4 R_s R_a))),

with salt as Srinivasan et al. (1999, Theor. Chem. Acc. 101, 426) and eps_in = 4. The source charges are X's charges
minus Ala's backbone charges on X's backbone atoms. The Born radii R come from each atom's burial in the built
structure: the r^-6 volume integral (Grycuk 2003, J. Chem. Phys. 119, 4817) over the solvent outside the
solvent-excluded surface (1.4 A probe), on a 0.3 A grid. The coil reference is the same quantity in all-PPII and
all-beta chains, over the same residue range with the same caps.

Structures: PeptideBuilder ideal geometry (helix -60/-45; N-cap -167/147; C-cap -60/150; an extended residue before the
N-cap and a PPII residue after the C-cap, outside the sum). Caps: "res" = an Ala N-cap or C-cap residue as above; "Ac" =
the N-cap replaced by an acetyl group (and no residue before it); "Am" = the C-cap replaced by an amide NH2 (and no
residue after it). AMBER ff14SB charges and radii from pdb2pqr; the acetyl (ACE) and amide (NHE) caps take AMBER's
library values. Forms: Lys (LYS, LYN), Arg (ARG), His (HIP, HID, HIE), Asp (ASP, ASH), Glu (GLU, GLH), Cys (CYM, CYS),
Tyr (Tyr-, TYR). AMBER has no tyrosinate: Tyr- is AMBER's Tyr without its hydroxyl proton, the remaining -0.6008 spread
over the ring and oxygen in the pattern of PARSE's tyrosinate (Sitkoff, Sharp & Honig 1994; TY- minus TYR), so that the
residue carries exactly -1. The forms of one residue share each structure (heavy atoms) and its Born radii, computed
on the union of their atoms.

Rotamers: each chi1 well (-60, 180, 60) is represented by its nearest sterically allowed member. The canonical member
is chi1 = the well; Asp chi2 180; His chi2 180; Tyr chi2 90; Glu chi2 180, chi3 180; Lys and Arg chi2-chi4 180 (Arg's
guanidinium 0/180); Cys chi1 only. If it clashes, members are tried in order of distance from the canonical one: chi1
+-10, +-20; the scanned dihedral (Asp chi2, His chi2, Tyr chi2, Glu chi3 in 30-degree steps; Lys and Arg chi2 at
+-60) at cost 1 per 30 degrees against 1 per 10 degrees of chi1. The raw stage allows contacts down to 2.8 A, or
2.5 A between a donor nitrogen (backbone N, Lys NZ, Arg NE/NH1/NH2, His ND1/NE2) and an oxygen (a hydrogen bond); the
strict rule (the package default) takes 2.8 A for every pair. The energy holds the side chain's interaction with the
backbone but not its own desolvation, which a hydrogen-bonded contact pays; equal weights would overstate such
members, hence the strict default.
Pairs within one residue, backbone O-N pairs, and pairs of adjacent residues other than X's side chain beyond C-beta
are not checked. A well with no allowed member keeps its canonical structure, flagged by a negative margin.

Raw output: one row per (form, state, conf, ncap, ccap, L, k, well) with the angles used, the steric margin (smallest
distance minus its threshold), the old single-threshold contact, and
    E_in = sum 332.06 q_s q_a (1/r - 1/f) / eps_in            (kcal/mol; no salt or temperature dependence)
    W(kappa) = sum 332.06 q_s q_a exp(-kappa f) / f           (kcal/mol x eps_w; on a kappa grid in 1/A)
so that E(kappa, T) = E_in + W(kappa) / eps_w(T).

Package table (--derive): charged forms only (K, R, H+, D, E, C-, Y-). Per row, the mean over the wells whose member
is allowed under the rule (equal weights; n_rot = their number; with none, the least-clashing well and n_rot = 0, which the model
does not use). Helix rows: (res, res) for L = 4..40, k = 0..L+1; (Ac, res) for k = 1..8 from the acetyl,
(res, Am) for the last 8 positions before the amide, (Ac, Am) for L = 4..16, all k. Coil rows: L = 24 chains (res, res)
and (Ac, Am), k = 0..25, mean over PPII and beta.

Requirements (not package dependencies): PeptideBuilder 1.1, biopython, pdb2pqr 3.7, numpy, scipy.
Run:  OMP_NUM_THREADS=1 python gb_pipeline/helix_builds/build_gb_dipole_table.py --raw RAW.tsv.gz [--workers 12]
      python gb_pipeline/helix_builds/build_gb_dipole_table.py --derive RAW.tsv.gz --supplement SUP.tsv.gz
             [--rule strict] [--out pyagadir/params/sidechain_dipole_gb.tsv]
"""
import argparse
import gzip
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "pyagadir/params/sidechain_dipole_gb.tsv"
BACKBONE = {"N", "H", "CA", "HA", "C", "O"}
PDIE, PROBE, RB, H_GRID = 4.0, 1.4, 10.0, 0.3
CLASH, HBOND_MIN = 2.8, 2.5
KAPPA = np.round(np.concatenate([np.arange(0.0, 0.2, 0.01), np.arange(0.2, 0.8001, 0.025)]), 4)  # 1/A
WELLS = (-60.0, 180.0, 60.0)
# residue letter -> forms (table name, pdb2pqr residue name); the first is the charged form
FORMS = {"K": [("K", "LYS"), ("K0", "LYN")], "R": [("R", "ARG")], "H": [("H+", "HIP"), ("H0d", "HID"), ("H0e", "HIE")],
         "D": [("D", "ASP"), ("D0", "ASH")], "E": [("E", "GLU"), ("E0", "GLH")], "C": [("C-", "CYM"), ("C0", "CYS")],
         "Y": [("Y-", "TYR-"), ("Y0", "TYR")]}
CHARGED = [f[0][0] for f in FORMS.values()]
# canonical dihedrals after chi1 (PeptideBuilder order), the index of the scanned one, and its offsets (degrees)
CANON = {"K": [180.0, 180.0, 180.0], "R": [180.0, 180.0, 180.0, 0.0, 180.0], "H": [180.0], "D": [180.0],
         "E": [180.0, 180.0], "C": [], "Y": [90.0]}
SCAN = {"D": (0, (30, -30, 60, -60, 90)), "Y": (0, (30, -30, 60, -60, 90)), "E": (1, (30, -30, 60, -60, 90)),
        "H": (0, (30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180)), "K": (0, (120, -120)), "R": (0, (120, -120)),
        "C": (None, ())}
DONOR_N = {"N", "NZ", "NE", "NH1", "NH2", "ND1", "NE2"}
# Tyr-: AMBER Tyr minus HH, plus PARSE's deprotonation pattern (TY- minus TYR) scaled to make the residue -1 in total
_PARSE_DEPROT = {"CG": -0.070, "CD1": -0.070, "CE1": -0.070, "CZ": -0.125, "OH": -0.090, "CE2": -0.070, "CD2": -0.070}
# AMBER ff14SB library caps: acetyl and C-terminal amide (charge, radius)
ACE = {"CH3": (-0.3662, 1.908), "HH3": (0.1123, 1.487), "C": (0.5972, 1.908), "O": (-0.5679, 1.6612)}
NHE = {"N": (-0.4630, 1.824), "H": (0.2315, 0.6)}
COIL_L = 24
NEAR_CAP = 8


def layout(L, letter, idx, conf):
    """Residue letters and (phi, psi). Index 0 = flank, 1 = N-cap, 2..L+1 = helix, L+2 = C-cap, L+3 = flank."""
    n = L + 4
    seq = ["A"] * n
    seq[idx] = letter
    if conf == "ppii":
        return seq, [(-75.0, 145.0)] * n
    if conf == "beta":
        return seq, [(-120.0, 120.0)] * n
    ang = [(-60.0, -45.0)] * n
    ang[0], ang[1], ang[L + 2], ang[n - 1] = (-120.0, 120.0), (-167.0, 147.0), (-60.0, 150.0), (-75.0, 145.0)
    return seq, ang


def rotamer(letter, chi1, scan=0.0):
    tail = list(CANON[letter])
    i, _ = SCAN[letter]
    if i is not None:
        tail[i] = ((tail[i] + scan + 180.0) % 360.0) - 180.0
    return (chi1,) + tuple(tail)


def structure(seq, ang, idx, rot):
    import PeptideBuilder
    from PeptideBuilder import Geometry
    geos = []
    for r, letter in enumerate(seq):
        g = Geometry.geometry(letter)
        if r > 0:
            g.phi, g.psi_im1 = ang[r][0], ang[r - 1][1]
        if r == idx:
            g.inputRotamers(list(rot))
        geos.append(g)
    return PeptideBuilder.make_structure_from_geos(geos)


def unit(v):
    return v / np.linalg.norm(v)


def heavy_atoms(st, L, ncap, ccap):
    """(name, residue index, xyz) of the heavy atoms, with the caps applied."""
    out = []
    for a in st.get_atoms():
        n, r = a.get_id(), a.get_parent().get_id()[1] - 1
        if n.startswith("H"):
            continue
        if ncap == "Ac" and (r == 0 or (r == 1 and n not in ("CA", "C", "O"))):
            continue
        if ccap == "Am" and (r == L + 3 or (r == L + 2 and n != "N")):
            continue
        out.append((n, r, a.coord.astype(float)))
    return out


def steric(heavy, src):
    """(margin, smallest distance): margin = min over checked pairs of distance minus its threshold."""
    name = np.array([a[0] for a in heavy])
    res = np.array([a[1] for a in heavy])
    xyz = np.array([a[2] for a in heavy])
    sc = (res == src) & ~np.isin(name, list(BACKBONE | {"CB"}))
    gap = np.abs(res[:, None] - res[None, :])
    on = ((name[:, None] == "O") & (name[None, :] == "N")) | ((name[:, None] == "N") & (name[None, :] == "O"))
    ok = ((gap > 1) | ((gap == 1) & (sc[:, None] | sc[None, :]))) & ~on & np.triu(np.ones(gap.shape, bool), 1)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    don, oxy = np.isin(name, list(DONOR_N)), np.char.startswith(name.astype(str), "O")
    hb = (don[:, None] & oxy[None, :]) | (oxy[:, None] & don[None, :])
    thr = np.where(hb, HBOND_MIN, CLASH)
    return float((d - thr)[ok].min()), float(d[ok].min())


def choose(letter, seq, ang, idx, L, ncap, ccap, well, strict=False):
    """Nearest allowed member of a chi1 well: (rotamer, d_chi1, d_scan, margin, old contact). strict: 2.8 A for every
    pair (no hydrogen-bond allowance); the margin is then the smallest distance minus 2.8."""
    i, offs = SCAN[letter]
    cands = [(abs(d1) / 10.0 + abs(ds) / 30.0, d1, ds) for d1 in (0, 10, -10, 20, -20) for ds in (0,) + tuple(offs)]
    cands.sort(key=lambda c: c[0])
    first = None
    for _, d1, ds in cands:
        rot = rotamer(letter, well + d1, ds)
        m, c = steric(heavy_atoms(structure(seq, ang, idx, rot), L, ncap, ccap), idx)
        if strict:
            m = c - CLASH
        if first is None:
            first = (rot, d1, ds, m, c)
        if m >= 0.0:
            return rot, d1, ds, m, c
    return first


def pqr_forms(st, letter, idx, L, ncap, ccap, tmp):
    """Charged atom lists {form: [(name, resname, residue index, xyz, q, radius)]} of one structure, caps applied."""
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(st)
    base = os.path.join(tmp, "p.pdb")
    io.save(base)
    lines = open(base).read().splitlines(keepends=True)
    out = {}
    for form, resname in FORMS[letter]:
        name = "TYR" if resname == "TYR-" else resname
        pdb, pqr = os.path.join(tmp, f"{form}.pdb"), os.path.join(tmp, f"{form}.pqr")
        open(pdb, "w").write("".join(ln[:17] + name + ln[20:] if ln.startswith(("ATOM", "HETATM"))
                                     and int(ln[22:26]) == idx + 1 else ln for ln in lines))
        subprocess.run([str(Path(sys.executable).parent / "pdb2pqr"), "--ff", "AMBER", "--nodebump", "--noopt",
                        "--keep-chain", "--whitespace", pdb, pqr], check=True, capture_output=True)
        atoms = []
        for line in open(pqr):
            if line.startswith(("ATOM", "HETATM")):
                f = line.split()
                atoms.append([f[2], f[3], int(f[-6]) - 1, np.array([float(t) for t in f[-5:-2]]), float(f[-2]),
                              float(f[-1])])
        x = [a for a in atoms if a[2] == idx]
        if x[0][1] != name:
            raise RuntimeError(f"pdb2pqr built {x[0][1]} for {name}")
        if resname == "TYR-":
            hh = [a for a in x if a[0] == "HH"]
            scale = (1.0 - hh[0][4]) / -sum(_PARSE_DEPROT.values())
            atoms = [a for a in atoms if not (a[2] == idx and a[0] == "HH")]
            for a in atoms:
                if a[2] == idx:
                    a[4] += scale * _PARSE_DEPROT.get(a[0], 0.0)
        net = sum(a[4] for a in atoms if a[2] == idx)
        want = -1.0 if form in ("D", "E", "C-", "Y-") else (1.0 if form in ("K", "R", "H+") else 0.0)
        if abs(net - want) > 0.01:
            raise RuntimeError(f"{form}: net charge {net:.4f}, expected {want}")
        out[form] = cap_atoms(atoms, L, ncap, ccap)
    return out


def cap_atoms(atoms, L, ncap, ccap):
    get = {(a[2], a[0]): a for a in atoms}
    keep = []
    for a in atoms:
        if ncap == "Ac" and a[2] in (0, 1):
            continue
        if ccap == "Am" and a[2] in (L + 2, L + 3):
            continue
        keep.append(tuple(a))
    if ncap == "Ac":
        ca, n_, cb = get[(1, "CA")][3], get[(1, "N")][3], get[(1, "CB")][3]
        keep += [("CH3", "ACE", 1, ca, *ACE["CH3"]), ("HH31", "ACE", 1, get[(1, "HA")][3], *ACE["HH3"]),
                 ("HH32", "ACE", 1, ca + 1.09 * unit(n_ - ca), *ACE["HH3"]),
                 ("HH33", "ACE", 1, ca + 1.09 * unit(cb - ca), *ACE["HH3"]),
                 ("C", "ACE", 1, get[(1, "C")][3], *ACE["C"]), ("O", "ACE", 1, get[(1, "O")][3], *ACE["O"])]
    if ccap == "Am":
        n_, ca = get[(L + 2, "N")][3], get[(L + 2, "CA")][3]
        keep += [("N", "NHE", L + 2, n_, *NHE["N"]), ("H", "NHE", L + 2, get[(L + 2, "H")][3], *NHE["H"]),
                 ("H2", "NHE", L + 2, n_ + 1.01 * unit(ca - n_), *NHE["H"])]
    return keep


def born_radii(xyz, rad, need):
    from scipy.ndimage import distance_transform_edt
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


def energies(forms, idx, L):
    """{form: (E_in, W)} between X's source charges and the backbone of the N-cap..C-cap (and any cap), X excluded.
    The forms share Born radii computed on the union of their atoms."""
    union, key = [], {}
    for atoms in forms.values():
        for a in atoms:
            if (a[2], a[0]) not in key:
                key[(a[2], a[0])] = len(union)
                union.append(a)
    first = next(iter(forms.values()))
    ala = {}
    for a in first:
        if a[1] == "ALA" and a[2] != idx and a[0] in BACKBONE:
            ala.setdefault(a[2], {})[a[0]] = a[4]
    full = sorted(r for r, d in ala.items() if set(d) == BACKBONE)
    ala_bb = ala[full[len(full) // 2]]
    partner = [key[(a[2], a[0])] for a in first if a[2] != idx and (a[1] in ("ACE", "NHE") or
               (a[0] in BACKBONE and 1 <= a[2] <= L + 2))]
    need = set(partner) | {key[(a[2], a[0])] for atoms in forms.values() for a in atoms if a[2] == idx}
    R = born_radii(np.array([a[3] for a in union]), np.array([a[5] for a in union]), sorted(need))
    xb = np.array([union[j][3] for j in partner])
    qb = np.array([union[j][4] for j in partner])
    rb = np.array([R[j] for j in partner])
    out = {}
    for form, atoms in forms.items():
        src = [(a, a[4] - ala_bb[a[0]] if a[0] in BACKBONE else a[4]) for a in atoms if a[2] == idx]
        src = [(a, q) for a, q in src if q != 0.0]
        xs = np.array([a[3] for a, _ in src])
        qs = np.array([q for _, q in src])
        rs = np.array([R[key[(a[2], a[0])]] for a, _ in src])
        r = np.linalg.norm(xs[:, None, :] - xb[None, :, :], axis=-1)
        rr = rs[:, None] * rb[None, :]
        f = np.sqrt(r * r + rr * np.exp(-r * r / (4.0 * rr)))
        qq = 332.06 * qs[:, None] * qb[None, :]
        out[form] = (float(np.sum(qq * (1.0 / r - 1.0 / f)) / PDIE),
                     np.array([float(np.sum(qq * np.exp(-k * f) / f)) for k in KAPPA]))
    return out


def task(job):
    letter, state, conf, ncap, ccap, L, k = job[:7]
    wells, strict = (job[7], job[8]) if len(job) > 7 else (WELLS, False)
    idx = 1 + k
    seq, ang = layout(L, letter, idx, conf)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for well in wells:
            rot, d1, ds, margin, contact = choose(letter, seq, ang, idx, L, ncap, ccap, well, strict)
            forms = pqr_forms(structure(seq, ang, idx, rot), letter, idx, L, ncap, ccap, tmp)
            for form, (e_in, w) in energies(forms, idx, L).items():
                rows.append((form, state, conf, ncap, ccap, L, k, well, rot[0], rot[1] if len(rot) > 1 else 0.0,
                             d1, ds, margin, contact, e_in, w))
    return rows


def jobs(lmax):
    out = []
    for letter in FORMS:
        for L in range(4, lmax + 1):
            out += [(letter, "helix", "helix", "res", "res", L, k) for k in range(0, L + 2)]
            out += [(letter, "helix", "helix", "Ac", "res", L, k) for k in range(1, min(NEAR_CAP, L + 1) + 1)]
            out += [(letter, "helix", "helix", "res", "Am", L, k) for k in range(max(0, L + 1 - NEAR_CAP), L + 1)]
            if L <= 16:
                out += [(letter, "helix", "helix", "Ac", "Am", L, k) for k in range(1, L + 1)]
        for conf in ("ppii", "beta"):
            out += [(letter, "coil", conf, "res", "res", COIL_L, k) for k in range(0, COIL_L + 2)]
            out += [(letter, "coil", conf, "Ac", "Am", COIL_L, k) for k in range(1, COIL_L + 1)]
    return sorted(out, key=lambda j: -j[5])


RAW_HEAD = ["form", "state", "conf", "ncap", "ccap", "L", "k", "well", "chi1", "chi2", "d_chi1", "d_scan",
            "margin", "old_contact", "E_in"] + [f"W_{x:.4f}" for x in KAPPA]


def write_raw(path, lmax, workers):
    js = jobs(lmax)
    with ProcessPoolExecutor(workers) as ex, gzip.open(path, "wt") as fh:
        fh.write("\t".join(RAW_HEAD) + "\n")
        for n, rows in enumerate(ex.map(task, js, chunksize=1)):
            for r in rows:
                fh.write("\t".join([str(x) for x in r[:8]] + [f"{x:.1f}" for x in r[8:12]] + [f"{r[12]:.3f}",
                         f"{r[13]:.3f}", f"{r[14]:.5f}"] + [f"{v:.5f}" for v in r[15]]) + "\n")
            if n % 500 == 0:
                print(f"{n}/{len(js)}", flush=True)
    print("wrote", path, len(js), "tasks")


def write_supplement(raw, path, workers):
    """For every well whose member is allowed only through the hydrogen-bond allowance (contact < 2.8 A), the nearest
    member allowed under the strict rule (2.8 A for every pair), in the raw format."""
    import pandas as pd
    d = pd.read_csv(raw, sep="\t", usecols=["form", "state", "conf", "ncap", "ccap", "L", "k", "well", "margin",
                                            "old_contact"])
    d = d[d.form.isin(CHARGED) & (d.margin >= 0.0) & (d.old_contact < CLASH)]
    letter = {f[0][0]: x for x, f in FORMS.items()}
    js = sorted({(letter[r.form], r.state, r.conf, r.ncap, r.ccap, int(r.L), int(r.k), (float(r.well),), True)
                 for r in d.itertuples()}, key=lambda j: -j[5])
    with ProcessPoolExecutor(workers) as ex, gzip.open(path, "wt") as fh:
        fh.write("\t".join(RAW_HEAD) + "\n")
        for rows in ex.map(task, js, chunksize=1):
            for r in rows:
                fh.write("\t".join([str(x) for x in r[:8]] + [f"{x:.1f}" for x in r[8:12]] + [f"{r[12]:.3f}",
                         f"{r[13]:.3f}", f"{r[14]:.5f}"] + [f"{v:.5f}" for v in r[15]]) + "\n")
    print("wrote", path, len(js), "wells")


NEUTRAL = ["K0", "D0", "E0", "H0d", "H0e"]  # the neutral forms of the residues whose Table 1 cells are charged-form


def derive(raw, out, rule="strict", supplement=None, forms=None):
    """rule "hbond": wells allowed with the hydrogen-bond allowance (raw margin >= 0). rule "strict": 2.8 A for every
    pair; a raw member allowed only through the allowance is replaced by the supplement's strict member.
    forms: the table's forms (default: the charged forms; NEUTRAL for params/sidechain_dipole_gb_neutral.tsv)."""
    import pandas as pd
    forms = forms or CHARGED
    d = pd.read_csv(raw, sep="\t")
    d = d[d.form.isin(forms)]
    if rule == "strict":
        wk = ["state", "conf", "ncap", "ccap", "L", "k", "well"]
        sup = pd.read_csv(supplement, sep="\t")
        sup = sup[sup.form.isin(forms)]
        swap = d[(d.margin >= 0.0) & (d.old_contact < CLASH)][wk + ["form"]]
        d = d.merge(swap.assign(_swap=True), on=wk + ["form"], how="left")
        d = pd.concat([d[d._swap.isna()].drop(columns="_swap"), sup])
        d["margin"] = d.old_contact - CLASH
    elif rule != "hbond":
        raise ValueError(rule)
    wc = [c for c in d.columns if c.startswith("W_")]
    keys = ["form", "state", "ncap", "ccap", "L", "k"]

    def mean_allowed(g):
        ok = g[g.margin >= 0.0]
        use = ok if len(ok) else g[g.margin == g.margin.max()]  # none allowed: the least-clashing well, n_rot = 0
        return pd.Series([len(ok), use.E_in.mean()] + list(use[wc].mean()), index=["n_rot", "E_in"] + wc)
    h = d[d.state == "helix"].groupby(keys).apply(mean_allowed).reset_index()
    c = d[d.state == "coil"].groupby(keys + ["conf"]).apply(mean_allowed).reset_index()
    c = c.groupby(keys).agg({"n_rot": "min", "E_in": "mean", **{w: "mean" for w in wc}}).reset_index()
    t = pd.concat([c, h])
    t["n_rot"] = t.n_rot.astype(int)
    t["order"] = t.form.map({f: i for i, f in enumerate(forms)})
    t = t.sort_values(["order", "state", "ncap", "ccap", "L", "k"], ascending=[True, False, True, True, True, True])
    with open(out, "w") as fh:
        fh.write("# Generalized Born field of the helix backbone on a " + ("charged" if forms == CHARGED else "neutral")
                 + " side chain (gb_pipeline/helix_builds/build_gb_dipole_table.py).\n")
        fh.write("# E(kappa, T) = E_in + W(kappa) / eps_w(T), kcal/mol, " + ("charged form" if forms == CHARGED else
                 "neutral forms " + " ".join(forms)) + "; W columns at kappa in 1/A.\n")
        fh.write("# helix: L helical residues, k = 0 (N-cap) .. L + 1 (C-cap); coil: L = 24 chains, PPII/beta mean.\n")
        fh.write("# ncap/ccap: res = residue cap, Ac = acetyl, Am = amide; n_rot = allowed chi1 wells averaged.\n")
        fh.write(f"# rotamer rule: {rule}" + (" (2.8 A for every heavy-atom pair)" if rule == "strict" else
                 " (2.8 A; 2.5 A for donor N - O pairs)") + "\n")
        t[keys + ["n_rot", "E_in"] + wc].to_csv(fh, sep="\t", index=False, float_format="%.5f")
    print("wrote", out, len(h), "helix rows,", len(c), "coil rows; wells allowed:", dict(h.n_rot.value_counts()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="compute every structure and write the raw rows here (.tsv.gz)")
    ap.add_argument("--derive", help="raw file to average into the package table")
    ap.add_argument("--supplement", help="strict-rule members for the wells allowed only by the hydrogen-bond "
                    "allowance: written from --derive's raw file if it does not exist, read by --rule strict")
    ap.add_argument("--rule", default="strict", choices=["strict", "hbond"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--lmax", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--neutral", action="store_true", help="--derive the neutral forms (" + " ".join(NEUTRAL) + ") "
                    "for params/sidechain_dipole_gb_neutral.tsv instead of the charged forms")
    a = ap.parse_args()
    if a.raw:
        write_raw(a.raw, a.lmax, a.workers)
    if a.derive:
        if a.rule == "strict" and not Path(a.supplement).exists():
            write_supplement(a.derive, a.supplement, a.workers)
        derive(a.derive, a.out, a.rule, a.supplement, NEUTRAL if a.neutral else None)


if __name__ == "__main__":
    main()
