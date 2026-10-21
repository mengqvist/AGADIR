"""Build pyagadir/params/sidechain_pair_gb.tsv: the Generalized Born interaction between two charged side chains in
the helix and in the coil, for the side chain-side chain electrostatics of the GB laws (the companion of
gb_pipeline/helix_builds/build_gb_dipole_table.py, whose structures, charges, rotamer rule and Born radii it uses
unchanged).

Physics (continuum electrostatics, nothing fitted)
--------------------------------------------------
Residues X (position i) and Y (position i + k) sit in an ideal poly-Ala chain (build_gb_dipole_table.layout and
structure; residue caps; flanks outside). Their interaction is the Generalized Born cross term between X's and Y's
charges, each taken minus Ala's backbone charges on its backbone atoms (the source convention of the dipole table and
of the APBS pair calculations it was checked against):

    E = sum_{s in X, t in Y} 332.06 q_s q_t [ (1/r - 1/f) / eps_in  +  exp(-kappa f) / (eps_w f) ],
    f = sqrt(r^2 + R_s R_t exp(-r^2 / (4 R_s R_t))),

eps_in from --eps-in (default 4; the gbp table uses 16), salt as Srinivasan et al. (1999), Born radii from burial in the
built two-residue structure (Grycuk r^-6 integral over the solvent outside the 1.4 A solvent-excluded surface, 0.3 A
grid). The raw rows also hold both side chains' desolvation (the change in each one's own GB self-energy when the other
is near), which --derive and --derive-grid add unless --no-desolv.

Charged forms: Lys (LYS), Arg (ARG), His+ (HIP), Asp (ASP), Glu (GLU), AMBER ff14SB charges and radii from pdb2pqr.
Separations k = 1..12 (the range of the point-charge pair term it replaces).

Rows (state, class):
  helix int   X at N4 of an L = k + 7 helix (k0 = 4), Y at k0 + k (both inside, away from the caps)
  helix Ncap  X is the N-cap residue (k0 = 0), Y at N_k, L = max(k + 4, 8)
  helix Ccap  Y is the C-cap residue (k0 = L + 1), X at C-cap - k, L = max(k + 4, 8)
  coil        X at position 8 of a 24-residue PPII chain and of a beta chain, Y at 8 + k; the row is their mean
Rotamers: each residue's chi1 wells (-60, 180, 60) are represented by build_gb_dipole_table.choose's nearest sterically
allowed member under the strict rule (2.8 A for every heavy-atom pair), chosen with the other residue as Ala; a well
combination whose two side chains then come closer than 2.8 A (beyond C-beta) is dropped, as in the APBS pair runs.
The row is the equal-weight mean over the remaining combinations (n_comb); with none, the least-clashing one
(n_comb = 0). Salt-bridge hydrogen bonds are therefore not in the table: they are the Table IV acid-base cells.

Raw output (--raw, .tsv.gz): one row per (X, Y, k, state, class, conf, well pair) with E_in and W(kappa).
Package table (--derive): one row per (X, Y, k, state, class), E(kappa, T) = E_in + W(kappa)/eps_w(T), kcal/mol,
charged forms, on the dipole table's kappa grid (the N180 candidate; superseded by --derive-grid).

dipole_law="gbp" (params/sidechain_pair_gb.tsv, params/README.md "gbp")
--------------------------------------------------------------------------
--raw --forms: every protonation form of both partners (K/K0, R, H+/H0d/H0e, D/D0, E/E0; LYN, HID, HIE, ASH, GLH),
  one pdb2pqr run per form; all forms of a residue share one set of Born radii on the union of their atoms (the fixed-
  boundary convention of the dipole builder), each form's charges at its own positions; the cross term and both
  residues' desolvation (D_in, DW: the change of their GB self-energies) per well combination.
--derive-grid RAW: F = -RT ln <exp(-E/RT)> over the allowed combinations (the rotamer free energy; --mean for the
  mean), E = cross term + desolvation (--no-desolv: cross only) at --eps-in, on a (T, kappa) grid (T_GRID x
  KAPPA_GRID; the package interpolates bilinearly). Neutral His is the HID/HIE mixture with intrinsic weights
  0.2/0.8 (form H0). --helix-md / --coil-md replace the raw file's helix / coil rows by records computed on other
  backbones (comma-separated .npz files: the same structures' energies in the raw-forms layout, with "state" and
  "cls" arrays for helix records). The shipped table: --eps-in 16, desolvation, free energy, helix rows from the
  ideal helices of this builder recomputed with all-Ala hydrogens fixed, coil rows from an MD coil ensemble of
  Ac-A16-NH2 (CHARMM36m, 278 K; side chains threaded onto non-helical frames with this builder's rotamer rule).
Requirements (not package dependencies): PeptideBuilder 1.1, biopython, pdb2pqr 3.7, numpy, scipy, pandas.
Run:  OMP_NUM_THREADS=1 python gb_pipeline/helix_builds/build_gb_pair_table.py --raw RAW.tsv.gz --forms [--workers 16]
      python gb_pipeline/helix_builds/build_gb_pair_table.py --derive-grid RAW.tsv.gz --eps-in 16 [--helix-md H.npz]
          [--coil-md C1.npz,...] [--out pyagadir/params/sidechain_pair_gb.tsv]
"""
import argparse
import gzip
import itertools
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gb_dipole_table as D  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "pyagadir/params/sidechain_pair_gb.tsv"
PAIR_RES = {"K": "LYS", "R": "ARG", "H": "HIP", "D": "ASP", "E": "GLU"}
Z = {"K": 1.0, "R": 1.0, "H": 1.0, "D": -1.0, "E": -1.0}
KMAX = 12
COIL_X = 8


def placement(cls, k):
    """(L, helix index of X, helix index of Y) in build_gb_dipole_table.layout's indexing (1 = N-cap)."""
    if cls == "int":
        L = k + 7
        return L, 1 + 4, 1 + 4 + k
    L = max(k + 4, 8)
    if cls == "Ncap":
        return L, 1, 1 + k
    if cls == "Ccap":
        return L, L + 2 - k, L + 2
    raise ValueError(cls)


def structure2(seq, ang, rots):
    import PeptideBuilder
    from PeptideBuilder import Geometry
    geos = []
    for r, letter in enumerate(seq):
        g = Geometry.geometry(letter)
        if r > 0:
            g.phi, g.psi_im1 = ang[r][0], ang[r - 1][1]
        if r in rots:
            g.inputRotamers(list(rots[r]))
        geos.append(g)
    return PeptideBuilder.make_structure_from_geos(geos)


@lru_cache(maxsize=None)
def member(letter, L, idx, conf, well):
    """The dipole table's member of a well for one residue alone in the chain (strict rule)."""
    seq, ang = D.layout(L, letter, idx, conf)
    rot, _, _, margin, contact = D.choose(letter, seq, ang, idx, L, "res", "res", well, strict=True)
    return rot, contact - D.CLASH


def side_contact(st, i1, i2):
    sc = {i1: [], i2: []}
    for a in st.get_atoms():
        r = a.get_parent().get_id()[1] - 1
        n = a.get_id()
        if r in sc and not n.startswith("H") and n not in D.BACKBONE | {"CB"}:
            sc[r].append(a.coord.astype(float))
    if not sc[i1] or not sc[i2]:
        return 99.0
    x, y = np.array(sc[i1]), np.array(sc[i2])
    return float(np.min(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)))


CATION_N = {"K": ("NZ",), "R": ("NE", "NH1", "NH2"), "H": ("ND1", "NE2")}
CARBOXYL_O = {"D": ("OD1", "OD2"), "E": ("OE1", "OE2")}
SB_CUT = 4.0  # A: Baker et al. 2015 (Methods): Lys N-zeta to the centroid of Glu O-epsilon1/2 <= 4 A


def salt_bridge_distance(st, l1, i1, l2, i2):
    """Smallest distance from the cation's side-chain nitrogens to the carboxylate's oxygen centroid; 99 for a pair
    that is not acid-base."""
    if l1 in CATION_N and l2 in CARBOXYL_O:
        cat, an = (l1, i1), (l2, i2)
    elif l2 in CATION_N and l1 in CARBOXYL_O:
        cat, an = (l2, i2), (l1, i1)
    else:
        return 99.0
    xyz = {}
    for a in st.get_atoms():
        xyz[(a.get_parent().get_id()[1] - 1, a.get_id())] = a.coord.astype(float)
    cen = np.mean([xyz[(an[1], o)] for o in CARBOXYL_O[an[0]]], axis=0)
    return float(min(np.linalg.norm(xyz[(cat[1], n)] - cen) for n in CATION_N[cat[0]]))


def pqr_pair(st, l1, i1, l2, i2, tmp):
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(st)
    pdb, pqr = os.path.join(tmp, "p.pdb"), os.path.join(tmp, "p.pqr")
    io.save(pdb)
    names = {i1 + 1: PAIR_RES[l1], i2 + 1: PAIR_RES[l2]}
    lines = open(pdb).read().splitlines(keepends=True)
    open(pdb, "w").write("".join(ln[:17] + names[int(ln[22:26])] + ln[20:] if ln.startswith(("ATOM", "HETATM"))
                                 and int(ln[22:26]) in names else ln for ln in lines))
    subprocess.run([str(Path(sys.executable).parent / "pdb2pqr"), "--ff", "AMBER", "--nodebump", "--noopt",
                    "--keep-chain", "--whitespace", pdb, pqr], check=True, capture_output=True)
    atoms = []
    for line in open(pqr):
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            atoms.append((f[2], f[3], int(f[-6]) - 1, np.array([float(t) for t in f[-5:-2]]), float(f[-2]),
                          float(f[-1])))
    for letter, idx in ((l1, i1), (l2, i2)):
        x = [a for a in atoms if a[2] == idx]
        if x[0][1] != PAIR_RES[letter]:
            raise RuntimeError(f"pdb2pqr built {x[0][1]} for {PAIR_RES[letter]}")
        net = sum(a[4] for a in x)
        if abs(net - Z[letter]) > 0.01:
            raise RuntimeError(f"{letter}{idx}: net charge {net:.4f}")
    return atoms


def cross(atoms, i1, i2):
    """(E_in, W[kappa]) between residue i1's and residue i2's source charges."""
    ala = {}
    for a in atoms:
        if a[1] == "ALA" and a[0] in D.BACKBONE:
            ala.setdefault(a[2], {})[a[0]] = a[4]
    full = sorted(r for r, d in ala.items() if set(d) == D.BACKBONE)
    ala_bb = ala[full[len(full) // 2]]
    idx = {i: [j for j, a in enumerate(atoms) if a[2] == i] for i in (i1, i2)}
    R = D.born_radii(np.array([a[3] for a in atoms]), np.array([a[5] for a in atoms]), sorted(idx[i1] + idx[i2]))

    def src(i):
        out = []
        for j in idx[i]:
            a = atoms[j]
            q = a[4] - ala_bb[a[0]] if a[0] in D.BACKBONE else a[4]
            if q != 0.0:
                out.append((a[3], q, R[j]))
        return out
    s, t = src(i1), src(i2)
    xs, qs, rs = np.array([x for x, _, _ in s]), np.array([q for _, q, _ in s]), np.array([r for _, _, r in s])
    xt, qt, rt = np.array([x for x, _, _ in t]), np.array([q for _, q, _ in t]), np.array([r for _, _, r in t])
    r = np.linalg.norm(xs[:, None, :] - xt[None, :, :], axis=-1)
    rr = rs[:, None] * rt[None, :]
    f = np.sqrt(r * r + rr * np.exp(-r * r / (4.0 * rr)))
    qq = 332.06 * qs[:, None] * qt[None, :]
    return (float(np.sum(qq * (1.0 / r - 1.0 / f)) / D.PDIE),
            np.array([float(np.sum(qq * np.exp(-kk * f) / f)) for kk in D.KAPPA]))


DESOLV_RANGE = 8.0  # A: side-chain contact beyond which the partner's desolvation is negligible (not computed)


def self_parts(atoms, idx, R, q):
    """GB self-energy of one charge set (all pairs, i = j included), split as E_in + W(kappa)/eps_w:
    E_self = -1/2 sum 332 q_i q_j [1/eps_in - exp(-kappa f_ij)/eps_w] / f_ij."""
    xs = np.array([atoms[i][3] for i in idx])
    qs = np.array([q[i] for i in idx])
    rs = np.array([R[i] for i in idx])
    r = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
    rr = rs[:, None] * rs[None, :]
    f = np.sqrt(r * r + rr * np.exp(-r * r / (4.0 * rr)))
    qq = 332.06 * qs[:, None] * qs[None, :]
    return float(-0.5 * np.sum(qq / f) / D.PDIE), np.array([float(0.5 * np.sum(qq * np.exp(-kk * f) / f))
                                                           for kk in D.KAPPA])


def desolvation(atoms, i1, i2):
    """(D_in, DW[kappa]): the change of X's and of Y's GB self-energy caused by the other's side chain (atoms beyond
    C-beta present vs absent), i.e. the desolvation each pays for the other's presence."""
    ala = {}
    for a in atoms:
        if a[1] == "ALA" and a[0] in D.BACKBONE:
            ala.setdefault(a[2], {})[a[0]] = a[4]
    full = sorted(r for r, d in ala.items() if set(d) == D.BACKBONE)
    ab = ala[full[len(full) // 2]]
    q = {j: (a[4] - ab[a[0]] if a[0] in D.BACKBONE else a[4]) for j, a in enumerate(atoms)}
    X = [j for j, a in enumerate(atoms) if a[2] == i1 and q[j] != 0.0]
    Y = [j for j, a in enumerate(atoms) if a[2] == i2 and q[j] != 0.0]
    xyz = np.array([a[3] for a in atoms])
    rad = np.array([a[5] for a in atoms])

    def side(i):
        return {j for j, a in enumerate(atoms) if a[2] == i and a[0] not in D.BACKBONE | {"CB", "HB2", "HB3"}}

    def radii(drop, need):
        keep = [j for j in range(len(atoms)) if j not in drop]
        mp = {j: n for n, j in enumerate(keep)}
        Rk = D.born_radii(xyz[keep], rad[keep], [mp[j] for j in need])
        return {j: Rk[mp[j]] for j in need}
    Rf = radii(set(), X + Y)
    RnoY, RnoX = radii(side(i2), X), radii(side(i1), Y)
    d_in, dw = 0.0, np.zeros(len(D.KAPPA))
    for idx, Ra, Rb in ((X, Rf, RnoY), (Y, Rf, RnoX)):
        a_in, a_w = self_parts(atoms, idx, Ra, q)
        b_in, b_w = self_parts(atoms, idx, Rb, q)
        d_in += a_in - b_in
        dw += a_w - b_w
    return d_in, dw


# --forms: every protonation form of each partner (the dipole table's FORMS), for the exact two-state coupling
PAIR_FORMS = {"K": [("K", "LYS"), ("K0", "LYN")], "R": [("R", "ARG")], "H": [("H+", "HIP"), ("H0d", "HID"), ("H0e", "HIE")],
              "D": [("D", "ASP"), ("D0", "ASH")], "E": [("E", "GLU"), ("E0", "GLH")]}
FORM_Z = {"K": 1.0, "R": 1.0, "H+": 1.0, "D": -1.0, "E": -1.0, "K0": 0.0, "H0d": 0.0, "H0e": 0.0, "D0": 0.0, "E0": 0.0}


def _pqr(st, names, tmp):
    from Bio.PDB import PDBIO
    io = PDBIO()
    io.set_structure(st)
    pdb, pqr = os.path.join(tmp, "p.pdb"), os.path.join(tmp, "p.pqr")
    io.save(pdb)
    lines = open(pdb).read().splitlines(keepends=True)
    open(pdb, "w").write("".join(ln[:17] + names[int(ln[22:26])] + ln[20:] if ln.startswith(("ATOM", "HETATM"))
                                 and int(ln[22:26]) in names else ln for ln in lines))
    subprocess.run([str(Path(sys.executable).parent / "pdb2pqr"), "--ff", "AMBER", "--nodebump", "--noopt",
                    "--keep-chain", "--whitespace", pdb, pqr], check=True, capture_output=True)
    atoms = []
    for line in open(pqr):
        if line.startswith(("ATOM", "HETATM")):
            f = line.split()
            atoms.append((f[2], f[3], int(f[-6]) - 1, np.array([float(t) for t in f[-5:-2]]), float(f[-2]),
                          float(f[-1])))
    return atoms


def pqr_forms_pair(st, l1, i1, l2, i2, tmp):
    """(union atoms, {form of X: X's atoms}, {form of Y: Y's atoms}). One pdb2pqr run per form of each partner (the
    other in its charged form); every other atom is the same in all runs. The union holds every atom of every form
    (keyed by residue and name, first occurrence), so that all forms share one set of Born radii, as in the dipole
    table's builder."""
    fx, fy = {}, {}
    base = None
    for f, rn in PAIR_FORMS[l1]:
        atoms = _pqr(st, {i1 + 1: rn, i2 + 1: PAIR_FORMS[l2][0][1]}, tmp)
        fx[f] = [a for a in atoms if a[2] == i1]
        if base is None:
            base = [a for a in atoms if a[2] not in (i1, i2)]
            fy[PAIR_FORMS[l2][0][0]] = [a for a in atoms if a[2] == i2]
    for f, rn in PAIR_FORMS[l2][1:]:
        atoms = _pqr(st, {i1 + 1: PAIR_FORMS[l1][0][1], i2 + 1: rn}, tmp)
        fy[f] = [a for a in atoms if a[2] == i2]
    for forms, letter in ((fx, l1), (fy, l2)):
        for f, x in forms.items():
            rn = dict(PAIR_FORMS[letter])[f]
            if x[0][1] != rn:
                raise RuntimeError(f"pdb2pqr built {x[0][1]} for {rn}")
            if abs(sum(a[4] for a in x) - FORM_Z[f]) > 0.01:
                raise RuntimeError(f"{f}: net charge {sum(a[4] for a in x):.4f}")
    union, seen = list(base), {(a[2], a[0]) for a in base}
    for forms in (fx, fy):
        for x in forms.values():
            for a in x:
                if (a[2], a[0]) not in seen:
                    seen.add((a[2], a[0]))
                    union.append(a)
    return union, fx, fy


def _gb_sum(xs, qs, rs, xt, qt, rt):
    r = np.linalg.norm(xs[:, None, :] - xt[None, :, :], axis=-1)
    rr = rs[:, None] * rt[None, :]
    f = np.sqrt(r * r + rr * np.exp(-r * r / (4.0 * rr)))
    qq = 332.06 * qs[:, None] * qt[None, :]
    return float(np.sum(qq * (1.0 / r - 1.0 / f)) / D.PDIE), np.array([float(np.sum(qq * np.exp(-kk * f) / f))
                                                                     for kk in D.KAPPA])


def _self_parts(xs, qs, rs):
    r = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
    rr = rs[:, None] * rs[None, :]
    f = np.sqrt(r * r + rr * np.exp(-r * r / (4.0 * rr)))
    qq = 332.06 * qs[:, None] * qs[None, :]
    return float(-0.5 * np.sum(qq / f) / D.PDIE), np.array([float(0.5 * np.sum(qq * np.exp(-kk * f) / f))
                                                           for kk in D.KAPPA])


def energies_forms(union, fx, fy, i1, i2, near):
    """{(form X, form Y): (E_in, W, D_in, DW)}: the cross term of every form combination and, if near, each residue's
    desolvation by the other's side chain (union atoms beyond C-beta present vs absent), on the union's radii. Source
    charges: each form's charges, minus Ala's backbone charges on its backbone atoms."""
    ala = {}
    for a in union:
        if a[1] == "ALA" and a[0] in D.BACKBONE:
            ala.setdefault(a[2], {})[a[0]] = a[4]
    full = sorted(r for r, d in ala.items() if set(d) == D.BACKBONE)
    ab = ala[full[len(full) // 2]]
    key = {(a[2], a[0]): j for j, a in enumerate(union)}
    xyz = np.array([a[3] for a in union])
    rad = np.array([a[5] for a in union])
    need = sorted({key[(a[2], a[0])] for forms in (fx, fy) for x in forms.values() for a in x})
    R = D.born_radii(xyz, rad, need)

    def src(x, Rk):
        s = [(a, a[4] - ab[a[0]] if a[0] in D.BACKBONE else a[4]) for a in x]
        s = [(a, q) for a, q in s if q != 0.0]
        return (np.array([a[3] for a, _ in s]), np.array([q for _, q in s]),
                np.array([Rk[key[(a[2], a[0])]] for a, _ in s]))
    dself = {}
    if near:
        def side(i):
            return {j for j, a in enumerate(union) if a[2] == i and a[0] not in D.BACKBONE | {"CB", "HB2", "HB3"}}

        def radii(drop, idx):
            keep = [j for j in range(len(union)) if j not in drop]
            mp = {j: n for n, j in enumerate(keep)}
            Rk = D.born_radii(xyz[keep], rad[keep], [mp[j] for j in idx])
            return {j: Rk[mp[j]] for j in idx}
        idx_x = sorted({key[(a[2], a[0])] for x in fx.values() for a in x})
        idx_y = sorted({key[(a[2], a[0])] for x in fy.values() for a in x})
        RnoY, RnoX = radii(side(i2), idx_x), radii(side(i1), idx_y)
        for forms, Rb in ((fx, RnoY), (fy, RnoX)):
            for f, x in forms.items():
                a_in, a_w = _self_parts(*src(x, R))
                b_in, b_w = _self_parts(*src(x, Rb))
                dself[f] = (a_in - b_in, a_w - b_w)
    out = {}
    for f1, x1 in fx.items():
        s1 = src(x1, R)
        for f2, x2 in fy.items():
            e_in, w = _gb_sum(*s1, *src(x2, R))
            if near:
                d_in, dw = dself[f1][0] + dself[f2][0], dself[f1][1] + dself[f2][1]
            else:
                d_in, dw = 0.0, np.zeros(len(D.KAPPA))
            out[(f1, f2)] = (e_in, w, d_in, dw)
    return out


def task_forms(job):
    """As task, for every form combination: rows (X, Y, fX, fY, k, ...)."""
    l1, l2, k, state, cls, conf = job
    if state == "helix":
        L, i1, i2 = placement(cls, k)
    else:
        L, i1, i2 = D.COIL_L, 1 + COIL_X, 1 + COIL_X + k
    seq, ang = D.layout(L, "A", 0, conf if state == "coil" else "helix")
    seq[i1], seq[i2] = l1, l2
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for w1, w2 in itertools.product(D.WELLS, D.WELLS):
            r1, m1 = member(l1, L, i1, conf if state == "coil" else "helix", w1)
            r2, m2 = member(l2, L, i2, conf if state == "coil" else "helix", w2)
            st = structure2(seq, ang, {i1: r1, i2: r2})
            contact = side_contact(st, i1, i2)
            sb = salt_bridge_distance(st, l1, i1, l2, i2)
            union, fx, fy = pqr_forms_pair(st, l1, i1, l2, i2, tmp)
            for (f1, f2), (e_in, w, d_in, dw) in energies_forms(union, fx, fy, i1, i2, contact < DESOLV_RANGE).items():
                rows.append((l1, l2, f1, f2, k, state, cls, conf, w1, w2, round(min(m1, m2), 3),
                             round(contact - D.CLASH, 3), round(sb, 3), e_in, w, d_in, dw))
    return rows


def task(job):
    l1, l2, k, state, cls, conf = job
    if state == "helix":
        L, i1, i2 = placement(cls, k)
    else:
        L, i1, i2 = D.COIL_L, 1 + COIL_X, 1 + COIL_X + k
    seq, ang = D.layout(L, "A", 0, conf if state == "coil" else "helix")
    seq[i1], seq[i2] = l1, l2
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for w1, w2 in itertools.product(D.WELLS, D.WELLS):
            r1, m1 = member(l1, L, i1, conf if state == "coil" else "helix", w1)
            r2, m2 = member(l2, L, i2, conf if state == "coil" else "helix", w2)
            st = structure2(seq, ang, {i1: r1, i2: r2})
            contact = side_contact(st, i1, i2)
            sb = salt_bridge_distance(st, l1, i1, l2, i2)
            atoms = pqr_pair(st, l1, i1, l2, i2, tmp)
            e_in, w = cross(atoms, i1, i2)
            if contact < DESOLV_RANGE:
                d_in, dw = desolvation(atoms, i1, i2)
            else:
                d_in, dw = 0.0, np.zeros(len(D.KAPPA))
            rows.append((l1, l2, k, state, cls, conf, w1, w2, round(min(m1, m2), 3), round(contact - D.CLASH, 3),
                         round(sb, 3), e_in, w, d_in, dw))
    return rows


def jobs():
    out = []
    for l1, l2 in itertools.product(PAIR_RES, PAIR_RES):
        for k in range(1, KMAX + 1):
            out += [(l1, l2, k, "helix", cls, "helix") for cls in ("int", "Ncap", "Ccap")]
            out += [(l1, l2, k, "coil", "coil", conf) for conf in ("ppii", "beta")]
    return out


RAW_HEAD = ["X", "Y", "k", "state", "cls", "conf", "well1", "well2", "margin_self", "margin_pair", "sb_dist", "E_in"] + \
    [f"W_{x:.4f}" for x in D.KAPPA] + ["D_in"] + [f"DW_{x:.4f}" for x in D.KAPPA]


RAW_HEAD_FORMS = RAW_HEAD[:2] + ["fX", "fY"] + RAW_HEAD[2:]


def write_raw(path, workers, only=None, forms=False):
    js = jobs()
    if only:
        js = [j for j in js if only(j)]
    n0 = 13 if forms else 11  # leading label columns before E_in
    with ProcessPoolExecutor(workers) as ex, gzip.open(path, "wt") as fh:
        fh.write("\t".join(RAW_HEAD_FORMS if forms else RAW_HEAD) + "\n")
        for n, rows in enumerate(ex.map(task_forms if forms else task, js, chunksize=1)):
            for r in rows:
                fh.write("\t".join(str(x) for x in r[:n0]) + f"\t{r[n0]:.5f}\t"
                         + "\t".join(f"{v:.5f}" for v in r[n0 + 1]) + f"\t{r[n0 + 2]:.5f}\t"
                         + "\t".join(f"{v:.5f}" for v in r[n0 + 3]) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(js)}", flush=True)
    print("wrote", path, len(js), "tasks")


def annotate_task(job):
    """sb_dist of every well combination of one task (structures only; no charges or energies)."""
    l1, l2, k, state, cls, conf = job
    if state == "helix":
        L, i1, i2 = placement(cls, k)
    else:
        L, i1, i2 = D.COIL_L, 1 + COIL_X, 1 + COIL_X + k
    seq, ang = D.layout(L, "A", 0, conf if state == "coil" else "helix")
    seq[i1], seq[i2] = l1, l2
    out = {}
    for w1, w2 in itertools.product(D.WELLS, D.WELLS):
        r1, _ = member(l1, L, i1, conf if state == "coil" else "helix", w1)
        r2, _ = member(l2, L, i2, conf if state == "coil" else "helix", w2)
        out[(w1, w2)] = salt_bridge_distance(structure2(seq, ang, {i1: r1, i2: r2}), l1, i1, l2, i2)
    return job, out


def annotate(raw, out, workers):
    import pandas as pd
    d = pd.read_csv(raw, sep="\t")
    js = [j for j in jobs() if (j[0] in CATION_N and j[1] in CARBOXYL_O) or (j[1] in CATION_N and j[0] in CARBOXYL_O)]
    sb = {}
    with ProcessPoolExecutor(workers) as ex:
        for job, res in ex.map(annotate_task, js, chunksize=1):
            for (w1, w2), v in res.items():
                sb[job + (w1, w2)] = v
    key = list(zip(d.X, d.Y, d.k, d.state, d.cls, d.conf, d.well1, d.well2))
    d.insert(10, "sb_dist", [round(sb.get(kk, 99.0), 3) for kk in key])
    d.to_csv(out, sep="\t", index=False, compression="gzip", float_format="%.5f")
    print("wrote", out, "salt-bridged combinations:", int((d.sb_dist <= SB_CUT).sum()))


def derive(raw, out, eps_in=D.PDIE, desolv=True):
    """eps_in: the interior dielectric; the E_in and D_in parts scale exactly as PDIE/eps_in (W does not depend on it).
    desolv: add each residue's desolvation by the other (raw files with D_in)."""
    import pandas as pd
    d = pd.read_csv(raw, sep="\t")
    wc = [c for c in d.columns if c.startswith("W_")]
    d["E_in"] = d.E_in * D.PDIE / eps_in
    if desolv and "D_in" in d.columns:  # full GB pair free energy: cross term + both residues' desolvation
        d["E_in"] = d.E_in + d.D_in * D.PDIE / eps_in
        for c in wc:
            d[c] = d[c] + d["D" + c]
    keys = ["X", "Y", "k", "state", "cls"]

    sb = d.sb_dist if "sb_dist" in d.columns else 99.0

    d = d.assign(_sb=sb)

    def mean_allowed(g):
        ok = g[(g.margin_pair >= 0.0) & (g.margin_self >= 0.0) & (g._sb > SB_CUT)]
        use = ok if len(ok) else g[g.margin_pair == g.margin_pair.max()]
        return pd.Series([len(ok), use.E_in.mean()] + list(use[wc].mean()), index=["n_comb", "E_in"] + wc)
    h = d[d.state == "helix"].groupby(keys).apply(mean_allowed).reset_index()
    c = d[d.state == "coil"].groupby(keys + ["conf"]).apply(mean_allowed).reset_index()
    c = c.groupby(keys).agg({"n_comb": "min", "E_in": "mean", **{w: "mean" for w in wc}}).reset_index()
    t = pd.concat([h, c])
    t["n_comb"] = t.n_comb.astype(int)
    t = t.sort_values(["X", "Y", "k", "state", "cls"])
    with open(out, "w") as fh:
        fh.write("# Generalized Born interaction of two charged side chains X(i), Y(i+k) (gb_pipeline/helix_builds/build_gb_pair_table.py).\n")
        fh.write("# E(kappa, T) = E_in + W(kappa) / eps_w(T), kcal/mol, charged forms; W columns at kappa in 1/A.\n")
        fh.write("# helix cls: int (both inside), Ncap (X is the N-cap residue), Ccap (Y is the C-cap residue);\n")
        fh.write("# coil: mean of a PPII and a beta 24-residue chain; n_comb = allowed chi1 well combinations averaged.\n")
        fh.write(f"# eps_in {eps_in:g}; energy: GB cross term" + (" + each residue's desolvation by the other's side chain "
                 f"(computed within {DESOLV_RANGE} A)" if desolv and "D_in" in d.columns else " only") + ".\n")
        fh.write(f"# excluded: steric contacts < {D.CLASH} A, and salt-bridged combinations (cation N to carboxylate O\n")
        fh.write(f"# centroid <= {SB_CUT} A, Baker et al. 2015), whose energy is the Table IV acid-base cell.\n")
        t[keys + ["n_comb", "E_in"] + wc].to_csv(fh, sep="\t", index=False, float_format="%.5f")
    print("wrote", out, len(h), "helix rows,", len(c), "coil rows; combos allowed:", dict(h.n_comb.value_counts()))


# --derive-grid: the package table of the GB pair law (dipole_law="gbp"): per (X, Y, fX, fY, k, state, cls) the free
# energy over the allowed rotamer combinations, F = -RT ln < exp(-E/RT) > (equal prior weights; or the mean, --mean),
# E = (4/eps_in)(E_in + D_in) + (W + DW)(kappa)/eps_w(T), tabulated on a (T, kappa) grid that the law interpolates.
T_GRID = (273.15, 278.15, 288.15, 298.15, 313.15, 333.15, 363.15)
KAPPA_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4,
              0.5, 0.6, 0.7, 0.8)


def eps_water(T):
    """pyagadir.chemistry.calculate_permittivity (Malmberg & Maryott 1950 fit), repeated so the tool needs no package."""
    return 5321 / T + 233.76 - 0.9297 * T + 0.1417e-2 * T ** 2 - 0.8292e-6 * T ** 3


def _grid_energies(e_in, w, d_in, dw, kap, eps_in, desolv):
    """E[comb, T, kappa] on the grid (kcal/mol)."""
    s = D.PDIE / eps_in
    a = s * (e_in + (d_in if desolv else 0.0))
    ww = w + (dw if desolv else 0.0)
    wg = np.stack([np.array([np.interp(k, kap, r) for k in KAPPA_GRID]) for r in ww])  # comb x kappa
    eps = np.array([eps_water(T) for T in T_GRID])
    return a[:, None, None] + wg[:, None, :] / eps[None, :, None]


def _grid_free(E, mean):
    if mean:
        return E.mean(0)
    RT = 1.9872e-3 * np.array(T_GRID)[:, None]
    m = E.min(0)
    return m - RT * np.log(np.mean(np.exp(-(E - m[None]) / RT[None]), axis=0))


def coil_md_rows(npz):
    """Records computed on other backbones (MD coil frames, or helices recomputed with fixed hydrogens; several files
    comma-separated, frames numbered on) as a DataFrame in the raw-forms layout. The records are made by the
    frame pipeline (gb_pipeline/postprocessing/n181_coil_gb.py, n181_coil_run.py, n181_helix_run.py)."""
    import pandas as pd
    if "," in str(npz):
        parts = [coil_md_rows(p) for p in str(npz).split(",")]
        off, ds = 0, []
        for d, _, _, _ in parts:
            ds.append(d.assign(frame=d.frame + off))
            off += int(d.frame.max()) + 1
        return (pd.concat(ds, ignore_index=True), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]), parts[0][3])
    z = np.load(npz)
    lab, num = z["lab"], z["num"]
    cols = dict(zip([str(c) for c in z["num_cols"]], num.T))
    state = z["state"] if "state" in z.files else np.array(["coil"] * len(lab))
    cls = z["cls"] if "cls" in z.files else np.array(["coil"] * len(lab))
    d = pd.DataFrame({"X": lab["X"], "Y": lab["Y"], "fX": lab["fX"], "fY": lab["fY"], "k": lab["k"].astype(int),
                      "state": state, "cls": cls, "conf": "md", "frame": lab["frame"], "well1": lab["w1"],
                      "well2": lab["w2"], "margin_self": cols["margin_self"], "margin_pair": cols["margin_pair"],
                      "sb_dist": cols["sb_dist"], "E_in": cols["E_in"].astype(float), "D_in": cols["D_in"].astype(float)})
    kap = np.array(z["kappa"], float)
    return d, z["W"].astype(float), z["DW"].astype(float), kap


def derive_grid(raw, out, eps_in=D.PDIE, desolv=True, mean=False, coil_md=None, his_e=0.8, helix_md=None):
    """Package table from a raw-forms file (helix and coil rows) and optionally MD coil records (replacing the raw
    file's PPII/beta coil rows). Exclusions as derive (steric margins >= 0, salt bridges > 4 A; none allowed: the
    least-clashing combinations). Neutral His: the two tautomers' free energies combined with intrinsic weights
    (HIE his_e, HID 1 - his_e) into form H0."""
    import pandas as pd
    d = pd.read_csv(raw, sep="\t")
    wc = [c for c in d.columns if c.startswith("W_")]
    kap = np.array([float(c[2:]) for c in wc])
    W, DW = d[wc].to_numpy(float), d[["D" + c for c in wc]].to_numpy(float)
    d = d.drop(columns=wc + ["D" + c for c in wc])
    d["frame"] = 0
    if helix_md:  # helix rows from records (coil_md_rows)
        parts = [coil_md_rows(helix_md)]
    else:
        parts = [(d[d.state == "helix"], W[(d.state == "helix").to_numpy()], DW[(d.state == "helix").to_numpy()], kap)]
    if coil_md:
        parts.append(coil_md_rows(coil_md))
    else:
        m = (d.state == "coil").to_numpy()
        parts.append((d[m], W[m], DW[m], kap))
    rows = []
    for g, Wp, DWp, kp in parts:
        g = g.reset_index(drop=True)
        g["_ok"] = (g.margin_pair >= 0.0) & (g.margin_self >= 0.0) & (g.sb_dist > SB_CUT)
        for key, h in g.groupby(["X", "Y", "k", "state", "cls"]):
            # the structure (frame, conf, wells) decides allowance, the same for every form combination
            ok = h[h._ok]
            if len(ok) == 0:
                ok = h[h.margin_pair == h.margin_pair.max()]
            n_struct = len(ok.groupby(["conf", "frame", "well1", "well2"]))
            for (fx, fy), u in ok.groupby(["fX", "fY"]):
                ix = u.index.to_numpy()
                E = _grid_energies(u.E_in.to_numpy(), Wp[ix], u.D_in.to_numpy(), DWp[ix], kp, eps_in, desolv)
                rows.append((*key, fx, fy, n_struct if len(h[h._ok]) else 0, _grid_free(E, mean)))
    # neutral His: tautomer mixture
    T = np.array(T_GRID)[:, None]
    RT = 1.9872e-3 * T
    tab = {(r[0], r[1], r[2], r[3], r[4], r[5], r[6]): r for r in rows}
    merged = []
    for key, r in tab.items():
        X, Y, k, state, cls, fx, fy = key
        if fx == "H0e" or fy == "H0e" or fx == "H0d" or fy == "H0d":
            continue
        merged.append(r)

    def mix(X, Y, k, state, cls, fxs, fys, wx, wy):
        acc = 0.0
        for ax, px in zip(fxs, wx):
            for ay, py in zip(fys, wy):
                acc = acc + px * py * np.exp(-tab[(X, Y, k, state, cls, ax, ay)][8] / RT)
        return -RT * np.log(acc)
    for (X, Y, k, state, cls) in {key[:5] for key in tab}:
        hx = ["H0d", "H0e"] if X == "H" else None
        hy = ["H0d", "H0e"] if Y == "H" else None
        wt = [1.0 - his_e, his_e]
        fxs_all = [f for f, _ in PAIR_FORMS[X]]
        fys_all = [f for f, _ in PAIR_FORMS[Y]]
        nst = tab[(X, Y, k, state, cls, fxs_all[0], fys_all[0])][7]
        for fx in ([f for f in fxs_all if f not in ("H0d", "H0e")] + (["H0"] if hx else [])):
            for fy in ([f for f in fys_all if f not in ("H0d", "H0e")] + (["H0"] if hy else [])):
                if fx != "H0" and fy != "H0":
                    continue
                xs, wx = (hx, wt) if fx == "H0" else ([fx], [1.0])
                ys, wy = (hy, wt) if fy == "H0" else ([fy], [1.0])
                F = mix(X, Y, k, state, cls, xs, ys, wx, wy)
                merged.append((X, Y, k, state, cls, fx, fy, nst, F))
    cols = [f"F_{T:.2f}_{k:.3f}" for T in T_GRID for k in KAPPA_GRID]
    t = pd.DataFrame([r[:8] for r in merged], columns=["X", "Y", "k", "state", "cls", "fX", "fY", "n_comb"])
    t = pd.concat([t, pd.DataFrame(np.array([r[8].ravel() for r in merged]), columns=cols)], axis=1)
    t = t.sort_values(["X", "Y", "k", "state", "cls", "fX", "fY"])
    with open(out, "w") as fh:
        fh.write("# Generalized Born interaction of two ionisable side chains X(i), Y(i+k), every protonation form "
                 "(gb_pipeline/helix_builds/build_gb_pair_table.py --derive-grid).\n")
        fh.write(f"# F = -RT ln <exp(-E/RT)> over the allowed rotamer combinations{' (mean, not free energy)' if mean else ''}; "
                 f"E = cross term{' + both residues desolvation' if desolv else ''}, eps_in {eps_in:g}; kcal/mol.\n")
        fh.write("# columns F_<T K>_<kappa 1/A>: the (T, kappa) grid, interpolated linearly. Forms: K K0 R H+ H0 D D0 E E0; "
                 f"H0 = HID/HIE mixture (HIE {his_e:g}).\n")
        md = ",".join(Path(p).name for p in str(coil_md).split(",")) if coil_md else ""
        fh.write(f"# coil: {'MD coil frames (' + md + ')' if coil_md else 'mean of PPII and beta 24-residue chains'}; "
                 "helix cls: int, Ncap (X is the N-cap), Ccap (Y is the C-cap).\n")
        fh.write(f"# excluded: steric contacts < {D.CLASH} A and salt-bridged combinations (cation N to carboxylate-O "
                 f"centroid <= {SB_CUT} A): the Table IV acid-base cells.\n")
        t.to_csv(fh, sep="\t", index=False, float_format="%.5f")
    print("wrote", out, len(t), "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--derive")
    ap.add_argument("--annotate", nargs=2, metavar=("RAW", "OUT"), help="add sb_dist to a raw file written without it")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--eps-in", type=float, default=D.PDIE)
    ap.add_argument("--no-desolv", action="store_true")
    ap.add_argument("--only", default=None, help="X,Y,k,state,cls filter for a quick check, e.g. E,K,4,,")
    ap.add_argument("--forms", action="store_true", help="--raw for every protonation form combination (fX, fY)")
    ap.add_argument("--derive-grid", help="raw-forms file -> the gbp package table (F on a (T, kappa) grid)")
    ap.add_argument("--coil-md", help="with --derive-grid: MD coil records (.npz, comma-separated) for the coil rows")
    ap.add_argument("--mean", action="store_true", help="with --derive-grid: mean energy instead of free energy")
    ap.add_argument("--helix-md", help="with --derive-grid: helix records (.npz) for the helix rows")
    a = ap.parse_args()
    if a.raw:
        only = None
        if a.only:
            f = a.only.split(",")
            only = lambda j: all(not v or str(j[i]) == v for i, v in enumerate(f))  # noqa: E731
        write_raw(a.raw, a.workers, only, a.forms)
    if a.annotate:
        annotate(a.annotate[0], a.annotate[1], a.workers)
    if a.derive:
        derive(a.derive, a.out, a.eps_in, not a.no_desolv)
    if a.derive_grid:
        derive_grid(a.derive_grid, a.out, a.eps_in, not a.no_desolv, a.mean, a.coil_md, helix_md=a.helix_md)


if __name__ == "__main__":
    main()
