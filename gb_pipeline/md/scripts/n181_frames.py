"""N181: coil frames from the cA16 runs (runs/n181/<job>/traj/chunk_*.xtc) -> results/n181_frames_<tag>.npz.

Usage: ../.venv/bin/python n181_frames.py TAG [--every-ps 500] [--discard-ns 10] [JOB ...]   (default: every cA16 job)
Per kept frame: all peptide atom coordinates (A), phi/psi per residue, and the helical-segment flag per residue
(pre-registered in N181, amendment 08:43 UTC): alpha-R basin (phi in [-100, -30], psi in [-80, -10]) AND one of
residues i+1..i+3 of a backbone H-bond O(i)...H-N(i+4), d(H,O) < 2.5 A and N-H...O > 120 deg (acetyl O = residue 0,
amide NT/HT1/HT2 = residue 17). Atom names are read from the raw peptide.pdb text (OpenMM's and mdtraj's readers rename
the cap hydrogens). Frame time t = (chunk - 1) * 2 ns + frame * 20 ps.
"""
import glob
import json
import os
import sys

import mdtraj as md
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs", "n181")
args = sys.argv[1:]
tag, args = args[0], args[1:]
every_ps, discard_ns = 500.0, 10.0
while args and args[0].startswith("--"):
    if args[0] == "--every-ps":
        every_ps, args = float(args[1]), args[2:]
    elif args[0] == "--discard-ns":
        discard_ns, args = float(args[1]), args[2:]
    else:
        sys.exit(f"unknown option {args[0]}")
jobs = args or sorted(os.path.basename(p) for p in glob.glob(os.path.join(RUNS, "cA16_coil*")))


def names(pdb):
    out = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")):
            out.append((int(line[22:26]), line[12:16].strip()))
    return out


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1)))


coords, meta, phis, psis, hel = [], [], [], [], []
atom_names = None
for job in jobs:
    jd = os.path.join(RUNS, job)
    nm = names(os.path.join(jd, "peptide.pdb"))
    resids = sorted({r for r, _ in nm})
    r0 = resids[0]
    lab = [(r - r0 + 1, n) for r, n in nm]  # residues 1..16
    if atom_names is None:
        atom_names = lab
    assert lab == atom_names, job
    ix = {k: i for i, k in enumerate(lab)}
    nres = max(r for r, _ in lab)
    chunks = sorted(glob.glob(os.path.join(jd, "traj", "chunk_*.xtc")))
    # only chunks whose state was saved (the run deletes later ones on resume; the newest may be mid-write)
    states = sorted(glob.glob(os.path.join(jd, "state_*.xml")))
    last = int(os.path.basename(states[-1])[6:10]) if states else 0
    chunks = [c for c in chunks if int(os.path.basename(c)[6:10]) <= last]
    top = md.load_topology(os.path.join(jd, "peptide.pdb"))
    for c in chunks:
        n = int(os.path.basename(c)[6:10])
        t = md.load(c, top=top)
        times = (n - 1) * 2000.0 + 20.0 * np.arange(1, t.n_frames + 1)  # ps
        keep = (times > discard_ns * 1000.0 + 1e-6) & (np.abs(times / every_ps - np.round(times / every_ps)) < 1e-6)
        if not keep.any():
            continue
        X = t.xyz[keep] * 10.0  # nm -> A
        for f, tt in zip(X, times[keep]):
            def g(r, a):
                return f[ix[(r, a)]]
            phi = [dihedral(g(1, "CY"), g(1, "N"), g(1, "CA"), g(1, "C"))]
            phi += [dihedral(g(r - 1, "C"), g(r, "N"), g(r, "CA"), g(r, "C")) for r in range(2, nres + 1)]
            psi = [dihedral(g(r, "N"), g(r, "CA"), g(r, "C"), g(r + 1, "N")) for r in range(1, nres)]
            psi += [dihedral(g(nres, "N"), g(nres, "CA"), g(nres, "C"), g(nres, "NT"))]
            phi, psi = np.array(phi), np.array(psi)
            alpha = (phi >= -100) & (phi <= -30) & (psi >= -80) & (psi <= -10)
            # acceptors O(i), i = 0 (acetyl OY) .. 16; donors H-N(j), j = 2..16, and 17 = amide (NT; HT1, HT2)
            acc = {0: g(1, "OY")}
            acc.update({r: g(r, "O") for r in range(1, nres + 1)})
            don = {r: [(g(r, "N"), g(r, "H"))] for r in range(2, nres + 1)}
            don[nres + 1] = [(g(nres, "NT"), g(nres, "HT1")), (g(nres, "NT"), g(nres, "HT2"))]
            seg = np.zeros(nres, bool)
            for i in range(0, nres - 2):
                j = i + 4
                if j not in don:
                    continue
                ok = False
                for N, H in don[j]:
                    d = np.linalg.norm(H - acc[i])
                    v1, v2 = N - H, acc[i] - H
                    ang = np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
                    ok |= d < 2.5 and ang > 120.0
                if ok:
                    for r in (i + 1, i + 2, i + 3):
                        if 1 <= r <= nres:
                            seg[r - 1] = True
            coords.append(f.astype(np.float32))
            phis.append(phi)
            psis.append(psi)
            hel.append(seg & alpha)
            meta.append((job, float(tt)))
out = os.path.join(ROOT, "results", f"n181_frames_{tag}.npz")
np.savez_compressed(out, coords=np.array(coords), phi=np.array(phis), psi=np.array(psis), helical=np.array(hel),
                    job=np.array([m[0] for m in meta]), time_ps=np.array([m[1] for m in meta]),
                    atom_res=np.array([r for r, _ in atom_names]), atom_name=np.array([n for _, n in atom_names]))
h = np.array(hel)
print(f"wrote {out}: {len(coords)} frames from {len(jobs)} jobs; helical-segment fraction per residue:",
      np.round(h.mean(0), 3) if len(h) else "-")
