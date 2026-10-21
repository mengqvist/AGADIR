"""N181 step 2 validation (pre-registered in N181, cycle 421 + amendment 08:43 UTC): MD coil charge-group distances
against Lacroix 1998 Table VI coil (params/table_6_coil_lacroix.tsv; arithmetic means between Asp CG, Glu CD, Arg CZ,
His CE1, Lys NZ in PDB strand/turn/coil segments).

Ours: on every frame of an n181_frames npz, every placement (i, i + k) with 2 <= i, i + k <= 15 and no helical-segment
residue in i..i+k, X and Y threaded by the pair builder's rule (n181_coil_gb.Frame.member: nearest allowed member per
chi1 well, partner as Ala), the 9 well combinations equally weighted, combinations with side chains < 2.8 A dropped.
Pass: over the 16 K/R/D/E pair types x k = 2..8, RMS relative deviation <= 20 % and no k with a mean signed deviation
beyond +-20 %; k = 1 reported only. Also: phi/psi basin shares of residues 3-14 (alpha-R, PPII, beta; information).
Run: OMP_NUM_THREADS=1 .venv-build/bin/python n181_coil_validate.py FRAMES.npz [--workers 6]
"""
import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n181_coil_gb as G  # noqa: E402

LETTERS = "KRDE"
LO, HI = 2, 15
LACROIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pyagadir", "params", "table_6_coil_lacroix.tsv")


def work(arg):
    fid, xyz, res, names, helical = arg
    fr = G.Frame(xyz, res, names)
    out = {}  # (X, Y, k) -> [sum, n]
    for k in range(1, 13):
        for i in range(LO, HI - k + 1):
            if helical[i - 1:i + k].any():
                continue
            j = i + k
            for X, Y in itertools.product(LETTERS, LETTERS):
                s, n = 0.0, 0
                for w1, w2 in itertools.product(G.D.WELLS, G.D.WELLS):
                    _, m1, sc1 = fr.member(i, X, w1)
                    _, m2, sc2 = fr.member(j, Y, w2)
                    if m1 < 0 or m2 < 0:
                        continue
                    f1 = np.array([x for nm, x in sc1.items() if nm != "CB"])
                    f2 = np.array([x for nm, x in sc2.items() if nm != "CB"])
                    if np.min(np.linalg.norm(f1[:, None] - f2[None], axis=-1)) < G.D.CLASH:
                        continue
                    s += float(np.linalg.norm(sc1[G.CHARGE_ATOM[X]] - sc2[G.CHARGE_ATOM[Y]]))
                    n += 1
                if n:
                    acc = out.setdefault((X, Y, k), [0.0, 0])
                    acc[0] += s / n  # each placement weighs equally (its allowed combinations averaged)
                    acc[1] += 1
    return fid, out


# Grdadolnik, Mohacek-Grosev, Baldwin & Avbelj 2011, PNAS 108:1794 (amide III IR/Raman of 19 dipeptides): basin
# centres P_II (-75, 145), beta (-120, 120), alpha-R (-60, -40); Ala dipeptide at 30 C: P_II 0.60-0.62, beta 0.29,
# alpha-R 0.09-0.11. Residues are assigned to the nearest centre (periodic phi/psi distance) for a 3-state share.
CENTRES = {"PPII": (-75.0, 145.0), "beta": (-120.0, 120.0), "alphaR": (-60.0, -40.0)}


def basins(phi, psi):
    d = []
    for c in CENTRES.values():
        dp = (phi - c[0] + 180.0) % 360.0 - 180.0
        ds = (psi - c[1] + 180.0) % 360.0 - 180.0
        d.append(dp ** 2 + ds ** 2)
    near = np.argmin(np.array(d), axis=0)
    return {name: float(np.mean(near == n)) for n, name in enumerate(CENTRES)}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.frames)
    jobs = z["job"]
    args = [(f, z["coords"][f], z["atom_res"], z["atom_name"], z["helical"][f]) for f in range(len(z["coords"]))]
    per_frame = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for fid, out in ex.map(work, args, chunksize=4):
            per_frame[fid] = out
    lac = pd.read_csv(LACROIX, sep="\t").set_index("Pos")
    rows = []
    for X, Y in itertools.product(LETTERS, LETTERS):
        for k in range(1, 13):
            vals = [o[(X, Y, k)][0] / o[(X, Y, k)][1] for o in per_frame.values() if (X, Y, k) in o]
            by_job = {}
            for f, o in per_frame.items():
                if (X, Y, k) in o:
                    by_job.setdefault(jobs[f], []).append(o[(X, Y, k)][0] / o[(X, Y, k)][1])
            jm = [np.mean(v) for v in by_job.values()]
            ref = float(lac.loc[X + Y, f"i+{k}"])
            rows.append(dict(X=X, Y=Y, k=k, md=np.mean(vals), sem=np.std(jm) / np.sqrt(len(jm)) if len(jm) > 1 else np.nan,
                             n_frames=len(vals), n_jobs=len(jm), lacroix=ref, rel=(np.mean(vals) - ref) / ref))
    t = pd.DataFrame(rows)
    g = t[t.k.between(2, 8)]
    rms = float(np.sqrt(np.mean(g.rel ** 2)))
    byk = g.groupby("k").rel.mean()
    passed = rms <= 0.20 and (byk.abs() <= 0.20).all()
    lines = [f"frames {len(args)} from jobs {sorted(set(jobs))}",
             "mean signed relative deviation (MD - Lacroix) / Lacroix by k (16 K/R/D/E types):",
             "  " + "  ".join(f"k{k}: {v:+.1%}" for k, v in t.groupby("k").rel.mean().items()),
             f"k = 2..8: RMS relative deviation {rms:.1%} (pass <= 20 %), worst k mean {byk.abs().max():.1%} "
             f"(pass <= 20 %): {'PASS' if passed else 'FAIL'}",
             "MD mean distance (A) / Lacroix, k = 1..8:"]
    for (X, Y), h in t[t.k <= 8].groupby(["X", "Y"]):
        lines.append(f"  {X}{Y}: " + " ".join(f"{r.md:5.1f}/{r.lacroix:4.1f}" for r in h.itertuples()))
    phi, psi = z["phi"][:, 2:14], z["psi"][:, 2:14]
    hel = z["helical"][:, 2:14]
    b = basins(phi[~hel], psi[~hel])
    lines.append("phi/psi of non-helical residues 3-14, nearest of Grdadolnik et al. 2011's basin centres: "
                 + ", ".join(f"{k} {v:.3f}" for k, v in b.items())
                 + f" (their Ala dipeptide, 30 C: PPII 0.60-0.62, beta 0.29, alphaR 0.09-0.11); helical-segment "
                 f"fraction {hel.mean():.3f}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
        t.to_csv(a.out.replace(".txt", ".csv"), index=False)


if __name__ == "__main__":
    main()
