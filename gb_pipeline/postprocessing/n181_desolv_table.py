"""N181 (cycle 431): the dipole-law desolvation table averaged over structure orientations (Born-radii grid alignment).
Reads data/n181_dipole_desolv_L24.txt (orientation 0) and _L24_r1..r8.txt (n181_dipole_desolv.py, N181_DESOLV_ROT), per
(form, position): mean and standard error over orientations of dX + dB. Positions at least INTERIOR from both ends of
the L = 24 helix are structurally equivalent (a full helical turn and more on both sides), so their mean is the
interior row; positions 0..INTERIOR-1 from an end keep their own value. Writes data/n181_dipole_desolv_avg.tsv (the
package layout: form, end, dist, desolv, se) and prints the table with the interior check.
usage: python n181_desolv_table.py [INTERIOR=5]"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INTERIOR = int(sys.argv[1]) if len(sys.argv) > 1 else 5
L = 24
files = [f"{D}/n181_dipole_desolv_L24.txt"] + sorted(glob.glob(f"{D}/n181_dipole_desolv_L24_r*.txt"))
rows = []
for n, f in enumerate(files):
    for line in open(f):
        m = re.match(r"\s+(\S+)\s+(\S+)\s+cross ([+-][\d.]+) \| dX ([+-][\d.]+) \| dB ([+-][\d.]+)", line)
        if m:
            form, pos, cr, dx, db = m.groups()
            end, dist = ("mid", -1) if pos == "mid" else (pos[0], int(pos[1:]))
            rows.append((n, form, end, dist, float(dx) + float(db)))
d = pd.DataFrame(rows, columns=["rot", "form", "end", "dist", "desolv"])
g = d.groupby(["form", "end", "dist"]).desolv.agg(["mean", "std", "count"]).reset_index()
g["se"] = g["std"] / np.sqrt(g["count"])
interior = g[(g.end == "mid") | (g.dist >= INTERIOR)]
inner = interior.groupby("form")["mean"].agg(["mean", "std", "count"])
print(f"{len(files)} orientations; interior = positions >= {INTERIOR} from both ends (N{INTERIOR}..N8, mid, C8..C{INTERIOR})")
print("per-position mean (se) of dX + dB, kcal/mol:")
piv = g.assign(v=[f"{m:+.2f}({s:.2f})" for m, s in zip(g["mean"], g.se)]).pivot_table(
    index=["end", "dist"], columns="form", values="v", aggfunc="first")
print(piv.to_string())
print("\ninterior mean, spread of the interior positions (SD), n:")
print(inner.round(3).to_string())
out = []
for (form, end, dist), r in g.set_index(["form", "end", "dist"]).iterrows():
    if end != "mid" and dist < INTERIOR:
        out.append((form, end, dist, r["mean"], r["se"]))
for form, r in inner.iterrows():
    out.append((form, "mid", -1, r["mean"], r["std"] / np.sqrt(r["count"])))
t = pd.DataFrame(out, columns=["form", "end", "dist", "desolv", "se"])
with open(f"{D}/n181_dipole_desolv_avg.tsv", "w") as fh:
    fh.write("# GB desolvation of a side chain, helix minus coil (dX + dB of n181_dipole_desolv.py), averaged over "
             f"{len(files)} orientations of the\n# structures (Born-radii grid alignment); positions 0..{INTERIOR - 1} "
             f"from the nearer end keep their own value, positions >= {INTERIOR} from both ends\n# take the interior row "
             "(their mean). L = 24 helix, residue caps, eps_in 4, 278 K, 0.1 M; kcal/mol.\n")
    t.to_csv(fh, sep="\t", index=False, float_format="%.4f")
print(f"\nwrote {D}/n181_dipole_desolv_avg.tsv ({len(t)} rows)")
