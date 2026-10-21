"""N181: GB pair rows on the builder's ideal helices with the frame pipeline (n181_coil_gb), so that helix and MD-coil
rows share one method: side chains threaded onto a fixed all-Ala structure (pdb2pqr hydrogens of the all-Ala chain;
the builder instead re-places every hydrogen per structure, and its context-dependent Ala methyl orientations move
contact desolvation by up to 0.4 kcal/mol, N181 cycle 423).
Usage: OMP_NUM_THREADS=1 .venv-build/bin/python n181_helix_run.py OUT.npz [--chi2-kmax 5] [--workers 10]
Tasks: every ordered K/R/H/D/E pair, k = 1..12, classes int / Ncap / Ccap (build_gb_pair_table.placement); (chi1,
chi2) wells for k <= chi2-kmax, chi1 wells beyond. Output: n181_coil_run's record layout plus state/cls arrays.
Resumable: tasks already in OUT are skipped.
"""
import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n181_coil_gb as G  # noqa: E402

CLASSES = ("int", "Ncap", "Ccap")
_FRAMES = {}


def frame(L):
    if L not in _FRAMES:
        _FRAMES[L] = G.ideal_frame(L, "helix")
    return _FRAMES[L]


def work(task):
    X, Y, k, cls, chi2 = task
    t0 = time.time()
    L, i1, i2 = G.P.placement(cls, k)
    fr = frame(L)
    cache = {}

    def singles(i, letter, rot):
        key = (i, letter, rot)
        if key not in cache:
            cache[key] = G.Single(fr, i, letter, rot)
        return cache[key]
    return task, G.pair_rows(fr, i1, i2, X, Y, singles, chi2), time.time() - t0


def pack(task, rows, tid):
    X, Y, k, cls, chi2 = task
    lab = np.array([(tid, r[4], r[0], r[1], r[2], r[3], r[8], r[9]) for r in rows],
                   dtype=[("frame", "i4"), ("k", "i2"), ("X", "U1"), ("Y", "U1"), ("fX", "U3"), ("fY", "U3"),
                          ("w1", "f4"), ("w2", "f4")])
    num = np.array([[r[10], r[11], r[12], r[17], r[13], r[15]] for r in rows], dtype=np.float32)
    return dict(lab=lab, num=num, W=np.array([r[14] for r in rows], np.float32),
                DW=np.array([r[16] for r in rows], np.float32), state=np.array(["helix"] * len(rows)),
                cls=np.array([cls] * len(rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--chi2-kmax", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    tasks = [(X, Y, k, cls, k <= a.chi2_kmax) for X, Y in itertools.product(G.LETTERS, G.LETTERS)
             for k in range(1, 13) for cls in CLASSES]
    tasks.sort(key=lambda t: -t[4])  # the expensive chi2 tasks first
    parts, done = [], set()
    if os.path.exists(a.out):
        old = np.load(a.out)
        parts = [{f: old[f] for f in ("lab", "num", "W", "DW", "state", "cls")}]
        done = {(str(x), str(y), int(k), str(c)) for x, y, k, c in
                zip(old["lab"]["X"], old["lab"]["Y"], old["lab"]["k"], old["cls"])}
    todo = [t for t in tasks if t[:4] not in done]
    print(f"{len(todo)} tasks to do ({len(tasks) - len(todo)} done)", flush=True)
    tid0 = int(parts[0]["lab"]["frame"].max()) + 1 if parts else 0
    with ProcessPoolExecutor(a.workers) as ex:
        for m, (task, rows, dt) in enumerate(ex.map(work, todo, chunksize=1)):
            parts.append(pack(task, rows, tid0 + m))
            if m % 20 == 0:
                print(f"{m + 1}/{len(todo)} {task}: {len(rows)} rows, {dt:.0f} s", flush=True)
            if (m + 1) % 25 == 0 or m + 1 == len(todo):
                cat = {f: np.concatenate([p[f] for p in parts]) for f in ("lab", "num", "W", "DW", "state", "cls")}
                tmp = a.out + ".tmp.npz"
                np.savez(tmp, **cat, kappa=G.D.KAPPA, num_cols=np.array(["margin_self", "margin_pair", "sb_dist",
                                                                        "d_charge", "E_in", "D_in"]))
                os.replace(tmp, a.out)
                parts = [cat]


if __name__ == "__main__":
    main()
