"""N181 step 2 driver: GB pair rows on MD coil frames (n181_coil_gb), one worker per frame.

Usage: OMP_NUM_THREADS=1 .venv-build/bin/python gb_pipeline/postprocessing/n181_coil_run.py FRAMES.npz OUT.npz
           [--workers 6] [--frames a:b] [--seed 181]
Placements: per frame and separation k = 1..12, one position pair (i, i + k) drawn uniformly from those with
2 <= i, i + k <= 15 and no helical-segment residue in i..i+k (n181_frames.py's flag); none -> that k is skipped for the
frame. The same placement serves all 25 ordered letter pairs (K R H D E). Output (npz, float32 energies): one record per
(frame, k, X, Y, fX, fY, well1, well2) with margins, salt-bridge distance, charge-atom distance, E_in, W[kappa], D_in,
DW[kappa] (build_gb_pair_table conventions, eps_in 4). Frames already in OUT are skipped (resumable).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n181_coil_gb as G  # noqa: E402

KMAX = 12
LO, HI = 2, 15


def placements(helical, rng, kmax=KMAX):
    out = []
    for k in range(1, KMAX + 1):
        ok = [i for i in range(LO, HI - k + 1) if not helical[i - 1:i + k].any()]
        if ok:
            i = int(rng.choice(ok))  # drawn for every k, so the placements do not depend on kmax
            if k <= kmax:
                out.append((i, i + k))
    return out


def work(arg):
    fid, xyz, res, names, place, chi2 = arg
    t0 = time.time()
    fr = G.Frame(xyz, res, names)
    rows = G.frame_rows(fr, place, chi2=chi2)
    return fid, rows, time.time() - t0


def pack(fid, rows):
    lab = np.array([(fid, r[4], r[0], r[1], r[2], r[3], r[8], r[9]) for r in rows],
                   dtype=[("frame", "i4"), ("k", "i2"), ("X", "U1"), ("Y", "U1"), ("fX", "U3"), ("fY", "U3"),
                          ("w1", "f4"), ("w2", "f4")])
    num = np.array([[r[10], r[11], r[12], r[17], r[13], r[15]] for r in rows], dtype=np.float32)
    W = np.array([r[14] for r in rows], dtype=np.float32)
    DW = np.array([r[16] for r in rows], dtype=np.float32)
    return lab, num, W, DW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--frames-range", default=None, help="a:b slice of the frames file")
    ap.add_argument("--seed", type=int, default=181)
    ap.add_argument("--kmax", type=int, default=KMAX, help="only placements with k <= kmax")
    ap.add_argument("--chi2", action="store_true", help="(chi1, chi2) wells (n181_coil_gb.CHI2_WELLS)")
    ap.add_argument("--every", type=int, default=1, help="every n-th frame of the frames file")
    a = ap.parse_args()
    z = np.load(a.frames)
    n = len(z["coords"])
    sl = range(n) if not a.frames_range else range(*[int(x) for x in a.frames_range.split(":")])
    sl = [f for f in sl if f % a.every == 0]
    rng = np.random.default_rng(a.seed)
    place = [placements(z["helical"][f], rng, a.kmax) for f in range(n)]  # drawn for all frames: stable under slicing
    done = {}
    if os.path.exists(a.out):
        old = np.load(a.out)
        done = {k: old[k] for k in old.files}
    have = set(done["lab"]["frame"].tolist()) if done else set()
    todo = [f for f in sl if f not in have]
    print(f"{len(todo)} frames to do ({len(have)} done), workers {a.workers}", flush=True)
    parts = [done] if done else []
    args = [(f, z["coords"][f], z["atom_res"], z["atom_name"], place[f], a.chi2) for f in todo]
    with ProcessPoolExecutor(a.workers) as ex:
        for m, (fid, rows, dt) in enumerate(ex.map(work, args, chunksize=1)):
            lab, num, W, DW = pack(fid, rows)
            parts.append(dict(lab=lab, num=num, W=W, DW=DW))
            print(f"frame {fid} ({m + 1}/{len(todo)}): {len(rows)} rows, {dt:.0f} s", flush=True)
            if (m + 1) % 5 == 0 or m + 1 == len(todo):
                cat = {k: np.concatenate([p[k] for p in parts]) for k in ("lab", "num", "W", "DW")}
                tmp = a.out + ".tmp.npz"
                np.savez(tmp, **cat, kappa=G.D.KAPPA, num_cols=np.array(["margin_self", "margin_pair", "sb_dist",
                                                                        "d_charge", "E_in", "D_in"]),
                         frames_file=os.path.abspath(a.frames), job=z["job"], time_ps=z["time_ps"])
                os.replace(tmp, a.out)
                parts = [cat]


if __name__ == "__main__":
    main()
