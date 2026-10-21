"""N181 (owner: "do we need desolvation for h-bonds too? Look into the COOH interaction with first-turn amides"): the
desolvation part of the GB side chain-backbone (dipole) law (N181_DESOLV_ROT=seed: the structures randomly
rotated, to average the Born-radii grid alignment; see n181_desolv_table.py), which the dipole table leaves out (it holds the cross
term between the side chain's source charges and the backbone only).

On the builder's ideal structures (frame pipeline n181_coil_gb: L = 16 helix with residue caps; 24-residue PPII and
beta coil chains, X mid-chain), for each form of X and each chi1 well (strict rule), helix minus coil:
  cross  = GB cross term X(source charges) - backbone (N-cap..C-cap, X excluded)   [the dipole table's quantity]
  dX     = X's own GB self-energy in the structure                                  [X's desolvation by its environment]
  dB     = backbone self-energy with X's side chain minus with Ala at X            [the backbone's desolvation by X]
at eps_in 4, 278 K, kappa 0.1 / A (0.1 M). Coil: X at index 9, backbone partners over the same offsets as in the helix.
Reported per position: equal-weight mean over allowed wells (the table's rule) of cross, dX, dB and their sum, and
the free energy of cross + dX + dB. Run in the build environment (gb_pipeline/env/build.txt); out: data/n181_dipole_desolv.txt."""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import n181_coil_gb as G  # noqa: E402

D, P = G.D, G.P
import os
L = int(os.environ.get("N181_DESOLV_L", "16"))
T = 278.15
RT = 1.9872e-3 * T
EPS_W = 5321 / T + 233.76 - 0.9297 * T + 0.1417e-2 * T ** 2 - 0.8292e-6 * T ** 3
kx = list(D.KAPPA).index(0.1)
FORMS = {"D": ["D", "D0"], "E": ["E", "E0"], "K": ["K", "K0"], "H": ["H+", "H0d", "H0e"], "R": ["R"]}
POS = ({"Ncap": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 4, "mid": 8, "C3": L - 2, "C2": L - 1, "C1": L, "Ccap": L + 1}
       if not os.environ.get("N181_DESOLV_TABLE") else
       {**{f"N{d}": d for d in range(0, 9)}, "mid": L // 2, **{f"C{d}": L + 1 - d for d in range(8, -1, -1)}})
DIPOLE_FORMS = {"D": [("D", "ASP"), ("D0", "ASH")], "E": [("E", "GLU"), ("E0", "GLH")], "K": [("K", "LYS"), ("K0", "LYN")],
                "H": [("H+", "HIP"), ("H0d", "HID"), ("H0e", "HIE")], "R": [("R", "ARG")]}
P.PAIR_FORMS.update(DIPOLE_FORMS)  # the templates' forms (same as the pair builder's, H0d/H0e kept apart)
P.FORM_Z.update({"H0d": 0.0, "H0e": 0.0})


def val(e_in, w):
    return e_in + w[kx] / EPS_W


def energy_terms(fr, idx, letter, rot, lo, hi):
    """{form: (cross, dX, dB)} for X at idx with rotamer rot; partners = backbone atoms of residues lo..hi except idx."""
    atoms, fa, _ = fr.threaded({idx: (letter, rot)})
    bare = [(a[0], a[1], a[2], a[4]) for a in fr.atoms]
    bb_t = [n for n, a in enumerate(atoms) if a[0] != idx and lo <= a[0] <= hi and a[1] in D.BACKBONE]
    bb_b = [n for n, a in enumerate(bare) if a[0] != idx and lo <= a[0] <= hi and a[1] in D.BACKBONE]
    q_t = {n: next(x[3] for x in fr.atoms if (x[0], x[1]) == (atoms[n][0], atoms[n][1])) for n in bb_t}
    need = sorted({e[0] for v in fa[idx].values() for e in v} | set(bb_t))
    R = G.fast_radii(fr, atoms, {idx}, need)
    Rb = G.fast_radii(fr, bare, set(), bb_b) if bb_b else {}
    xb = np.array([atoms[n][2] for n in bb_t])
    qb = np.array([q_t[n] for n in bb_t])
    rb = np.array([R[n] for n in bb_t])
    sB_t = P._self_parts(xb, qb, rb)
    sB_b = P._self_parts(np.array([bare[n][2] for n in bb_b]), np.array([fr.atoms[n][3] for n in bb_b]),
                         np.array([Rb[n] for n in bb_b]))
    dB = val(*sB_t) - val(*sB_b)
    out = {}
    for f, ent in fa[idx].items():
        xs, qs, rs = G._src(ent, R)
        cr = P._gb_sum(xs, qs, rs, xb, qb, rb)
        dX = val(*P._self_parts(xs, qs, rs))
        out[f] = (val(*cr), dX, dB)
    return out


def wells_terms(fr, idx, letter, lo, hi):
    res = []
    for w in D.WELLS:
        rot, m, _ = fr.member(idx, letter, w)
        if m < 0:
            continue
        res.append(energy_terms(fr, idx, letter, rot, lo, hi))
    return res


def main():
    rot = int(os.environ.get("N181_DESOLV_ROT", "0"))  # orientation seed (Born-radii grid alignment)
    helix = G.ideal_frame(L, "helix", rot)
    coils = [G.ideal_frame(D.COIL_L, c, rot) for c in ("ppii", "beta")]
    lines = [f"GB dipole-law terms, helix minus coil, L = {L} (res caps), eps_in 4, 278 K, 0.1 M (kcal/mol):",
             "  mean over allowed wells of cross | dX (own) | dB (backbone's) | sum ; free energy of the sum"]
    for letter, forms in FORMS.items():
        coil_ref = {}
        for name, k in POS.items():
            idx = 1 + k
            lo, hi = 1, L + 2
            h = wells_terms(helix, idx, letter, lo, hi)
            cidx = 9
            c = [t for fr in coils for t in wells_terms(fr, cidx, letter, max(1, cidx - k), min(D.COIL_L + 2, cidx + (L + 1 - k)))]
            for f in forms:
                hv = np.array([t[f] for t in h])
                cv = np.array([t[f] for t in c])
                dm = hv.mean(0) - cv.mean(0)
                sh, sc = hv.sum(1), cv.sum(1)
                fe = lambda e: float(e.min() - RT * np.log(np.mean(np.exp(-(e - e.min()) / RT))))  # noqa: E731
                lines.append(f"  {f:4s} {name:5s} cross {dm[0]:+.3f} | dX {dm[1]:+.3f} | dB {dm[2]:+.3f} | sum {dm.sum():+.3f} ;"
                             f" FE {fe(sh) - fe(sc):+.3f}   (wells {len(hv)}/{len(cv)})")
            print("\n".join(lines[-len(forms):]), flush=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", os.environ.get("N181_DESOLV_OUT", "n181_dipole_desolv.txt")), "w").write(
        "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
