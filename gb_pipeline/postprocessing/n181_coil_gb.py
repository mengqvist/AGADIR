"""N181 step 2: GB side chain-side chain energies on MD coil frames (the coil reference of the GB pair law).

Each frame is a poly-Ala backbone from the cA16 coil MD (gb_pipeline/md/scripts/n181_frames.py: all peptide atoms, A).
Two charged side chains X (residue i) and Y (residue i + k) are threaded onto it by tools/build_gb_pair_table.py's
rule, and their GB cross term and desolvation are computed exactly as there, for every protonation-form combination:
  - side chains: PeptideBuilder ideal geometry for the rotamer, placed by superposing the residue's N, CA, C (the
    builder places them from the same three atoms); backbone atoms N, H, CA, HA, C, O stay at the MD positions;
  - rotamers: each chi1 well's nearest allowed member (build_gb_dipole_table.choose's candidate order, strict 2.8 A
    rule for the side chain beyond C-beta against every heavy atom of the other residues), chosen with the partner as
    Ala; well combinations whose side chains come closer than 2.8 A are flagged (margin_pair < 0);
  - charges and radii: AMBER ff14SB from pdb2pqr, run once per (letter, rotamer, form) on an A-X-A template; Ala and
    the caps (acetyl, amide) take AMBER library values (build_gb_dipole_table.ACE / NHE); source charges are each
    form's charges minus Ala's backbone charges on its backbone atoms;
  - Born radii: build_gb_dipole_table.born_radii (Grycuk r^-6 integral, SES 1.4 A, 0.3 A grid) on the atoms within
    CROP of the needed atoms (the far solute's r^-6 share is negligible; checked in n181_coil_gb_check.py); all
    forms of a residue share one set of radii (the union of their atoms), as in the builders;
  - desolvation (contact < 8 A): each residue's self-energy with the partner's side chain present minus with the
    partner as Ala. Beyond 8 A the cross term uses each residue's radii with the partner as Ala.
Output rows in the forms raw format of build_gb_pair_table (state "coil", cls "coil", conf "md") plus frame, i and
the charge-atom distance (Lacroix 1998: Asp CG, Glu CD, Arg CZ, His CE1, Lys NZ) for the validation.
Run in the build environment (gb_pipeline/env/build.txt) (PeptideBuilder, pdb2pqr, scipy).
"""
import itertools
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helix_builds"))
import build_gb_dipole_table as D  # noqa: E402
import build_gb_pair_table as P  # noqa: E402

LETTERS = "KRHDE"
# chi2 wells for the chi2 extension (N181): Lys/Arg/Glu chi2 staggered; Asp's carboxylate (period 180) and His's ring
CHI2_WELLS = {"K": (180.0, -60.0, 60.0), "R": (180.0, -60.0, 60.0), "E": (180.0, -60.0, 60.0),
              "D": (180.0, 60.0, -60.0), "H": (180.0, 90.0, -90.0)}
CHARGE_ATOM = {"D": "CG", "E": "CD", "R": "CZ", "H": "CE1", "K": "NZ"}
CROP = float(os.environ.get("N181_CROP", "1000"))  # A; cropping biases desolvation (N181), so off by default
FAST = not os.environ.get("N181_SLOW")
# desolvation reference: "drop" = the builder's (partner's side chain beyond CB, HB2, HB3 removed from the pair structure),
# "ala" = partner as Ala (the cached single-residue structure; one radii computation per combination instead of three)
DESOLV_REF = os.environ.get("N181_DESOLV_REF", "drop")  # FrameGrid radii (default) or build_gb_dipole_table.born_radii per structure
ACE_MAP = {"CAY": "CH3", "HY1": "HH3", "HY2": "HH3", "HY3": "HH3", "CY": "C", "OY": "O"}
NHE_MAP = {"NT": "N", "HT1": "H", "HT2": "H"}


def _kabsch(A, B):
    """R, t with R @ a + t ~ b for the rows of A onto B."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


@lru_cache(maxsize=None)
def _heavy_template(letter, rot):
    """X's atoms of an ideal A-X-A chain: {name: xyz} (heavy atoms, PeptideBuilder)."""
    st = P.structure2(["A", letter, "A"], [(-75.0, 145.0)] * 3, {1: rot})
    return {a.get_id(): a.coord.astype(float) for a in st.get_atoms() if a.get_parent().get_id()[1] == 2}


@lru_cache(maxsize=None)
def _form_template(letter, rot):
    """{form: [(name, q, radius, xyz)]} of X in the A-X-A template, from pdb2pqr (AMBER)."""
    st = P.structure2(["A", letter, "A"], [(-75.0, 145.0)] * 3, {1: rot})
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f, rn in P.PAIR_FORMS[letter]:
            atoms = P._pqr(st, {2: rn}, tmp)
            x = [(a[0], a[4], a[5], a[3]) for a in atoms if a[2] == 1]
            if abs(sum(a[1] for a in x) - P.FORM_Z[f]) > 0.01:
                raise RuntimeError(f"{letter} {f}: net charge")
            out[f] = x
    return out


@lru_cache(maxsize=None)
def _ala_params():
    """AMBER Ala {name: (q, radius)} (interior residue of a pdb2pqr'd Ala chain) and the cap tables."""
    st = P.structure2(list("AAAAA"), [(-75.0, 145.0)] * 5, {})
    with tempfile.TemporaryDirectory() as tmp:
        atoms = P._pqr(st, {}, tmp)
    ala = {a[0]: (a[4], a[5]) for a in atoms if a[2] == 2}
    return ala, D.ACE, D.NHE


class Frame:
    """One MD frame: heavy atoms for sterics, all atoms with AMBER charges and radii for the energies. pqr: instead,
    the atoms of a pdb2pqr'd structure [(name, resname, residue, xyz, q, radius)] (the builders' own all-Ala chains)."""

    def __init__(self, xyz=None, res=None, names=None, pqr=None):
        ala, ace, nhe = _ala_params()
        if pqr is not None:
            self.atoms = [(a[2], a[0], np.array(a[3], float), a[4], a[5]) for a in pqr]
            self.nres = max(a[0] for a in self.atoms)
            self.bb = {(a[0], a[1]): a[2] for a in self.atoms}
            self.ala_bb = {n: ala[n][0] for n in D.BACKBONE}
            self.grid = FrameGrid(np.array([a[2] for a in self.atoms]), np.array([a[4] for a in self.atoms]))
            return
        self.nres = int(res.max())
        self.atoms = []  # (res, name, xyz, q, radius)
        for r, n, x in zip(res, names, xyz):
            r, n = int(r), str(n)
            if n in ACE_MAP:
                q, rad = ace[ACE_MAP[n]]
                self.atoms.append((0, n, x.astype(float), q, rad))
            elif n in NHE_MAP:
                q, rad = nhe[NHE_MAP[n]]
                self.atoms.append((self.nres + 1, n, x.astype(float), q, rad))
            else:
                q, rad = ala[n]
                self.atoms.append((r, n, x.astype(float), q, rad))
        self.bb = {(a[0], a[1]): a[2] for a in self.atoms}
        self.ala_bb = {n: ala[n][0] for n in D.BACKBONE}
        self.grid = FrameGrid(np.array([a[2] for a in self.atoms]), np.array([a[4] for a in self.atoms]))

    def _place(self, i, letter, rot):
        T = _heavy_template(letter, rot)
        R, t = _kabsch(np.array([T["N"], T["CA"], T["C"]]),
                       np.array([self.bb[(i, "N")], self.bb[(i, "CA")], self.bb[(i, "C")]]))
        return R, t

    def side_heavy(self, i, letter, rot):
        """{name: xyz} of X's side-chain heavy atoms (CB and beyond) on residue i."""
        R, t = self._place(i, letter, rot)
        return {n: R @ x + t for n, x in _heavy_template(letter, rot).items() if n not in D.BACKBONE}

    def margin(self, i, sc):
        """Strict-rule margin: smallest distance from the side chain beyond C-beta to any heavy atom of another
        residue (caps included), minus 2.8 A."""
        far = np.array([x for n, x in sc.items() if n != "CB"])
        if len(far) == 0:
            return 99.0
        other = np.array([a[2] for a in self.atoms if a[0] != i and not a[1].startswith("H")])
        return float(np.min(np.linalg.norm(far[:, None] - other[None], axis=-1))) - D.CLASH

    @lru_cache(maxsize=None)
    def member(self, i, letter, well):
        """(rot, margin, side heavy atoms) of the well's nearest allowed member on residue i (partner = Ala). well: a
        chi1 well (the builders' rule), or (chi1, chi2) wells (CHI2_WELLS): chi1 moved by up to 20 and chi2 by up to 30
        degrees (cost 1 per 10 and per 15 degrees), the dihedrals after chi2 canonical, Glu's chi3 scanned as before."""
        if isinstance(well, tuple):
            return self._member_chi2(i, letter, well)
        si, offs = D.SCAN[letter]
        cands = [(abs(d1) / 10.0 + abs(ds) / 30.0, d1, ds) for d1 in (0, 10, -10, 20, -20) for ds in (0,) + tuple(offs)]
        cands.sort(key=lambda c: c[0])
        first = None
        for _, d1, ds in cands:
            rot = D.rotamer(letter, well + d1, ds)
            sc = self.side_heavy(i, letter, rot)
            m = self.margin(i, sc)
            if first is None:
                first = (rot, m, sc)
            if m >= 0.0:
                return rot, m, sc
        return first

    def _member_chi2(self, i, letter, well):
        w1, w2 = well
        e3 = D.SCAN["E"][1] if letter == "E" else ()
        cands = [(abs(d1) / 10.0 + abs(d2) / 15.0 + abs(d3) / 30.0, d1, d2, d3) for d1 in (0, 10, -10, 20, -20)
                 for d2 in (0, 15, -15, 30, -30) for d3 in (0,) + tuple(e3)]
        cands.sort(key=lambda c: c[0])
        first = None
        for _, d1, d2, d3 in cands:
            tail = list(D.CANON[letter])
            tail[0] = ((w2 + d2 + 180.0) % 360.0) - 180.0
            if letter == "E":
                tail[1] = ((tail[1] + d3 + 180.0) % 360.0) - 180.0
            rot = (w1 + d1,) + tuple(tail)
            sc = self.side_heavy(i, letter, rot)
            m = self.margin(i, sc)
            if first is None:
                first = (rot, m, sc)
            if m >= 0.0:
                return rot, m, sc
        return first

    def threaded(self, subs):
        """Union atom list with side chains threaded: subs = {residue: (letter, rot)}. Returns (atoms, {residue:
        {form: [(union index, own xyz, source charge)]}}): atoms = [(res, name, xyz, radius)] holds every atom of every
        form once (first occurrence by name, for the shared Born radii); each form's charges sit at its own positions
        (its hydrogens can differ, e.g. LYN's NZ hydrogens), as in the builders."""
        atoms, forms_at = [], {}
        for a in self.atoms:
            if a[0] in subs:
                continue  # the substituted residue is added below with its own atoms and charges
            atoms.append((a[0], a[1], a[2], a[4]))
        for i, (letter, rot) in subs.items():
            R, t = self._place(i, letter, rot)
            ft = _form_template(letter, rot)
            idx = {}
            forms_at[i] = {}
            for f, xs in ft.items():
                forms_at[i][f] = []
                for n, q, rad, x in xs:
                    pos = self.bb[(i, n)] if n in D.BACKBONE else R @ x + t
                    if n not in idx:
                        idx[n] = len(atoms)
                        atoms.append((i, n, pos, rad))
                    forms_at[i][f].append((idx[n], pos, q - self.ala_bb[n] if n in D.BACKBONE else q))
        return atoms, forms_at, None


def radii(atoms, need, drop=(), around=None):
    """Born radii of the atoms `need`, on the atoms within CROP of the atoms `around` (default: need), minus `drop`."""
    xyz = np.array([a[2] for a in atoms])
    rad = np.array([a[3] for a in atoms])
    nx = xyz[need if around is None else around]
    d = np.min(np.linalg.norm(xyz[:, None] - nx[None], axis=-1), axis=1)
    drop = set(drop)
    sel = [j for j in np.where(d < CROP)[0] if j not in drop]
    mp = {j: n for n, j in enumerate(sel)}
    R = D.born_radii(xyz[sel], rad[sel], [mp[j] for j in need])
    return {j: R[mp[j]] for j in need}


def _src(entries, R):
    """Source arrays (xyz, q, Born radius) of one form: entries = [(union index, own xyz, source charge)]."""
    s = [e for e in entries if e[2] != 0.0]
    return np.array([e[1] for e in s]), np.array([e[2] for e in s]), np.array([R[e[0]] for e in s])


class FrameGrid:
    """The Born-radii grid of one frame (build_gb_dipole_table.born_radii's method and constants). The bare frame's
    solvent-excluded surface (SES) is computed once; a structure with residues substituted differs only near the
    changed atoms, so its SES is the bare one with a box around the changes recomputed (SAS from every atom reaching
    the box, distance transform over the box plus PROBE + 2 voxels, so the in-box SES is exact). One fixed grid per
    frame (origin from the bare atoms with EXTRA room for side chains): all structures of a frame share its alignment,
    so with/without-partner differences carry no grid-alignment noise (the builders' per-structure grids carry
    SD ~0.05 kcal/mol in desolvation, n181_grid_noise.txt)."""
    EXTRA = 8.0

    def __init__(self, xyz, rad):
        from scipy.ndimage import distance_transform_edt
        self.edt = distance_transform_edt
        h = D.H_GRID
        pad = D.RB + 1.0 + self.EXTRA
        self.lo = xyz.min(0) - pad
        self.n = np.ceil((xyz.max(0) + pad - self.lo) / h).astype(int) + 1
        self.sas = self._mark(xyz, rad, np.zeros(3, int), self.n)
        self.ses = self.sas & (self.edt(self.sas, sampling=h) > D.PROBE)

    def _mark(self, xyz, rad, b0, b1):
        h = D.H_GRID
        out = np.zeros(b1 - b0, bool)
        for c, r in zip(xyz, rad):
            rr = r + D.PROBE
            i0 = np.maximum(np.floor((c - rr - self.lo) / h).astype(int), b0)
            i1 = np.minimum(np.ceil((c + rr - self.lo) / h).astype(int) + 1, b1)
            if np.any(i1 <= i0):
                continue
            gx, gy, gz = (self.lo[d] + h * np.arange(i0[d], i1[d]) for d in range(3))
            d2 = (gx[:, None, None] - c[0]) ** 2 + (gy[None, :, None] - c[1]) ** 2 + (gz[None, None, :] - c[2]) ** 2
            out[i0[0] - b0[0]:i1[0] - b0[0], i0[1] - b0[1]:i1[1] - b0[1], i0[2] - b0[2]:i1[2] - b0[2]] |= d2 < rr * rr
        return out

    def ses_for(self, xyz, rad, changed):
        """SES of the structure (xyz, rad), which differs from the bare frame only at the points `changed`."""
        h = D.H_GRID
        m = 2.0 + 2.0 * D.PROBE + 2.0 * h  # largest radius (< 2 A) + probe, + the probe's reach, + 2 voxels
        b0 = np.maximum(np.floor((changed.min(0) - m - self.lo) / h).astype(int), 0)
        b1 = np.minimum(np.ceil((changed.max(0) + m - self.lo) / h).astype(int) + 1, self.n)
        e = int(np.ceil((D.PROBE + 2 * h) / h))
        c0, c1 = np.maximum(b0 - e, 0), np.minimum(b1 + e, self.n)
        lim0, lim1 = self.lo + h * c0, self.lo + h * (c1 - 1)
        reach = rad + D.PROBE
        on = np.all((xyz + reach[:, None] >= lim0) & (xyz - reach[:, None] <= lim1), axis=1)
        sas = self._mark(xyz[on], rad[on], c0, c1)
        ses_box = sas & (self.edt(sas, sampling=h) > D.PROBE)
        out = self.ses.copy()
        s0, s1 = b0 - c0, b1 - c0
        out[b0[0]:b1[0], b0[1]:b1[1], b0[2]:b1[2]] = ses_box[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]
        return out

    def radii(self, ses, xyz, rad, need):
        """build_gb_dipole_table.born_radii's integrals on a given SES grid."""
        h, RB = D.H_GRID, D.RB
        n, lo = self.n, self.lo
        w = int(np.ceil(RB / h))
        solute = lo + h * np.argwhere(ses)
        out = {}
        for i in need:
            c = xyz[i]
            ic = np.round((c - lo) / h).astype(int)
            i0, i1 = np.maximum(ic - w, 0), np.minimum(ic + w + 1, n)
            sub = ses[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
            gx, gy, gz = (lo[d] + h * np.arange(i0[d], i1[d]) - c[d] for d in range(3))
            d2 = gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2
            near = (~sub) & (d2 < RB * RB) & (d2 > 1e-9)
            s_in = float(np.sum(h ** 3 / d2[near] ** 3))
            dd = np.sum((solute - c) ** 2, axis=1)
            s_far = float(np.sum(h ** 3 / dd[dd >= RB * RB] ** 3))
            inv = 3.0 / (4.0 * np.pi) * (s_in + 4.0 * np.pi / (3.0 * RB ** 3) - s_far)
            out[i] = max(min(inv ** (-1.0 / 3.0), 50.0) if inv > 0 else 50.0, rad[i])
        return out


def fast_radii(frame, atoms, subs, need, drop=()):
    """Born radii of `need` in a threaded structure (atoms from Frame.threaded with substitutions `subs`), minus the
    atom indices `drop`, on the frame's grid."""
    keep = [j for j in range(len(atoms)) if j not in set(drop)]
    xyz = np.array([atoms[j][2] for j in keep])
    rad = np.array([atoms[j][3] for j in keep])
    mp = {j: n for n, j in enumerate(keep)}
    changed = [a[2] for a in frame.atoms if a[0] in subs] + [atoms[j][2] for j in range(len(atoms)) if atoms[j][0] in subs]
    ses = frame.grid.ses_for(xyz, rad, np.array(changed)) if changed else frame.grid.ses  # no substitution: the bare SES
    R = frame.grid.radii(ses, xyz, rad, [mp[j] for j in need])
    return {j: R[mp[j]] for j in need}


class Single:
    """X alone on residue i (partner positions Ala): atoms, per-form indices and charges, union radii."""

    def __init__(self, frame, i, letter, rot):
        self.atoms, fa, _ = frame.threaded({i: (letter, rot)})
        self.forms = fa[i]
        need = sorted({e[0] for v in self.forms.values() for e in v})
        self.R = fast_radii(frame, self.atoms, {i}, need) if FAST else radii(self.atoms, need)
        self.i = i

    def src(self, f):
        return _src(self.forms[f], self.R)


def wells_of(letter, chi2):
    return [(a, b) for a in D.WELLS for b in CHI2_WELLS[letter]] if chi2 else list(D.WELLS)


def well_code(w):
    """A (chi1, chi2) well as one float (chi1 + chi2 / 1000; unique for the wells used), a chi1 well as itself."""
    return w[0] + w[1] / 1000.0 if isinstance(w, tuple) else w


def pair_rows(frame, i, j, X, Y, singles, chi2=False):
    """Rows for every well combination and form combination of X(i), Y(j) on one frame (chi2: (chi1, chi2) wells)."""
    k = j - i
    out = []
    for w1, w2 in itertools.product(wells_of(X, chi2), wells_of(Y, chi2)):
        r1, m1, sc1 = frame.member(i, X, w1)
        r2, m2, sc2 = frame.member(j, Y, w2)
        f1 = np.array([x for n, x in sc1.items() if n != "CB"])
        f2 = np.array([x for n, x in sc2.items() if n != "CB"])
        contact = float(np.min(np.linalg.norm(f1[:, None] - f2[None], axis=-1)))
        sb = 99.0
        if X in P.CATION_N and Y in P.CARBOXYL_O or Y in P.CATION_N and X in P.CARBOXYL_O:
            cat, an = (sc1, sc2) if X in P.CATION_N else (sc2, sc1)
            cl, al = (X, Y) if X in P.CATION_N else (Y, X)
            cen = np.mean([an[o] for o in P.CARBOXYL_O[al]], axis=0)
            sb = float(min(np.linalg.norm(cat[n] - cen) for n in P.CATION_N[cl]))
        dist = float(np.linalg.norm(sc1[CHARGE_ATOM[X]] - sc2[CHARGE_ATOM[Y]]))
        s1 = singles(i, X, r1)
        s2 = singles(j, Y, r2)
        near = contact < P.DESOLV_RANGE
        if near:
            atoms, fa, _ = frame.threaded({i: (X, r1), j: (Y, r2)})
            need = sorted({e[0] for v in list(fa[i].values()) + list(fa[j].values()) for e in v})
            R = fast_radii(frame, atoms, {i, j}, need) if FAST else radii(atoms, need)
            dself = {}
            # the builder's desolvation: the same atoms with the partner's side chain (beyond CB, HB2, HB3) removed
            for res, other, sgl in ((i, j, s1), (j, i, s2)):
                if DESOLV_REF == "ala":  # partner as Ala: the cached single-residue structure
                    for f in fa[res]:
                        a_in, a_w = P._self_parts(*_src(fa[res][f], R))
                        b_in, b_w = P._self_parts(*sgl.src(f))
                        dself[(res, f)] = (a_in - b_in, a_w - b_w)
                    continue
                own = sorted({e[0] for v in fa[res].values() for e in v})
                side = [n for n, a in enumerate(atoms) if a[0] == other and a[1] not in D.BACKBONE | {"CB", "HB2", "HB3"}]
                Rb = (fast_radii(frame, atoms, {i, j}, own, drop=side) if FAST
                      else radii(atoms, own, drop=side, around=need))
                for f in fa[res]:
                    a_in, a_w = P._self_parts(*_src(fa[res][f], R))
                    b_in, b_w = P._self_parts(*_src(fa[res][f], Rb))
                    dself[(res, f)] = (a_in - b_in, a_w - b_w)
        for fx in s1.forms:
            for fy in s2.forms:
                if near:
                    e_in, w = P._gb_sum(*_src(fa[i][fx], R), *_src(fa[j][fy], R))
                    d_in = dself[(i, fx)][0] + dself[(j, fy)][0]
                    dw = dself[(i, fx)][1] + dself[(j, fy)][1]
                else:
                    e_in, w = P._gb_sum(*s1.src(fx), *s2.src(fy))
                    d_in, dw = 0.0, np.zeros(len(D.KAPPA))
                out.append((X, Y, fx, fy, k, "coil", "coil", "md", well_code(w1), well_code(w2), round(min(m1, m2), 3),
                            round(contact - D.CLASH, 3), round(sb, 3), e_in, w, d_in, dw, round(dist, 3)))
    return out


def frame_rows(frame, placements, letters=LETTERS, chi2=False):
    """placements: [(i, j)]; every ordered letter pair at each."""
    cache = {}

    def singles(i, letter, rot):
        key = (i, letter, rot)
        if key not in cache:
            cache[key] = Single(frame, i, letter, rot)
        return cache[key]
    rows = []
    for i, j in placements:
        for X, Y in itertools.product(letters, letters):
            rows += pair_rows(frame, i, j, X, Y, singles, chi2)
    return rows


def random_rotation(seed):
    """A uniformly random rotation matrix (seed 0: the identity)."""
    if not seed:
        return np.eye(3)
    q = np.random.default_rng(seed).normal(size=4)
    a, b, c, d = q / np.linalg.norm(q)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def ideal_frame(L, conf, rot_seed=0):
    """A builder chain (build_gb_dipole_table.layout: all Ala, residue caps and flanks) as a Frame, with pdb2pqr's
    hydrogens and every atom of the structure (as the builders compute their radii). rot_seed: the structure rotated
    by a random rotation (a different Born-radii grid alignment; seed 0 = as built)."""
    import tempfile
    seq, ang = D.layout(L, "A", 0, conf)
    st = P.structure2(seq, ang, {})
    with tempfile.TemporaryDirectory() as tmp:
        atoms = P._pqr(st, {}, tmp)
    Rm = random_rotation(rot_seed)
    return Frame(pqr=[(a[0], a[1], a[2], Rm @ np.asarray(a[3], float), a[4], a[5]) for a in atoms])


def fmt(r):
    return ("\t".join(str(x) for x in r[:13]) + f"\t{r[13]:.5f}\t" + "\t".join(f"{v:.5f}" for v in r[14])
            + f"\t{r[15]:.5f}\t" + "\t".join(f"{v:.5f}" for v in r[16]) + f"\t{r[17]}")


HEAD = P.RAW_HEAD_FORMS + ["d_charge"]
