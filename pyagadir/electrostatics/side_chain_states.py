"""
Acid-base side chains of the helix as states.

One side chain, one state. At any moment a side chain holds a set of contacts with partners three or four residues
away (partner slots -4, -3, +3, +4), or, at N3/N4, its capping motif, or nothing; ionisable side chains also carry their
charge state. Every acid-base interaction reads that state: a contact is the pair's hydrogen bond and its close-range
Coulomb term together, a capping motif (the capping box with an N3 acid, the charged staple and the Petukhov motif at N4)
excludes every contact of its side chain. The helix free energy of a segment sums over all joint states (and charge
states) by a transfer matrix along the sequence, exactly for sites within EXACT_RANGE positions; sites further apart
interact through a mean field taken at the ionisation solver's charges (the free energy is stationary in the charges,
so the error is second order: <= 0.01 kcal/mol in the cases checked, at net charge +12 and 0.015 M). Summing, not
selecting, keeps the configurational entropy of alternative networks and lets pH, salt and temperature shift which
network is populated.

Which contact sets one side chain can hold (the compatibility table; single contacts are always allowed, at most two):
- Lys or His holding two acids: only both C-terminal, (+3, +4). Wolny et al. (2017, Sci. Rep. 7, 44341, p. 9; molecular
  dynamics of EK3/EK2R1/ER3): "Lys only interacts with two Glu residues that are both C-terminal to the Lys
  (K->E(+3) & E(+4))"; the opposite-face combinations (-4, +4) and (-3, +3) "were not significantly populated, in
  agreement with the experimental finding of non-cooperativity" in an Ala-based K->E(-4) & E(+4) triplet (Iqbalsyah &
  Doig 2005, Biochemistry 44, 10449). No rotamer of an ideal helix reaches both opposite-face partners.
- Arg holding two acids: the one-face straddles (-4, +3) and (-3, +4) (Wolny: "R->E(-4) & E(+3) or R->E(-3) & E(+4) ...
  up to 10 % of the time"); i-4 and i+3 (and i-3 and i+4) lie on one face of the helix, about 20 degrees apart.
- Asp or Glu holding two Lys/His: only the straddle (-4, +3) (Wolny: "With Lys, only those using the E->K(-4) & K(+3)
  combination formed (7-10 %)"); holding two Arg: (-4, -3), (+3, +4), (-4, +3), (-4, +4), (-3, +3) (Wolny's same-side
  E->R pairs; Olson et al. 2001, Proteins 44, 123: cooperative R-E-R triads at i, i+-4 and i, i+-3); one Lys/His and
  one Arg: the straddle (-4, +3) only (no data).
- A side-chain hydrogen bond with a titrating partner that is not an acid-base pair (polar_bond: Table IV cells of Ser,
  Thr, Asn, Gln with Asp, Glu, Lys, Arg, His, or of two like-charged acids, and the Table V motifs Gln-Asp-, Glu--Asn,
  Gln-Glu-) is a contact too, and is held alone: a side chain in such a contact holds no other. Ser, Thr, Asn and Gln
  join the model as sites with one (uncharged) state. Hydrogen bonds between two side chains that do not titrate stay
  energy terms (and a Gln or Glu side chain's hydrogen bond to the main chain is not modelled here).
- Aromatic–His contacts carry their combined Table IV/V energy in the same state sum. Phe/Trp join as neutral sites;
  Tyr retains its titration state. Mixed acid/aromatic contacts use the existing compatibility table, including His
  (+3,+4) doubles; aromatic centers hold one contact. This extends the table's coarse geometry rule, rather than
  introducing a newly measured compatibility rule for aromatic rings. Table IV excludes caps; Table V retains them.
Rotamer populations do not appear: every separation is reachable from the populated chi1 wells, so a side chain's
direction is a matter of its contacts, and the cost of adopting a contact is inside the contact's calibrated free energy.

Free energy of a contact (candidate pair three or four apart, both inside the helix, caps excluded). The isolated pair's
free energy is taken as measured: the bond H when both partners are charged, the bond Hn when one is neutral (the
neutral partner's hydrogen bond; Scholtz et al. 1993, Biochemistry 32, 9668: "the ionization of one side chain increases
the strength of the hydrogen bond"; Fossat et al. 2023, ChemPhysChem 24, e202200746: (E4K4)3 with every Glu neutral,
net +12, is as helical as with every Glu charged), plus, when both are charged, the screened Coulomb term at the pair's Lacroix et al.
(1998) Table VI helix distance. The contact state's weight is set so that the isolated pair, summed over contact and no
contact, reproduces that value exactly in every charge state, salt and temperature:
    exp(-G_contact / RT) = exp(-G_iso / RT) - exp(-G_open / RT),
G_open being the pair without its contact: the Coulomb term at the open geometry, the mean of the two like-pair Table VI
rows of the same residues (the geometry without the pair's own attraction; Glu-Lys i+4 5.9 A in Table VI against 7.8
and 8.2 A for Glu-Glu and Lys-Lys), 0 when a partner is neutral. Where Table VI does not hold the pair closer than the
open geometry (Asp/Glu(i)-His(i+3/i+4), Lys/Arg(i)-Asp(i+3)), the open reference is the Table VI term itself and the
contact carries the bond alone. H and Hn are re-derived from isolated pairs (params/acid_base_pairs.tsv: Smith & Scholtz
1998; Iqbalsyah & Doig 2005; Huyghues-Despointes et al. 1993); acid-base classes without an entry there (His pairs) use
the Table IV cell for both. Both are free energies at 0-5 C and temperature-independent (no measured enthalpy; the
Coulomb term keeps water's eps(T)). A capping motif's state is calibrated the same
way, K_motif = exp(-g/RT) - 1 beside the free state, so an isolated motif reproduces its free energy g exactly.

The coil state holds no contacts. The correction a segment's free energy takes is
    Delta(i, j) = [F(helix) - F(coil)] - [F_MF(helix) - F_MF(coil)] - sum over motif sites of g_motif(neutral),
F_MF being the mean-field charge-dependent free energy of the segment as the energy terms give it (with the acid-base
pairs left to this module and the capping motifs additive, see EnergyCalculator.get_dG_i3): the
module replaces that part of the segment's free energy, and the charge-independent part of each motif it carries.
"""
import itertools
import math
from functools import lru_cache
from importlib.resources import files

import numpy as np
import pandas as pd

EXACT_RANGE = 4
LIKE_ROW = {"D": "DD", "E": "EE", "K": "KK", "R": "RR"}
ACIDS, BASES = set("DE"), set("KRH")
POLAR = set("STNQ")  # side chains that hydrogen-bond but do not titrate here
AROMATIC = set("FYW")
R_KCAL = 0.0019872043
SLOT = {-4: 0, -3: 1, 3: 2, 4: 3}
NBITS, CBITS = 0b0011, 0b1100


@lru_cache(maxsize=1)
def acid_base_pairs() -> dict:
    """Re-derived pair strengths: (acid, base, separation, first) -> (H, Hn) in kcal/mol at 0 C, first being "acid",
    "base" (which residue comes first in the sequence) or "any"."""
    t = pd.read_csv(files("pyagadir.params").joinpath("acid_base_pairs.tsv"), sep="\t")
    return {(r.acid, r.base, int(r.separation), r.first): (float(r.H_kcal_mol), float(r.Hn_kcal_mol))
            for r in t.itertuples()}


def pair_strength(A: str, B: str, k: int):
    """(H, Hn) of the acid-base pair A(i)-B(i+k) from acid_base_pairs(), or None for a class without an entry."""
    if not ((A in ACIDS and B in BASES) or (A in BASES and B in ACIDS)):
        return None
    acid, base = (A, B) if A in ACIDS else (B, A)
    p = acid_base_pairs()
    return p.get((acid, base, k, "acid" if A in ACIDS else "base"), p.get((acid, base, k, "any")))


def owns_pair(A: str, B: str, k: int, cell: float) -> bool:
    """Whether this module carries the acid-base pair A(i)-B(i+k) (Table IV cell `cell`, kcal/mol): a re-derived class,
    or a favourable published cell. The energy terms then leave the pair out."""
    return pair_strength(A, B, k) is not None or cell < 0


@lru_cache(maxsize=1)
def _table_iv() -> dict:
    """Lacroix 1998 supplementary Table IV (kcal/mol, i+3 and i+4), read once: {k: DataFrame}."""
    return {k: pd.read_csv(files("pyagadir.params").joinpath(f"table_4{'a' if k == 3 else 'b'}_lacroix.tsv"),
                           index_col="AA", sep="\t").astype(float) / 100.0 for k in (3, 4)}


def table_v(A: str, B: str, k: int):
    """The Table V hydrogen-bond motif of A(i)-B(i+4): (value in kcal/mol, which residue's charge it needs: 0 for A,
    1 for B), or None. Gln-Asp-, Glu--Asn and Gln-Glu- (Lacroix 1998 supplementary Table V; Gln-Asp- re-derived,
    energies.TABLE_V_GLN_ASP)."""
    if k != 4:
        return None
    from pyagadir.energies import TABLE_V_GLN_ASP
    return {("Q", "D"): (TABLE_V_GLN_ASP, 1), ("E", "N"): (-0.5, 0), ("Q", "E"): (-0.1, 1)}.get((A, B))


def polar_bond(A: str, B: str, k: int, cell: float):
    """The side-chain hydrogen bond of a pair A(i)-B(i+k) that is not an acid-base pair, at least one of whose residues
    titrates and both of which hydrogen-bond (S T N Q D E K R H): bond[s_A][s_B] in kcal/mol by charge state, the
    Table IV cell (in the states that are not both charged, when both titrate) plus a Table V motif when its residue is
    charged; or None where no state bonds (a contact needs a favourable bond)."""
    charged = ACIDS | BASES
    if (A in ACIDS and B in BASES) or (A in BASES and B in ACIDS):
        return None
    if not ({A, B} & charged and A in POLAR | charged and B in POLAR | charged):
        return None
    bond = np.full((2, 2), cell)
    if A in charged and B in charged:
        bond[1, 1] = 0.0
    tv = table_v(A, B, k)
    if tv is not None:
        v, who = tv
        if who == 0:
            bond[1, :] += v
        else:
            bond[:, 1] += v
    return bond if (bond < 0).any() else None


def owns_polar(A: str, B: str, k: int, cell: float) -> bool:
    """Whether this module carries the side-chain hydrogen bond of the pair A(i)-B(i+k) (polar_bond): the energy terms
    then leave its Table IV cell and Table V motif out."""
    return polar_bond(A, B, k, cell) is not None


def is_aromatic_his(A: str, B: str) -> bool:
    return (A in AROMATIC and B == "H") or (A == "H" and B in AROMATIC)


def aromatic_bond(calc, i: int, k: int):
    """Combined Table IV/V aromatic–His energy in each binary charge state. Table IV excludes caps; the forward
    i+4 Table V increment also reaches caps and keeps its C1/C-cap scaling. Tyr's charged state is retained."""
    j = i + k
    A, B = calc.seq_list[i], calc.seq_list[j]
    if not is_aromatic_his(A, B):
        return None
    table = calc.table_4a_lacroix if k == 3 else calc.table_4b_lacroix
    cell = float(table.loc[A, B]) / 100.0 if i != calc.ncap_idx and j != calc.ccap_idx else 0.0
    bond = np.full((2, 2), cell)
    if "Y" in (A, B):
        bond[1, 1] = 0.0  # original Table IV "not both charged" convention
    if k == 4 and A in AROMATIC and B == "H":
        from pyagadir.energies import TABLE_V_AROMATIC_HIS
        bond[:, 1] += TABLE_V_AROMATIC_HIS * (1.0 if calc.ccap_idx - j <= 1 else 1.0 / 3.0)
    return bond if (bond < 0).any() else None


def has_candidates(seq: str) -> bool:
    """Whether a sequence has a candidate acid-base, polar hydrogen-bond or aromatic–His contact at i+3/i+4."""
    t4 = _table_iv()
    for i, a in enumerate(seq):
        for k in (3, 4):
            if i + k < len(seq):
                b = seq[i + k]
                if is_aromatic_his(a, b):
                    return True
                if (a in ACIDS and b in BASES) or (a in BASES and b in ACIDS):
                    return True
                if a in t4[k].index and b in t4[k].columns and owns_polar(a, b, k, float(t4[k].loc[a, b])):
                    return True
    return False


def allowed_doubles(center: str, partners: dict) -> set:
    """Two-contact masks a side chain of residue `center` may hold, partners = {slot offset: residue}."""
    out = set()
    for o1, o2 in itertools.combinations(sorted(partners), 2):
        pair = (o1, o2)
        if center in ("K", "H"):
            ok = pair == (3, 4)
        elif center == "R":
            ok = pair in ((-4, 3), (-3, 4))
        elif center in ACIDS:
            kinds = {"K" if partners[o] in ("K", "H") else "R" for o in pair}
            if kinds == {"R"}:
                ok = pair in ((-4, -3), (3, 4), (-4, 3), (-4, 4), (-3, 3))
            else:
                ok = pair == (-4, 3)
        else:
            ok = False
        if ok:
            out.add((1 << SLOT[o1]) | (1 << SLOT[o2]))
    return out


def _sites(calc, helix: bool):
    """The ionisation solver's model of one state of a segment: sites, charges z, ln of the intrinsic charged odds,
    sequence positions, the field term h_F = z phi, the non-Coulomb coefficients (h_C, C), and the Coulomb couplings
    zzW = z_a z_b W_ab (SegmentBatch's distances; pairs at 40 A or more do not interact)."""
    sites = [(k, i) for k, i in calc._ionizable_sites()[0]]
    n = len(sites)
    z = np.array([calc._site_full_charge(k, i) for k, i in sites], float)
    pka = np.array([calc._site_base_pka(k, i) for k, i in sites], float)
    lnx = np.where(z > 0, pka - calc.pH, calc.pH - pka) * math.log(10)
    L = len(calc.seq_list)
    pos = np.array([0 if k == "Nterm" else L - 1 if k == "Cterm" else i for k, i in sites], float)
    W = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            r = calc._site_pair_distance(sites[a][0], sites[a][1], sites[b][0], sites[b][1], helix)
            if r < 40.0:
                W[a, b] = W[b, a] = calc._electrostatic_interaction_energy(1.0, 1.0, r)
    phi = np.zeros(n)
    hC = np.zeros(n)
    C = np.zeros((n, n))
    if helix:
        for a, (k, i) in enumerate(sites):
            if k == "Nterm":
                phi[a] = calc._electrostatic_interaction_energy(qi=calc.mu_helix, qj=1.0,
                                                                r=calc.terminal_macrodipole_distance_nterm)
            elif k == "Cterm":
                phi[a] = calc._electrostatic_interaction_energy(qi=-calc.mu_helix, qj=1.0,
                                                                r=calc.terminal_macrodipole_distance_cterm)
            else:
                phi[a] = calc.sidechain_dipole_potential[i]
        ca, cab = calc._nonCoulomb_charge_coefficients()
        where = {i: a for a, (k, i) in enumerate(sites) if k == "SC"}
        for i, v in ca.items():
            hC[where[i]] += v
        for (i, j), v in cab.items():
            C[where[i], where[j]] += v
            C[where[j], where[i]] += v
    return dict(sites=sites, n=n, z=z, lnx=lnx, pos=pos, hF=z * phi, hC=hC, C=C, zzW=np.outer(z, z) * W,
                RT=R_KCAL * calc.T_kelvin, res=[calc.seq_list[i] if k == "SC" else "" for k, i in sites],
                charged=np.ones(n, bool), contacts={}, motif={})


def _add_polar_site(m, i, aa):
    """A non-titrating contact partner (S T N Q F W) joins with one neutral state, no field or Coulomb coupling.
    Returns its site index."""
    n = m["n"]
    for key in ("z", "lnx", "hF", "hC"):
        m[key] = np.append(m[key], 0.0)
    m["pos"] = np.append(m["pos"], float(i))
    for key in ("C", "zzW"):
        m[key] = np.pad(m[key], ((0, 1), (0, 1)))
    m["charged"] = np.append(m["charged"], False)
    m["sites"] = m["sites"] + [("POL", i)]
    m["res"] = m["res"] + [aa]
    m["n"] = n + 1
    return n


def _mean_field(m, tol=1e-12, max_sweeps=10000):
    """Self-consistent mean-field charged fractions and free energy (relative to the intrinsic ionisation) of the
    Ising model E(s) = s.(h_F + h_C) + 1/2 s (zzW + C) s, by Gauss-Seidel sweeps."""
    RT, lnx = m["RT"], m["lnx"]
    J = m["zzW"] + m["C"]
    h = m["hF"] + m["hC"]
    p = 1.0 / (1.0 + np.exp(-lnx))
    for _ in range(max_sweeps):
        mx = 0.0
        for a in range(m["n"]):
            new = 1.0 / (1.0 + math.exp(-(lnx[a] - (h[a] + J[a] @ p) / RT)))
            mx = max(mx, abs(new - p[a]))
            p[a] = new
        if mx < tol:
            break
    psi = h + J @ p
    F = float(np.sum(-RT * (np.logaddexp(0.0, lnx - psi / RT) - np.logaddexp(0.0, lnx)))) - 0.5 * float(p @ J @ p)
    return F, p


def _distance_of(calc, A: str, B: str, k: int) -> float:
    """The open distance of an acid-base pair: the mean of the two like-pair Table VI helix rows (HelixRest for His)."""
    hel = calc.table_6_helix_lacroix
    col = f"i+{k}"
    ra = hel.loc[LIKE_ROW[A], col] if A in LIKE_ROW else hel.loc["HelixRest", col]
    rb = hel.loc[LIKE_ROW[B], col] if B in LIKE_ROW else hel.loc["HelixRest", col]
    return 0.5 * (float(ra) + float(rb))


def _add_contacts(calc, m):
    """The candidate pairs (a, b) (site indices, a before b) with their bonds H (both charged), Hn (one neutral) and the
    Coulomb couplings at contact (Table VI) and open; the open geometry goes into zzW."""
    where = {i: a for a, (k, i) in enumerate(m["sites"]) if k == "SC"}
    inner = set(calc.helix_indices[1:-1])
    tables = {3: calc.table_4a_lacroix, 4: calc.table_4b_lacroix}
    for i in sorted(where):
        for k in (3, 4):
            j = i + k
            if j not in where or i not in inner or j not in inner:
                continue
            A, B = calc.seq_list[i], calc.seq_list[j]
            if not ((A in ACIDS and B in BASES) or (A in BASES and B in ACIDS)):
                continue
            cell = float(tables[k].loc[A, B]) / 100.0
            if not owns_pair(A, B, k, cell):
                continue
            H, Hn = pair_strength(A, B, k) or (cell, cell)
            a, b = where[i], where[j]
            r_vi = float(calc.sidechain_sidechain_distances_hel[i, j])
            r_open = max(_distance_of(calc, A, B, k), r_vi)
            Wc = calc._electrostatic_interaction_energy(1.0, 1.0, r_vi)
            Wo = calc._electrostatic_interaction_energy(1.0, 1.0, r_open)
            m["zzW"][a, b] = m["zzW"][b, a] = m["z"][a] * m["z"][b] * Wo
            m["contacts"][(a, b)] = dict(kind="ab", H=H, Hn=Hn, Wc=Wc)
    # side-chain hydrogen bonds with a titrating partner (polar_bond): the bond alone is the contact; the pair's Coulomb
    # term, where both are charged, stays at its Table VI distance
    for i in sorted(inner):
        for k in (3, 4):
            j = i + k
            if j not in inner:
                continue
            A, B = calc.seq_list[i], calc.seq_list[j]
            if A not in tables[k].index or B not in tables[k].columns:
                continue
            bond = polar_bond(A, B, k, float(tables[k].loc[A, B]) / 100.0)
            if bond is None:
                continue
            a = where.get(i)
            if a is None:
                a = where[i] = _add_polar_site(m, i, A)
            b = where.get(j)
            if b is None:
                b = where[j] = _add_polar_site(m, j, B)
            m["contacts"][(a, b)] = dict(kind="polar", bond=bond)

    # One combined ring contact carries Table IV and V, so neither can coexist independently with a competing bond.
    # Unlike the hydrogen-bond loop, include caps here: aromatic_bond retains each term's own cap convention.
    for i in calc.helix_indices:
        for k in (3, 4):
            j = i + k
            if j > calc.ccap_idx:
                continue
            bond = aromatic_bond(calc, i, k)
            if bond is None:
                continue
            for pos in (i, j):
                if pos not in where:
                    where[pos] = _add_polar_site(m, pos, calc.seq_list[pos])
            m["contacts"][(where[i], where[j])] = dict(kind="polar", bond=bond, aromatic=True)


def _contact_energy(m, a, b):
    """e[s_a][s_b]: the contact's free energy relative to the open pair, calibrated on the isolated pair."""
    ct, RT = m["contacts"][(a, b)], m["RT"]
    e = np.zeros((2, 2))
    if ct["kind"] == "polar":  # exp(-e/RT) = exp(-bond/RT) - 1 beside the open pair: the isolated bond is reproduced
        for sa in (0, 1):
            for sb in (0, 1):
                g = ct["bond"][sa, sb]
                e[sa, sb] = -RT * math.log(math.expm1(-g / RT)) if g < 0 else math.inf
        return e
    zz = m["z"][a] * m["z"][b]
    for sa in (0, 1):
        for sb in (0, 1):
            tt = sa * sb
            G_open = tt * m["zzW"][a, b]
            G_iso = (ct["H"] if tt else ct["Hn"]) + tt * zz * ct["Wc"]
            x = math.exp(-G_iso / RT) - math.exp(-G_open / RT)
            e[sa, sb] = (-RT * math.log(x) - G_open) if x > 0 else math.inf
    return e


def _motif_energies(calc):
    """{residue index: (g neutral, g charged)} of the segment's capping motifs as the energy terms give them (kcal/mol):
    the N3 acid's capping box and Nc-3 column gain, the charged staple, the Petukhov motif. The N3 acid's motif also
    carries the capping-box half of the hydrophobic staple (get_dG_staple: x1 behind an N, D, S or T N-cap with an
    N3 acid, x0.5 otherwise): the box positions N' and N4, so the staple keeps its full value only while the N3 acid caps."""
    out = {}
    ion = calc.modified_seq_ionization_hel
    for st in calc._capping_motif_sites():
        i = st["idx"]
        saved = float(ion[i])
        sign = -1.0 if calc.seq_list[i] in ACIDS else 1.0
        staple_half = 0.5 * float(calc.get_dG_staple()) if st["kind"] == "n3" and calc.Ncap_AA in ("N", "D", "S", "T") else 0.0
        g = []
        for s in (0.0, 1.0):
            ion[i] = s * sign
            if st["kind"] == "n3":
                g.append(calc._capping_n3_energy(st) + staple_half)
            elif st["kind"] == "cstaple":
                g.append(float(calc.get_dG_charged_staple()))
            else:
                g.append(float(calc.get_dG_petukhov_motif()))
        ion[i] = saved
        out[i] = (g[0], g[1])
    return out


def _add_motifs(calc, m):
    """The capping motifs as states of their side chains: their charge dependence moves out of h_C into the state."""
    where = {i: a for a, (k, i) in enumerate(m["sites"]) if k == "SC"}
    for i, (g0, g1) in _motif_energies(calc).items():
        if i in where:
            a = where[i]
            m["hC"][a] -= g1 - g0
            m["motif"][a] = (g0, g1)


def _site_masks(m):
    """Allowed contact masks per site: 0, every single candidate contact, the table's doubles."""
    cand = {a: {} for a in range(m["n"])}
    compatible = {a: {} for a in range(m["n"])}
    for (a, b), ct in m["contacts"].items():
        k = int(round(m["pos"][b] - m["pos"][a]))
        cand[a][k] = m["res"][b]
        cand[b][-k] = m["res"][a]
        if ct["kind"] == "ab" or ct.get("aromatic"):
            compatible[a][k] = m["res"][b]
            compatible[b][-k] = m["res"][a]
    masks = []
    for a in range(m["n"]):
        ms = {0} | {1 << SLOT[o] for o in cand[a]}
        ms |= allowed_doubles(m["res"][a], compatible[a])  # other polar hydrogen bonds are held alone
        masks.append(sorted(ms))
    return masks


def _motif_logw(g, states, RT):
    """Log-weights of a capping-motif site's states: the motif state carries K = exp(-g(s)/RT) - 1 beside the free
    state (an isolated motif reproduces exp(-g(s)/RT)); where g(s) >= 0 the motif is no state and its energy applies
    to the non-motif states, as the energy terms apply it."""
    out = np.zeros(len(states))
    for i, (s_, mk, mo) in enumerate(states):
        gs = g[1] if s_ else g[0]
        if mo:
            out[i] = math.log(math.expm1(-gs / RT)) if gs < 0 else -np.inf
        elif gs >= 0:
            out[i] = -gs / RT
    return out


def _lse(x, axis):
    mx = np.max(x, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(mx, axis=axis) + np.log(np.sum(np.exp(x - mx), axis=axis))


def _transfer(m, field, track=False):
    """-RT ln Z over charge states x contact sets (x motif state), exact within EXACT_RANGE positions, with the
    long-range field on the charges; and <s> when track. States per site: (s, contact mask, motif). A contact (a, b),
    b = a + k, sets a's +k slot and b's -k slot."""
    n, RT = m["n"], m["RT"]
    order = list(np.argsort(m["pos"], kind="stable"))
    masks = _site_masks(m)
    sts = []
    for a in range(n):
        st = [(s, mk, 0) for s in ((0, 1) if m["charged"][a] else (0,)) for mk in masks[a]]
        if a in m["motif"]:
            st += [(s, 0, 1) for s in (0, 1)]
        sts.append(st)
    S = [np.array([x[0] for x in st], float) for st in sts]
    pos_of = [{x: i for i, x in enumerate(st)} for st in sts]
    relab = {}
    for a in range(n):
        for cbit in (1 << SLOT[3], 1 << SLOT[4]):
            src, tgt = [], []
            for i_s, (s_, mk, mo) in enumerate(sts[a]):
                if not mo and not mk & cbit and (s_, mk | cbit, 0) in pos_of[a]:
                    src.append(i_s)
                    tgt.append(pos_of[a][(s_, mk | cbit, 0)])
            relab[(a, cbit)] = (np.array(src, int), np.array(tgt, int))
    ec = {key: _contact_energy(m, *key) for key in m["contacts"]}
    win, logA, EX, lnZ = [], np.zeros(()), np.zeros(n), 0.0

    def remove(k):
        nonlocal logA, EX, win
        tot = _lse(logA, k)
        if track:
            with np.errstate(invalid="ignore"):
                w = np.where(np.isfinite(logA), np.exp(logA - np.expand_dims(tot, k)), 0.0)
            EX = np.sum(EX * np.expand_dims(w, -1), axis=k)
        logA = tot
        win = win[:k] + win[k + 1:]

    for b in order:
        while win and m["pos"][win[0]] < m["pos"][b] - EXACT_RANGE:
            remove(0)
        nb, nd = len(sts[b]), logA.ndim
        own = S[b] * m["lnx"][b] - S[b] * (m["hC"][b] + m["hF"][b] + field[b]) / RT
        if b in m["motif"]:
            own = own + _motif_logw(m["motif"][b], sts[b], RT)
        base = np.expand_dims(logA, -1) + own.reshape([1] * nd + [nb])
        for k, a in enumerate(win):
            pair = -np.outer(S[a], S[b]) * (m["C"][a, b] + m["zzW"][a, b]) / RT
            shp = [1] * (nd + 1)
            shp[k], shp[-1] = len(sts[a]), nb
            base = base + pair.reshape(shp)
        if track:
            EXb = np.broadcast_to(np.expand_dims(EX, nd), EX.shape[:nd] + (nb, n)).copy()
            EXb[..., b] = S[b].reshape([1] * nd + [nb])
        # the choices Q of N-side contacts give disjoint sets of b's states (each fixes b's N-side slots)
        P = [(k, a) for k, a in enumerate(win) if (a, b) in m["contacts"]]
        new = np.full(base.shape, -np.inf)
        newEX = np.zeros(EXb.shape) if track else None
        for r in range(len(P) + 1):
            for Q in itertools.combinations(P, r):
                nbits = 0
                for _, a in Q:
                    nbits |= 1 << SLOT[-int(round(m["pos"][b] - m["pos"][a]))]
                bidx = np.array([i for i, x in enumerate(sts[b])
                                 if (x[2] == 0 and (x[1] & NBITS) == nbits and (x[1] & CBITS) == 0)
                                 or (x[2] == 1 and nbits == 0)], int)
                if not len(bidx):
                    continue
                lw = base[..., bidx]
                ex = EXb[..., bidx, :] if track else None
                for k, a in Q:
                    sa = np.array([x[0] for x in sts[a]], int)
                    sb = np.array([sts[b][i][0] for i in bidx], int)
                    shp = [1] * (nd + 1)
                    shp[k], shp[-1] = len(sts[a]), len(bidx)
                    lw = lw + (-ec[(a, b)][sa][:, sb] / RT).reshape(shp)
                    src, tgt = relab[(a, 1 << SLOT[int(round(m["pos"][b] - m["pos"][a]))])]
                    L_ = np.moveaxis(lw, k, 0)
                    T_ = np.full(L_.shape, -np.inf)
                    T_[tgt] = L_[src]
                    lw = np.moveaxis(T_, 0, k)
                    if track:
                        X_ = np.moveaxis(ex, k, 0)
                        TX_ = np.zeros(X_.shape)
                        TX_[tgt] = X_[src]
                        ex = np.moveaxis(TX_, 0, k)
                new[..., bidx] = lw
                if track:
                    newEX[..., bidx, :] = ex
        logA, EX = new, newEX
        win = win + [b]
        mx = float(np.max(logA))
        logA = logA - mx
        lnZ += mx
    while win:
        remove(0)
    lnZ += float(logA)
    F = -RT * lnZ + RT * float(np.sum(np.logaddexp(0.0, m["lnx"][m["charged"]])))
    return F, (EX.copy() if track else None)


def _long_range(m):
    d = np.abs(m["pos"][:, None] - m["pos"][None, :])
    return np.where((d > EXACT_RANGE) & ~np.eye(m["n"], dtype=bool), m["zzW"], 0.0)


def _free_energy_at(m, t0):
    """F with the long-range mean field at the charges t0 (no refinement)."""
    J = _long_range(m)
    t0 = np.asarray(t0, float)
    return _transfer(m, J @ t0)[0] - 0.5 * float(t0 @ J @ t0)


def _free_energy_selfconsistent(m, tol=1e-7, max_iter=200):
    """F with a self-consistent long-range mean field (used once per prediction, for the coil)."""
    J = _long_range(m)
    if not J.any():
        return _transfer(m, np.zeros(m["n"]))[0]
    t = np.full(m["n"], 0.5)
    for it in range(max_iter):
        _, t_new = _transfer(m, J @ t, track=True)
        if np.max(np.abs(t_new - t)) < tol:
            t = t_new
            break
        t = 0.5 * t + 0.5 * t_new if it > 20 else t_new
    return _transfer(m, J @ t)[0] - 0.5 * float(t @ J @ t)


class SideChainStates:
    """Per prediction: the coil terms (computed once) and the correction of each segment."""

    def __init__(self, seq_letters: str):
        self.active = has_candidates(seq_letters)
        self._coil = None

    def segment_correction(self, calc) -> float:
        if not self.active:
            return 0.0
        if self._coil is None:
            mc = _sites(calc, False)
            self._coil = (_mean_field(mc)[0], _free_energy_selfconsistent(mc))
        F_mf_c, F_c = self._coil
        mh = _sites(calc, True)
        F_mf_h, p_h = _mean_field(mh)
        _add_contacts(calc, mh)
        _add_motifs(calc, mh)
        F_h = _free_energy_at(mh, np.pad(p_h, (0, mh["n"] - len(p_h))))  # side chains without charge: 0
        g0 = sum(g[0] for g in mh["motif"].values())
        return (F_h - F_c) - (F_mf_h - F_mf_c) - g0
