"""
Salt-bridge contacts in the helix: the positional exclusivity of doubly charged acid-base pairs (default law).

An acid-base pair of side chains three or four residues apart interacts through two channels with different
geometry.

- The hydrogen bond is a contact. Its strength does not depend on salt (Scholtz et al. 1993, Biochemistry 32,
  9668), and the carboxylate's two oxygens let one rotamer bond the i+3 and i+4 partners on the same side. It is the
  Table IV term (energies.get_dG_i3 / get_dG_i4).
- The Coulomb attraction is an average over where the two charges are, and salt screens it. The Lacroix et al. (1998)
  Table VI helix distance of an acid-base pair is that of a pair that sits close because it attracts (Glu-Lys i+4
  5.9 A, against 7.8 / 8.2 A for Glu-Glu / Lys-Lys). A charged side chain can sit at that distance from one partner at a
  time; charging every pair of a charge-dense block that close-range attraction at once over-counts it.

This module makes the second channel exclusive. In the helix state of a segment each charged side chain is either
free or in one salt-bridge contact with a charged partner three or four residues away (both inside the helix, caps
excluded, with a favourable Table IV cell: the candidates). A pair in contact has its Table VI (contact) Coulomb; a
candidate pair not in contact has the open geometry, the mean of the two like-pair Table VI rows of the same residues
at that separation (the geometry without the pair's own attraction). The contact state's free energy is set so that
an isolated pair, summed over contact and no contact, reproduces its Table VI Coulomb exactly, in every salt and
temperature:

    exp(-G_contact / RT) = exp(-G_TableVI / RT) - exp(-G_open / RT).

So isolated pairs and chains of pairs that never compete for a partner keep the published values; only side chains
with competing partners (charge-dense blocks such as (E4K4)n) change. The partition function over the charge states
and the contact configurations is summed exactly for every pair of sites within EXACT_RANGE residues by a transfer
matrix along the sequence; sites further apart interact through a mean field from the ionisation solver's charges,
refined once (the free energy is stationary in the charges, so the error is second order). The coil state has no
contacts.

The correction a segment's free energy takes is

    Delta(i, j) = [F_contacts(helix) - F_contacts(coil)] - [F_MF(helix) - F_MF(coil)],

F_MF being the mean-field charge-dependent free energy of the segment under the additive rule, which is exactly the
charge-dependent part of the segment free energy the energy terms give (ionisation relative to the intrinsic one,
Coulomb, side chain-macrodipole and the charge-dependent non-Coulomb terms). The two use the same model of the sites
(SegmentBatch's distances, macrodipole potentials and non-Coulomb coefficients), so a peptide without candidate pairs is
left unchanged (Delta = 0 by construction, and it is not computed). Development, checks and scores: reasoning/nodes/
N202.md, N204.md (cycles 464-472).
"""
import math

import numpy as np

EXACT_RANGE = 4
LIKE_ROW = {"D": "DD", "E": "EE", "K": "KK", "R": "RR"}
ACIDS, BASES = set("DE"), set("KRH")
R_KCAL = 0.0019872043


def has_candidates(seq: str) -> bool:
    """Whether a sequence has an acid-base pair three or four residues apart (the only place the rule acts)."""
    for i, a in enumerate(seq):
        for k in (3, 4):
            if i + k < len(seq):
                b = seq[i + k]
                if (a in ACIDS and b in BASES) or (a in BASES and b in ACIDS):
                    return True
    return False


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
                RT=R_KCAL * calc.T_kelvin)


def _open_distance(calc, A: str, B: str, k: int) -> float:
    hel = calc.table_6_helix_lacroix
    col = f"i+{k}"
    ra = hel.loc[LIKE_ROW[A], col] if A in LIKE_ROW else hel.loc["HelixRest", col]
    rb = hel.loc[LIKE_ROW[B], col] if B in LIKE_ROW else hel.loc["HelixRest", col]
    return 0.5 * (float(ra) + float(rb))


def _contacts(calc, m):
    """The candidate pairs (a, b) (site indices, a before b) with their calibrated contact free energy relative to the
    open pair (both charged), and the open geometry put into zzW."""
    where = {i: a for a, (k, i) in enumerate(m["sites"]) if k == "SC"}
    inner = set(calc.helix_indices[1:-1])
    tables = {3: calc.table_4a_lacroix, 4: calc.table_4b_lacroix}
    RT = m["RT"]
    out = {}
    for i in sorted(where):
        for k in (3, 4):
            j = i + k
            if j not in where or i not in inner or j not in inner:
                continue
            A, B = calc.seq_list[i], calc.seq_list[j]
            if not ((A in ACIDS and B in BASES) or (A in BASES and B in ACIDS)):
                continue
            if float(tables[k].loc[A, B]) >= 0:
                continue
            a, b = where[i], where[j]
            zz = m["z"][a] * m["z"][b]
            g_cur = zz * calc._electrostatic_interaction_energy(1.0, 1.0, float(calc.sidechain_sidechain_distances_hel[i, j]))
            g_open = zz * calc._electrostatic_interaction_energy(1.0, 1.0, _open_distance(calc, A, B, k))
            x = math.exp(-g_cur / RT) - math.exp(-g_open / RT)
            m["zzW"][a, b] = m["zzW"][b, a] = g_open
            out[(a, b)] = (-RT * math.log(x) - g_open) if x > 0 else math.inf
    return out


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


def _transfer(m, contacts, field, track):
    """-RT ln Z of the charge and contact states, exact for the pairs within EXACT_RANGE positions, with the long-range
    field on the charges; and <s> when track. Site states: 0 neutral, 1 charged and free, 2 charged in a contact (one
    contact per side chain, with a candidate partner three or four residues away)."""
    n, RT = m["n"], m["RT"]
    order = list(np.argsort(m["pos"], kind="stable"))
    capable = {a for key in contacts for a in key}
    states = [[0, 1, 2] if a in capable else [0, 1] for a in range(n)]
    S = [np.array([0.0 if s == 0 else 1.0 for s in st]) for st in states]
    win, logA, EX, lnZ = [], np.zeros(()), np.zeros(n), 0.0

    def drop(k):
        nonlocal logA, EX, win
        lw = logA
        mx = np.max(lw, axis=k, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tot = np.squeeze(mx, axis=k) + np.log(np.sum(np.exp(lw - mx), axis=k))
            if track:
                w = np.where(np.isfinite(lw), np.exp(lw - np.expand_dims(tot, k)), 0.0)
                EX = np.sum(EX * np.expand_dims(w, -1), axis=k)
        logA = tot
        win = win[:k] + win[k + 1:]

    for b in order:
        while win and m["pos"][win[0]] < m["pos"][b] - EXACT_RANGE:
            drop(0)
        nb, nd = len(states[b]), logA.ndim
        Sb = S[b]
        own = Sb * m["lnx"][b] - Sb * (m["hC"][b] + m["hF"][b] + field[b]) / RT
        base = np.expand_dims(logA, -1) + own.reshape([1] * nd + [nb])
        for k, a in enumerate(win):
            pair = -np.outer(S[a], Sb) * (m["C"][a, b] + m["zzW"][a, b]) / RT
            shp = [1] * (nd + 1)
            shp[k], shp[-1] = len(states[a]), nb
            base = base + pair.reshape(shp)
        if track:
            EXb = np.broadcast_to(np.expand_dims(EX, nd), EX.shape[:nd] + (nb, n)).copy()
            EXb[..., b] = Sb.reshape([1] * nd + [nb])
        else:
            EXb = None
        # b free (states 0, 1)
        free = np.array([s != 2 for s in states[b]])
        new = np.where(free.reshape([1] * nd + [nb]), base, -np.inf)
        new_ex = EXb
        # b in a contact (state 2) with one window partner a in state 1, which becomes state 2
        if 2 in states[b]:
            ib2 = states[b].index(2)
            for k, a in enumerate(win):
                if (a, b) not in contacts or not np.isfinite(contacts[(a, b)]):
                    continue
                ia1, ia2 = states[a].index(1), states[a].index(2)
                src = [slice(None)] * (nd + 1)
                src[k], src[-1] = ia1, ib2
                dst = [slice(None)] * (nd + 1)
                dst[k], dst[-1] = ia2, ib2
                add = base[tuple(src)] - contacts[(a, b)] / RT
                cur = new[tuple(dst)]
                with np.errstate(invalid="ignore"):
                    tot = np.logaddexp(cur, add)
                    if track:
                        wc = np.where(np.isfinite(cur), np.exp(cur - tot), 0.0)
                        wa = np.where(np.isfinite(add), np.exp(add - tot), 0.0)
                        new_ex = new_ex.copy() if new_ex is EXb else new_ex
                        new_ex[tuple(dst)] = new_ex[tuple(dst)] * wc[..., None] + EXb[tuple(src)] * wa[..., None]
                new[tuple(dst)] = tot
        logA, EX = new, new_ex
        win = win + [b]
        mx = float(np.max(logA))
        logA = logA - mx
        lnZ += mx
    while win:
        drop(0)
    lnZ += float(logA)
    F = -RT * lnZ + RT * float(np.sum(np.logaddexp(0.0, m["lnx"])))
    return F, (EX.copy() if track else None)


def _free_energy(m, contacts, t0, refine):
    """F of the contact model: exact within EXACT_RANGE, mean field beyond (one refinement from t0, or self-consistent
    when refine is None)."""
    d = np.abs(m["pos"][:, None] - m["pos"][None, :])
    J = np.where((d > EXACT_RANGE) & ~np.eye(m["n"], dtype=bool), m["zzW"], 0.0)
    if not J.any():
        return _transfer(m, contacts, np.zeros(m["n"]), False)[0]
    if refine == "one_step":
        _, t1 = _transfer(m, contacts, J @ np.asarray(t0, float), True)
        return _transfer(m, contacts, J @ t1, False)[0] - 0.5 * float(t1 @ J @ t1)
    t = np.asarray(t0, float)
    for it in range(200):
        _, t_new = _transfer(m, contacts, J @ t, True)
        if np.max(np.abs(t_new - t)) < 1e-6:
            t = t_new
            break
        t = 0.5 * t + 0.5 * t_new if it > 20 else t_new
    return _transfer(m, contacts, J @ t, False)[0] - 0.5 * float(t @ J @ t)


class SaltBridgeContacts:
    """Per prediction: the coil terms (computed once) and the correction of each segment."""

    def __init__(self, seq_letters: str):
        self.active = has_candidates(seq_letters)
        self._coil = None

    def segment_correction(self, calc) -> float:
        if not self.active:
            return 0.0
        if self._coil is None:
            mc = _sites(calc, False)
            F_mf_c, p_c = _mean_field(mc)
            F_ct_c = _free_energy(mc, {}, p_c, refine=None)
            self._coil = (F_mf_c, F_ct_c)
        F_mf_c, F_ct_c = self._coil
        mh = _sites(calc, True)
        F_mf_h, p_h = _mean_field(mh)
        contacts = _contacts(calc, mh)
        F_ct_h = _free_energy(mh, contacts, p_h, refine="one_step")
        return (F_ct_h - F_ct_c) - (F_mf_h - F_mf_c)
