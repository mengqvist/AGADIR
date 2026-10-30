"""Acid-base side chains as states (electrostatics/side_chain_states.py): a peptide without acid-base pairs three or
four apart is unchanged by the module; the pair table and the compatibility table are what the module states;
a contact and a capping motif are calibrated so that the isolated case reproduces its free energy; the transfer matrix
sums the states exactly (brute-force enumeration); and an isolated pair without a re-derived strength keeps its
published value."""
import contextlib
import io
import itertools
import math

import numpy as np
import pytest

from pyagadir import models
from pyagadir.electrostatics import side_chain_states as S
from pyagadir.energies import EnergyCalculator


def _helix(seq, nc, cc, T, M, pH):
    with contextlib.redirect_stdout(io.StringIO()):
        return models.AGADIR(method="1s", T=T, M=M, pH=pH).predict(seq, ncap=nc, ccap=cc).percent_helix


def _without_states(monkeypatch):
    """The model with the module switched off: every acid-base pair on its Table IV cell in every charge state and its
    Coulomb term at the Table VI distance, summed additively (the rule before 2026-10-26, for an isolated pair)."""
    from pyagadir import energies
    monkeypatch.setattr(energies, "owns_pair", lambda *a: False)
    monkeypatch.setattr(energies, "owns_polar", lambda *a: False)
    monkeypatch.setattr(S.SideChainStates, "segment_correction", lambda self, calc: 0.0)


@pytest.mark.parametrize("seq,nc,cc,pH", [
    ("AAKAAAAKAAAAKAAY", "Ac", "Am", 7.0),   # Lys only
    ("AKAAKAKAAKAKAAKA", "Ac", "Am", 7.0),   # like charges only
    ("AEAAAAEAAAAEAAAA", "Ac", "Am", 4.5),   # Glu only
    ("AEAAAAAKAAAAAAAY", "Ac", "Am", 7.0),   # an acid-base pair six apart
])
def test_peptides_without_candidate_pairs_are_unchanged(seq, nc, cc, pH, monkeypatch):
    with_states = _helix(seq, nc, cc, 5.0, 0.05, pH)
    _without_states(monkeypatch)
    assert with_states == _helix(seq, nc, cc, 5.0, 0.05, pH)


def test_pair_table():
    p = S.acid_base_pairs()
    assert set(p) >= {("E", "K", 3, "any"), ("E", "K", 4, "acid"), ("E", "K", 4, "base"), ("D", "K", 3, "any"),
                      ("D", "K", 4, "any"), ("E", "R", 3, "any"), ("E", "R", 4, "any"), ("D", "R", 3, "any"),
                      ("D", "R", 4, "any")}
    assert all(H <= 0 and Hn <= 0 for H, Hn in p.values())
    assert S.pair_strength("E", "K", 4) == p[("E", "K", 4, "acid")]       # Glu first
    assert S.pair_strength("K", "E", 4) == p[("E", "K", 4, "base")]       # Lys first
    assert S.pair_strength("K", "E", 3) == S.pair_strength("E", "K", 3) == p[("E", "K", 3, "any")]
    assert S.pair_strength("H", "E", 4) is None and S.pair_strength("E", "E", 4) is None
    assert S.owns_pair("H", "E", 3, -0.18) and not S.owns_pair("H", "D", 3, 0.0)


def test_compatibility_table():
    m = lambda *o: sum(1 << S.SLOT[x] for x in o)  # noqa: E731
    lys = S.allowed_doubles("K", {-4: "E", -3: "E", 3: "E", 4: "E"})
    assert lys == {m(3, 4)}                                   # Lys: two acids only both C-terminal
    arg = S.allowed_doubles("R", {-4: "E", -3: "E", 3: "E", 4: "E"})
    assert arg == {m(-4, 3), m(-3, 4)}                        # Arg: the one-face straddles
    glu_k = S.allowed_doubles("E", {-4: "K", -3: "K", 3: "K", 4: "K"})
    assert glu_k == {m(-4, 3)}                                # Glu with two Lys: the straddle only
    glu_r = S.allowed_doubles("E", {-4: "R", -3: "R", 3: "R", 4: "R"})
    assert glu_r == {m(-4, -3), m(3, 4), m(-4, 3), m(-4, 4), m(-3, 3)}
    assert S.allowed_doubles("E", {-4: "K", 3: "R"}) == {m(-4, 3)}
    assert S.allowed_doubles("E", {3: "R", 4: "K"}) == set()  # one Lys and one Arg on one side: not allowed


def _segment(seq, pH=7.0, T=5.0, I=0.05, ncap=None, ccap="Am"):
    c = EnergyCalculator(seq=seq, i=0, j=len(seq) + (ccap is not None), pH=pH, T=T, ionic_strength=I, ncap=ncap,
                         ccap=ccap)
    m = S._sites(c, True)
    S._add_contacts(c, m)
    S._add_motifs(c, m)
    return c, m


@pytest.mark.parametrize("seq", ["AAAEAAAKAAAAY", "AAAKAAAEAAAAY", "AAADAAKAAAAY", "AAARAAEAAAAY"])
def test_an_isolated_contact_reproduces_its_pair(seq):
    """Contact and no contact of an isolated pair sum to exp(-G_iso/RT) in every charge state: G_iso = H + Coulomb at
    the Table VI distance when both are charged, Hn otherwise (relative to the pair's open Coulomb term)."""
    c, m = _segment(seq)
    assert len(m["contacts"]) == 1
    (a, b), ct = next(iter(m["contacts"].items()))
    e, RT, zz = S._contact_energy(m, a, b), m["RT"], m["z"][a] * m["z"][b]
    for sa, sb in itertools.product((0, 1), repeat=2):
        tt = sa * sb
        G_open = tt * m["zzW"][a, b]
        G_iso = (ct["H"] if tt else ct["Hn"]) + tt * zz * ct["Wc"]
        tot = math.exp(-G_open / RT) * (1.0 + math.exp(-e[sa, sb] / RT))
        assert tot == pytest.approx(math.exp(-G_iso / RT), rel=1e-12)


def test_a_motif_state_reproduces_its_free_energy():
    states = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
    for g in ((-0.05, -0.85), (0.05, -0.85), (-0.3, -0.3)):
        lw = S._motif_logw(g, states, 0.55)
        for s in (0, 1):
            free, motif = lw[s], lw[2 + s]
            assert np.logaddexp(free, motif) == pytest.approx(-g[s] / 0.55, abs=1e-12)


def _brute(m, field):
    """-RT ln Z by enumeration: charge states x contact sets allowed by the table x motif states, the pairs within
    EXACT_RANGE exact, the long-range field on the charges."""
    n, RT = m["n"], m["RT"]
    masks = S._site_masks(m)
    keys = list(m["contacts"])
    ec = {k: S._contact_energy(m, *k) for k in keys}
    near = [(a, b) for a in range(n) for b in range(a + 1, n) if abs(m["pos"][a] - m["pos"][b]) <= S.EXACT_RANGE]
    terms = []
    for s in itertools.product(*[(0, 1) if m["charged"][a] else (0,) for a in range(n)]):
        base = sum(s[a] * (m["lnx"][a] * RT - m["hC"][a] - m["hF"][a] - field[a]) for a in range(n))
        base -= sum(s[a] * s[b] * (m["C"][a, b] + m["zzW"][a, b]) for a, b in near)
        mot = list(m["motif"])
        for on in itertools.product((0, 1), repeat=len(mot)):
            onset = {a for a, x in zip(mot, on) if x}
            gm = 0.0
            for a in mot:
                g = m["motif"][a][s[a]]
                gm += (RT * math.log(math.expm1(-g / RT)) if g < 0 else -math.inf) if a in onset else (-g if g >= 0 else 0.0)
            if not math.isfinite(gm):
                continue
            for r in range(len(keys) + 1):
                for Q in itertools.combinations(keys, r):
                    mk = [0] * n
                    if any(a in onset or b in onset for a, b in Q):
                        continue
                    for a, b in Q:
                        k = int(round(m["pos"][b] - m["pos"][a]))
                        mk[a] |= 1 << S.SLOT[k]
                        mk[b] |= 1 << S.SLOT[-k]
                    if any(mk[a] not in masks[a] for a in range(n)):
                        continue
                    e = sum(ec[(a, b)][s[a]][s[b]] for a, b in Q)
                    if math.isfinite(e):
                        terms.append((base + gm - e) / RT)
    t = np.array(terms)
    lnZ = float(np.max(t) + np.log(np.sum(np.exp(t - np.max(t)))))
    return -RT * lnZ + RT * float(np.sum(np.logaddexp(0.0, m["lnx"][m["charged"]])))


@pytest.mark.parametrize("seed", range(10))
def test_transfer_matrix_against_enumeration(seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        seq = "S" + "".join(rng.choice(list("EKRDAAH"), 9))
        c, m = _segment(seq, pH=float(rng.uniform(2, 11)), T=float(rng.uniform(0, 40)), I=float(rng.uniform(0.01, 1)))
        if m["n"] <= 9 and m["contacts"]:
            break
    field = rng.normal(0, 0.2, m["n"])
    assert S._transfer(m, field)[0] == pytest.approx(_brute(m, field), abs=1e-10)


@pytest.mark.parametrize("X,Y,k", [("H", "E", 3), ("H", "E", 4)])
def test_an_isolated_pair_without_a_rederived_strength_keeps_its_value(X, Y, k, monkeypatch):
    """A class without an entry in the pair table (His-first Glu pairs) keeps its published cell for both bonds and its
    Table VI Coulomb term: the result equals the additive treatment of the isolated pair up to the exact short-range
    correlation of the charges."""
    seq = "AAQAA" + X + "A" * (k - 1) + Y + "AAQAAY"
    conds = ((5.0, 0.005), (7.0, 0.005), (3.0, 0.1))
    states = [_helix(seq, "Ac", "Am", 0.0, M, pH) for pH, M in conds]
    _without_states(monkeypatch)
    for (pH, M), v in zip(conds, states):
        assert abs(v - _helix(seq, "Ac", "Am", 0.0, M, pH)) < 0.05


@pytest.mark.parametrize("seed", range(10))
def test_transfer_matrix_against_enumeration_with_polar_contacts(seed):
    """Side-chain hydrogen bonds with titrating partners (Gln, Asn, Ser, Thr as uncharged sites; like-charged pairs):
    the transfer matrix still equals brute-force enumeration."""
    rng = np.random.default_rng(100 + seed)
    for _ in range(50):
        seq = "S" + "".join(rng.choice(list("EKRDQNSTAA"), 9))
        c, m = _segment(seq, pH=float(rng.uniform(2, 11)), T=float(rng.uniform(0, 40)), I=float(rng.uniform(0.01, 1)))
        if m["n"] <= 9 and any(ct["kind"] == "polar" for ct in m["contacts"].values()):
            break
    assert any(ct["kind"] == "polar" for ct in m["contacts"].values())
    field = rng.normal(0, 0.2, m["n"])
    assert S._transfer(m, field)[0] == pytest.approx(_brute(m, field), abs=1e-10)


@pytest.mark.parametrize("X,Y,k,pH", [("Q", "E", 4, 7.0), ("E", "N", 4, 7.0), ("E", "N", 4, 3.0), ("R", "Q", 3, 7.0),
                                      ("Q", "D", 4, 7.0), ("E", "E", 4, 2.5)])
def test_an_isolated_polar_bond_keeps_its_value(X, Y, k, pH, monkeypatch):
    """An isolated side-chain hydrogen bond with a titrating partner reproduces its Table IV cell plus its Table V motif
    (the additive treatment) up to the exact short-range correlation of the charges."""
    seq = "AAAA" + X + "A" * (k - 1) + Y + "AAAAY"
    v = _helix(seq, "Ac", "Am", 0.0, 0.05, pH)
    _without_states(monkeypatch)
    assert abs(v - _helix(seq, "Ac", "Am", 0.0, 0.05, pH)) < 0.05


def test_a_carboxyl_pair_half_charged_gains_from_the_exact_sum(monkeypatch):
    """Glu-Glu i+4 at pH 4.5 (both about half charged): the bond forms when one is neutral, so it steers which one is
    protonated. The exact state sum is lower in free energy than the mean-field (1 - p p) treatment (a mean-field free
    energy is an upper bound), so the helix is a little more stable."""
    seq = "AAAAEAAAEAAAAY"
    v = _helix(seq, "Ac", "Am", 0.0, 0.05, 4.5)
    _without_states(monkeypatch)
    base = _helix(seq, "Ac", "Am", 0.0, 0.05, 4.5)
    assert base < v < base + 1.0


def test_a_polar_contact_is_held_alone():
    """A Glu with a Lys four on and a Gln four back: each contact alone, never both (one side chain, one state)."""
    c, m = _segment("AAAQAAAEAAAKAAAY")
    kinds = sorted(ct["kind"] for ct in m["contacts"].values())
    assert kinds == ["ab", "polar"]
    glu = next(a for a, (k, i) in enumerate(m["sites"]) if k == "SC" and c.seq_list[i] == "E")
    masks = S._site_masks(m)[glu]
    assert masks == sorted({0, 1 << S.SLOT[-4], 1 << S.SLOT[4]})


def test_the_staple_half_rides_on_the_capping_box():
    """The hydrophobic staple keeps its capping-box half only while the N3 acid caps: that half is part of the N3
    motif's free energy."""
    c = EnergyCalculator(seq="ALSAAELAAAKAAAAAY", i=3, j=14, pH=7.0, T=5.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    assert c.Ncap_AA == "S" and c.N3_AA == "E" and c.get_dG_staple() < 0
    (i, (g0, g1)), = S._motif_energies(c).items()
    st = next(x for x in c._capping_motif_sites() if x["kind"] == "n3")
    ion = c.modified_seq_ionization_hel
    saved = float(ion[i])
    ion[i] = 0.0
    try:
        assert g0 == pytest.approx(c._capping_n3_energy(st) + 0.5 * c.get_dG_staple(), abs=1e-12)
    finally:
        ion[i] = saved


@pytest.mark.parametrize("aromatic", ["F", "Y", "W"])
@pytest.mark.parametrize("reverse,k", [(False, 3), (False, 4), (True, 3), (True, 4)])
@pytest.mark.parametrize("cap", [False, True])
def test_aromatic_contact_reproduces_isolated_binary_pair(aromatic, reverse, k, cap):
    """The combined IV/V contact reproduces each charge-state free energy, including Tyr and the cap gate;
    its two contributions never survive as separate additive energies."""
    A, B = ("H", aromatic) if reverse else (aromatic, "H")
    seq = "AAA" + A + "A" * (k - 1) + B + "AAA"
    i, j = 4, 4 + k  # acetyl modification occupies index 0
    c = EnergyCalculator(seq=seq, i=1, j=j if cap else len(seq), pH=6., T=5., ionic_strength=.05,
                         ncap="Ac", ccap="Am")
    m = S._sites(c, True)
    S._add_contacts(c, m)
    contacts = [(edge, ct) for edge, ct in m['contacts'].items() if ct.get('aromatic')]
    if (reverse and k == 3) or (cap and (reverse or k == 3)):
        assert not contacts  # Table IV excludes the cap; only forward i+4 has a Table V increment
        return
    assert len(contacts) == 1
    (a, b), ct = contacts[0]
    base = 0. if cap else (-.3 if reverse else (-.25 if k == 3 else (-.6 if aromatic == 'Y' else -.4)))
    increment = 0. if reverse or k == 3 else (-1. if c.ccap_idx - j <= 1 else -1. / 3.)
    e = S._contact_energy(m, a, b)
    for sa, sb in itertools.product([0, 1], repeat=2):
        expected = (0. if aromatic == 'Y' and sa and sb else base) + (increment if sb else 0.)
        assert -m['RT'] * np.logaddexp(0., -e[sa, sb] / m['RT']) == pytest.approx(expected, abs=1e-12)
    assert (c.get_dG_i3() if k == 3 else c.get_dG_i4())[i] == 0.


def _ring_competition_model(permitted=False):
    """Fixed-charge model: use two known favorable contacts to isolate contact entropy from ionization."""
    rt = .0019872043 * 273.15
    return dict(n=3, RT=rt, pos=np.array([0., 3., 4.] if permitted else [0., 4., 7.]),
                res=['H', 'E', 'F'] if permitted else ['W', 'H', 'E'],
                charged=np.zeros(3, bool), z=np.zeros(3), lnx=np.zeros(3), hF=np.zeros(3), hC=np.zeros(3),
                C=np.zeros((3, 3)), zzW=np.zeros((3, 3)), motif={},
                contacts=({(0, 1):dict(kind='ab', H=-.18, Hn=-.18, Wc=0.),
                           (0, 2):dict(kind='polar', aromatic=True, bond=np.full((2, 2), -.4))} if permitted else
                          {(0, 1):dict(kind='polar', aromatic=True, bond=np.full((2, 2), -.4)),
                           (1, 2):dict(kind='ab', H=-.18, Hn=-.18, Wc=0.)}))


def test_aromatic_and_acid_contacts_sum_exclusive_alternatives():
    m = _ring_competition_model()
    hmask = (1 << S.SLOT[-4]) | (1 << S.SLOT[3])
    assert hmask not in S._site_masks(m)[1]
    ka, kb = math.expm1(.4 / m['RT']), math.expm1(.18 / m['RT'])
    # Free + W-H + H-E, with no W-H/H-E double: sum all alternatives, never select only the best.
    expected = -m['RT'] * math.log(1 + ka + kb)
    f = S._transfer(m, np.zeros(3))[0]
    assert f == pytest.approx(expected, abs=1e-12)
    assert -.58 < f < -.4


def test_aromatic_mixed_double_allowed_by_table_is_retained():
    m = _ring_competition_model(permitted=True)
    assert (1 << S.SLOT[3]) | (1 << S.SLOT[4]) in S._site_masks(m)[0]
    assert S._transfer(m, np.zeros(3))[0] == pytest.approx(-.58, abs=1e-12)


@pytest.mark.parametrize("seq", ["AAAWAAAHAAEAAAA", "AAAYAAAHAAEAAAA", "AAAHAAEFAAAA"])
def test_transfer_with_aromatic_contacts_matches_enumeration(seq):
    c, m = _segment(seq, pH=6.5)
    assert len(m['contacts']) == 2
    assert any(ct.get('aromatic') for ct in m['contacts'].values())
    his = next(a for a, aa in enumerate(m['res']) if aa == 'H')
    double = ((1 << S.SLOT[3]) | (1 << S.SLOT[4]) if seq == 'AAAHAAEFAAAA' else
              (1 << S.SLOT[-4]) | (1 << S.SLOT[3]))
    assert (double in S._site_masks(m)[his]) == (seq == 'AAAHAAEFAAAA')
    assert S._transfer(m, np.zeros(m['n']))[0] == pytest.approx(_brute(m, np.zeros(m['n'])), abs=1e-10)


@pytest.mark.parametrize("partner,g", [("T", -.20), ("N", -.60), ("Q", -.15)])
@pytest.mark.parametrize("temperature", [0., 25., 60.])
def test_neutral_polar_pair_keeps_its_energy(partner, g, temperature):
    seq = "AAAQAAA" + partner + "AAA"
    assert S.has_candidates(seq)  # activate the solver even without titrating residues
    c, m = _segment(seq, T=temperature, ncap="Ac")
    assert len(m["contacts"]) == 1
    assert not m["charged"].any()
    assert c.get_dG_i4().sum() == 0.  # no additive duplicate
    assert S._transfer(m, np.zeros(m["n"]))[0] == pytest.approx(g, abs=1e-12)
    assert S.SideChainStates(seq).segment_correction(c) == pytest.approx(g, abs=1e-12)


@pytest.mark.parametrize("partner", ["T", "N", "Q"])
@pytest.mark.parametrize("cap", ["N", "C"])
def test_neutral_polar_pair_excludes_caps(partner, cap):
    seq = "AAAQAAA" + partner + "AAA"
    # With Ac at index 0, Q is index 4 and its partner is index 8.
    c = EnergyCalculator(seq=seq, i=4 if cap == "N" else 1, j=8,
                         pH=7., T=5., ionic_strength=.05, ncap="Ac", ccap="Am")
    m = S._sites(c, True)
    S._add_contacts(c, m)
    assert not m["contacts"]
    assert c.get_dG_i4().sum() == 0.
    assert S.SideChainStates(seq).segment_correction(c) == pytest.approx(0., abs=1e-12)


def test_neutral_polar_ownership_preserves_table_orientation_and_unfavorable_cells():
    tables = S._table_iv()
    new = {(a, b, k) for k, table in tables.items() for a in S.POLAR for b in S.POLAR
           if S.owns_polar(a, b, k, float(table.loc[a, b]))}
    assert new == {("Q", "T", 4), ("Q", "N", 4), ("Q", "Q", 4)}
    c, m = _segment("AAATAAAQAAA", ncap="Ac")
    assert not m["contacts"]  # reverse T-Q is zero, not Q-T's favorable cell
    c, m = _segment("AAANAAANAAA", ncap="Ac")
    assert not m["contacts"]
    assert c.get_dG_i4().sum() == pytest.approx(.20)  # unfavorable N-N stays additive


@pytest.mark.parametrize("seq", ["AAQAAAQAAAQAA", "AAQAAAQAAADAA", "AAQAAAQAAAEAA", "AARAAQAAATAA"])
def test_neutral_polar_competition_matches_enumeration(seq):
    c, m = _segment(seq, ncap="Ac")
    assert len(m["contacts"]) == 2
    assert any(not m["charged"][a] and not m["charged"][b] for a, b in m["contacts"])
    for a, aa in enumerate(m["res"]):
        if aa in S.POLAR:
            assert all(mask.bit_count() <= 1 for mask in S._site_masks(m)[a])
    assert S._transfer(m, np.zeros(m["n"]))[0] == pytest.approx(_brute(m, np.zeros(m["n"])), abs=1e-10)
    if seq == "AAQAAAQAAAQAA":
        k = math.expm1(.15 / m["RT"])
        expected = -m["RT"] * math.log1p(2 * k)  # free + left bond + right bond; never both
        assert S._transfer(m, np.zeros(m["n"]))[0] == pytest.approx(expected, abs=1e-12)
