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
    for s in itertools.product((0, 1), repeat=n):
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
    return -RT * lnZ + RT * float(np.sum(np.logaddexp(0.0, m["lnx"])))


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
