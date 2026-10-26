"""Capping motifs and acid-base bonds under pair_rule "additive" (energies.CAPPING_EXCLUSIVITY; under the default "states"
rule the motif is a state of its side chain, tests/test_side_chain_states.py): a side chain that caps the helix by pointing back
toward the N-terminus (an N3 acid behind the N-cap, an N4 charged-staple base or Petukhov acid) does not also take an
acid-base bond forward; "exclusive" sums the two as exclusive states of one side chain, "max" keeps the stronger,
"additive" counts both. Also: the ionisation solver's coefficients equal whole-energy differences under the rule."""
import numpy as np
import pytest

import pyagadir.energies as energies
from pyagadir.energies import EnergyCalculator


def _calc(seq, i, ncap, ccap, pH=7.0, T=4.0, I=0.01):
    L = len(seq) + (ncap is not None) + (ccap is not None)
    return EnergyCalculator(seq=seq, i=i, j=L - i, pH=pH, T=T, ionic_strength=I, ncap=ncap, ccap=ccap,
                            pair_rule="additive")


def _rules(monkeypatch, seq, i, ncap, ccap, **kw):
    out = {}
    for rule in ("additive", "max", "exclusive"):
        monkeypatch.setattr(energies, "CAPPING_EXCLUSIVITY", rule)
        out[rule] = _calc(seq, i, ncap, ccap, **kw)
        out[rule]._acid_base_hbonds()
    return out


def _under(monkeypatch, rule, fn):
    """Evaluate fn() with CAPPING_EXCLUSIVITY = rule (the energy methods read the switch when called)."""
    monkeypatch.setattr(energies, "CAPPING_EXCLUSIVITY", rule)
    return fn()


def test_capping_box_glu_does_not_also_bridge(monkeypatch):
    """Lyu et al.'s Sc-YS-EEEEKKKK...: Ser N-cap (index 2), Glu at N3 (index 5) with Lys at i+3 and i+4. The capping box
    (0.9 + Ser's Nc-1 - Nc-3) outweighs the bonds (0.15 + 0.20): the Glu caps."""
    c = _rules(monkeypatch, "YSEEEEKKKKEEEEKKKK", 2, "Sc", "Am")
    assert {(5, 8), (5, 9)} <= c["additive"]._ab_hbonds
    for rule in ("max", "exclusive"):
        assert not {b for b in c[rule]._ab_hbonds if b[0] == 5}
        assert c[rule]._capping_state["n3"][0] == "motif"
    nc = 2
    x = c["exclusive"]  # one calculator (one set of solved charges), read under each switch
    add, mx, ex = (_under(monkeypatch, r, lambda: x.get_dG_Ncap()[nc]) for r in ("additive", "max", "exclusive"))
    assert mx == pytest.approx(add, abs=1e-12)  # the cap is kept whole
    monkeypatch.setattr(energies, "CAPPING_EXCLUSIVITY", "exclusive")
    g_bond = sum(c["exclusive"]._bond_increment(i, j) for i, j in ((5, 8), (5, 9)))
    g_cap = c["exclusive"]._capping_n3_energy(c["exclusive"]._capping_state["n3"][1])
    g_excl, p_bridge = c["exclusive"]._capping_exclusive(g_cap, g_bond)
    assert ex == pytest.approx(add + g_excl - g_cap, abs=1e-12)
    assert g_cap + g_bond < g_excl < g_cap < 0.0  # less than both counted, more than the cap alone
    assert 0.0 < p_bridge < 0.5


def test_a_stronger_bond_wins_over_a_weak_cap(monkeypatch):
    """Thr N-cap with Asp at N3: the cap is only Thr's Nc-3 gain (0.10); the Asp->Lys(i+4) bond (0.15) wins under "max",
    and the N-cap reverts to Nc-1."""
    c = _rules(monkeypatch, "ATAADAAAKAAAAY", 2, "Ac", "Am")
    assert (5, 9) in c["additive"]._ab_hbonds and (5, 9) in c["max"]._ab_hbonds
    assert c["max"]._capping_state["n3"][0] == "bridge"
    t1 = c["max"].table_1_lacroix
    gain = float(t1.loc["T", "Nc-1"]) - float(t1.loc["T", "Nc-3"])
    assert gain == pytest.approx(0.10)
    d = (_under(monkeypatch, "max", lambda: c["max"].get_dG_Ncap()[2])
         - _under(monkeypatch, "additive", lambda: c["max"].get_dG_Ncap()[2]))
    assert d == pytest.approx(c["max"]._hbond_temperature(gain), abs=1e-12)


def test_charged_staple_and_petukhov_motifs(monkeypatch):
    """N4 motifs: the charged staple (Ser N-cap, Lys at N4 with a Glu at i+4; 0.3 against the Table IV bond 0.10) and
    the Petukhov motif (free N-terminus, capping box, Asp at N4 with a Lys at i+4; 1.0 against 0.15) keep their motif."""
    c = _rules(monkeypatch, "ASAAAKAAAEAAAAY", 2, "Ac", "Am")
    assert (6, 10) in c["additive"]._ab_hbonds and (6, 10) not in c["max"]._ab_hbonds
    sta = {r: _under(monkeypatch, r, c["exclusive"].get_dG_charged_staple) for r in c}
    assert sta["max"] == pytest.approx(sta["additive"], abs=1e-12)
    assert sta["exclusive"] < sta["max"]
    p = _rules(monkeypatch, "SAAEDAAAKAAAAY", 0, None, None)
    assert (4, 8) in p["additive"]._ab_hbonds and (4, 8) not in p["max"]._ab_hbonds
    assert p["max"]._capping_state["petukhov"][0] == "motif"
    pet = {r: _under(monkeypatch, r, p["exclusive"].get_dG_petukhov_motif) for r in p}
    assert pet["max"] == pytest.approx(pet["additive"], abs=1e-12)


def test_additive_counts_both(monkeypatch):
    """"additive" sets no capping state and leaves every motif term as it was."""
    c = _rules(monkeypatch, "YSEEEEKKKKEEEEKKKK", 2, "Sc", "Am")
    assert c["additive"]._capping_state == {}


# Peptides whose capping-motif side chains have one rival bond partner (an N3 acid with two rivals makes the motif term
# trilinear in the charges, which the solver's bilinear coefficients approximate; params/README.md).
SINGLE_RIVAL = [("ATAAEAAKAADAAKY", "Ac", "Am", 4.5, 10.0, 2.0), ("ASAAAKAAAEAAAAY", "Ac", "Am", 7.0, 4.0, 0.01),
                ("SAAEDAAAKAAAAY", None, None, 7.0, 4.0, 0.01), ("ATAADAAAKAAAAY", "Ac", "Am", 7.0, 4.0, 0.01)]


@pytest.mark.parametrize("rule", ["max", "exclusive"])
@pytest.mark.parametrize("case", SINGLE_RIVAL, ids=lambda c: c[0])
def test_coefficients_equal_whole_energy_differences(monkeypatch, rule, case):
    """The seven-pattern coefficients (EnergyCalculator._nonCoulomb_charge_coefficients) against the whole
    charge-dependent energy toggled one residue (c_a) or one i+3/i+4 pair (c_ab) at a time, every segment."""
    monkeypatch.setattr(energies, "CAPPING_EXCLUSIVITY", rule)
    seq, nc, cc, pH, T, I = case
    L = len(seq) + (nc is not None) + (cc is not None)
    for j in range(6, L + 1):
        for i in range(0, L - j + 1):
            c = EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=T, ionic_strength=I, ncap=nc, ccap=cc, pair_rule="additive")
            la, lab = c._nonCoulomb_charge_coefficients()
            charged = set(c.neg_charge_aa + c.pos_charge_aa)
            titr = [k for k in range(int(c.ncap_idx), int(c.ccap_idx) + 1) if c.seq_list[k] in charged]
            saved, n = c.modified_seq_ionization_hel, len(c.seq_list)

            def G(on):
                c.modified_seq_ionization_hel = np.array([1.0 if k in on else 0.0 for k in range(n)])
                return sum(float(np.sum(v)) for v in c._charge_dependent_nonCoulomb_terms().values())

            try:
                g0 = G(())
                for a in titr:
                    assert la.get(a, 0.0) == pytest.approx(G((a,)) - g0, abs=1e-12), (i, j, a)
                for a in titr:
                    for d in (3, 4):
                        b = a + d
                        if b in titr:
                            direct = G((a, b)) - G((a,)) - G((b,)) + g0
                            assert lab.get((a, b), 0.0) == pytest.approx(direct, abs=1e-12), (i, j, a, b)
            finally:
                c.modified_seq_ionization_hel = saved
