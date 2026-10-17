"""The opt-in Generalized Born side chain-macrodipole law (dipole_law="gb"; params/README.md)."""
import math

import numpy as np
import pytest

import pyagadir.energies as E
from pyagadir.models import AGADIR


def calc(seq, law="gb", T=0.0, I=0.01, pH=7.0):
    """Ac-seq-Am with the whole chain as one helix segment (acetyl N-cap, amide C-cap)."""
    return E.EnergyCalculator(seq=seq, i=0, j=len(seq) + 2, pH=pH, T=T, ionic_strength=I, ncap="Ac", ccap="Am",
                              dipole_law=law)


def energy_at(aa, k, L=16, caps="res", **kw):
    """Charged-form energy (kcal/mol) of residue aa at helix position k of an L-residue Ala helix, through the model.
    caps="res": Ala N-cap and C-cap residues; caps="AcAm": acetyl N-cap and amide C-cap."""
    if caps == "AcAm":
        seq = "A" * (k - 1) + aa + "A" * (L - k)
        c, idx = calc(seq, **kw), k
    else:
        seq = "A" + "A" * k + aa + "A" * (L + 1 - k) + "A"
        law, T, I, pH = kw.get("law", "gb"), kw.get("T", 0.0), kw.get("I", 0.01), kw.get("pH", 7.0)
        c = E.EnergyCalculator(seq=seq, i=1, j=L + 2, pH=pH, T=T, ionic_strength=I, dipole_law=law)
        idx = 1 + k
    assert c.seq_list[idx] == aa
    return E.GB_DIPOLE_CHARGE[aa] * sum(c._sidechain_dipole_potential(idx))


def test_default_law_is_unchanged():
    seq = "AKAAAAEAAAAKAAAAEA"
    a = E.EnergyCalculator(seq=seq, i=0, j=len(seq) + 2, pH=7.0, T=5.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    b = calc(seq, law="munoz", T=5.0, I=0.05)
    assert a.dipole_law == "munoz" and AGADIR().dipole_law == "munoz"
    assert np.allclose(a.sidechain_dipole_potential, b.sidechain_dipole_potential)


def test_unknown_law_is_refused():
    with pytest.raises(ValueError):
        AGADIR(dipole_law="dmd")
    with pytest.raises(ValueError):
        calc("AAAAAAAA", law="dmd")


def test_table_covers_every_position_and_cap():
    tab = E.gb_dipole_table()
    assert tab["lmax"] >= 36  # the longest peptide in data/
    for form in E.GB_DIPOLE_FORM.values():
        for L in range(4, tab["lmax"] + 1):
            for k in range(L + 2):
                assert (form, "res", "res", L, k) in tab["helix"]
            for k in range(1, min(E.GB_NEAR_CAP, L + 1) + 1):
                assert (form, "Ac", "res", L, k) in tab["helix"]
            for k in range(max(0, L + 1 - E.GB_NEAR_CAP), L + 1):
                assert (form, "res", "Am", L, k) in tab["helix"]
            if L <= E.GB_BOTH_CAPS:
                for k in range(1, L + 1):
                    assert (form, "Ac", "Am", L, k) in tab["helix"]
        assert set(tab["coil"][(form, "res", "res")]) == set(range(26))
        assert set(tab["coil"][(form, "Ac", "Am")]) == set(range(1, 25))


def test_asp_first_turn_is_contact_electrostatics_that_salt_barely_screens():
    # Poisson-Boltzmann (APBS) on the same structures: Asp at N1 minus N6 is about -1 kcal/mol at 0.01 and 2.5 M
    low = energy_at("D", 1, I=0.01) - energy_at("D", 6, I=0.01)
    high = energy_at("D", 1, I=2.5) - energy_at("D", 6, I=2.5)
    assert -1.4 < low < -0.6 and -1.4 < high < -0.5
    assert abs(high - low) < 0.35
    # the default law's first-turn effect is a third of that, and salt removes it
    m_low = energy_at("D", 1, law="munoz", I=0.01) - energy_at("D", 6, law="munoz", I=0.01)
    m_high = energy_at("D", 1, law="munoz", I=2.5) - energy_at("D", 6, law="munoz", I=2.5)
    assert abs(m_low) < 0.4 and abs(m_high) < 0.1


def test_signs_at_the_two_ends():
    # a cation is disfavoured in the first turn and favoured in the last; an anion the reverse
    for aa in ("K", "R", "H"):
        assert energy_at(aa, 2) - energy_at(aa, 8) > 0.0
        assert energy_at(aa, 16) - energy_at(aa, 8) < 0.0
    for aa in ("D", "E", "C", "Y"):
        assert energy_at(aa, 2) - energy_at(aa, 8) < 0.0
        assert energy_at(aa, 16) - energy_at(aa, 8) > 0.0


def test_caps_select_their_own_rows():
    # an acetyl N-cap has no N-H: a Glu at N3 loses its contact with it
    assert energy_at("E", 3, caps="AcAm") != energy_at("E", 3)
    # far from both ends the cap makes no difference
    assert np.isclose(energy_at("K", 20, L=40, caps="AcAm"), energy_at("K", 20, L=40), atol=1e-9)


def test_coil_reference_spans_the_helix_range():
    c = calc("AAAAAKAAAAAAAAAA")
    kap = c.kappa * 1e-10
    # mid-chain: no end correction; next to an end: the chain's own end value
    mid = c._gb_coil("K", "res", "res", 12, 12, kap)
    assert np.isclose(c._gb_coil("K", "res", "res", 9, 30, kap), mid)
    tab = E.gb_dipole_table()
    e_in, w = tab["coil"][("K", "Ac", "Am")][2]
    end = e_in + np.interp(kap, tab["kappa"], w) / c.epsilon_r
    assert np.isclose(c._gb_coil("K", "Ac", "Am", 2, 30, kap), end)


def test_energy_is_charge_times_potential_and_has_no_extra_temperature_factor():
    seq = "AEAAKAADAARAAHAAEA"
    c = calc(seq, T=25.0, I=0.1, pH=6.0)
    assert c.dipole_temperature_factor() == 1.0
    eN, eC = c.get_dG_sidechain_macrodipole()
    for idx, aa in enumerate(c.seq_list):
        if aa in E.GB_DIPOLE_CHARGE:
            q = float(c.modified_seq_ionization_hel[idx])
            assert np.isclose(eN[idx] + eC[idx], q * sum(c._sidechain_dipole_potential(idx)), atol=1e-12)
            assert np.isclose(c.sidechain_dipole_potential[idx], sum(c._sidechain_dipole_potential(idx)), atol=1e-12)


def test_temperature_enters_through_the_water_permittivity():
    # E = E_in + W / eps_w(T): the change from 0 to 50 C is W (1/eps_w(50) - 1/eps_w(0)), small and finite
    e0, e50 = energy_at("E", 1, T=0.0), energy_at("E", 1, T=50.0)
    assert math.isfinite(e0) and math.isfinite(e50) and 0.0 < abs(e50 - e0) < 0.3


def test_flanks_keep_the_default_law_and_tyr_uses_the_table():
    # seq_list = Ac K A x10 Y A Am; the segment idx 2..12 has its N-cap at idx 2 and its C-cap at the Tyr (idx 12)
    seq = "KAAAAAAAAAAYA"
    g = E.EnergyCalculator(seq=seq, i=2, j=11, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac", ccap="Am",
                           dipole_law="gb")
    m = E.EnergyCalculator(seq=seq, i=2, j=11, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    assert g.ncap_idx == 2 and g.ccap_idx == 12 and g.seq_list[1] == "K" and g.seq_list[12] == "Y"
    assert m.sidechain_dipole_potential[1] != 0.0  # Lys in the N-terminal flank: default law, same value
    assert np.isclose(g.sidechain_dipole_potential[1], m.sidechain_dipole_potential[1])
    assert not np.isclose(g.sidechain_dipole_potential[12], m.sidechain_dipole_potential[12])  # Tyr-: the table
    # Tyr- as the N-cap residue (cycle 398: an allowed structure exists): the table, not the default law
    g2 = E.EnergyCalculator(seq="AYAAAAAAAAAA", i=2, j=10, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac",
                            ccap="Am", dipole_law="gb")
    m2 = E.EnergyCalculator(seq="AYAAAAAAAAAA", i=2, j=10, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    assert g2.ncap_idx == 2 and g2.seq_list[2] == "Y"
    assert not np.isclose(g2.sidechain_dipole_potential[2], m2.sidechain_dipole_potential[2])


def test_long_helices_are_end_anchored():
    tab = E.gb_dipole_table()
    L = tab["lmax"] + 10
    for k in (1, 3):
        assert np.isclose(energy_at("K", k, L=L), energy_at("K", k, L=tab["lmax"]))
    assert np.isclose(energy_at("E", L, L=L), energy_at("E", tab["lmax"], L=tab["lmax"]))


def test_agadir_runs_with_the_gb_law():
    seq = "AEAAAKEAAAKEAAAKA"
    p_gb = AGADIR(T=0.0, M=0.01, dipole_law="gb").predict(seq, ncap="Ac", ccap="Am").get_percent_helix()
    p_mu = AGADIR(T=0.0, M=0.01).predict(seq, ncap="Ac", ccap="Am").get_percent_helix()
    assert 0.0 <= p_gb <= 100.0 and p_gb != p_mu
