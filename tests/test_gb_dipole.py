"""The opt-in Generalized Born side chain-macrodipole law (dipole_law="gb"; params/README.md)."""
import math

import numpy as np
import pytest

import pyagadir.energies as E
from pyagadir.models import AGADIR


def calc(seq, law="gb", T=0.0, I=0.01, pH=7.0):
    return E.EnergyCalculator(seq=seq, i=0, j=len(seq) + 2, pH=pH, T=T, ionic_strength=I, ncap="Ac", ccap="Am",
                              dipole_law=law)


def energy_at(aa, k, L=16, **kw):
    """Charged-form energy (kcal/mol) of residue aa at helix position k (Ac N-cap = 0) of an L-residue Ala helix."""
    seq = "A" * (k - 1) + aa + "A" * (L - k)
    c = calc(seq, **kw)
    z = E.GB_DIPOLE_CHARGE[aa]
    return z * sum(c._sidechain_dipole_potential(k))


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


def test_table_covers_every_position():
    tab = E.gb_dipole_table()
    assert tab["lmax"] >= 36  # the longest peptide in data/
    for aa in E.GB_DIPOLE_CHARGE:
        assert aa in tab["coil"]
        for L in range(4, tab["lmax"] + 1):
            e_in, w, n_rot = tab["helix"][(aa, L)]
            assert len(e_in) == L + 2 and w.shape == (L + 2, len(tab["kappa"])) and len(n_rot) == L + 2
            # every row rests on clash-free structures, except Tyr- at the N-cap (falls back to the default law)
            assert all(n_rot[1:] > 0) and (n_rot[0] > 0 or aa == "Y")


def test_asp_first_turn_is_contact_electrostatics_that_salt_barely_screens():
    # Poisson-Boltzmann (APBS) on the same structures, same rotamer average: Asp at N1 minus N6 is -0.96 at 0.01 M
    # and -0.88 at 2.5 M; this table gives -0.83 and -0.67
    low = energy_at("D", 1, I=0.01) - energy_at("D", 6, I=0.01)
    high = energy_at("D", 1, I=2.5) - energy_at("D", 6, I=2.5)
    assert -1.2 < low < -0.6 and -1.2 < high < -0.5
    assert abs(high - low) < 0.3
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


def test_flanks_keep_the_default_law_and_cys_tyr_use_the_table():
    # seq_list = Ac K A x10 Y A Am; the segment idx 2..12 has its N-cap at idx 2 and its C-cap at the Tyr (idx 12)
    seq = "KAAAAAAAAAAYA"
    g = E.EnergyCalculator(seq=seq, i=2, j=11, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac", ccap="Am",
                           dipole_law="gb")
    m = E.EnergyCalculator(seq=seq, i=2, j=11, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    assert g.ncap_idx == 2 and g.ccap_idx == 12 and g.seq_list[1] == "K" and g.seq_list[12] == "Y"
    assert m.sidechain_dipole_potential[1] != 0.0  # Lys in the N-terminal flank: default law, same value
    assert np.isclose(g.sidechain_dipole_potential[1], m.sidechain_dipole_potential[1])
    tab = E.gb_dipole_table()  # Tyr- at the C-cap: the table
    e_in, w, _ = tab["helix"][("Y", 9)]
    c_in, c_w = tab["coil"]["Y"]
    kap = g.kappa * 1e-10
    e = (e_in[10] - c_in) + (np.interp(kap, tab["kappa"], w[10]) - np.interp(kap, tab["kappa"], c_w)) / g.epsilon_r
    assert np.isclose(g.sidechain_dipole_potential[12], -e)
    # Tyr- at the N-cap has no clash-free structure in the table: the default law there
    g2 = E.EnergyCalculator(seq="YAAAAAAAAAA", i=1, j=10, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac",
                            ccap="Am", dipole_law="gb")
    m2 = E.EnergyCalculator(seq="YAAAAAAAAAA", i=1, j=10, pH=10.0, T=20.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    assert g2.ncap_idx == 1 and g2.seq_list[1] == "Y"
    assert np.isclose(g2.sidechain_dipole_potential[1], m2.sidechain_dipole_potential[1])


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
