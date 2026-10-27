"""The helix macrodipole (electrostatics.point, _macrodipole_potential): one rule, a +1/2 and a -1/2 charge at the helix
ends felt by screened Coulomb in water at water's eps(T); inside the helix both ends at their Table VII distances plus the
residue's Table 1 carry-over (params/macrodipole_transfer.tsv); flanking residues feel the nearby end at 6 A + 3 A per
further position; no cutoff inside the helix, so the field changes sign once, across the middle."""
import math

import pytest

import pyagadir.energies as E
import pyagadir.electrostatics.point as P
from pyagadir.chemistry import calculate_permittivity


def _calc(seq, i, j, T=5.0, I=0.05, nc="Ac", cc="Am"):
    return E.EnergyCalculator(seq=seq, i=i, j=j, pH=7.0, T=T, ionic_strength=I, ncap=nc, ccap=cc)


def _half(c, pole, r):
    """+1/2 or -1/2 charge against a +1 probe at r A, screened Coulomb in water (the law of every pair)."""
    return 332.0637 * pole * 0.5 / (c.epsilon_r * r) * math.exp(-c.kappa * r * 1e-10)


def test_helix_residue_feels_both_ends_at_table_vii_distances():
    seq = "AAAAEAAAAAAAAAAAAAKA"  # Glu at N4 (Ncen class), Lys near the C end
    c = _calc(seq, 0, len(seq) + 2)
    idx = 5  # Glu: tokens Ac(0) A A A A E(5)
    n_pos, c_pos = idx - c.ncap_idx, c.ccap_idx - idx
    phi_n, phi_c = c._macrodipole_potential(idx)
    d = P.macrodipole_transfer()[("E", P.table1_position_class("E", n_pos, c_pos, "Ac"))]
    expect_n = _half(c, 1, float(c.table_7_ncap_lacroix.loc["E", f"N{n_pos}"])) + d
    expect_c = _half(c, -1, float(c.table_7_ccap_lacroix.loc["E", "C13"]) + 1.5 * (c_pos - 13))
    assert phi_n == pytest.approx(expect_n, rel=1e-4)
    assert phi_c == pytest.approx(expect_c, rel=1e-4)


def test_no_cutoff_and_one_sign_change_across_a_long_helix():
    """Before 2026-10-27 the nearest end acted by MS95-II eq. 11 with a nine-position cutoff while the far end stayed:
    in the middle of a long helix the field pointed the wrong way. Now it changes sign once, across the middle."""
    seq = "K" * 31
    c = _calc(seq, 1, 31, I=0.01)
    phis = [sum(c._macrodipole_potential(i)) for i in range(c.ncap_idx + 5, c.ccap_idx - 4)]  # the Ncen class only
    signs = [x > 0 for x in phis]
    assert signs[0] and not signs[-1]
    assert sum(a != b for a, b in zip(signs, signs[1:])) == 1


def test_flanks_feel_the_nearby_end():
    seq = "AEEAAAAAAAAAAAAAAA"  # Glu at tokens 2 and 3; the helix starts at token 4 (N-cap), so they are N'' and N'
    c = _calc(seq, 4, 12)
    assert c._macrodipole_potential(3) == pytest.approx((_half(c, 1, 6.0), 0.0), rel=1e-4)
    assert c._macrodipole_potential(2) == pytest.approx((_half(c, 1, 9.0), 0.0), rel=1e-4)


def test_water_dielectric_with_temperature():
    """The macrodipole takes water's eps(T) like every other charge pair (no separate temperature factor)."""
    seq = "AAAAAAAAAKAAAAAAAAAA"
    lo, hi = _calc(seq, 0, 22, T=0.0, I=1e-7), _calc(seq, 0, 22, T=50.0, I=1e-7)
    idx = 10
    d = P.macrodipole_transfer()[("K", "Ncen")]
    ratio = (sum(hi._macrodipole_potential(idx)) - d) / (sum(lo._macrodipole_potential(idx)) - d)
    assert ratio == pytest.approx(calculate_permittivity(273.15) / calculate_permittivity(323.15), rel=1e-3)


def test_carry_over_by_class():
    """An Asp at N1 behind an acetyl N-cap takes the N1 offset; behind a residue N-cap, N1r."""
    t = P.macrodipole_transfer()
    assert P.table1_position_class("D", 1, 10, "Ac") == "N1" and P.table1_position_class("D", 1, 10, "S") == "N1r"
    assert P.table1_position_class("E", 3, 10, "S") == "N3cb" and P.table1_position_class("K", 2, 10, "Ac") == "Ncen"
    assert t[("D", "N1")] == pytest.approx(0.0853, abs=1e-4)
    assert set(cls for _, cls in t) == {"Ncap", "Ccap", "Ncen", "N3cb"} | {f"N{k}{s}" for k in (1, 2, 3, 4) for s in ("", "r")}
