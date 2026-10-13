"""First-turn N2 offset (energies.FIRST_TURN_N2_OFFSET): added to the Table 1 N2 cell only where the residue is at N2
of the segment (not N1, N3, the interior or C1-C3), and the same for the charged and the neutral form."""
import pytest

import pyagadir.energies as E


def _int_energy(seq, pH, monkeypatch=None, off=None):
    if monkeypatch is not None:
        monkeypatch.setattr(E, "FIRST_TURN_N2_OFFSET", off)
    n = len(seq) + 2  # Ac and Am tokens
    return E.EnergyCalculator(seq=seq, i=0, j=n, pH=pH, T=0.0, ionic_strength=0.05, ncap="Ac", ccap="Am").get_dG_Int()


@pytest.mark.parametrize("pH", [7.0, 12.5])
def test_lys_n2_offset_charge_independent(pH, monkeypatch):
    seq = "AKAAAAAAAAAA"  # Ac = N-cap (index 0), A1 = N1, K2 = N2
    k_off = E.FIRST_TURN_N2_OFFSET["K"]
    with_off = _int_energy(seq, pH)
    without = _int_energy(seq, pH, monkeypatch, {})
    d = with_off - without
    assert d[2] == pytest.approx(k_off * 273.15 / 273.0, abs=1e-9)
    assert abs(d).sum() == pytest.approx(abs(d[2]), abs=1e-12)


@pytest.mark.parametrize("seq", ["KAAAAAAAAAAA", "AAKAAAAAAAAA", "AAAAAAKAAAAA", "AAAAAAAAAKAA"])
def test_no_offset_off_n2(seq, monkeypatch):
    with_off = _int_energy(seq, 7.0)
    without = _int_energy(seq, 7.0, monkeypatch, {})
    assert abs(with_off - without).max() == 0.0


def test_measured_cells_untouched():
    for aa in "ALIVMGSTNQDEP":
        assert aa not in E.FIRST_TURN_N2_OFFSET
