"""One temperature law for every hydrogen-bond term (energies.EnergyCalculator._hbond_temperature): Munoz & Serrano
1995-III eq. (8) for the backbone bond, and the same relative change for the hydrogen-bonding caps and end groups, the
side-chain hydrogen bonds of the Table IV/V pair terms and the Petukhov and charged-staple motifs."""
import pytest

import pyagadir.energies as E


def _calc(seq, T, ncap="Ac", ccap="Am", pH=7.0):
    n = len(seq) + (ncap is not None) + (ccap is not None)
    return E.EnergyCalculator(seq=seq, i=0, j=n, pH=pH, T=T, ionic_strength=0.05, ncap=ncap, ccap=ccap)


@pytest.mark.parametrize("T", [0.0, 25.0, 60.0])
def test_backbone_is_eq8(T):
    c = _calc("A" * 14, T)
    n_hb = c.j - 6
    assert c.get_dG_Hbond() == pytest.approx((-0.898 + c.dCp * (c.T_kelvin - 273.0)) * n_hb, abs=1e-12)


def test_factor_is_one_at_273_K():
    c = _calc("A" * 10, 0.0)
    c.T_kelvin = 273.0  # the law's reference temperature (the constructor does not accept T < 0 C)
    assert c._hbond_temperature(-0.7) == pytest.approx(-0.7, abs=1e-12)


def test_ser_ncap_advantage_follows_the_backbone_law():
    seq = "SAAAAAAAAAAA"  # free N-terminus: Ser is the N-cap (index 0)
    lo, hi = _calc(seq, 0.0, ncap=None), _calc(seq, 60.0, ncap=None)
    ref = lo.table_1_lacroix.loc["A", "Nc-1"]
    adv_lo, adv_hi = lo.get_dG_Ncap()[0] - ref, hi.get_dG_Ncap()[0] - ref
    assert adv_hi / adv_lo == pytest.approx(hi._hbond_temperature(1.0) / lo._hbond_temperature(1.0))
    assert adv_hi < adv_lo < 0.0  # more favourable at higher temperature, as the backbone term


def test_ala_ncap_is_temperature_independent():
    seq = "AAAAAAAAAAAA"
    assert _calc(seq, 0.0, ncap=None).get_dG_Ncap()[0] == _calc(seq, 60.0, ncap=None).get_dG_Ncap()[0]


def test_hydrophobic_pair_keeps_its_entropic_law():
    seq = "AAALAAALAAAA"  # Leu(i)-Leu(i+4): hydrophobic, dG_ref * t/t_ref
    lo, hi = _calc(seq, 0.0), _calc(seq, 60.0)
    i = 4  # token index of the first Leu
    assert hi.get_dG_i4()[i] / lo.get_dG_i4()[i] == pytest.approx(hi.T_kelvin / lo.T_kelvin)


def test_polar_pair_takes_the_hbond_law():
    seq = "AAAQAAANAAAA"  # Gln(i)-Asn(i+4), Table IVb -0.60: a side-chain hydrogen bond
    lo, hi = _calc(seq, 0.0), _calc(seq, 60.0)
    i = 4
    assert hi.get_dG_i4()[i] / lo.get_dG_i4()[i] == pytest.approx(hi._hbond_temperature(1.0) / lo._hbond_temperature(1.0))
