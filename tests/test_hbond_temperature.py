"""One temperature law for every hydrogen-bond term (energies.EnergyCalculator._hbond_temperature): Munoz & Serrano
1995-III eq. (8) for the backbone bond, and the same relative change for the hydrogen-bonding caps and end groups, the
side-chain hydrogen bonds of the Table IV/V pair terms and the Petukhov and charged-staple motifs."""
import pytest

import pyagadir.energies as E
from pyagadir.chemistry import T_REF


def _calc(seq, T, ncap="Ac", ccap="Am", pH=7.0):
    n = len(seq) + (ncap is not None) + (ccap is not None)
    return E.EnergyCalculator(seq=seq, i=0, j=n, pH=pH, T=T, ionic_strength=0.05, ncap=ncap, ccap=ccap)


@pytest.mark.parametrize("T", [0.0, 25.0, 60.0])
def test_backbone_is_eq8(T):
    c = _calc("A" * 14, T)
    n_hb = c.j - 6
    assert c.get_dG_Hbond() == pytest.approx((E.HBOND_DH_REF + c.dCp * (c.T_kelvin - T_REF)) * n_hb, abs=1e-12)


def test_factor_is_one_at_the_reference_temperature():
    c = _calc("A" * 10, 0.0)  # 0 C = T_REF, the law's reference temperature
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


def test_hydrophobic_staple_takes_the_hydrophobic_pair_law():
    """The N'-N4 hydrophobic staple scales like the hydrophobic pairs, dG_ref * T / T_REF (Lacroix 1998: "similar
    to that for hydrophobic side-chain-side-chain interactions")."""
    seq = "ALSAAELAAAAAA"  # Ac A L S A A E L ...: N' = L (index 2), N-cap S (3), N3 = E, N4 = L (7)
    lo = E.EnergyCalculator(seq=seq, i=3, j=10, pH=7.0, T=0.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    hi = E.EnergyCalculator(seq=seq, i=3, j=10, pH=7.0, T=60.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    assert lo.seq_list[lo.ncap_idx - 1] == "L" and lo.N4_AA == "L"
    assert lo.get_dG_staple() < 0.0
    assert hi.get_dG_staple() / lo.get_dG_staple() == pytest.approx(hi.T_kelvin / lo.T_kelvin)


@pytest.mark.parametrize("j", [6, 7, 10, 16])
def test_heat_capacity_counted_once_per_hydrogen_bond(j):
    """dCp enters the backbone enthalpy once per hydrogen bond (j - 6, eq. 8) and the residue entropy (eq. 9) the
    same number of times: the four nucleus residues carry neither."""
    import math
    seq = "A" * 16
    hot = E.EnergyCalculator(seq=seq, i=0, j=j, pH=7.0, T=50.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    cold = E.EnergyCalculator(seq=seq, i=0, j=j, pH=7.0, T=0.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    T = hot.T_kelvin
    entropic = cold.get_dG_Int().sum() * T / T_REF
    cp_entropy = hot.get_dG_Int().sum() - entropic
    assert cp_entropy == pytest.approx(-(j - 6) * T * hot.dCp * math.log(T / T_REF), abs=1e-12)
    cp_enthalpy = hot.get_dG_Hbond() - cold.get_dG_Hbond()
    assert cp_enthalpy == pytest.approx((j - 6) * hot.dCp * (T - T_REF), abs=1e-12)
