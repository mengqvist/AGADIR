"""Temperature laws of the hydrogen-bond terms. The backbone bond follows Munoz & Serrano 1995-III eq. (8) (enthalpy with
a heat capacity, its entropy in the intrinsic term). Every side-chain and capping hydrogen bond (caps and end groups,
the capping box, Table IV/V pairs, the N4 motifs, the side-chain states' bonds H and Hn) is a free energy at 0-5 C and
temperature-independent: no source measures its enthalpy in a helix (params/README.md, "Hydrogen-bond temperature
dependence")."""
import pytest

import pyagadir.energies as E
from pyagadir.chemistry import T_REF
from pyagadir.electrostatics import side_chain_states as S


def _calc(seq, T, ncap="Ac", ccap="Am", pH=7.0):
    n = len(seq) + (ncap is not None) + (ccap is not None)
    return E.EnergyCalculator(seq=seq, i=0, j=n, pH=pH, T=T, ionic_strength=0.05, ncap=ncap, ccap=ccap)


@pytest.mark.parametrize("T", [0.0, 25.0, 60.0])
def test_backbone_is_eq8(T):
    c = _calc("A" * 14, T)
    n_hb = c.j - 6
    assert c.get_dG_Hbond() == pytest.approx((E.HBOND_DH_REF + c.dCp * (c.T_kelvin - T_REF)) * n_hb, abs=1e-12)


@pytest.mark.parametrize("seq", ["SAAAAAAAAAAA", "TAAEAAAAAAAA", "AAAAAAAAAAAA"])
def test_ncap_is_temperature_independent(seq):
    """A Ser N-cap, a Thr N-cap with its capping box (Glu at N3), and an Ala N-cap (free N-terminus: the first residue
    is the N-cap)."""
    lo, hi = _calc(seq, 0.0, ncap=None), _calc(seq, 60.0, ncap=None)
    # the capping box is weighted by Glu's helix-state charge, which moves slightly with T
    assert hi.get_dG_Ncap()[0] == pytest.approx(lo.get_dG_Ncap()[0], rel=1e-3)


@pytest.mark.parametrize("seq, ccap", [("AAAAAAAAAAAK", None), ("AAAAAAAAAAAA", "Am")])
def test_ccap_and_end_groups_are_temperature_independent(seq, ccap):
    lo, hi = _calc(seq, 0.0, ccap=ccap), _calc(seq, 60.0, ccap=ccap)
    assert hi.get_dG_Ncap().sum() == pytest.approx(lo.get_dG_Ncap().sum(), abs=1e-12)  # the Ac end group
    assert hi.get_dG_Ccap().sum() == pytest.approx(lo.get_dG_Ccap().sum(), abs=1e-12)


def test_hydrophobic_pair_keeps_its_entropic_law():
    seq = "AAALAAALAAAA"  # Leu(i)-Leu(i+4): hydrophobic, dG_ref * t/t_ref
    lo, hi = _calc(seq, 0.0), _calc(seq, 60.0)
    i = 4  # token index of the first Leu
    assert hi.get_dG_i4()[i] / lo.get_dG_i4()[i] == pytest.approx(hi.T_kelvin / lo.T_kelvin)


def test_polar_pairs_and_table_v_are_temperature_independent():
    """Gln(i)-Asn(i+4) (Table IVb -0.60; no titrating partner, an energy term) and Glu-(i)-Asn(i+4) (Table IV plus the
    Table V motif; a side-chain state, whose bond is a constant): side-chain hydrogen bonds."""
    lo, hi = _calc("AAAQAAANAAAA", 0.0), _calc("AAAQAAANAAAA", 60.0)
    assert lo.get_dG_i4()[4] < 0.0
    assert hi.get_dG_i4()[4] == pytest.approx(lo.get_dG_i4()[4], abs=1e-12)
    assert S.polar_bond("E", "N", 4, -0.30)[1, 0] == pytest.approx(-0.80)


def test_side_chain_state_bonds_are_temperature_independent():
    """The contact bonds H and Hn of the side-chain states are the tabulated 0 C values at any temperature."""
    seq = "AAAAEAAAKAAAAA"  # Glu(i)-Lys(i+4)
    H, Hn = S.pair_strength("E", "K", 4)
    for T in (0.0, 60.0):
        c = _calc(seq, T)
        m = S._sites(c, True)
        S._add_contacts(c, m)
        (ct,) = m["contacts"].values()
        assert (ct["H"], ct["Hn"]) == (H, Hn)


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
