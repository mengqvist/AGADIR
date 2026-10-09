import pytest
import numpy as np
from pyagadir.energies import EnergyCalculator, PrecomputeParams

# Define common test parameters
SEQ = "AAAAAA"
PH = 7.0
TEMP = 0.0 # 0°C
IONIC = 0.05 # 0.05 M

@pytest.fixture(autouse=True)
def cleanup_params():
    """Reset params to prevent test pollution."""
    PrecomputeParams._params = None
    yield
    PrecomputeParams._params = None

def get_calculator(ncap, ccap):
    """Helper to instantiate EnergyCalculator for the AAAAAA helix."""
    # Determine indices for the AAAAAA helix
    # If ncap is present, seq is [Cap, A, A, A, A, A, A, ...]. Helix starts at 1.
    # If ncap is None, seq is [A, A, A, A, A, A, ...]. Helix starts at 0.
    start_idx = 1 if ncap is not None else 0
    length = 6
    
    calc = EnergyCalculator(
        seq=SEQ,
        i=start_idx,
        j=length,
        pH=PH,
        T=TEMP,
        ionic_strength=IONIC,
        ncap=ncap,
        ccap=ccap
    )
    return calc, start_idx, start_idx + length - 1

def test_capping_NN():
    """Test Free N-term, Free C-term (NN)."""
    calc, n_idx, c_idx = get_calculator(None, None)
    
    dG_Ncap = calc.get_dG_Ncap()[n_idx]
    dG_Ccap = calc.get_dG_Ccap()[c_idx]
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()
    
    # N-cap: 0.40
    # C-cap: 0.40 (Lacroix 1998 published value); code produces 0.40.
    # N-dipole 0.60, C-dipole 0.81: the free N-terminus is partly charged at pH 7. Its base pKa
    # is 8.00 (Thurlkill et al. 2006, unstructured peptides), giving a helix-state pKa of 7.45
    # and q = 0.74. With the earlier base pKa of 7.9 these were 0.50 and 0.77.

    assert np.isclose(dG_Ncap, 0.40, atol=0.01)
    assert np.isclose(dG_Ccap, 0.40, atol=0.01)
    assert np.isclose(dG_N_dip[n_idx], 0.60, atol=0.05)
    assert np.isclose(dG_C_dip[c_idx], 0.81, atol=0.05)

def test_capping_NA():
    """Test Acetylated N-term, Free C-term (NA)."""
    calc, n_idx, c_idx = get_calculator("Ac", None)
    
    dG_Ncap = calc.get_dG_Ncap()[n_idx]
    dG_Ccap = calc.get_dG_Ccap()[c_idx]
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()
    
    # Values from NA file:
    # N-dipole: 0.00 (Acetylated)
    # C-dipole: 0.7691
    
    assert np.isclose(dG_Ncap, 0.40, atol=0.01)
    assert np.isclose(dG_Ccap, 0.40, atol=0.01)
    assert np.isclose(dG_N_dip[n_idx], 0.00, atol=0.01)
    assert np.isclose(dG_C_dip[c_idx], 0.7691, atol=0.05)

def test_capping_NS():
    """Test Succinylated N-term, Free C-term (NS)."""
    calc, n_idx, c_idx = get_calculator("Sc", None)
    
    dG_Ncap = calc.get_dG_Ncap()[n_idx]
    dG_Ccap = calc.get_dG_Ccap()[c_idx]
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()
    
    # Values from NS file:
    # N-dipole: -0.3405 (Succinyl is negative)
    
    assert np.isclose(dG_Ncap, 0.40, atol=0.01)
    assert np.isclose(dG_Ccap, 0.40, atol=0.01)
    assert np.isclose(dG_N_dip[n_idx], -0.3405, atol=0.05)
    assert np.isclose(dG_C_dip[c_idx], 0.7691, atol=0.05)

def test_capping_YA():
    """Test Acetylated N-term, Amidated C-term (YA)."""
    calc, n_idx, c_idx = get_calculator("Ac", "Am")
    
    dG_Ncap = calc.get_dG_Ncap()[n_idx]
    dG_Ccap = calc.get_dG_Ccap()[c_idx]
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()
    
    # Values from YA file:
    # N-dipole: 0.00
    # C-dipole: 0.00 (Amidated)
    
    assert np.isclose(dG_Ncap, 0.40, atol=0.01)
    assert np.isclose(dG_Ccap, 0.40, atol=0.01)
    assert np.isclose(dG_N_dip[n_idx], 0.00, atol=0.01)
    assert np.isclose(dG_C_dip[c_idx], 0.00, atol=0.01)

def test_capping_YN():
    """Test Free N-term, Amidated C-term (YN)."""
    calc, n_idx, c_idx = get_calculator(None, "Am")
    
    dG_Ncap = calc.get_dG_Ncap()[n_idx]
    dG_Ccap = calc.get_dG_Ccap()[c_idx]
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()
    
    # N-dipole 0.56 (free N-terminus, base pKa 8.00 as in test_capping_NN; the uncharged amidated
    # C-terminus leaves it slightly less charged than in NN); C-dipole 0.00

    assert np.isclose(dG_Ncap, 0.40, atol=0.01)
    assert np.isclose(dG_Ccap, 0.40, atol=0.01)
    assert np.isclose(dG_N_dip[n_idx], 0.56, atol=0.05)
    assert np.isclose(dG_C_dip[c_idx], 0.00, atol=0.01)

def test_capping_YS():
    """Test Succinylated N-term, Amidated C-term (YS)."""
    calc, n_idx, c_idx = get_calculator("Sc", "Am")
    
    dG_Ncap = calc.get_dG_Ncap()[n_idx]
    dG_Ccap = calc.get_dG_Ccap()[c_idx]
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()
    
    # Values from YS file:
    # N-dipole: -0.3405
    # C-dipole: 0.00
    
    assert np.isclose(dG_Ncap, 0.40, atol=0.01)
    assert np.isclose(dG_Ccap, 0.40, atol=0.01)
    assert np.isclose(dG_N_dip[n_idx], -0.3405, atol=0.05)
    assert np.isclose(dG_C_dip[c_idx], 0.00, atol=0.01)

@pytest.mark.parametrize("ncap,factor", [("A", 0.5), ("G", 0.5), ("S", 0.5), ("P", 0.5)])
def test_staple_after_any_ncap(ncap, factor):
    """Hydrophobic staple (Leu N' - Leu N4, table 2: -0.90) is half strength after a non-polar N-cap (Viguera & Serrano
    1999, Protein Sci. 8, 1733, Table 3 note c), as after a polar N-cap without the capping box; the published
    supplement gave non-polar N-caps nothing."""
    seq = "L" + ncap + "AAALAAAAAAAA"
    calc = EnergyCalculator(seq=seq, i=1, j=12, pH=PH, T=TEMP, ionic_strength=IONIC, ncap=None, ccap=None,
                            params=EnergyCalculator.snapshot_params())
    assert calc.get_dG_staple() == pytest.approx(factor * -0.90, abs=1e-6)


def test_staple_full_with_capping_box():
    """Ser N-cap with Glu at N3 (capping box): full staple."""
    calc = EnergyCalculator(seq="LSAAELAAAAAAAA", i=1, j=12, pH=PH, T=TEMP, ionic_strength=IONIC, ncap=None, ccap=None,
                            params=EnergyCalculator.snapshot_params())
    assert calc.get_dG_staple() == pytest.approx(-0.90, abs=1e-6)


def test_first_turn_cation_near_field():
    """A helical Lys at N2 carries the FIRST_TURN_CATION near field on top of eq. 11; at N5 it does not."""
    from pyagadir.energies import FIRST_TURN_CATION

    def phi(pos):
        seq = "A" * (pos - 1) + "K" + "A" * (12 - pos)
        c = EnergyCalculator(seq=seq, i=0, j=14, pH=7.0, T=0.0, ionic_strength=0.1, ncap="Ac", ccap="Am")
        return sum(c._sidechain_dipole_potential(pos))

    def phi_ala_host_eq11(pos):
        from pyagadir import energies as E
        saved = dict(E.FIRST_TURN_CATION)
        E.FIRST_TURN_CATION.clear()
        try:
            return phi(pos)
        finally:
            E.FIRST_TURN_CATION.update(saved)

    assert abs((phi(2) - phi_ala_host_eq11(2)) - FIRST_TURN_CATION["K"][1]) < 1e-9
    assert abs(phi(5) - phi_ala_host_eq11(5)) < 1e-12
