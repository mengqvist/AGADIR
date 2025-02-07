import pytest
import numpy as np

from pyagadir.chemistry import (
    calculate_ionic_strength,
    adjust_pKa,
    acidic_residue_ionization,
    basic_residue_ionization,
    calculate_permittivity,
    debye_screening_kappa
)


def test_calculate_ionic_strength():
    # Test with monovalent ions
    assert calculate_ionic_strength({"Na+": 0.1, "Cl-": 0.1}) == 0.1
    
    # Test with divalent ions
    assert calculate_ionic_strength({"Ca2+": 0.1, "SO42-": 0.1}) == 0.4
    
    # Test with mixed ions
    assert calculate_ionic_strength({"Na+": 0.1, "Ca2+": 0.1, "Cl-": 0.3}) == 0.4
    
    # Test warning for unrecognized ions
    with pytest.warns(UserWarning):
        calculate_ionic_strength({"Unknown": 0.1})
    
    # Test error for negative concentrations
    with pytest.raises(ValueError):
        calculate_ionic_strength({"Na+": -0.1})


def test_adjust_pKa():
    T = 298.15
    pKa_ref = 7.0
    
    # 1) Acidic residue tests
    # ------------------------------------------------
    # 1a) No energy difference => pKa should be unchanged
    adjusted = adjust_pKa(T=T, pKa_ref=pKa_ref, deltaG=0.0, is_basic=False)
    assert adjusted == pytest.approx(pKa_ref, abs=1e-6), \
        f"Acidic, ΔG=0. Expected {pKa_ref}, got {adjusted}"
    
    # 1b) Negative (favorable) ΔG => pKa should decrease (stabilizes deprotonated form)
    adjusted = adjust_pKa(T=T, pKa_ref=pKa_ref, deltaG=-1.0, is_basic=False)
    assert adjusted < pKa_ref, \
        f"Acidic, ΔG<0. Expected pKa < {pKa_ref}, got {adjusted}"
    
    # 1c) Positive (unfavorable) ΔG => pKa should increase
    adjusted = adjust_pKa(T=T, pKa_ref=pKa_ref, deltaG=1.0, is_basic=False)
    assert adjusted > pKa_ref, \
        f"Acidic, ΔG>0. Expected pKa > {pKa_ref}, got {adjusted}"
    
    # 2) Basic residue tests
    # ------------------------------------------------
    # 2a) No energy difference => pKa should be unchanged
    adjusted = adjust_pKa(T=T, pKa_ref=pKa_ref, deltaG=0.0, is_basic=True)
    assert adjusted == pytest.approx(pKa_ref, abs=1e-6), \
        f"Basic, ΔG=0. Expected {pKa_ref}, got {adjusted}"
    
    # 2b) Negative (favorable) ΔG => pKa should increase (stabilizes protonated form)
    adjusted = adjust_pKa(T=T, pKa_ref=pKa_ref, deltaG=-1.0, is_basic=True)
    assert adjusted > pKa_ref, \
        f"Basic, ΔG<0. Expected pKa > {pKa_ref}, got {adjusted}"
    
    # 2c) Positive (unfavorable) ΔG => pKa should decrease
    adjusted = adjust_pKa(T=T, pKa_ref=pKa_ref, deltaG=1.0, is_basic=True)
    assert adjusted < pKa_ref, \
        f"Basic, ΔG>0. Expected pKa < {pKa_ref}, got {adjusted}"


def test_acidic_residue_ionization():
    # Test fully deprotonated (pH >> pKa)
    assert np.isclose(acidic_residue_ionization(pH=14, pKa=5), -1.0, atol=0.001)
    
    # Test neutral (pH << pKa)
    assert np.isclose(acidic_residue_ionization(pH=0, pKa=5), 0.0, atol=0.001)
    
    # Test half ionized (pH = pKa)
    assert np.isclose(acidic_residue_ionization(pH=5, pKa=5), -0.5, atol=0.001)


def test_basic_residue_ionization():
    # Test fully protonated (pH << pKa)
    assert np.isclose(basic_residue_ionization(pH=0, pKa=10), 1.0, atol=0.001)
    
    # Test neutral (pH >> pKa)
    assert np.isclose(basic_residue_ionization(pH=14, pKa=10), 0.0, atol=0.001)
    
    # Test half ionized (pH = pKa)
    assert np.isclose(basic_residue_ionization(pH=10, pKa=10), 0.5, atol=0.001)


def test_calculate_permittivity():
    # Test at freezing point
    assert np.isclose(calculate_permittivity(273.15), 88.0, atol=0.2)

    # Test at room temperature
    assert np.isclose(calculate_permittivity(298.15), 78.3, atol=0.2)
    
    # Test at boiling point
    assert np.isclose(calculate_permittivity(373.15), 55.3, atol=0.2)
    
    # Test temperature dependence
    assert calculate_permittivity(273.15) > calculate_permittivity(298.15)


def test_debye_screening_length():
    # Test typical physiological conditions
    kappa = debye_screening_kappa(ionic_strength=0.15, T=298)
    assert 0.5e9 < kappa < 2e9  # Should be around 1/nm in SI units
    
    # Test ionic strength dependence
    kappa1 = debye_screening_kappa(ionic_strength=0.15, T=298)
    kappa2 = debye_screening_kappa(ionic_strength=0.30, T=298)
    assert kappa2 > kappa1  # Higher ionic strength = stronger screening
    
    # Test temperature dependence
    kappa1 = debye_screening_kappa(ionic_strength=0.15, T=298)
    kappa2 = debye_screening_kappa(ionic_strength=0.15, T=310)
    assert kappa1 < kappa2  # Higher temperature = weaker permittivity = larger kappa

    # Test minimum ionic strength. Kappa can never be zero due to autoionization of water.
    kappa = debye_screening_kappa(ionic_strength=0.0, T=298)
    assert kappa != 0.0
