import numpy as np

# If you already have get_calculator in tests/test_capping.py, import it.
# Otherwise, copy/paste your get_calculator helper here.
from tests.test_capping import get_calculator


def test_r_ladder_matches_lacroix():
    """
    Lacroix distance ladder: 2.1, 4.1, 6.1... Å for N = 0,1,2...  :contentReference[oaicite:1]{index=1}
    """
    calc, _, _ = get_calculator(None, None)
    assert np.isclose(calc._calculate_r(0), 2.1, atol=1e-12)
    assert np.isclose(calc._calculate_r(1), 4.1, atol=1e-12)
    assert np.isclose(calc._calculate_r(2), 6.1, atol=1e-12)
    assert np.isclose(calc._calculate_r(3), 8.1, atol=1e-12)


def test_terminal_macrodipole_distances_full_length_helix_are_minimum():
    """
    Full-length helix => N=0 at both ends => r=2.0 Å at both ends.
    The polynomial distance model gives 2.0 Å for N=0 (an empirical
    calibration) instead of the 2.1 Å stated in the paper text.
    """
    calc, _, _ = get_calculator(None, None)

    assert np.isclose(calc.terminal_macrodipole_distance_nterm, 2.0, atol=1e-6)
    assert np.isclose(calc.terminal_macrodipole_distance_cterm, 2.0, atol=1e-6)


def test_terminal_macrodipole_energies_match_expectation_NN():
    """
    Regression vs expectation for NN. The N-terminal energy follows the free N-terminus'
    helix-state ionization (base pKa 8.00, Thurlkill et al. 2006 -> helix pKa 7.45 at pH 7).
    """
    calc, n_idx, c_idx = get_calculator(None, None)
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()

    assert np.isclose(dG_N_dip[n_idx], 0.60, atol=0.05)
    assert np.isclose(dG_C_dip[c_idx], 0.81, atol=0.05)


def test_free_Nterm_effective_charge_fraction_matches_expectation():
    """
    In NN, r is the same at both ends (2.0 Å), and mu_helix magnitude is the same.
    So (N-term energy)/(C-term energy) should equal the *effective N-term charge fraction*
    relative to a fully charged C-terminus.

    With corrected distances (2.0 Å) and the N-terminal base pKa 8.00 (Thurlkill et al. 2006),
    the charge fraction is ~0.740 (it was ~0.694 with base pKa 7.9).
    """
    calc, n_idx, c_idx = get_calculator(None, None)
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()

    frac = dG_N_dip[n_idx] / dG_C_dip[c_idx]
    assert np.isclose(frac, 0.740, atol=0.02)


def test_free_Nterm_effective_charge_fraction_matches_energy_ratio():
    calc, n_idx, c_idx = get_calculator(None, None)
    dG_N_dip, dG_C_dip = calc.get_dG_terminals_macrodipole()

    # Since r is 2.0 Å at both ends in NN, the only difference is q_Nterm vs q_Cterm (~1)
    frac_from_energies = dG_N_dip[n_idx] / dG_C_dip[c_idx]

    assert np.isclose(frac_from_energies, calc.modified_nterm_ionization_hel, atol=0.01)


def test_inferred_Nterm_pKa_helix_matches_expectation():
    calc, n_idx, c_idx = get_calculator(None, None)

    q = calc.modified_nterm_ionization_hel
    pH = 7.0  # pH = 7.0

    # invert q = 1 / (1 + 10^(pH - pKa))  (basic group)
    pKa_hel = pH - np.log10((1.0 / q) - 1.0)

    # With corrected macrodipole distance (2.0 Å) the helix-state pKa sits 0.55 below the base
    # pKa: 7.45 for base 8.00 (Thurlkill et al. 2006); it was 7.35 for the earlier base 7.9.
    assert np.isclose(pKa_hel, 7.45, atol=0.05)
