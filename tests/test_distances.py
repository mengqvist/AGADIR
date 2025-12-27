import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pyagadir.energies import EnergyCalculator, PrecomputeParams

# --- 1. MOCK DATA SETUP ---

# Create mock dataframes (same as before)
aa = ['A', 'G', 'K', 'R', 'D', 'E', 'C', 'Y', 'Nterm', 'Cterm', 'Ac', 'Am', 'Sc']
cols = [f'i+{k}' for k in range(30)]

helix_df = pd.DataFrame(10.0, index=['AA', 'RK', 'DA', 'KK', 'HelixRest'], columns=cols)
coil_df  = pd.DataFrame(20.0, index=['AA', 'RK', 'DA', 'KK', 'RcoilRest'], columns=cols)
n_macro_df = pd.DataFrame(30.0, index=aa, columns=['Ncap'] + [f'N{k}' for k in range(1, 30)])
c_macro_df = pd.DataFrame(30.0, index=aa, columns=['Ccap'] + [f'C{k}' for k in range(1, 30)])
pka_df   = pd.DataFrame(7.0,  index=aa, columns=['pKa'])
pka_df.loc['Nterm', 'pKa'] = 8.0
pka_df.loc['Cterm', 'pKa'] = 3.5
zeros_df = pd.DataFrame(0.0,  index=aa, columns=aa)
t1_cols = ['N1', 'N2', 'N3', 'N4', 'Ncen', 'Neutral', 'Nc-1', 'Nc-2', 'Nc-3', 'Nc-4', 'Cc-1', 'Cc-2']

MOCK_PARAMS = {
    "table_1_lacroix": pd.DataFrame(0.0, index=aa, columns=t1_cols),
    "table_2_lacroix": zeros_df,
    "table_3_lacroix": zeros_df,
    "table_4a_lacroix": zeros_df,
    "table_4b_lacroix": zeros_df,
    "table_6_helix_lacroix": helix_df,
    "table_6_coil_lacroix": coil_df,
    "table_7_ncap_lacroix": n_macro_df,
    "table_7_ccap_lacroix": c_macro_df,
    "pka_values": pka_df
}

# --- 2. SUBCLASSING FOR ISOLATION ---

class MockEnergyCalculator(EnergyCalculator):
    """
    Subclass that overrides load_params to return MOCK_PARAMS.
    This ensures we NEVER touch the global PrecomputeParams._params class variable,
    preventing test pollution completely.
    """
    def load_params(self):
        # Return the local mock dictionary directly
        return MOCK_PARAMS

@pytest.fixture
def calculator():
    """
    Instantiates MockEnergyCalculator.
    Sequence: Ac - K(0) A(1) A(2) A(3) K(4) K(5) A(6) K(7) K(8) - Am
    (Indices shifted by 1 because of Ac cap at 0)
    Real Indices: 0(Ac), 1(K), 2(A)...
    Helix Definition: Start=2 (A), Len=6.
    """
    calc = MockEnergyCalculator(
        seq="KAAAKKAKK", 
        i=2, j=6,
        pH=7.0, 
        T=25.0, 
        ionic_strength=0.1, 
        ncap="Ac", 
        ccap="Am"
    )
    return calc

@pytest.fixture
def calculator_NN():
    calc = MockEnergyCalculator(
        seq="AAAAAA",
        i=0, j=6,
        pH=7.0, 
        T=25.0, 
        ionic_strength=0.1,
        ncap=None,
        ccap=None
    )
    return calc

@pytest.fixture
def calculator_AcAm():
    calc = MockEnergyCalculator(
        seq="AAAAAA",
        i=0, j=8,
        pH=7.0, 
        T=25.0, 
        ionic_strength=0.1,
        ncap="Ac",
        ccap="Am"
    )
    return calc

# --- 2. TESTS ---

def test_terminal_macrodipole_distances_NN(calculator_NN):
    """Verify the terminal macrodipole distances for a free N-term and free C-term.
    """
    assert np.isclose(calculator_NN.terminal_macrodipole_distance_nterm, 2.1, atol=1e-6)
    assert np.isclose(calculator_NN.terminal_macrodipole_distance_cterm, 2.1, atol=1e-6)

def test_helix_state_distances(calculator):
    """
    Verify distances between two residues INSIDE the helix.
    Indices 5 (K) and 6 (K) are both in the helix [2..7].
    Should use Table 6 Helix (Mock value = 10.0).
    """
    dist = calculator.sidechain_sidechain_distances_hel[5, 6]
    assert dist == 10.0, f"Expected Helix Table value (10.0), got {dist}"

def test_coil_state_distances(calculator):
    """
    Verify distances between two residues in the Random Coil state.
    Indices 1 (K, N') and 5 (K, Helix). 
    In RC state, they are just two charges.
    Should use Table 6 Coil (Mock value = 20.0).
    """
    dist = calculator.charged_sidechain_distances_rc[1, 5]
    assert dist == 20.0, f"Expected Coil Table value (20.0), got {dist}"

def test_helix_coil_interface_restriction(calculator):
    """
    Verify Helix-Coil interactions are ONLY calculated for N' and C'.
    Helix: [2, 3, 4, 5, 6, 7]
    N' = 1 (K)
    C' = 8 (K)
    C''= 9 (K)
    Target Helix Residue = 5 (K)
    """
    
    # Case A: N'(1) to Helix(5) -> Valid
    # Path: Coil(1->2) + Helix(2->5). 
    # Dist: MockCoil(20.0) + MockHelix(10.0) = 30.0
    dist_n_prime = calculator.sidechain_sidechain_distances_hel[1, 5]
    assert dist_n_prime == 30.0, f"Expected Sum (30.0) for N'->Helix, got {dist_n_prime}"

    # Case B: C'(8) to Helix(5) -> Valid
    # Path: Coil(8->7) + Helix(7->5). 
    # Dist: MockCoil(20.0) + MockHelix(10.0) = 30.0
    dist_c_prime = calculator.sidechain_sidechain_distances_hel[5, 8]
    assert dist_c_prime == 30.0, f"Expected Sum (30.0) for C'->Helix, got {dist_c_prime}"

    # Case C: "Phantom" Interaction C''(9) to Helix(5)
    # 9 is NOT N' or C'. Should be 99 (ignored).
    dist_phantom = calculator.sidechain_sidechain_distances_hel[5, 9]
    assert dist_phantom == 99.0, f"Expected Phantom Interaction to be 99.0, got {dist_phantom}"

def test_terminal_sidechain_rc_linear_formula(calculator):
    """
    Verify Random Coil interactions involving the N-terminus use 
    the linear distance formula.
    Target: Index 5 (K).
    N-term is at -1 (conceptually/physically at 0).
    
    Correct Physics: Separation = Index 5 - Index 0 = 5 residues.
    Formula: r = 0.1 + (N+1)*2
    r = 0.1 + (5+1)*2 = 12.1 A.
    """
    # Check the stored distance in the calculator
    calc_dist = calculator.terminal_sidechain_distances_nterm[5]
    
    # With the fix in `_assign...` (using idx), this should be 12.1
    expected = 12.1
    
    assert np.isclose(calc_dist, expected), f"Expected linear dist {expected}, got {calc_dist}"

def test_electrostatic_matrix_symmetry(calculator):
    """
    Verify that the electrostatic matrix is symmetric and has correct shape.
    Sequence length is 11 (9 AA + 2 Caps).
    """
    matrix = calculator.get_dG_sidechain_sidechain_electrost()
    
    assert matrix.shape == (11, 11)
    
    # Check a value that should be non-zero (K at 1 and K at 5)
    # Both charged in mock.
    val = matrix[1, 5]
    assert val != 0.0, "Expected non-zero energy for charged pair"
    assert matrix[5, 1] == val, "Matrix should be symmetric"