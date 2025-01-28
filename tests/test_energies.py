import pytest
import numpy as np
from pyagadir.energies import PrecomputeParams

@pytest.fixture
def precompute_instance():
    """Fixture to create an instance of PrecomputeParams with test inputs."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    return (PrecomputeParams(seq, i=0, j=20, pH=0.0, T=25.0, ionic_strength=0.1),
           PrecomputeParams(seq, i=0, j=22, pH=14.0, T=25.0, ionic_strength=0.1, ncap="Ac", ccap="Am"),
           PrecomputeParams(seq, i=2, j=16, pH=14.0, T=25.0, ionic_strength=0.1),
           PrecomputeParams(seq, i=2, j=18, pH=14.0, T=25.0, ionic_strength=0.1, ncap="Sc", ccap="Am"),)


def test_seq_list(precompute_instance):
    """Test the seq_list property."""
    instance1, instance2, instance3, instance4 = precompute_instance
    assert instance1.seq_list[0] == 'A'
    assert instance1.seq_list[1] == 'C'
    assert instance1.seq_list[2] == 'D'
    assert instance1.seq_list[3] == 'E'
    assert instance1.seq_list[4] == 'F'
    assert instance1.seq_list[5] == 'G'
    assert instance1.seq_list[6] == 'H'
    assert instance1.seq_list[7] == 'I'
    assert instance1.seq_list[8] == 'K'
    assert instance1.seq_list[9] == 'L'
    assert instance1.seq_list[10] == 'M'
    assert instance1.seq_list[11] == 'N'
    assert instance1.seq_list[12] == 'P'
    assert instance1.seq_list[13] == 'Q'
    assert instance1.seq_list[14] == 'R'
    assert instance1.seq_list[15] == 'S'
    assert instance1.seq_list[16] == 'T'
    assert instance1.seq_list[17] == 'V'
    assert instance1.seq_list[18] == 'W'
    assert instance1.seq_list[19] == 'Y'

    assert instance2.seq_list[0] == 'Ac'
    assert instance2.seq_list[21] == 'Am'

    assert instance3.seq_list[0] == 'A'
    
    assert instance4.seq_list[0] == 'Sc'

def test_find_charged_pairs(precompute_instance):
    """Test the _find_charged_pairs method."""
    instance1, instance2, instance3, instance4 = precompute_instance
    instance1._find_charged_pairs()
    charged_pairs1 = instance1.charged_pairs
    instance2._find_charged_pairs()
    charged_pairs2 = instance2.charged_pairs
    instance3._find_charged_pairs()
    charged_pairs3 = instance3.charged_pairs
    instance4._find_charged_pairs()
    charged_pairs4 = instance4.charged_pairs

    # Verify expected charged pairs are found
    expected_pairs1 = [
        ("D", "E", 2, 3),  # Example pair
        ("H", "K", 6, 8),  # Example pair
    ]
    expected_pairs2 = [
        ("D", "E", 3, 4),  # Example pair
        ("H", "K", 7, 9),  # Example pair
    ]
    expected_pairs3 = [
        ("D", "E", 2, 3),  # Example pair
        ("H", "K", 6, 8),  # Example pair
    ]
    expected_pairs4 = [
        ("D", "E", 3, 4),  # Example pair
        ("H", "K", 7, 9),  # Example pair
    ]
    assert all(pair in charged_pairs1 for pair in expected_pairs1)
    assert all(pair in charged_pairs2 for pair in expected_pairs2)
    assert all(pair in charged_pairs3 for pair in expected_pairs3)
    assert all(pair in charged_pairs4 for pair in expected_pairs4)

def test_calculate_r(precompute_instance):
    """Test the _calculate_r method."""
    instance1, instance2, instance3, instance4 = precompute_instance

    # Test for a range of inputs
    assert np.isclose(instance1._calculate_r(0), 2.1)
    assert np.isclose(instance1._calculate_r(1), 4.1)
    assert np.isclose(instance1._calculate_r(2), 6.1)
    assert np.isclose(instance1._calculate_r(10), 22.1)

def test_electrostatic_interaction_energy(precompute_instance):
    """Test the _electrostatic_interaction_energy method."""
    instance1, instance2, instance3, instance4 = precompute_instance

    # Input values
    qi = 1.0
    qj = -1.0
    r = 5.0  # Ångströms

    # Negative energy for opposite charges, i.e. attraction
    energy = instance1._electrostatic_interaction_energy(qi, qj, r)
    assert energy < 0
    assert np.isclose(energy, -0.503, atol=0.001)
    energy = instance1._electrostatic_interaction_energy(qj, qi, r)
    assert energy < 0

    # Positive energy for same charge, i.e. repulsion
    energy = instance1._electrostatic_interaction_energy(qi, qi, r)
    assert energy > 0
    energy = instance1._electrostatic_interaction_energy(qj, qj, r)
    assert energy > 0

def test_assign_pka_values(precompute_instance):
    """Test the _assign_pka_values method."""
    instance1, instance2, instance3, instance4 = precompute_instance

    # Verify pKa values are assigned correctly
    assert instance1.seq_pka is not None
    assert len(instance1.seq_pka) == len(instance1.seq_list)

def test_assign_ionization_states(precompute_instance):
    """Test the _assign_ionization_states method."""
    instance1, instance2, instance3, instance4 = precompute_instance

    # Verify ionization states are assigned correctly, pH = 0.0
    assert instance1.seq_ionization is not None
    assert len(instance1.seq_ionization) == len(instance1.seq_list)
    assert np.isclose(instance1.seq_ionization[0], 0.0, atol=0.001) # A
    assert np.isclose(instance1.seq_ionization[1], 0.0, atol=0.001) # C
    assert np.isclose(instance1.seq_ionization[2], 0.0, atol=0.001) # D
    assert np.isclose(instance1.seq_ionization[3], 0.0, atol=0.001) # E
    assert np.isclose(instance1.seq_ionization[4], 0.0, atol=0.001) # F
    assert np.isclose(instance1.seq_ionization[5], 0.0, atol=0.001) # G
    assert np.isclose(instance1.seq_ionization[6], 1.0, atol=0.001) # H
    assert np.isclose(instance1.seq_ionization[7], 0.0, atol=0.001) # I
    assert np.isclose(instance1.seq_ionization[8], 1.0, atol=0.001) # K
    assert np.isclose(instance1.seq_ionization[9], 0.0, atol=0.001) # L
    assert np.isclose(instance1.seq_ionization[10], 0.0, atol=0.001) # M
    assert np.isclose(instance1.seq_ionization[11], 0.0, atol=0.001) # N
    assert np.isclose(instance1.seq_ionization[12], 0.0, atol=0.001) # P
    assert np.isclose(instance1.seq_ionization[13], 0.0, atol=0.001) # Q
    assert np.isclose(instance1.seq_ionization[14], 1.0, atol=0.001) # R
    assert np.isclose(instance1.seq_ionization[15], 0.0, atol=0.001) # S
    assert np.isclose(instance1.seq_ionization[16], 0.0, atol=0.001) # T
    assert np.isclose(instance1.seq_ionization[17], 0.0, atol=0.001) # V
    assert np.isclose(instance1.seq_ionization[18], 0.0, atol=0.001) # W
    assert np.isclose(instance1.seq_ionization[19], 0.0, atol=0.001) # Y
    assert np.isclose(instance1.nterm_ionization, 1.0, atol=0.001)
    assert np.isclose(instance1.cterm_ionization, 0.0, atol=0.001)

    # Verify ionization states are assigned correctly, pH = 14.0
    assert instance2.seq_ionization is not None
    assert len(instance2.seq_ionization) == len(instance2.seq_list)
    assert np.isclose(instance2.seq_ionization[0], 0.0, atol=0.001) # Ac
    assert np.isclose(instance2.seq_ionization[1], 0.0, atol=0.001) # A
    assert np.isclose(instance2.seq_ionization[2], -1.0, atol=0.001) # C
    assert np.isclose(instance2.seq_ionization[3], -1.0, atol=0.001) # D
    assert np.isclose(instance2.seq_ionization[4], -1.0, atol=0.001) # E
    assert np.isclose(instance2.seq_ionization[5], 0.0, atol=0.001) # F
    assert np.isclose(instance2.seq_ionization[6], 0.0, atol=0.001) # G
    assert np.isclose(instance2.seq_ionization[7], 0.0, atol=0.001) # H
    assert np.isclose(instance2.seq_ionization[9], 0.0, atol=0.001) # K
    assert np.isclose(instance2.seq_ionization[10], 0.0, atol=0.001) # L
    assert np.isclose(instance2.seq_ionization[11], 0.0, atol=0.001) # M
    assert np.isclose(instance2.seq_ionization[12], 0.0, atol=0.001) # N
    assert np.isclose(instance2.seq_ionization[13], 0.0, atol=0.001) # P
    assert np.isclose(instance2.seq_ionization[14], 0.0, atol=0.001) # Q
    assert np.isclose(instance2.seq_ionization[15], 0.4, atol=0.1) # R
    assert np.isclose(instance2.seq_ionization[16], 0.0, atol=0.001) # S
    assert np.isclose(instance2.seq_ionization[17], 0.0, atol=0.001) # T
    assert np.isclose(instance2.seq_ionization[18], 0.0, atol=0.001) # V
    assert np.isclose(instance2.seq_ionization[19], 0.0, atol=0.001) # W
    assert np.isclose(instance2.seq_ionization[20], -1.0, atol=0.001) # Y
    assert np.isclose(instance2.seq_ionization[21], 0.0, atol=0.001) # Am
    assert instance2.nterm_ionization == 0.0
    assert instance2.cterm_ionization == 0.0

    assert np.isclose(instance3.nterm_ionization, 0.0, atol=0.001)
    assert np.isclose(instance3.cterm_ionization, -1.0, atol=0.001)

    assert np.isclose(instance4.nterm_ionization, -1.0, atol=0.001) # Sc is acidic and charged!
    assert np.isclose(instance4.cterm_ionization, 0.0, atol=0.001)

def test_assign_modified_ionization_states(precompute_instance):
    """Test the _assign_modified_ionization_states method."""
    instance1, instance2, instance3, instance4 = precompute_instance

    # Test instance 1
    # Verify modified ionization states for instance1 (pH = 0.0)
    assert instance1.modified_seq_ionization_hel is not None
    assert len(instance1.modified_seq_ionization_hel) == len(instance1.seq_list)

    # Modified ionization should reflect interaction with the helix macrodipole
    # Specific residues should remain ionized at low pH
    assert np.isclose(instance1.modified_seq_ionization_hel[2], 0.0, atol=0.1)  # D (Asp)
    assert np.isclose(instance1.modified_seq_ionization_hel[6], 1.0, atol=0.1)  # H (His)
    assert np.isclose(instance1.modified_seq_ionization_hel[8], 1.0, atol=0.1)  # K (Lys)

    # Test instance 2 (pH = 14.0, Acetylated N-term and Amide C-term)
    assert instance2.modified_seq_ionization_hel is not None
    assert len(instance2.modified_seq_ionization_hel) == len(instance2.seq_list)

    # Modified ionization states for highly basic pH
    assert np.isclose(instance2.modified_seq_ionization_hel[2], -1.0, atol=0.1)  # C (Cys)
    assert np.isclose(instance2.modified_seq_ionization_hel[3], -1.0, atol=0.1)  # D (Asp)
    assert np.isclose(instance2.modified_seq_ionization_hel[4], -1.0, atol=0.1)  # E (Glu)
    assert np.isclose(instance2.modified_seq_ionization_hel[15], 0.4, atol=0.1)  # R (Arg)

    # Test instance 4 (N-term = Sc, acidic modification)
    assert instance4.modified_seq_ionization_hel is not None
    assert len(instance4.modified_seq_ionization_hel) == len(instance4.seq_list)

    # Sc at N-term should be negatively charged
    assert np.isclose(instance4.modified_nterm_ionization_hel, -1.0, atol=0.1)

    # Ensure convergence of iterative modification
    prev_states = instance1.modified_seq_ionization_hel
    instance1._assign_modified_ionization_states()  # Recompute
    assert np.allclose(instance1.modified_seq_ionization_hel, prev_states, atol=0.01)

# def test_assign_sidechain_macrodipole_distances(precompute_instance):
#     """Test the _assign_sidechain_macrodipole_distances method."""
#     instance = precompute_instance
#     instance._assign_sidechain_macrodipole_distances()

#     # Verify distances are assigned
#     assert instance.sidechain_macrodipole_distances_nterm is not None
#     assert instance.sidechain_macrodipole_distances_cterm is not None

def test_assign_terminal_macrodipole_distances(precompute_instance):
    """Test the _assign_terminal_macrodipole_distances method."""
    instance1, instance2, instance3, instance4 = precompute_instance

    # Verify terminal distances are calculated
    assert instance1.terminal_macrodipole_distances_nterm is not None
    assert instance1.terminal_macrodipole_distances_cterm is not None
    assert np.isclose(instance1.terminal_macrodipole_distances_nterm, 2.1, atol=0.001)
    assert np.isclose(instance1.terminal_macrodipole_distances_cterm, 2.1, atol=0.001)

    assert instance2.terminal_macrodipole_distances_nterm is not None
    assert instance2.terminal_macrodipole_distances_cterm is not None
    assert np.isclose(instance2.terminal_macrodipole_distances_nterm, 2.1, atol=0.001)
    assert np.isclose(instance2.terminal_macrodipole_distances_cterm, 2.1, atol=0.001)

    assert instance3.terminal_macrodipole_distances_nterm is not None
    assert instance3.terminal_macrodipole_distances_cterm is not None
    assert np.isclose(instance3.terminal_macrodipole_distances_nterm, 6.1, atol=0.001)
    assert np.isclose(instance3.terminal_macrodipole_distances_cterm, 6.1, atol=0.001)

    assert instance4.terminal_macrodipole_distances_nterm is not None
    assert instance4.terminal_macrodipole_distances_cterm is not None
    assert np.isclose(instance4.terminal_macrodipole_distances_nterm, 6.1, atol=0.001)
    assert np.isclose(instance4.terminal_macrodipole_distances_cterm, 6.1, atol=0.001)

# def test_assign_charged_sidechain_distances(precompute_instance):
#     """Test the _assign_charged_sidechain_distances method."""
#     instance = precompute_instance
#     instance._assign_charged_sidechain_distances()

#     # Verify charged sidechain distances are assigned
#     assert instance.charged_sidechain_distances_hel is not None
#     assert instance.charged_sidechain_distances_rc is not None


