import pytest
import numpy as np
from pyagadir.models import AGADIR, ModelResult

# Test data for reuse
@pytest.fixture
def basic_model():
    """Basic AGADIR model with default parameters."""
    return AGADIR()

@pytest.fixture
def test_sequence():
    """Basic test sequence."""
    return "AAAAAAAAAA"  # Simple 10-residue polyalanine sequence

def test_model_initialization():
    """Test model initialization with default and custom parameters."""
    # Test default initialization
    model = AGADIR()
    assert model._method == "1s"
    assert model.T_celsius == 4.0
    assert model.molarity == 0.15
    assert model.pH == 7.0
    
    # Test custom initialization
    model = AGADIR(method="r", T=25.0, M=0.1, pH=6.0)
    assert model._method == "r"
    assert model.T_celsius == 25.0
    assert model.molarity == 0.1
    assert model.pH == 6.0

def test_invalid_model_parameters():
    """Test that invalid parameters raise appropriate exceptions."""
    with pytest.raises(ValueError):
        AGADIR(method="invalid")
    
    with pytest.raises(ValueError):
        AGADIR(T=-274)  # Below absolute zero
        
    with pytest.raises(ValueError):
        AGADIR(M=-1)  # Negative molarity
        
    with pytest.raises(ValueError):
        AGADIR(pH=15)  # pH out of range

def test_invalid_sequence(basic_model):
    """Test invalid sequence inputs."""
    with pytest.raises(ValueError):
        basic_model.predict("AAA1AA")  # Invalid character
        
    with pytest.raises(ValueError):
        basic_model.predict("AAA")  # Too short
        
    with pytest.raises(ValueError):
        basic_model.predict("")  # Empty sequence

def test_invalid_terminal_modifications(basic_model):
    """Test invalid terminal modification inputs."""
    seq = "AAAAAAAAAA"
    
    with pytest.raises(ValueError):
        basic_model.predict(seq, ncap="invalid")
        
    with pytest.raises(ValueError):
        basic_model.predict(seq, ccap="invalid")

def test_prediction_output_format(basic_model, test_sequence):
    """Test that prediction output has correct format."""
    result = basic_model.predict(test_sequence)
    
    assert isinstance(result, ModelResult)
    assert isinstance(result.helical_propensity, np.ndarray)
    assert len(result.helical_propensity) == len(test_sequence)
    assert isinstance(result.percent_helix, float)
    assert 0 <= result.percent_helix <= 100

def test_terminal_modifications(basic_model, test_sequence):
    """Test that terminal modifications are handled correctly."""
    # Test acetylation
    result_ac = basic_model.predict(test_sequence, ncap="Ac")
    assert result_ac.seq_list[0] == "Ac"
    
    # Test succinylation
    result_sc = basic_model.predict(test_sequence, ncap="Sc")
    assert result_sc.seq_list[0] == "Sc"
    
    # Test amidation
    result_am = basic_model.predict(test_sequence, ccap="Am")
    assert result_am.seq_list[-1] == "Am"

def test_method_consistency(test_sequence):
    """Test that different methods give reasonable results."""
    model_r = AGADIR(method="r")
    model_1s = AGADIR(method="1s")
    
    result_r = model_r.predict(test_sequence)
    result_1s = model_1s.predict(test_sequence)
    
    # Results should be different but both valid
    assert not np.array_equal(result_r.helical_propensity, result_1s.helical_propensity)
    assert all(0 <= x <= 100 for x in result_r.helical_propensity)
    assert all(0 <= x <= 100 for x in result_1s.helical_propensity)

def test_temperature_effect(test_sequence):
    """Test that temperature changes affect the prediction."""
    model_cold = AGADIR(T=4.0)
    model_hot = AGADIR(T=37.0)
    
    result_cold = model_cold.predict(test_sequence)
    result_hot = model_hot.predict(test_sequence)
    
    # Higher temperature should generally result in lower helicity
    assert result_cold.percent_helix != result_hot.percent_helix

def test_pH_effect(test_sequence):
    """Test that pH changes affect the prediction for charged sequences."""
    charged_seq = "AEAAAKAAAA"  # Sequence with charged residues
    
    model_acidic = AGADIR(pH=4.0)
    model_basic = AGADIR(pH=9.0)
    
    result_acidic = model_acidic.predict(charged_seq)
    result_basic = model_basic.predict(charged_seq)
    
    # pH should affect helicity for charged sequences
    assert result_acidic.percent_helix != result_basic.percent_helix
    