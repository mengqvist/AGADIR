"""validation.titrated_ionic_strength: the ionic strength of a sample whose pH was set with HCl or NaOH."""
import pytest

from pyagadir.validation import load_buffer, titrated_ionic_strength

LACROIX = {"sodium_phosphate_M": 0.0025, "prepared_pH": 7.0}


def test_no_buffer_adds_the_free_acid_or_base():
    assert titrated_ionic_strength(0.01, 2.0) == pytest.approx(0.01 + 0.01, rel=1e-6)
    assert titrated_ionic_strength(0.01, 12.0) == pytest.approx(0.01 + 0.01, rel=1e-6)
    assert titrated_ionic_strength(0.01, 7.0) == pytest.approx(0.01, abs=1e-6)


@pytest.mark.parametrize("pH,expected", [(3.0, 0.00447), (5.0, 0.00349), (7.0, 0.00443), (9.0, 0.00744),
                                         (11.0, 0.00882), (12.0, 0.01982)])
def test_lacroix_phosphate_buffer(pH, expected):
    """2.5 mM sodium phosphate made up at pH 7.0, then titrated with HCl or NaOH (Lacroix et al. 1998)."""
    assert titrated_ionic_strength(0.005, pH, LACROIX) == pytest.approx(expected, abs=2e-5)


def test_lacroix_files_record_their_buffer():
    for f in ("lacroix_figure_3_data.json", "lacroix_figure_4_data.json"):
        b = load_buffer(f)
        assert b["sodium_phosphate_M"] == 0.0025 and b["prepared_pH"] == 7.0
