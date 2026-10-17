"""dipole_law="gbx": the GB law with the Table 1 values that were measured with the default law in the loop
transferred to it, and the parameters re-derived under it (params/README.md "gbx")."""
import pytest

import pyagadir.energies as E
from pyagadir.models import AGADIR
from pyagadir.validation import titrated_ionic_strength


def potential(seq, i, j, idx, law, T, I, pH, ncap="Ac", ccap="Am"):
    ec = E.EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=T, ionic_strength=I, ncap=ncap, ccap=ccap, dipole_law=law)
    return float(ec.sidechain_dipole_potential[idx])


@pytest.mark.parametrize("aa,k", [("D", 1), ("D", 2), ("E", 2), ("E", 3), ("D", 4)])
def test_first_turn_reproduces_default_at_reference(aa, k):
    """Petukhov et al. 1998's host, 5 C, 5 mM sodium phosphate pH 7, full-length segment: the transferred GB potential
    equals the default law's (the offset is a mean over the reference's segments, hence the tolerance)."""
    s = list("AAAAAAAARAAARGGY")
    s[k - 1] = aa
    seq = "".join(s)
    I = titrated_ionic_strength(0.005, 7.0, {"sodium_phosphate_M": 0.005, "prepared_pH": 7.0})
    a = potential(seq, 0, len(seq) + 2, k, "munoz", 5.0, I, 7.0)
    b = potential(seq, 0, len(seq) + 2, k, "gbx", 5.0, I, 7.0)
    assert b == pytest.approx(a, abs=0.03)


@pytest.mark.parametrize("aa", ["D", "E", "K", "R", "H"])
def test_ncap_reproduces_default_at_reference(aa):
    """Doig & Baldwin 1995 Table 1 host, 0 C, 1 M, N-cap = the free N-terminal residue."""
    seq = aa + "AKAAAAKAAAAKAAGY"
    I = titrated_ionic_strength(1.0, 5.0)
    a = potential(seq, 0, len(seq) + 1, 0, "munoz", 0.0, I, 5.0, ncap=None)
    b = potential(seq, 0, len(seq) + 1, 0, "gbx", 0.0, I, 5.0, ncap=None)
    assert b == pytest.approx(a, abs=0.03)


def test_gbx_differs_from_gb_only_by_class_offsets():
    seq = "AAAADAAAAAAAAAAA"
    g = potential(seq, 0, 18, 5, "gb", 0.0, 0.01, 7.0)
    x = potential(seq, 0, 18, 5, "gbx", 0.0, 0.01, 7.0)
    assert x - g == pytest.approx(E.gbx_tables()["transfer"][("D", "Ncen")], abs=1e-9)


def test_position_classes():
    c = E.gbx_position_class
    assert c("D", 0, 9, "Ac") == "Ncap" and c("D", 9, 0, "Ac") == "Ccap"
    assert c("D", 2, 9, "Ac") == "N2" and c("D", 2, 9, "S") == "N2r"
    assert c("E", 3, 9, "S") == "N3cb" and c("E", 3, 9, "A") == "N3r" and c("E", 3, 9, "Ac") == "N3"
    assert c("K", 2, 9, "Ac") == "Ncen" and c("E", 7, 2, "Ac") == "Ncen"


def test_overrides_apply_only_under_gbx():
    tabs = E.gbx_tables()
    if not tabs["overrides"]:
        pytest.skip("no table overrides")
    table, row, col, value = tabs["overrides"][0]
    assert E.EnergyCalculator.law_params("gbx")[table].loc[row, col] == pytest.approx(value)
    assert E.EnergyCalculator.load_params()[table].loc[row, col] != pytest.approx(value)


def test_first_turn_n2_offset_is_law_specific():
    ec = {law: E.EnergyCalculator(seq="AKAAAAKAAAAKAAGY", i=0, j=18, pH=7.0, T=0.0, ionic_strength=0.02, ncap="Ac",
                                  ccap="Am", dipole_law=law) for law in ("munoz", "gbx")}
    assert ec["munoz"]._first_turn_n2_offset("K") == pytest.approx(E.FIRST_TURN_N2_OFFSET["K"])
    assert ec["gbx"]._first_turn_n2_offset("K") == pytest.approx(0.0)
    assert ec["gbx"]._first_turn_n2_offset("F") == pytest.approx(E.FIRST_TURN_N2_OFFSET["F"])


def _helix(seq, law):
    m = AGADIR(method="1s", T=0.0, M=0.019, pH=7.01, dipole_law=law)
    return m.predict(seq, ncap="Ac", ccap="Am").get_percent_helix()


@pytest.mark.parametrize("host,guest,measured", [
    ("AAAAAQAAAAQAAGY", "KAAAAQAAAAQAAGY", -12.0), ("AAAAAQAAAAQAAGY", "RAAAAQAAAAQAAGY", -12.0),
    ("AAAAAAKAAAAKAAGY", "AKAAAAKAAAAKAAGY", -17.0), ("AAAAAAKAAAAKAAGY", "ARAAAAKAAAAKAAGY", -15.0)])
def test_first_turn_cation_contrast(host, guest, measured):
    """N1 scan of Cochran, Penel & Doig 2001 (Protein Sci. 10, 463) and N2 scan of Cochran & Doig 2001 (Protein Sci.
    10, 1305), Ac-...-NH2, 0 C, 0.019 M, pH 7.01: helix percent of Lys or Arg minus Ala in the same host (each
    measured +-3). The host's cells cancel and Lys/Arg have no first-turn cells of their own, so the contrast tests
    the side chain-macrodipole law."""
    assert _helix(guest, "gbx") - _helix(host, "gbx") == pytest.approx(measured, abs=3.0)


def test_gbx_runs_end_to_end():
    for seq, nc, cc in (("AKAAAEKAAAEKAAAEA", "Ac", "Am"), ("YGGSKAAAAAAARAAA", None, "Am"), ("EDAVEALRKLQAGGY", None, None)):
        h = AGADIR(method="1s", T=4.0, M=0.1, pH=7.0, dipole_law="gbx").predict(seq, ncap=nc, ccap=cc).get_percent_helix()
        assert 0.0 <= h <= 100.0
