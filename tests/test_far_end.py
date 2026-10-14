"""Far end of the helix macrodipole (energies.FEH_FAR_APBS): inside the helix, the far-end half charge scales with the
constant while the nearest-end (Munoz 1995-II eq. 11) term does not; flank residues outside the helix keep the
unscaled half-charge law."""
import pytest

import pyagadir.energies as E


def _potential(seq, i, j, idx):
    calc = E.EnergyCalculator(seq=seq, i=i, j=j, pH=7.0, T=5.0, ionic_strength=0.05, ncap="Ac", ccap="Am")
    return calc._sidechain_dipole_potential(idx)


SEQ = "AAEAAAAAAAAAAAAAKA"  # Glu near the N end, Lys near the C end (tokens: Ac at 0, Am at 19)


def test_constant_value():
    assert E.FEH_FAR_APBS == pytest.approx(0.82)


@pytest.mark.parametrize("idx, near, far", [(3, 0, 1), (17, 1, 0)])
def test_far_end_scales_near_end_does_not(idx, near, far, monkeypatch):
    n = len(SEQ) + 2
    full = _potential(SEQ, 0, n, idx)
    monkeypatch.setattr(E, "FEH_FAR_APBS", 1.0)
    unscaled = _potential(SEQ, 0, n, idx)
    monkeypatch.setattr(E, "FEH_FAR_APBS", 0.0)
    off = _potential(SEQ, 0, n, idx)
    assert full[near] == pytest.approx(unscaled[near]) == pytest.approx(off[near])
    assert off[far] == pytest.approx(0.0)
    assert unscaled[far] != pytest.approx(0.0)
    assert full[far] == pytest.approx(0.82 * unscaled[far])


def test_flank_unscaled(monkeypatch):
    seq = "AEAAAAAAAAAAAAAAAA"  # Glu at token 2; helix from token 4 to 15, so the Glu is an N-terminal flank residue
    a = _potential(seq, 4, 12, 2)
    monkeypatch.setattr(E, "FEH_FAR_APBS", 1.0)
    b = _potential(seq, 4, 12, 2)
    assert a[0] != pytest.approx(0.0)
    assert a == pytest.approx(b)
