"""Ionisation solver and electrostatic free energy (energies.py, chemistry.ionization_free_energy)."""
import math

import numpy as np
import pytest

from pyagadir.chemistry import ionization_free_energy
from pyagadir.energies import EnergyCalculator

RT = 1.9865e-3 * 273.15


@pytest.mark.parametrize("ln_x", [-6.0, -2.3, 0.0, 1.7, 8.0])
@pytest.mark.parametrize("psi", [-1.5, -0.4, 0.0, 0.3, 1.2])
def test_ionization_free_energy_completes_the_single_group_free_energy(ln_x, psi):
    """q * psi + I must equal the exact free energy of one group in a fixed field,
    -RT ln[(1 + x e^(-psi/RT)) / (1 + x)], with q the self-consistent charged fraction."""
    x = math.exp(ln_x)
    q = x * math.exp(-psi / RT) / (1 + x * math.exp(-psi / RT))
    exact = -RT * math.log((1 + x * math.exp(-psi / RT)) / (1 + x))
    I = ionization_free_energy(ln_x, psi, q, RT)
    assert I >= -1e-12
    assert math.isclose(q * psi + I, exact, abs_tol=1e-9)


def test_ionization_free_energy_vanishes_for_fixed_charge():
    """A group that stays fully charged (or neutral) pays no ionisation free energy."""
    assert abs(ionization_free_energy(25.0, -1.0, 1.0, RT)) < 1e-9
    assert abs(ionization_free_energy(-25.0, 1.0, 0.0, RT)) < 1e-9


def _calc(seq, ncap, ccap, pH, T=0.0, M=0.01, i=None, j=None):
    n = len(seq) + (ncap is not None) + (ccap is not None)
    return EnergyCalculator(seq=seq, i=0 if i is None else i, j=n if j is None else j, pH=pH, T=T,
                            ionic_strength=M, ncap=ncap, ccap=ccap)


def test_segment_ionization_free_energy_zero_when_groups_fully_charged():
    c = _calc("AKAAAAKAAAAKAAGY", "Ac", "Am", pH=7.0)
    assert abs(c.get_dG_ionization()) < 1e-3


def test_segment_ionization_free_energy_positive_for_titrating_acid_pairs():
    """Asp next to Arg at pH 2.5: the helix shifts Asp's ionisation, which costs free energy."""
    c = _calc("ADAAARDAAARDAAARY", "Ac", "Am", pH=2.5)
    assert c.get_dG_ionization() > 0.1


def test_terminal_sidechain_geometry_shared_by_solver_and_energy():
    """Free N-terminus on the N-cap: interior side chains use Lacroix Table VI 'N-cap f' in the
    helix and RcoilRest in the coil; the residue carrying the terminus uses 2.1 A in both states."""
    c = _calc("DAKAAAAKAAAAKAAGY", None, "Am", pH=7.0)
    hel6 = c.table_6_helix_lacroix
    coil6 = c.table_6_coil_lacroix
    assert c.terminal_sidechain_modelled_nterm[2]
    assert np.isclose(c.terminal_sidechain_distances_nterm[2], hel6.loc["N-cap f", "i+2"])
    assert np.isclose(c.terminal_sidechain_distances_nterm_rc[2], coil6.loc["RcoilRest", "i+2"])
    assert np.isclose(c.terminal_sidechain_distances_nterm[0], 2.1)
    assert np.isclose(c.terminal_sidechain_distances_nterm_rc[0], 2.1)
    assert not c.terminal_sidechain_modelled_nterm[0]


def test_solver_dipole_field_is_the_energy_law():
    """The helix-state field the solver uses equals the side chain-macrodipole energy per unit charge."""
    c = _calc("AAEAAAAKAAAAKAAGY", "Ac", "Am", pH=7.0)
    eN, eC = c.get_dG_sidechain_macrodipole()
    for idx in (3, 8):  # Glu at N2, Lys mid-helix (Ac at index 0)
        q = float(c.modified_seq_ionization_hel[idx])
        assert np.isclose(eN[idx] + eC[idx], q * sum(c._sidechain_dipole_potential(idx)), atol=1e-12)
        assert np.isclose(c.sidechain_dipole_potential[idx],
                          c.dipole_temperature_factor() * sum(c._sidechain_dipole_potential(idx)), atol=1e-12)


def test_n_terminal_tyr_alpha_amino_pka():
    """Free N-terminal Tyr uses its measured alpha-amino pKa (Lacroix 1998 Table 1); others Nterm."""
    assert np.isclose(_calc("YGGSAAAAAAAKRAAA", None, "Am", pH=7.0, T=5.0).nterm_pka, 7.2)
    assert np.isclose(_calc("AGGSAAAAAAAKRAAA", None, "Am", pH=7.0, T=5.0).nterm_pka, 8.00)
    assert np.isnan(_calc("YGGSAAAAAAAKRAAA", "Ac", "Am", pH=7.0, T=5.0).nterm_pka)
