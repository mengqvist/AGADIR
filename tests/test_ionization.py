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
    # Lys stays charged at pH 7, so the ionisation free energy is ~0. Not exactly 0: Lys2 sits at N2 in the
    # first-turn field (FIRST_TURN_CATION), which lowers its helix pKa by ~0.14 units, and the far-from-pKa
    # residual is ~0.001 kcal/mol.
    c = _calc("AKAAAAKAAAAKAAGY", "Ac", "Am", pH=7.0)
    assert abs(c.get_dG_ionization()) < 2e-3


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


def test_tyr_side_chain_coil_pka():
    """Tyr base pKa 9.5 (params README): in the unstructured control KR-1c (Lacroix 1998 Table 1, apparent 9.4
    +/- 0.1 at 278 K) the coil-state Tyr1 side chain is half ionised at pH 9.4."""
    c = _calc("YGGSAGAGAGAKRGAA", None, "Am", pH=9.4, T=5.0, M=0.005)
    assert abs(float(c.modified_seq_ionization_rc[0])) == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("pH", [3.0, 6.0, 6.5, 7.0, 8.0])
def test_his_his_pair_matches_exact_microstate_enumeration(pH):
    """His-His has no row of its own in Lacroix 1998 Table VI; it takes the HelixRest/RcoilRest
    distances ("charged pairs not included before"). Its electrostatic segment energy (pair term
    at the solver's fractional charges + macrodipole + ionisation free energy) must equal the
    exact free energy of the four protonation microstates, so no further protonation scaling
    belongs on the pair term."""
    seq = "AAAAAHAAAHAAAAA"
    c = EnergyCalculator(seq=seq, i=0, j=len(seq) + 2, pH=pH, T=25.0, ionic_strength=0.1, ncap="Ac", ccap="Am")
    his = [k for k, aa in enumerate(c.seq_list) if aa == "H"]
    n, cc = c.get_dG_sidechain_macrodipole()
    model = (np.sum(c.get_dG_sidechain_sidechain_electrost()) + np.sum(n + cc) * c.dipole_temperature_factor()
             + c.get_dG_ionization())

    rt = 1.9865e-3 * c.T_kelvin
    x = [10 ** (c.seq_pka[k] - pH) for k in his]

    def free_energy(helix):
        d = (c.sidechain_sidechain_distances_hel if helix else c.charged_sidechain_distances_rc)[his[0], his[1]]
        w = c._electrostatic_interaction_energy(1.0, 1.0, d)
        phi = [c.sidechain_dipole_potential[k] if helix else 0.0 for k in his]
        z = sum(x[0] ** s1 * x[1] ** s2 * math.exp(-(s1 * phi[0] + s2 * phi[1] + s1 * s2 * w) / rt)
                for s1 in (0, 1) for s2 in (0, 1))
        return -rt * math.log(z)

    assert model == pytest.approx(free_energy(True) - free_energy(False), abs=0.005)
