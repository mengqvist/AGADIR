"""Ionisation solver and electrostatic free energy (electrostatics/common.py, chemistry.ionization_free_energy)."""
import math

import numpy as np
import pytest

from pyagadir.chemistry import R_KCAL, ionization_free_energy
from pyagadir.energies import EnergyCalculator

RT = R_KCAL * 273.15


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
    """The three Lys at pH 7 are 99.9 % charged in both states; their ionisation cost is ~1e-3 kcal/mol."""
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
    belongs on the pair term. The charge-dependent non-Coulomb terms (here the His charged/neutral propensities) are
    part of both: in the model through the solver's pKa (point-charge law), in each microstate as its own energy."""
    seq = "AAAAAHAAAHAAAAA"
    c = EnergyCalculator(seq=seq, i=0, j=len(seq) + 2, pH=pH, T=25.0, ionic_strength=0.1, ncap="Ac", ccap="Am")
    his = [k for k, aa in enumerate(c.seq_list) if aa == "H"]
    n, cc = c.get_dG_sidechain_macrodipole()
    c_a, c_ab = c._nonCoulomb_charge_coefficients()
    p = [abs(float(c.modified_seq_ionization_hel[k])) for k in his]
    nonC = (sum(c_a.get(k, 0.0) * pk for k, pk in zip(his, p))
            + c_ab.get((his[0], his[1]), 0.0) * p[0] * p[1])
    model = (np.sum(c.get_dG_sidechain_sidechain_electrost()) + np.sum(n + cc) * c.dipole_temperature_factor()
             + c.get_dG_ionization() + nonC)

    rt = R_KCAL * c.T_kelvin
    x = [10 ** (c.seq_pka[k] - pH) for k in his]

    def free_energy(helix):
        d = (c.sidechain_sidechain_distances_hel if helix else c.charged_sidechain_distances_rc)[his[0], his[1]]
        w = c._electrostatic_interaction_energy(1.0, 1.0, d)
        phi = [c.sidechain_dipole_potential[k] + c_a.get(k, 0.0) if helix else 0.0 for k in his]
        w += c_ab.get((his[0], his[1]), 0.0) if helix else 0.0
        z = sum(x[0] ** s1 * x[1] ** s2 * math.exp(-(s1 * phi[0] + s2 * phi[1] + s1 * s2 * w) / rt)
                for s1 in (0, 1) for s2 in (0, 1))
        return -rt * math.log(z)

    assert model == pytest.approx(free_energy(True) - free_energy(False), abs=0.005)


def _solver_mean_field_energy(c, helix):
    """E = sum_a q_a phi_a + sum_{a<b} q_a q_b W_ab over the sites, distances (< 40 A) and field the ionisation solver
    uses, at its converged charges for that state."""
    import itertools
    n = len(c.seq_list)
    sites = []  # (kind, idx, charge)
    if c.seq_list[0] != "Ac":
        sites.append(("N", None, c.modified_nterm_ionization_hel if helix else c.modified_nterm_ionization_rc))
    q_sc = c.modified_seq_ionization_hel if helix else c.modified_seq_ionization_rc
    for k, aa in enumerate(c.seq_list):
        if aa in c.neg_charge_aa + c.pos_charge_aa:
            sites.append(("S", k, float(q_sc[k])))
    if c.seq_list[-1] != "Am":
        sites.append(("C", None, c.modified_cterm_ionization_hel if helix else c.modified_cterm_ionization_rc))

    def dist(a, b):
        if a[0] != "S" and b[0] != "S":
            return c._calculate_r(n - 1)
        if a[0] != "S" or b[0] != "S":
            t, k = (a[0], b[1]) if a[0] != "S" else (b[0], a[1])
            arr = {("N", True): c.terminal_sidechain_distances_nterm, ("N", False): c.terminal_sidechain_distances_nterm_rc,
                   ("C", True): c.terminal_sidechain_distances_cterm, ("C", False): c.terminal_sidechain_distances_cterm_rc}
            d = arr[(t, helix)][k]
        else:
            d = (c.sidechain_sidechain_distances_hel if helix else c.charged_sidechain_distances_rc)[a[1], b[1]]
        return 99.0 if np.isnan(d) else float(d)

    E = 0.0
    for a, b in itertools.combinations(sites, 2):
        r = dist(a, b)
        if r < 40.0:
            E += c._electrostatic_interaction_energy(a[2], b[2], r)
    if helix:
        for kind, k, q in sites:
            if kind == "N":
                E += c._electrostatic_interaction_energy(c.mu_helix, q, c.terminal_macrodipole_distance_nterm)
            elif kind == "C":
                E += c._electrostatic_interaction_energy(-c.mu_helix, q, c.terminal_macrodipole_distance_cterm)
            elif kind == "S":
                E += q * c.sidechain_dipole_potential[k]
    return E


@pytest.mark.parametrize("seq,ncap,ccap,pH,M,i,j", [
    ("KAAAAKAAAAKAAAAY", None, None, 7.0, 0.005, 0, 9),     # Lys1 and its own amine; t-t pair
    ("AEAAAKAAAAKAAAAD", None, None, 7.0, 0.005, 3, 8),     # side chain pairs without helix geometry
    ("AKAAAAKAAAAKAAGY", "Ac", "Am", 7.0, 0.01, 3, 8),      # straddling pair
    ("YGGSAEAAAKAAKRAAAE", None, None, 8.0, 0.005, 8, 9),   # ungated terminus-side chain pairs
    ("AEAAAKEAAAKEAAAKA", "Sc", "Am", 4.5, 0.01, 0, 9),     # succinyl
])
def test_point_energy_terms_are_the_solver_mean_field_energy(seq, ncap, ccap, pH, M, i, j):
    """Point-charge law: the electrostatic energy terms equal E(helix) - E(coil) of the solver's own pairs and field,
    so that with dG_ionization they are the mean-field free energy of the coupled titration."""
    c = EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=0.0, ionic_strength=M, ncap=ncap, ccap=ccap)
    tN, tC = c.get_dG_terminals_macrodipole()
    sN, sC = c.get_dG_sidechain_macrodipole()
    eN, eC = c.get_dG_terminals_sidechain_electrost()
    terms = (np.sum(tN) + np.sum(tC) + np.sum(sN + sC) * c.dipole_temperature_factor() + np.sum(eN + eC)
             + np.sum(c.get_dG_sidechain_sidechain_electrost()) + c.get_dG_terminal_terminal_electrost())
    assert terms == pytest.approx(_solver_mean_field_energy(c, True) - _solver_mean_field_energy(c, False), abs=1e-9)


@pytest.mark.parametrize("seq,ncap,ccap,pH,i,j", [
    ("SAAEAAAAKAAAAEAAKAQAAD", "Ac", "Am", 4.5, 1, 16),  # capping box Glu at N3, Lys/Glu interpolation
    ("AAQAADAAAAQAAEAANAAKA", "Ac", "Am", 4.0, 0, 21),   # Table V Gln-Asp-, Gln-Glu-, Glu--Asn
    ("DAKAAEDAAAKAAAAD", None, None, 4.2, 0, 16),       # free termini, Asp C-cap, Petukhov-type N-terminus
    ("YGGSHAAAAKAAAAEHAAAKA", "Ac", "Am", 6.3, 4, 15),   # His N-cap and interior His
])
def test_point_law_charges_minimise_the_mean_field_free_energy(seq, ncap, ccap, pH, i, j):
    """Point-charge law: at the solver's helix-state charges, the helix-state mean-field free energy (ionisation
    entropy + Coulomb energy terms + charge-dependent non-Coulomb terms) is stationary in every side chain's charged
    fraction, which is what feeding dG_nonC/dp into the pKa achieves (Lacroix 1998: such terms enter the pKa)."""
    c = EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=0.0, ionic_strength=0.01, ncap=ncap, ccap=ccap)
    rt = R_KCAL * c.T_kelvin

    def helix_energy():
        tN, tC = c.get_dG_terminals_macrodipole()
        sN, sC = c.get_dG_sidechain_macrodipole()
        eN, eC = c.get_dG_terminals_sidechain_electrost()
        return (np.sum(tN) + np.sum(tC) + np.sum(sN + sC) * c.dipole_temperature_factor() + np.sum(eN + eC)
                + np.sum(c.get_dG_sidechain_sidechain_electrost()) + c.get_dG_terminal_terminal_electrost()
                + sum(float(np.sum(v)) for v in c._charge_dependent_nonCoulomb_terms().values()))

    q0 = c.modified_seq_ionization_hel.copy()
    for k in range(c.ncap_idx, c.ccap_idx + 1):
        aa = c.seq_list[k]
        if aa not in c.neg_charge_aa + c.pos_charge_aa:
            continue
        z = -1.0 if aa in c.neg_charge_aa else 1.0
        p = abs(float(q0[k]))
        if not 0.02 < p < 0.98:
            continue
        h = 1e-4
        e = []
        for dp in (h, -h):
            c.modified_seq_ionization_hel = q0.copy()
            c.modified_seq_ionization_hel[k] = z * (p + dp)
            e.append(helix_energy())
        c.modified_seq_ionization_hel = q0
        ln_x = (c.seq_pka[k] - pH if z > 0 else pH - c.seq_pka[k]) * math.log(10)
        grad = rt * math.log(p / (1 - p)) - rt * ln_x + (e[0] - e[1]) / (2 * h)
        assert abs(grad) < 0.03, (aa, k, grad)
