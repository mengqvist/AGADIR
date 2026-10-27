"""The segments' ionisation solved together (electrostatics/common.py SegmentBatch, models.AGADIR._calc_partition_fxn):
each segment's charges, potentials, ionisation free energy and sweep counts equal those of the same segment solved on its
own (a batch of one), and so do the predictions; the premises the batch rests on (every interacting pair within
PAIR_RANGE residues; a segment-independent coil state); and the non-Coulomb coefficients against whole-energy
differences of the terms themselves.

(Until 2026-10-27 the batch was also checked against results recorded from the per-segment solver it replaced in
2026-10-22; those references were recorded under a macrodipole law and pair rule that no longer exist.)"""
import contextlib
import io

import numpy as np
import pytest

import pyagadir.models as models
from pyagadir.electrostatics.common import PAIR_RANGE
from pyagadir.energies import EnergyCalculator

# (sequence, ncap, ccap, pH, T, M): free, Ac and Sc ends, pH extremes, low and high salt, and every charge-dependent
# non-Coulomb term: Cys/His N-cap, capping box, Asp C-cap, Petukhov and charged-staple motifs, Table IV/V pairs
CASES = [
    ("YGGSAAAAAAAKRAAA", None, "Am", 3.0, 5.0, 0.0044),
    ("YGGSAAAAAAAKRAAA", None, "Am", 11.0, 5.0, 0.0044),
    ("AEAAAKEAAAKEAAAKA", "Ac", "Am", 2.5, 1.0, 0.01),
    ("DAQAAAAQAAAAQAAY", "Sc", "Am", 7.0, 0.0, 0.01),
    ("SAAEDKAAKAAKAY", None, None, 6.0, 0.0, 0.05),
    ("ASAAAKAAEAAKAAGY", "Ac", None, 7.0, 25.0, 0.1),
    ("ACAAEKAAKHAAEAAY", None, "Am", 7.5, 0.0, 0.001),
    ("ATAAEAAKAADAAKY", "Ac", "Am", 4.5, 10.0, 2.0),
    ("AQAAADKAAEFAAHKAAEAARY", "Ac", "Am", 6.0, 0.0, 0.1),
    ("DAKAAAAKAAE", None, None, 7.0, 0.0, 0.01),
    ("AEEEKKKAEEEKKKAEEEKKKAEEEKKKAW", None, None, 7.4, 60.0, 0.1),
]


def _ids(case):
    return f"{case[0]}-{case[3]}"


def _windows(case):
    seq, nc, cc, *_ = case
    L = len(seq) + (nc is not None) + (cc is not None)
    return [(i, j) for j in range(6, L + 1) for i in range(0, L - j + 1)]


def _alone(case, i, j, params):
    seq, nc, cc, pH, T, M = case
    return EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=T, ionic_strength=M, ncap=nc, ccap=cc, params=params)


def _batched(case, params):
    """The segments as models.AGADIR builds them: the first on its own, the rest on its batch, solved together."""
    seq, nc, cc, pH, T, M = case
    windows = _windows(case)
    first = _alone(case, *windows[0], params)
    batch = first.segment_batch
    rest = [EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=T, ionic_strength=M, ncap=nc, ccap=cc, params=params,
                             batch=batch) for i, j in windows[1:]]
    batch.solve_helix(rest)
    return [first] + rest


def _site_q(c, helix):
    seq_q = c.modified_seq_ionization_hel if helix else c.modified_seq_ionization_rc
    nq = c.modified_nterm_ionization_hel if helix else c.modified_nterm_ionization_rc
    cq = c.modified_cterm_ionization_hel if helix else c.modified_cterm_ionization_rc
    return np.array([float(nq) if k == "Nterm" else float(cq) if k == "Cterm" else float(seq_q[i])
                     for k, i in c._ionizable_sites()[0]])


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_every_segment_of_the_batch_equals_the_segment_alone(case):
    params = EnergyCalculator.snapshot_params()
    for b in _batched(case, params):
        a = _alone(case, b.i, b.j, params)
        assert np.allclose(_site_q(b, False), _site_q(a, False), rtol=0, atol=1e-12), (b.i, b.j)
        assert np.allclose(_site_q(b, True), _site_q(a, True), rtol=0, atol=1e-12), (b.i, b.j)
        assert abs(b.dG_ionization - a.dG_ionization) < 1e-12, (b.i, b.j)
        assert b.ionization_sweeps == a.ionization_sweeps, (b.i, b.j)


@pytest.mark.parametrize("case", CASES[:6], ids=_ids)
def test_predictions_equal_the_sum_of_segments_solved_alone(case):
    """models.AGADIR's % helix equals the one-sequence sum over segments each solved on its own."""
    seq, nc, cc, pH, T, M = case
    with contextlib.redirect_stdout(io.StringIO()):
        m = models.AGADIR(method="1s", T=T, M=M, pH=pH)
        r = m.predict(seq, ncap=nc, ccap=cc)
    params = EnergyCalculator.snapshot_params()
    L = r.seq_length
    k_arr, k_tot = np.zeros(L), 0.0
    for i, j in _windows(case):
        m.energy_calculator = _alone(case, i, j, params)
        K = m._calc_K(m._calc_dG_Hel(i, j))
        k_arr[i + 1:i + j - 1] += K
        k_tot += K
    prop = 100 * k_arr / (1.0 + k_tot)
    assert np.max(np.abs(prop - r.helical_propensity)) < 1e-9


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_noncoulomb_coefficients_equal_the_whole_energy_differences(case):
    """The seven-pattern coefficients (EnergyCalculator._nonCoulomb_charge_coefficients) against the whole
    charge-dependent non-Coulomb energy evaluated at single and pairwise charge patterns."""
    params = EnergyCalculator.snapshot_params()
    for c in _batched(case, params):
        la, lab = c._nonCoulomb_charge_coefficients()
        charged = set(c.neg_charge_aa + c.pos_charge_aa)
        titr = [k for k in range(c.ncap_idx, c.ccap_idx + 1) if c.seq_list[k] in charged]
        saved = c.modified_seq_ionization_hel

        def G(on):
            c.modified_seq_ionization_hel = np.array([1.0 if k in on else 0.0 for k in range(len(c.seq_list))])
            t = c._charge_dependent_nonCoulomb_terms()
            return float(sum(np.sum(v) for v in t.values()))
        try:
            g0 = G(())
            single = {a: G((a,)) - g0 for a in titr}
            for a in titr:
                assert abs(single[a] - la.get(a, 0.0)) < 1e-12, (c.i, c.j, a)
            for a in titr:
                for d in (3, 4):
                    b = a + d
                    if b in titr:
                        cab = G((a, b)) - g0 - single[a] - single[b]
                        assert abs(cab - lab.get((a, b), 0.0)) < 1e-12, (c.i, c.j, a, b)
        finally:
            c.modified_seq_ionization_hel = saved


def test_every_tabulated_distance_is_within_range_and_inside_the_solver_cutoff():
    """Table VI ends at i+PAIR_RANGE, and every tabulated distance is below the solver's 40 A cutoff, so the pairs within
    PAIR_RANGE residues are exactly the pairs any term counts."""
    params = EnergyCalculator.snapshot_params()
    for name in ("table_6_helix_lacroix", "table_6_coil_lacroix"):
        t = params[name]
        cols = [c for c in t.columns if str(c).startswith("i+")]
        assert cols[-1] == f"i+{PAIR_RANGE}"
        values = [float(t.loc[row, col]) for row in t.index for col in cols]
        assert max(values) < 40.0, name


def test_pairs_beyond_the_range_are_unmodelled_in_both_states():
    case = CASES[-1]
    params = EnergyCalculator.snapshot_params()
    for i, j in _windows(case)[::7]:
        c = _alone(case, i, j, params)
        sites = c._ionizable_sites()[0]
        for k1, i1 in sites:
            for k2, i2 in sites:
                if k1 == "SC" and k2 == "SC" and abs(i1 - i2) > PAIR_RANGE:
                    assert c._site_pair_distance(k1, i1, k2, i2, True) == 99.0
                    assert c._site_pair_distance(k1, i1, k2, i2, False) == 99.0
                if k1 != "SC" and k2 == "SC":
                    x = i2 if k1 == "Nterm" else len(c.seq_list) - 1 - i2
                    if x > PAIR_RANGE:
                        assert c._site_pair_distance(k1, i1, k2, i2, True) == 99.0
                        assert c._site_pair_distance(k1, i1, k2, i2, False) == 99.0


@pytest.mark.parametrize("case", [CASES[0], CASES[3], CASES[6], CASES[-2]], ids=_ids)
def test_the_coil_state_is_the_same_for_every_segment(case):
    first = None
    for c in _batched(case, EnergyCalculator.snapshot_params()):
        coil = (c.modified_seq_ionization_rc, c.modified_nterm_ionization_rc, c.modified_cterm_ionization_rc,
                c.site_potential_rc, c.ionization_free_energy_states[False], c.ionization_sweeps[False])
        if first is None:
            first = coil
            continue
        assert np.array_equal(coil[0], first[0]) and coil[1:3] == first[1:3]
        assert np.array_equal(coil[3], first[3], equal_nan=True) and coil[4:] == first[4:]
