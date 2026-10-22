"""The default law's batched segments (electrostatics/common.py SegmentBatch, models.AGADIR._calc_partition_fxn_batched):
the same predictions as one segment at a time (AGADIR._calc_partition_fxn_per_segment, each segment a batch of one),
the same ionisation as the per-segment solver it replaces (Electrostatics._assign_modified_ionization_states, kept for
the GB laws), the premises they rest on (every interacting pair within PAIR_RANGE residues; a segment-independent coil
state), and the local non-Coulomb coefficients."""
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


def _predict(case, batched):
    seq, nc, cc, pH, T, M = case
    model = models.AGADIR(method="1s", T=T, M=M, pH=pH)
    if not batched:  # one segment at a time, each a batch of one
        params = EnergyCalculator.snapshot_params("munoz")
        model._calc_partition_fxn = lambda: model._calc_partition_fxn_per_segment(params)
    with contextlib.redirect_stdout(io.StringIO()):
        return model.predict(seq, ncap=nc, ccap=cc)


def _segments(case):
    seq, nc, cc, pH, T, M = case
    L = len(seq) + (nc is not None) + (cc is not None)
    for j in range(6, L + 1):
        for i in range(0, L - j + 1):
            yield EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=T, ionic_strength=M, ncap=nc, ccap=cc)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[3]}")
def test_batched_segments_reproduce_the_per_segment_solver(case):
    """Identical % helix; per-residue helicity equal to rounding (the local coefficients differ at the last bit)."""
    a = _predict(case, False)
    b = _predict(case, True)
    assert a.percent_helix == b.percent_helix
    assert np.max(np.abs(a.helical_propensity - b.helical_propensity)) < 1e-9
    assert abs(a.K_tot - b.K_tot) <= 1e-12 * max(1.0, abs(a.K_tot))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[3]}")
def test_every_segment_matches_the_per_segment_solver(case):
    """Each segment's ionisation from its SegmentBatch (a batch of one) against the per-segment solver
    (_assign_modified_ionization_states, kept until the GB laws are ported): the coil state to the last bit, the
    helix state to rounding (the solver's whole-energy non-Coulomb coefficients differ from the local ones there)."""
    for c in _segments(case):
        batch = (c.modified_seq_ionization_hel.copy(), c.modified_nterm_ionization_hel, c.modified_cterm_ionization_hel,
                 c.site_potential_hel.copy(), c.modified_seq_ionization_rc.copy(), c.modified_nterm_ionization_rc,
                 c.modified_cterm_ionization_rc, c.site_potential_rc.copy(), c.dG_ionization, dict(c.ionization_sweeps))
        c._assign_modified_ionization_states()
        assert np.array_equal(c.modified_seq_ionization_rc, batch[4]), (c.i, c.j)
        assert (c.modified_nterm_ionization_rc, c.modified_cterm_ionization_rc) == batch[5:7], (c.i, c.j)
        assert np.array_equal(c.site_potential_rc, batch[7], equal_nan=True), (c.i, c.j)
        assert np.allclose(c.modified_seq_ionization_hel, batch[0], rtol=0, atol=1e-12), (c.i, c.j)
        assert abs(c.modified_nterm_ionization_hel - batch[1]) < 1e-12 and abs(c.modified_cterm_ionization_hel - batch[2]) < 1e-12
        assert np.allclose(c.site_potential_hel, batch[3], rtol=0, atol=1e-12, equal_nan=True), (c.i, c.j)
        assert abs(c.dG_ionization - batch[8]) < 1e-12, (c.i, c.j)
        assert c.ionization_sweeps == batch[9], (c.i, c.j)


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[3]}")
def test_local_noncoulomb_coefficients_equal_the_whole_energy_differences(case):
    for c in _segments(case):
        a, ab = c._nonCoulomb_charge_coefficients()
        la, lab = c._nonCoulomb_charge_coefficients_local()
        assert set(a) == set(la) and set(ab) == set(lab), (c.i, c.j)
        for k in a:
            assert abs(a[k] - la[k]) < 1e-12, (c.i, c.j, k)
        for k in ab:
            assert abs(ab[k] - lab[k]) < 1e-12, (c.i, c.j, k)


def test_every_tabulated_distance_is_within_range_and_inside_the_solver_cutoff():
    """Table VI ends at i+PAIR_RANGE, and every tabulated distance is below the solver's 40 A cutoff, so the pairs within
    PAIR_RANGE residues are exactly the pairs any term counts."""
    params = EnergyCalculator.snapshot_params("munoz")
    for name in ("table_6_helix_lacroix", "table_6_coil_lacroix"):
        t = params[name]
        cols = [c for c in t.columns if str(c).startswith("i+")]
        assert cols[-1] == f"i+{PAIR_RANGE}"
        values = [float(t.loc[row, col]) for row in t.index for col in cols]
        assert max(values) < 40.0, name


def test_pairs_beyond_the_range_are_unmodelled_in_both_states():
    case = CASES[-1]
    for c in _segments(case):
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


@pytest.mark.parametrize("case", [CASES[0], CASES[3], CASES[6], CASES[-2]], ids=lambda c: f"{c[0]}-{c[3]}")
def test_the_coil_state_is_the_same_for_every_segment(case):
    first = None
    for c in _segments(case):
        coil = (c.modified_seq_ionization_rc, c.modified_nterm_ionization_rc, c.modified_cterm_ionization_rc,
                c.site_potential_rc, c.ionization_free_energy_states[False], c.ionization_sweeps[False])
        if first is None:
            first = coil
            continue
        assert np.array_equal(coil[0], first[0]) and coil[1:3] == first[1:3]
        assert np.array_equal(coil[3], first[3], equal_nan=True) and coil[4:] == first[4:]
