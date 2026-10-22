"""The segments' ionisation solved together (electrostatics/common.py SegmentBatch, models.AGADIR._calc_partition_fxn):
the results of the per-segment solver it replaced (deleted on 2026-10-22; its results for these peptides are in
tests/data/segment_batch_reference.json.gz, recorded at commit 0b7b056), the premises the batch rests on (every
interacting pair within PAIR_RANGE residues; a segment-independent coil state), and the non-Coulomb coefficients."""
import contextlib
import gzip
import io
import json
from pathlib import Path

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


# (case, law): every case under the default law, a subset under gbx (the GB field instead of Muñoz's)
LAW_CASES = [(c, "munoz") for c in CASES] + [(c, "gbx") for c in (CASES[0], CASES[2], CASES[3], CASES[6], CASES[8],
                                                                    CASES[-1])]


REFERENCE = json.load(gzip.open(Path(__file__).parent / "data" / "segment_batch_reference.json.gz", "rt"))


def _reference(case, law):
    """The per-segment solver's recorded results for (case, law)."""
    for r in REFERENCE["cases"]:
        if tuple(r["case"]) == tuple(case) and r["law"] == law:
            return r
    raise KeyError((case, law))


def _predict(case, law="munoz"):
    seq, nc, cc, pH, T, M = case
    with contextlib.redirect_stdout(io.StringIO()):
        return models.AGADIR(method="1s", T=T, M=M, pH=pH, dipole_law=law).predict(seq, ncap=nc, ccap=cc)


def _segments(case, law="munoz"):
    seq, nc, cc, pH, T, M = case
    L = len(seq) + (nc is not None) + (cc is not None)
    params = EnergyCalculator.snapshot_params(law)
    for j in range(6, L + 1):
        for i in range(0, L - j + 1):
            yield EnergyCalculator(seq=seq, i=i, j=j, pH=pH, T=T, ionic_strength=M, ncap=nc, ccap=cc, params=params,
                                   dipole_law=law)


def _ids(cl):
    return f"{cl[1]}-{cl[0][0]}-{cl[0][3]}"


@pytest.mark.parametrize("cl", LAW_CASES, ids=_ids)
def test_predictions_reproduce_the_per_segment_solver(cl):
    """Identical % helix; per-residue helicity equal to rounding (the coefficients differ from the per-segment solver's
    whole-energy differences at the last bit)."""
    case, law = cl
    ref = _reference(case, law)
    r = _predict(case, law)
    assert r.percent_helix == ref["percent_helix"]
    assert np.max(np.abs(r.helical_propensity - np.array(ref["helical_propensity"]))) < 1e-9


@pytest.mark.parametrize("cl", LAW_CASES, ids=_ids)
def test_every_segment_matches_the_per_segment_solver(cl):
    """Each segment's ionisation (a batch of one) against the per-segment solver's recorded results: the coil state to
    the last bit, the helix state and dG_ionization to rounding, the same sweep counts."""
    case, law = cl
    ref = _reference(case, law)
    segs = {(g["i"], g["j"]): g for g in ref["segments"]}
    n_seen = 0
    for c in _segments(case, law):
        g = segs[(c.i, c.j)]
        sites = c._ionizable_sites()[0]

        def site_q(helix):
            seq_q = c.modified_seq_ionization_hel if helix else c.modified_seq_ionization_rc
            nq = c.modified_nterm_ionization_hel if helix else c.modified_nterm_ionization_rc
            cq = c.modified_cterm_ionization_hel if helix else c.modified_cterm_ionization_rc
            return [float(nq) if k == "Nterm" else float(cq) if k == "Cterm" else float(seq_q[i]) for k, i in sites]
        assert site_q(False) == ref["q_coil"], (c.i, c.j)
        assert np.allclose(site_q(True), g["q_hel"], rtol=0, atol=1e-12), (c.i, c.j)
        assert abs(c.dG_ionization - g["dG_ionization"]) < 1e-12, (c.i, c.j)
        assert [c.ionization_sweeps[True], c.ionization_sweeps[False]] == g["sweeps"], (c.i, c.j)
        n_seen += 1
    assert n_seen == len(segs)


@pytest.mark.parametrize("cl", LAW_CASES, ids=_ids)
def test_noncoulomb_coefficients_equal_the_whole_energy_differences(cl):
    """The seven-pattern coefficients against the per-segment solver's recorded whole-energy differences."""
    case, law = cl
    segs = {(g["i"], g["j"]): g for g in _reference(case, law)["segments"]}
    for c in _segments(case, law):
        g = segs[(c.i, c.j)]
        la, lab = c._nonCoulomb_charge_coefficients()
        a = {int(k): v for k, v in g["c_a"].items()}
        ab = {tuple(int(x) for x in k.split(",")): v for k, v in g["c_ab"].items()}
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
