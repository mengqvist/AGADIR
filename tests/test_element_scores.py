"""pyagadir.element_scores: the loop and strand scores of Rocklin et al. (2017) and Kim et al. (2022)."""
import numpy as np
import pytest

from pyagadir.element_scores import (
    abego_from_torsions,
    abego_profile,
    core_participation,
    hydrophobic_core_clusters,
    load_abego_table,
    segments,
    sidechain_neighbors,
    strand_surface_score,
)

NAN = float("nan")


def test_segments_splits_runs():
    assert segments("LLHHHHL") == [("L", 0, 2), ("H", 2, 6), ("L", 6, 7)]
    assert segments("") == []


def test_abego_bins_and_boundaries():
    phi = [-60, -120, 60, 60, -60, NAN, -60, -60, 60, 60]
    psi = [-45, 130, 30, 180, -45, 100, -75, 50, -100, 100]
    omega = [180, 180, 180, 180, 0, 180, NAN, 180, 180, 180]
    # cis peptide -> O; undefined phi -> X; undefined omega counts as trans
    assert abego_from_torsions(phi, psi, omega) == "ABGEOXABGE"


def test_abego_profile_scores_central_residues_only():
    table = load_abego_table()
    result = abego_profile("AAAAAA", "AAAAAA")
    # the first two and last two residues are never scored
    assert np.isnan(result["per_residue"][[0, 1, 4, 5]]).all()
    assert result["per_residue"][2] == pytest.approx(table[("AAA", "A")])
    assert result["n_scored"] == 2
    # published convention: mean over all positions, unscored ones counting as zero
    assert result["mean"] == pytest.approx(2 * table[("AAA", "A")] / 6)

    loop = abego_profile("AAAAAA", "AAAAAA", positions=[2])
    assert loop["total"] == pytest.approx(table[("AAA", "A")])
    assert loop["mean"] == pytest.approx(table[("AAA", "A")])


def test_abego_profile_skips_triads_missing_from_table():
    result = abego_profile("AAAAAA", "AAXAAA")
    assert np.isnan(result["per_residue"][2]) and np.isnan(result["per_residue"][3])
    assert result["n_scored"] == 0


def test_strand_surface_score_sums_exposed_strand_residues():
    # exposed middle W + exposed edge Y; the buried P and the loop A contribute nothing
    result = strand_surface_score("WYPA", "EEEL", [True, True, False, True], ["middle", "edge", None, None])
    assert result["score"] == pytest.approx(0.13903)
    assert result["middle_score"] == pytest.approx(0.08690)
    assert result["edge_score"] == pytest.approx(0.05213)
    assert (result["n_middle"], result["n_edge"]) == (1, 1)


def test_strand_surface_score_reports_what_it_cannot_score():
    result = strand_surface_score("CWV", "EEE", [True, True, True], ["edge", None, "middle"])
    # no published cysteine effect, and an exposed strand residue without a class
    assert [position for position, _ in result["unscored"]] == [0, 1]
    assert result["score"] == pytest.approx(0.0712)
    with pytest.raises(ValueError):
        strand_surface_score("V", "E", [True], ["inner"])


def test_sidechain_neighbors_counts_residues_in_the_side_chain_cone():
    ca = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    cb = np.array([[1.5, 0.0, 0.0], [7.5, 0.0, 0.0], [20.0, 0.0, 0.0]])
    counts = sidechain_neighbors(ca, cb)
    # residue 0 points straight at residue 1 (4.5 A from its CB); residue 1 points away from 0
    assert counts[0] == pytest.approx(1 / (1 + np.exp(4.5 - 9.0)) + 1 / (1 + np.exp(18.5 - 9.0)))
    assert counts[1] == pytest.approx(1 / (1 + np.exp(12.5 - 9.0)))
    # CB on top of CA: no side-chain direction
    assert np.isnan(counts[2])


def test_core_participation_counts_buried_hydrophobics_per_element():
    sequence = "LAALAKVEVKLLAG"
    ss = "HHHHHLEEELHHLL"
    burial = [5, 1, 1, 4.5, 1, 1, 3, 1, 1, 1, 6, 6, 1, 1]
    result = core_participation(sequence, ss, burial)
    # the helix has L0 and L3 in the core, the strand only V6 in the boundary layer; the
    # two-residue helix is too short to count
    assert result["elements"] == [("H", 0, 5, 2, 2), ("E", 6, 9, 0, 1)]
    assert result["one_core_each"] == 0.5
    assert result["two_core_each"] == 0.5
    assert result["ss_contributes_core"] == 1.0


def test_hydrophobic_core_clusters_builds_the_contact_graph():
    sequence = "LAAVLAAAAAF"
    far = np.array([[100.0, 0.0, 0.0]])
    atoms = [np.empty((0, 3))] * len(sequence)
    atoms[0] = np.array([[0.0, 0.0, 0.0]])
    atoms[3] = np.array([[4.0, 0.0, 0.0]])   # 4.0 A from L0, three residues apart: contact
    atoms[4] = np.array([[0.0, 3.0, 0.0]])   # close to L0 and V3, but only one residue from V3
    atoms[10] = far
    result = hydrophobic_core_clusters(sequence, atoms)
    assert result["n_hydrophobic"] == 4
    assert result["contacts"] == [(0, 3), (0, 4)]
    assert result["clusters"] == [[0, 3, 4], [10]]
    assert (result["largest_cluster"], result["n_clusters"]) == (3, 2)
    assert result["contacts_per_residue"] == 0.5
