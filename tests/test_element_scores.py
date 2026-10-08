"""pyagadir.element_scores: the loop and strand scores of Rocklin et al. (2017) and Kim et al. (2022)."""
import numpy as np
import pytest

from pyagadir.element_scores import (
    STRAND_SURFACE_EFFECTS,
    abego_from_torsions,
    abego_profile,
    atom_radius,
    buried_unsatisfied_polar_atoms,
    core_participation,
    hydrophobic_core_clusters,
    load_abego_table,
    segments,
    sidechain_neighbors,
    solvent_accessible_area,
    strand_surface_score,
    surface_burial,
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
    middle, edge = STRAND_SURFACE_EFFECTS["middle"]["W"], STRAND_SURFACE_EFFECTS["edge"]["Y"]
    assert result["score"] == pytest.approx(middle + edge)
    assert result["middle_score"] == pytest.approx(middle)
    assert result["edge_score"] == pytest.approx(edge)
    assert (result["n_middle"], result["n_edge"]) == (1, 1)


def test_strand_surface_score_reports_what_it_cannot_score():
    result = strand_surface_score("CWV", "EEE", [True, True, True], ["edge", None, "middle"])
    # no cysteine effect (an assay artefact in the source data), and an exposed strand
    # residue without a class
    assert [position for position, _ in result["unscored"]] == [0, 1]
    assert result["score"] == pytest.approx(STRAND_SURFACE_EFFECTS["middle"]["V"])
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


def test_atom_radius_follows_protor_classes():
    assert atom_radius("A", "C") == 1.61       # carbonyl C, C3H0
    assert atom_radius("F", "CZ") == 1.76      # aromatic CH, C3H1
    assert atom_radius("L", "CD1") == 1.88     # methyl, C4H3
    assert atom_radius("S", "OG") == 1.46      # hydroxyl, O2H1
    assert atom_radius("D", "OD1") == 1.42     # carboxyl, O1H0
    assert atom_radius("K", "NZ") == 1.64
    assert atom_radius("M", "SD") == 1.77


def test_solvent_accessible_area_of_one_and_two_spheres():
    probe = 1.4
    single = solvent_accessible_area(np.zeros((1, 3)), [1.88], probe=probe)
    assert single[0] == pytest.approx(4 * np.pi * (1.88 + probe) ** 2)
    # two overlapping spheres: each loses a spherical cap of height h = R - d/2 (equal radii)
    d, R = 4.0, 1.88 + probe
    pair = solvent_accessible_area(np.array([[0.0, 0, 0], [d, 0, 0]]), [1.88, 1.88], probe=probe, n_points=2000)
    exact = 4 * np.pi * R ** 2 - 2 * np.pi * R * (R - d / 2)
    assert pair == pytest.approx([exact, exact], rel=0.01)


def test_surface_burial_buried_area_is_never_negative():
    rng = np.random.default_rng(0)
    sequence = "LVA"
    atoms = [{"N": rng.normal(size=3) * 3, "CA": rng.normal(size=3) * 3, "C": rng.normal(size=3) * 3,
              "O": rng.normal(size=3) * 3, "CB": rng.normal(size=3) * 3} for _ in sequence]
    result = surface_burial(sequence, atoms, bonded=[False, False])
    buried = result["reference_nonpolar"] - result["exposed_nonpolar"]
    assert (buried >= -1e-9).all()
    assert result["buried_npsa"] == pytest.approx(buried.sum())


def _polar_pair(distance):
    """A Lys NZ and an Asp OD1 `distance` apart, in residues that are not bonded."""
    atoms = [{"NZ": np.array([0.0, 0.0, 0.0])}, {"OD1": np.array([distance, 0.0, 0.0])}]
    buried = [{"NZ": 0.0}, {"OD1": 0.0}]
    return atoms, buried


def test_buried_unsatisfied_polar_atoms_pairs_donors_with_acceptors():
    atoms, area = _polar_pair(3.0)
    paired = buried_unsatisfied_polar_atoms("KD", atoms, area, bonded=[False])
    # the salt bridge satisfies both heavy atoms; two of Lys NZ's three hydrogens stay free
    assert (paired["n_sidechain"], paired["n_hydrogen"]) == (0, 2)

    atoms, area = _polar_pair(5.0)
    apart = buried_unsatisfied_polar_atoms("KD", atoms, area, bonded=[False])
    assert (apart["n_sidechain"], apart["n_hydrogen"]) == (2, 3)

    exposed = buried_unsatisfied_polar_atoms("KD", atoms, [{"NZ": 5.0}, {"OD1": 5.0}], bonded=[False])
    assert exposed["unsatisfied"] == []


def test_peptide_bonded_backbone_atoms_are_not_partners():
    atoms = [{"O": np.array([0.0, 0.0, 0.0])}, {"N": np.array([2.3, 0.0, 0.0])}]
    area = [{"O": 0.0}, {"N": 0.0}]
    result = buried_unsatisfied_polar_atoms("AA", atoms, area, bonded=[True])
    assert result["n_backbone"] == 2
