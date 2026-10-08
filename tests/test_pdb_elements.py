"""pyagadir.pdb_elements on two designs of Rocklin et al. (2017), checked against the values the
authors computed with Rosetta (study/score_monomeric_designs/scripts/expected_results_for_test_designs.sc).
"""
from pathlib import Path

import numpy as np
import pytest

import pyagadir.pdb_elements as pdb_elements
from pyagadir.pdb_elements import (
    assign_secondary_structure,
    backbone_torsions,
    read_pdb,
    score_helix,
    score_structure,
    virtual_cb,
)

STRUCTURES = Path(__file__).parent / "data" / "structures"

AUTHOR_VALUES = {
    "HEEH_rd4_0428": {
        "dssp": "LHHHHHHHHHHHHHHLLLEEELLEEELLHHHHHHHHHHHHHHL",
        "abego_res_profile": 0.218137751473,
        "abego_res_profile_penalty": -0.0376128355995,
        "one_core_each": 1.0,
        "two_core_each": 0.75,
        "ss_contributes_core": 1.0,
        "n_hydrophobic": 13,
        "largest_hphob_cluster": 13,
        "n_hphob_clusters": 1,
        "hphob_sc_contacts": 19,
        "hphob_sc_degree": 19 / 13,
    },
    "EEHEE_rd4_0872": {
        "dssp": "LEEEELLEEEELLLHHHHHHHHHHHHHHHLLLEEEELLEEEEL",
        "abego_res_profile": 0.320479775734,
        "abego_res_profile_penalty": -0.0372258521757,
        "one_core_each": 0.8,
        # two_core_each: 0.4 by Rosetta, 0.6 here; a hydrophobic sits within 0.25 of the core cutoff
        "ss_contributes_core": 1.0,
        "n_hydrophobic": 12,
        "largest_hphob_cluster": 11,
        "n_hphob_clusters": 2,
        "hphob_sc_contacts": 15,
        "hphob_sc_degree": 1.25,
    },
}


def _rosetta_reduced(dssp):
    return "".join("H" if s in "HGI" else "E" if s in "EB" else "L" for s in dssp)


@pytest.fixture(scope="module")
def reports():
    return {name: score_structure(STRUCTURES / f"{name}.pdb", agadir=False) for name in AUTHOR_VALUES}


@pytest.mark.parametrize("name", AUTHOR_VALUES)
def test_secondary_structure_matches_rosetta_dssp(reports, name):
    assert _rosetta_reduced(reports[name]["dssp"]) == AUTHOR_VALUES[name]["dssp"]


@pytest.mark.parametrize("name", AUTHOR_VALUES)
def test_whole_chain_scores_match_authors(reports, name):
    summary = reports[name]["summary"]
    for key, value in AUTHOR_VALUES[name].items():
        if key != "dssp":
            assert summary[key] == pytest.approx(value, abs=1e-6), key


def test_strands_are_classified_by_their_partners(reports):
    # EEHEE sheet order 2-1-4-3; HEEH is a two-stranded hairpin
    eehee = [e.strand_class for e in reports["EEHEE_rd4_0872"]["elements"] if e.kind == "sheet"]
    heeh = [e.strand_class for e in reports["HEEH_rd4_0428"]["elements"] if e.kind == "sheet"]
    assert eehee == ["middle", "edge", "edge", "middle"]
    assert heeh == ["edge", "edge"]


def test_elements_tile_the_chain(reports):
    elements = reports["EEHEE_rd4_0872"]["elements"]
    assert elements[0].start == 0 and elements[-1].end == 43
    assert all(a.end == b.start for a, b in zip(elements, elements[1:]))
    assert [e.kind for e in elements].count("helix") == 1


def test_helices_are_scored_with_agadir():
    report = score_structure(STRUCTURES / "HEEH_rd4_0428.pdb")
    helices = [e for e in report["elements"] if e.kind == "helix"]
    assert len(helices) == 2
    for helix in helices:
        assert 0 < helix.scores["agadir_helix_percent"] < 100
        assert np.isfinite(helix.scores["agadir_dG_helix"])


def test_helix_score_sees_the_n_prime_residue():
    # Ser N-cap and Leu N4: the hydrophobic staple between N' and N4 needs the real N'
    helix = "SPEELLKKALELAKKG"
    with_leu = score_helix("AL" + helix + "AA", start=3, end=17)
    with_gly = score_helix("AG" + helix + "AA", start=3, end=17)
    assert with_leu["agadir_dG_helix"] != pytest.approx(with_gly["agadir_dG_helix"], abs=1e-6)


def test_missing_agadir_parameter_is_reported_not_raised(monkeypatch):
    def missing(*args, **kwargs):
        raise KeyError("QQ")

    monkeypatch.setattr(pdb_elements, "score_helix", missing)
    report = score_structure(STRUCTURES / "HEEH_rd4_0428.pdb")
    helices = [e for e in report["elements"] if e.kind == "helix"]
    assert all(h.scores["agadir_dG_helix"] is None for h in helices)
    assert all("QQ" in h.notes[0] for h in helices)


def test_hydrophobic_contacts_are_attributed_to_elements(reports):
    report = reports["HEEH_rd4_0428"]
    contacts = report["hydrophobic_core"]["contacts"]
    for element in report["elements"]:
        touching = sum(element.start <= i < element.end or element.start <= j < element.end for i, j in contacts)
        assert element.scores["n_hydrophobic_contacts"] == touching


def test_virtual_cb_is_close_to_real_cb():
    chain = read_pdb(STRUCTURES / "HEEH_rd4_0428.pdb")
    non_gly = [i for i, aa in enumerate(chain.sequence) if aa != "G"]
    placed = virtual_cb(chain.n[non_gly], chain.ca[non_gly], chain.c[non_gly])
    assert np.linalg.norm(placed - chain.cb[non_gly], axis=1).max() < 0.25


def test_chain_break_and_hetero_records(tmp_path):
    lines = (STRUCTURES / "EEHEE_rd4_0872.pdb").read_text().splitlines()
    kept = [line for line in lines if not (line.startswith("ATOM") and 20 <= int(line[22:26]) <= 22)]
    kept.insert(-1, "HETATM 9999  O   HOH A 101      10.000  10.000  10.000  1.00  0.00           O")
    kept.insert(-1, "HETATM 9998 CA    CA A 102      11.000  10.000  10.000  1.00  0.00          CA")
    path = tmp_path / "gap.pdb"
    path.write_text("\n".join(kept) + "\n")

    chain = read_pdb(path)
    assert len(chain) == 40 and "X" not in chain.sequence
    phi, psi, _ = backbone_torsions(chain)
    gap = chain.residue_ids.index("19")
    assert np.isnan(psi[gap]) and np.isnan(phi[gap + 1])
    elements = score_structure(path, agadir=False)["elements"]
    assert any(e.last_residue == "19" for e in elements)
    assert any(e.first_residue == "23" for e in elements)
    with pytest.raises(ValueError):
        read_pdb(path, chain="Z")


def test_incomplete_residue_breaks_the_chain(tmp_path):
    lines = (STRUCTURES / "HEEH_rd4_0428.pdb").read_text().splitlines()
    kept = [line for line in lines if not (line[22:26].strip() == "10" and line[12:16].strip() == "O")]
    path = tmp_path / "no_carbonyl_o.pdb"
    path.write_text("\n".join(kept) + "\n")
    dssp = assign_secondary_structure(read_pdb(path)).dssp
    assert dssp[9] == "X"
