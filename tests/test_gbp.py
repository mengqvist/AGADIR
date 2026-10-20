"""dipole_law="gbp": gbx plus the Generalized Born side chain-side chain pair law, the exact two-state coupling of
every ionisable group to both GB laws, and the statistical salt-bridge network (params/README.md "gbp")."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import pyagadir.energies as E
import pyagadir.electrostatics.gb as GB
from pyagadir.models import AGADIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def calc(seq, pH, law="gbp", T=25.0, M=0.1, ncap="Ac", ccap="Am"):
    return E.EnergyCalculator(seq=seq, i=0, j=len(seq) + 2, pH=pH, T=T, ionic_strength=M, ncap=ncap, ccap=ccap,
                              dipole_law=law)


def test_pair_energy_is_the_table_value_at_the_residues_charges():
    """A fully charged Lys(i)-Glu(i+4) pair inside the helix: the pair term's helix-state energy is the table's charged
    pair free energy, interpolated at the instance's T and kappa."""
    c = calc("AAAAKAAAEAAAAAA", 7.0)
    i, j = [k for k, aa in enumerate(c.seq_list) if aa in "KE"]
    U, a1, a2, enn = c.gb_pairs[True][(i, j)]
    row = GB.gbp_tables()["pairs"][("K", "E", 4, "helix", "int")]
    e_cc = GB.gb_pair_grid_value(row[("K", "E")], c.T_kelvin, c.kappa * 1e-10)
    assert enn + a1 * 1.0 + a2 * -1.0 + U * -1.0 == pytest.approx(e_cc, abs=1e-12)


@pytest.mark.parametrize("seq,pH", [("AAAAKAAAEAAAAAA", 4.0), ("AAAAKAAAEAAAAAA", 10.5), ("AAAAHAAADAAAAAA", 3.5),
                                    ("AAAAHAAADAAAAAA", 6.5), ("AAKAEAAAAAAAAAA", 4.0)])
def test_coupling_matches_exact_microstate_enumeration(seq, pH):
    """The segment's electrostatic free energy (pair term + macrodipole with the neutral forms' energies + ionisation
    free energy) equals the exact free energy of the four protonation microstates, each with its own GB energies."""
    c = calc(seq, pH)
    n, cc = c.get_dG_sidechain_macrodipole()
    model = np.sum(c.get_dG_sidechain_sidechain_electrost()) + np.sum(n + cc) + c.get_dG_ionization()
    rt = 1.9865e-3 * c.T_kelvin
    idx = [k for k, aa in enumerate(c.seq_list) if aa in "KEHD"]
    z = [1.0 if c.seq_list[k] in "KH" else -1.0 for k in idx]
    x = [10 ** ((c.seq_pka[k] - pH) if zz > 0 else (pH - c.seq_pka[k])) for k, zz in zip(idx, z)]

    def free_energy(helix):
        U, a1, a2, enn = c.gb_pairs[helix][(idx[0], idx[1])]
        phi = [c.sidechain_dipole_potential[k] if helix else 0.0 for k in idx]
        const = sum(c.gb_neutral_energy[k] for k in idx) if helix else 0.0
        Z = 0.0
        for s1 in (0, 1):
            for s2 in (0, 1):
                q1, q2 = z[0] * s1, z[1] * s2
                e = enn + q1 * a1 + q2 * a2 + q1 * q2 * U + q1 * phi[0] + q2 * phi[1] + const
                Z += x[0] ** s1 * x[1] ** s2 * math.exp(-e / rt)
        return -rt * math.log(Z)
    assert model == pytest.approx(free_energy(True) - free_energy(False), abs=1e-3)


def test_salt_bridge_network_isolated_pair_and_chain():
    """An isolated pair keeps its cell; the alternating i+3 chain of seven Glu-Lys cells (-0.15 each, 2 C) is the
    exact transfer-matrix free energy, -0.883, not the one-partner rule's -0.60 or the additive -1.05."""
    RT = 1.9872e-3 * 275.15
    assert GB.sb_network_free_energy({(5, 8): -0.15}, RT) == pytest.approx(-0.15, abs=1e-12)
    chain = {(i, i + 3): -0.15 for i in range(0, 21, 3)}
    assert GB.sb_network_free_energy(chain, RT) == pytest.approx(-0.883, abs=0.001)


def test_point_law_outside_the_gb_pair_range():
    """Pairs with Cys or Tyr, more than 12 apart, or with one residue outside the segment keep the point law."""
    c = calc("AKAAYAAAAAAAAAAAAAAE", 7.0)
    assert not any(k for k in c.gb_pairs[True] if c.seq_list[k[0]] == "Y" or c.seq_list[k[1]] == "Y")
    c2 = E.EnergyCalculator(seq="KAAAAAAAAAAAAE", i=3, j=8, pH=7.0, T=25.0, ionic_strength=0.1, ncap="Ac", ccap="Am",
                            dipole_law="gbp")
    assert c2.gb_pairs[True] == {}


@pytest.mark.parametrize("cls,aa", [("Ncen", "E"), ("Ncap", "D"), ("Neutral", "E")])
def test_transfer_reproduces_default_at_reference(cls, aa):
    """At a Table 1 value's reference measurement (tools/build_gb_transfer.REFS), the guest-minus-Ala-host segment
    free energy under gbp equals the default model's, weighted over the segments that place the guest in the class,
    with the guest's interactions with charged host residues left out on both sides (tools/build_gbp_transfer.py).
    Class Neutral: Chakrabartty et al.'s Ncen host at a pH where the guest is neutral."""
    T = pytest.importorskip("build_gbp_transfer")
    ref = T.BT.REFS["Ncen" if cls == "Neutral" else cls]
    pH = T.NEUTRAL_PH[aa] if cls == "Neutral" else ref.get("pH_res", {}).get(aa, ref["pH"])
    seq, idx, segs = T.segments("Ncen" if cls == "Neutral" else cls, aa, ref, pH)
    host = ref["seq"].replace("X", "A")
    d = {law: T.seg_dG(seq, pH, segs, ref, law, idx)[0] - T.seg_dG(host, pH, segs, ref, law)[0]
         for law in ("munoz", "gbp")}
    assert d["gbp"] == pytest.approx(d["munoz"], abs=0.002)


def test_default_and_gbx_have_no_gb_pairs():
    for law in ("munoz", "gbx"):
        c = calc("AAAAKAAAEAAAAAA", 7.0, law=law)
        assert c.gb_pairs == {True: {}, False: {}}
        assert not np.any(c.gb_neutral_energy)


def test_gbp_runs_end_to_end():
    for law in ("gbx", "gbp"):
        r = AGADIR(method="1s", T=0.0, M=0.1, pH=7.0, dipole_law=law).predict("AEAAAKEAAAKEAAAKA", ncap="Ac", ccap="Am")
        assert 0.0 < r.get_percent_helix() < 100.0


@pytest.mark.parametrize("pos,key", [(2, ("K", "N", 2)), (3, ("K", "N", 3)), (8, ("K", "mid", -1)),
                                     (16, ("K", "C", 1))])
def test_desolvation_row_by_distance_from_the_nearer_end(pos, key):
    """Under gbp a side chain's helix-state GB energy carries its desolvation row: by distance from the nearer end up to
    GB_DESOLV_RANGE, the interior row beyond; gbx has none."""
    seq = "A" * (pos - 1) + "K" + "A" * (16 - pos)
    g, x = calc(seq, 7.0), calc(seq, 7.0, law="gbx")
    idx = g.seq_list.index("K")
    eg = g._gb_form_energy(GB.gb_dipole_table(), "K", idx)
    ex = x._gb_form_energy(GB.gb_dipole_table(), "K", idx)
    assert eg[0] - ex[0] == pytest.approx(GB.gbp_tables()["desolv"][key], abs=1e-12)
    assert eg[1] == pytest.approx(ex[1], abs=1e-12)
