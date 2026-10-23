"""Acid-base side-chain pairs under pair_rule 'contacts' (energies.get_dG_i3 / get_dG_i4, electrostatics/contacts.py):
a peptide without acid-base pairs three or four apart is unchanged; an isolated salt bridge keeps its published value;
the contact partition function is summed exactly; a pair with a neutral partner hydrogen-bonds on any side; and the
charge-dense blocks keep their helicity when the acids are neutralised (Lyu et al. 1992; Fossat et al. 2023)."""
import contextlib
import io
import itertools
import math

import numpy as np
import pytest

from pyagadir import models
from pyagadir.electrostatics import contacts as C
from pyagadir.energies import EnergyCalculator


def _helix(seq, nc, cc, T, M, pH, rule):
    with contextlib.redirect_stdout(io.StringIO()):
        return models.AGADIR(method="1s", T=T, M=M, pH=pH, pair_rule=rule).predict(seq, ncap=nc, ccap=cc).percent_helix


@pytest.mark.parametrize("seq,nc,cc,pH", [
    ("AAKAAAAKAAAAKAAY", "Ac", "Am", 7.0),   # Lys only
    ("AKAAKAKAAKAKAAKA", "Ac", "Am", 7.0),   # Marqusee & Baldwin 1989 6K: like charges only
    ("AEAAAAEAAAAEAAAA", "Ac", "Am", 4.5),   # Glu only
    ("AEAAAAAKAAAAAAAY", "Ac", "Am", 7.0),   # an acid-base pair six apart
])
def test_peptides_without_candidate_pairs_are_unchanged(seq, nc, cc, pH):
    assert _helix(seq, nc, cc, 5.0, 0.05, pH, "contacts") == _helix(seq, nc, cc, 5.0, 0.05, pH, "additive")


@pytest.mark.parametrize("seq,pH,M", [("AAQAAEAQAKAAQAAY", 7.0, 0.0172), ("AAQAAKAQAEAAQAAY", 7.0, 0.0172),
                                      ("AAQAAEAQAKAAQAAY", 2.5, 0.0141), ("AAQAAKAQEAAAQAAY", 7.0, 2.5)])
def test_an_isolated_salt_bridge_keeps_its_value(seq, pH, M):
    """Smith & Scholtz 1998 single pairs: the contact state is calibrated so that an isolated pair reproduces its
    Table VI Coulomb term; what remains is the exact short-range correlation of the charges (< 0.05 helix points)."""
    a = _helix(seq, "Ac", "Am", 0.0, M, pH, "contacts")
    b = _helix(seq, "Ac", "Am", 0.0, M, pH, "additive")
    assert abs(a - b) < 0.05


@pytest.mark.parametrize("X,Y", [(a, b) for a in "DE" for b in "KRH"] + [(b, a) for a in "DE" for b in "KRH"])
@pytest.mark.parametrize("k", [3, 4])
def test_every_isolated_acid_base_pair_type_keeps_its_value(X, Y, k):
    """Every acid-base type at i+3 and i+4, including the ones Table VI does not hold closer than the open geometry
    (Asp(i)-His(i+4)), which then have no contact state and keep their Table VI term."""
    seq = "AAQAA" + X + "A" * (k - 1) + Y + "AAQAAY"
    for pH, M in ((5.0, 0.005), (7.0, 0.005), (3.0, 0.1)):
        assert abs(_helix(seq, "Ac", "Am", 0.0, M, pH, "contacts") - _helix(seq, "Ac", "Am", 0.0, M, pH, "additive")) < 0.05


def _brute(m, contacts):
    """-RT ln Z by enumeration of the charge states and of every set of contacts in which each charged site has at most
    one partner, all pairs exact, no long-range field (the transfer matrix's model with EXACT_RANGE covering all)."""
    n, RT = m["n"], m["RT"]
    keys = [k for k, v in contacts.items() if np.isfinite(v)]
    lw = []
    for s in itertools.product((0.0, 1.0), repeat=n):
        s = np.array(s)
        E = s @ (m["hF"] + m["hC"]) + 0.5 * s @ (m["zzW"] + m["C"]) @ s
        base = float(s @ m["lnx"]) - E / RT
        for r in range(len(keys) + 1):
            for M in itertools.combinations(keys, r):
                used = [x for key in M for x in key]
                if len(used) != len(set(used)) or any(s[x] == 0 for x in used):
                    continue
                lw.append(base - sum(contacts[key] for key in M) / RT)
    return -RT * float(np.logaddexp.reduce(lw)) + RT * float(np.sum(np.logaddexp(0.0, m["lnx"])))


@pytest.mark.parametrize("pH", [2.5, 4.5, 7.4])
def test_the_contact_partition_function_is_summed_exactly(pH, monkeypatch):
    monkeypatch.setattr(C, "EXACT_RANGE", 100)
    c = EnergyCalculator(seq="GEEEEKKKKGW", i=0, j=13, pH=pH, T=20.0, ionic_strength=0.015, ncap="Ac", ccap="Am")
    m = C._sites(c, True)
    ct = C._contacts(c, m)
    assert ct, "Ac-GEEEEKKKKGW-Am has candidate pairs"
    F, _ = C._transfer(m, ct, np.zeros(m["n"]), False)
    assert F == pytest.approx(_brute(m, ct), abs=1e-9)


def test_the_contact_state_is_calibrated_on_the_isolated_pair():
    """exp(-G_contact/RT) + exp(-G_open/RT) = exp(-G_TableVI/RT) for an E-K i+4 pair, both charged."""
    c = EnergyCalculator(seq="AAQAAEAQAKAAQAAY", i=0, j=18, pH=7.0, T=0.0, ionic_strength=0.0172, ncap="Ac", ccap="Am")
    m = C._sites(c, True)
    ct = C._contacts(c, m)
    (a, b), e = next(iter(ct.items()))
    RT = m["RT"]
    i, j = m["sites"][a][1], m["sites"][b][1]
    g_tab = m["z"][a] * m["z"][b] * c._electrostatic_interaction_energy(1.0, 1.0, float(c.sidechain_sidechain_distances_hel[i, j]))
    g_open = m["zzW"][a, b]
    assert math.exp(-(g_open + e) / RT) + math.exp(-g_open / RT) == pytest.approx(math.exp(-g_tab / RT), rel=1e-12)


def test_a_pair_with_a_neutral_partner_bonds_on_any_side():
    """An Asp(i)-Lys(i+4) pair that _acid_base_hbonds does not select: its Table IV bond is 0 when both are charged
    and present when the acid is neutral (pair_rule 'contacts'); always 0 under 'additive'."""
    for rule, expect_neutral in (("contacts", True), ("additive", False)):
        c = EnergyCalculator(seq="DAKAAEDAAAKAAAAD", i=0, j=16, pH=4.2, T=0.0, ionic_strength=0.01, pair_rule=rule)
        assert (6, 10) not in c._acid_base_hbonds()
        q = c.modified_seq_ionization_hel.copy()
        c.modified_seq_ionization_hel = q.copy()
        c.modified_seq_ionization_hel[6], c.modified_seq_ionization_hel[10] = -1.0, 1.0
        both = float(c.get_dG_i4()[6])
        c.modified_seq_ionization_hel[6] = 0.0
        neutral = float(c.get_dG_i4()[6])
        c.modified_seq_ionization_hel = q
        assert both == pytest.approx(0.0, abs=1e-12)
        assert (neutral < -0.1) is expect_neutral


def test_blocks_keep_their_helicity_with_the_acids_neutral():
    """Fossat et al. 2023 (E4K4)3 and Lyu et al. 1992 E4K4 at low pH: measured 55 % and 47 %; the additive rule gives
    about 19 % for both; the contacts rule recovers most of it (cycle 470-473, N204)."""
    f = _helix("G" + "EEEEKKKK" * 3 + "GW", "Ac", "Am", 20.0, 0.015, 2.5, "contacts")
    lyu = _helix("YSEEEEKKKKEEEEKKKK", "Sc", "Am", 4.0, 0.0416, 1.5, "contacts")
    assert f > 35.0 and lyu > 35.0


def _hbond_calc(seq):
    return EnergyCalculator(seq=seq, i=0, j=len(seq), pH=7.0, T=5.0, ionic_strength=0.15, ncap="Ac", ccap="Am",
                            params=EnergyCalculator.snapshot_params("munoz"))


def _bond_sum(calc, pairs):
    return sum(float((calc.table_4a_lacroix if j - i == 3 else calc.table_4b_lacroix)
                     .loc[calc.seq_list[i], calc.seq_list[j]]) / 100.0 for i, j in pairs)


def test_hbond_set_is_the_best_compatible_set():
    """_acid_base_hbonds returns the compatible set with the largest summed Table IV value: every side chain bonds on
    one side only, and no assignment of sides does better (brute force over all of them)."""
    rng = np.random.default_rng(7)
    for _ in range(60):
        seq = "".join(rng.choice(list("AAEEKKDRH"), size=int(rng.integers(8, 15))))
        c = _hbond_calc(seq)
        chosen = c._acid_base_hbonds()
        side = {}
        for i, j in chosen:
            assert side.setdefault(i, "up") == "up" and side.setdefault(j, "down") == "down"
        interior = set(c.helix_indices[1:-1])
        w = {}
        for k, table in ((3, c.table_4a_lacroix), (4, c.table_4b_lacroix)):
            for i in interior:
                a, b = c.seq_list[i], c.seq_list[i + k] if i + k < len(c.seq_list) else "A"
                if i + k in interior and {a, b} & {"D", "E"} and {a, b} & {"K", "R", "H"}:
                    if float(table.loc[a, b]) < 0:
                        w[(i, i + k)] = float(table.loc[a, b]) / 100.0
        res = sorted({r for p in w for r in p})
        best = min((sum(v for (i, j), v in w.items() if d[res.index(i)] and not d[res.index(j)])
                    for d in itertools.product((True, False), repeat=len(res))), default=0.0)
        assert _bond_sum(c, chosen) == pytest.approx(best, abs=1e-9)


def test_hbond_set_keeps_both_lys_glu_blocks_in_k4e4_2():
    """Ac-GA4(K4E4)2A4GW-NH2: the central Glu->Lys i+4 bonds (-0.20 each) would exclude both Lys->Glu blocks if taken
    first (-1.25 kcal/mol); the two blocks together are -1.70."""
    c = _hbond_calc("GAAAAKKKKEEEEKKKKEEEEAAAAGW")
    assert _bond_sum(c, c._acid_base_hbonds()) == pytest.approx(-1.70)
    assert len(c._acid_base_hbonds()) == 14
