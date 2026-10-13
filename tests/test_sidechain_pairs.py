"""Acid-base side-chain pairs against Smith & Scholtz (1998, Biochemistry 37, 33).

The (i,i+5) peptides carry no interaction, so the helicity difference between an (i,i+k) peptide and its (i,i+5)
partner at the same salt and pH cancels the charge-macrodipole terms and isolates the pair. Measured values are read
from the tracked corpus file and converted with the paper's own equations 1-3.
"""
import contextlib
import csv
import io
from pathlib import Path

import pytest

from pyagadir.models import AGADIR

DATA = Path(__file__).resolve().parents[1] / "pyagadir/data/peptides/1998_smith_biochem37_table3.tsv"
THETA_H, THETA_C = -42500 * (1 - 3 / 16), 640.0


def _rows():
    with DATA.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _measured(peptide_id, state, nacl):
    for r in _rows():
        if r["peptide_id"] == peptide_id and f"state={state}" in r["flags"] and f"NaCl={nacl} M" in r["flags"]:
            pct = 100 * (-float(r["value"]) - THETA_C) / (THETA_H - THETA_C)
            return r["sequence"], float(r["pH"]), float(r["ionic_strength_M"]), pct
    raise KeyError((peptide_id, state, nacl))


def _model(seq, pH, ionic):
    with contextlib.redirect_stdout(io.StringIO()):
        return AGADIR(method="1s", T=0.0, M=ionic, pH=pH).predict(seq, ncap="Ac", ccap="Am").get_percent_helix()


@pytest.mark.parametrize("pair,state,k", [
    ("HE", "H+E-", 4),  # His(i)-Glu(i+4), table 4b: -0.21 (was 0; model 0.5 vs measured 5.7 points)
    ("HE", "H+E-", 3),  # His(i)-Glu(i+3), table 4a: -0.18 (was 0)
    ("HE", "H0E-", 4),  # neutral His: the interaction persists, as a hydrogen bond
    ("KD", "K+D-", 4),  # Lys(i)-Asp(i+4), table 4b: -0.24 (was -0.40; model 12.3 vs measured 6.8 points)
    ("EK", "E-K+", 3),  # published cell, unchanged: control
    ("EK", "E-K+", 4),  # published cell, unchanged: control
])
def test_dipole_cancelled_pair_contrast(pair, state, k):
    seq_k, pH, ionic, meas_k = _measured(f"{k}{pair}", state, 1.0)
    seq_5, _, _, meas_5 = _measured(f"5{pair}", state, 1.0)
    model = _model(seq_k, pH, ionic) - _model(seq_5, pH, ionic)
    assert model == pytest.approx(meas_k - meas_5, abs=1.5)


HUYGHUES = Path(__file__).resolve().parents[1] / "pyagadir/data/peptides/1993_huyghues_protsci2_aspargglurg.tsv"


def _huyghues(peptide_id, pH, nacl):
    """Huyghues-Despointes, Klingler & Baldwin 1993 (Protein Sci. 2, 80): helicity from -[theta]222 with the
    Scholtz 1991 conversion used for this corpus set (theta_H = -40,000(1 - 2.5/n), theta_C = +640)."""
    with HUYGHUES.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["peptide_id"] == peptide_id and r["quantity"] == "minus_theta222" and float(r["pH"]) == pH
                    and float(r["ionic_strength_M"]) == nacl):
                n = int(r["n_res"])
                th_h = -40000 * (1 - 2.5 / n)
                return r["sequence"], 100 * (-float(r["value"]) - 640.0) / (th_h - 640.0)
    raise KeyError((peptide_id, pH, nacl))


def test_glu_arg_i3_orientation_neutral_ph():
    """Glu(i)-Arg(i+3) = Arg(i)-Glu(i+3) = -0.20 (table 4a; published 0 and -0.35, which reversed the measured
    preference): at pH 7, 0.01 M NaCl the Glu-first peptide is the more helical one (measured +30.8 points)."""
    seq_ab, meas_ab = _huyghues("GluArg_i+3_AB", 7.0, 0.01)
    seq_ba, meas_ba = _huyghues("GluArg_i+3_BA", 7.0, 0.01)
    assert meas_ab - meas_ba > 0
    assert _model(seq_ab, 7.0, 0.01) - _model(seq_ba, 7.0, 0.01) > 0


def test_glu_arg_i3_levels():
    """The i,i+3 Glu-Arg cells are fixed jointly by the Huyghues-Despointes levels: all eight (two orientations x two pH
    x two NaCl) within 9 points RMSE. The published cells (0 / -0.35) give 15.5 and Arg-first -0.05 alone gives 14.9."""
    err = []
    for pid in ("GluArg_i+3_AB", "GluArg_i+3_BA"):
        for pH in (2.5, 7.0):
            for nacl in (0.01, 1.0):
                seq, meas = _huyghues(pid, pH, nacl)
                err.append(_model(seq, pH, nacl) - meas)
    assert (sum(e * e for e in err) / len(err)) ** 0.5 < 9.0


def test_asp_arg_i3_levels():
    """Asp(i)-Arg(i+3) = -0.15 (table 4a; published -0.30): the Asp-first Huyghues-Despointes peptide
    Ac-ADAARADAARADAARY-NH2 at pH 2.5 and 7.0, 0.01 and 1.0 M NaCl, within 6 points RMSE. The published value gives 11.9;
    -0.15 gives 2.8."""
    err = []
    for pH in (2.5, 7.0):
        for nacl in (0.01, 1.0):
            seq, meas = _huyghues("AspArg_i+3_AB", pH, nacl)
            err.append(_model(seq, pH, nacl) - meas)
    assert (sum(e * e for e in err) / len(err)) ** 0.5 < 6.0


@pytest.mark.xfail(strict=True, reason="Documented limitation (params/README.md): the measured Glu-Arg i+3 orientation "
                   "preference depends on the charge of Glu, which a cell applied in every ionisation state cannot "
                   "carry. With Glu neutral the Arg-first peptide is measured 6.7 points more helical; the model has "
                   "the two about equal.")
def test_glu_arg_i3_orientation_neutral_acid():
    """With Glu neutral (pH 2.5, 0.01 M) the Arg-first peptide is slightly more helical (measured -6.7 points)."""
    seq_ab, meas_ab = _huyghues("GluArg_i+3_AB", 2.5, 0.01)
    seq_ba, meas_ba = _huyghues("GluArg_i+3_BA", 2.5, 0.01)
    model = _model(seq_ab, 2.5, 0.01) - _model(seq_ba, 2.5, 0.01)
    assert model == pytest.approx(meas_ab - meas_ba, abs=5.0)


RICHARDSON = Path(__file__).resolve().parents[1] / "pyagadir/data/peptides"


def test_arg_glu_i3_levels_third_and_fourth_design():
    """Richardson & Makhatadze 2004 (J. Mol. Biol. 335, 1029), Table 2, twelve Y(XEARA)n peptides at pH 2.0, 0 C, and
    Richardson et al. 1999 (Biochemistry 38, 12869), NH2-Y(MEARA)6-CONH2 at 1 C, pH 2.0 and 7.0. Every peptide carries
    only Arg(i)-Glu(i+3) pairs (four or five), so its level fixes that cell alone. Mean model - measured within 5
    points; Arg(i)-Glu(i+3) = -0.05 gives -15 and the published -0.35 gives +9."""
    err = []
    for f, q in (("2004_richardson_jmb335_table2.tsv", "helix_fraction"),
                 ("1999_richardson_biochem38_meara6.tsv", "helix_percent")):
        with (RICHARDSON / f).open() as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["quantity"] != q:
                    continue
                meas = float(r["value"]) * (100 if q == "helix_fraction" else 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    p = AGADIR(method="1s", T=float(r["T_C"]), M=float(r["ionic_strength_M"]), pH=float(r["pH"])).predict(
                        r["sequence"], ncap=r["ncap"] or None, ccap=r["ccap"] or None).get_percent_helix()
                err.append(p - meas)
    assert len(err) == 14
    assert abs(sum(err) / len(err)) < 5.0


@pytest.mark.parametrize("pH,ionic", [(6.8, 0.0314), (2.0, 0.0183)])
def test_glu_arg_i3_orientation_second_host(pH, ionic):
    """Meuzelaar, Vreede & Woutersen 2016 (Biophys. J. 110, 2328), Table S1: Ac-A(AEAAR)3A-NH2 melts above
    Ac-A(ARAAE)3A-NH2 at pH 7.0 (289.1 vs 270.0 K) and 2.5 (275.4 vs 272.6 K). CD conditions: 20 mM phosphate at
    pH 6.8 or 2.0; ionic strength derived from the buffer speciation plus the added acid."""
    with contextlib.redirect_stdout(io.StringIO()):
        er = AGADIR(method="1s", T=5.0, M=ionic, pH=pH).predict("A" + "AEAAR" * 3 + "A", ncap="Ac", ccap="Am")
        re_ = AGADIR(method="1s", T=5.0, M=ionic, pH=pH).predict("A" + "ARAAE" * 3 + "A", ncap="Ac", ccap="Am")
    assert er.get_percent_helix() > re_.get_percent_helix()


def test_aromatic_his_table_v_full_value():
    """Table V aromatic(i)-His+(i+4) keeps its full -0.4 kcal/mol with His at C1 (no Coulomb coil share): the
    Phe-8...His-12+ contact of the C-peptide (Fairman et al. 1989). The His+ - His0 difference of the Phe's i,i+4
    term isolates it, since the Table IV base does not depend on pH; at 0 C no temperature correction applies."""
    from pyagadir.energies import EnergyCalculator

    def term(pH):
        # Ac A A A F A A A H A A Am; segment N-cap at index 1, C-cap at index 9, so His (index 8) is C1
        c = EnergyCalculator(seq="AAAFAAAHAA", i=1, j=9, pH=pH, T=0.0, ionic_strength=0.1, ncap="Ac", ccap="Am")
        return float(c.get_dG_i4()[4]), abs(float(c.modified_seq_ionization_hel[8]))

    e_low, p_low = term(4.0)
    e_high, p_high = term(10.0)
    assert p_low > 0.9 and p_high < 0.1
    assert abs((e_low - e_high) - (-0.4) * (p_low - p_high)) < 0.01
