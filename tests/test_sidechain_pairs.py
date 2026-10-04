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
    """Arg(i)-Glu(i+3) = -0.05 (table 4a; was -0.35, which reversed the measured preference): at pH 7, 0.01 M NaCl the
    Glu-first peptide is the more helical one (measured +30.8 points; the old cell gave -15.6)."""
    seq_ab, meas_ab = _huyghues("GluArg_i+3_AB", 7.0, 0.01)
    seq_ba, meas_ba = _huyghues("GluArg_i+3_BA", 7.0, 0.01)
    assert meas_ab - meas_ba > 0
    assert _model(seq_ab, 7.0, 0.01) - _model(seq_ba, 7.0, 0.01) > 0


def test_glu_arg_i3_orientation_neutral_acid():
    """With Glu neutral (pH 2.5, 0.01 M) the Arg-first peptide is only slightly more helical (measured -6.7 points);
    the old cell gave -27.2."""
    seq_ab, meas_ab = _huyghues("GluArg_i+3_AB", 2.5, 0.01)
    seq_ba, meas_ba = _huyghues("GluArg_i+3_BA", 2.5, 0.01)
    model = _model(seq_ab, 2.5, 0.01) - _model(seq_ba, 2.5, 0.01)
    assert model == pytest.approx(meas_ab - meas_ba, abs=5.0)


@pytest.mark.parametrize("pH,ionic", [(6.8, 0.0314), (2.0, 0.0183)])
def test_glu_arg_i3_orientation_second_host(pH, ionic):
    """Meuzelaar, Vreede & Woutersen 2016 (Biophys. J. 110, 2328), Table S1: Ac-A(AEAAR)3A-NH2 melts above
    Ac-A(ARAAE)3A-NH2 at pH 7.0 (289.1 vs 270.0 K) and 2.5 (275.4 vs 272.6 K). CD conditions: 20 mM phosphate at
    pH 6.8 or 2.0; ionic strength derived from the buffer speciation plus the added acid."""
    with contextlib.redirect_stdout(io.StringIO()):
        er = AGADIR(method="1s", T=5.0, M=ionic, pH=pH).predict("A" + "AEAAR" * 3 + "A", ncap="Ac", ccap="Am")
        re_ = AGADIR(method="1s", T=5.0, M=ionic, pH=pH).predict("A" + "ARAAE" * 3 + "A", ncap="Ac", ccap="Am")
    assert er.get_percent_helix() > re_.get_percent_helix()
