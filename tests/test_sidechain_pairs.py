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
