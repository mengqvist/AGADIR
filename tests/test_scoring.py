"""pyagadir.scoring scores every dataset at the experimental conditions recorded in its data file.

The model is replaced by a stub that records the conditions it is constructed with, so these
tests check the bookkeeping (which conditions reach the model) and not helicity values.
"""
import pytest

import pyagadir.scoring as scoring
from pyagadir.validation import titrated_ionic_strength


class _RecordingModel:
    calls = []

    def __init__(self, method="1s", T=None, M=None, pH=None):
        self.cond = (T, M, pH)

    def predict(self, seq, ncap=None, ccap=None):
        _RecordingModel.calls.append((seq, self.cond))
        return self

    def get_percent_helix(self):
        return 50.0


@pytest.fixture
def calls(monkeypatch):
    _RecordingModel.calls = []
    monkeypatch.setattr(scoring, "AGADIR", _RecordingModel)
    return _RecordingModel.calls


def test_score_all_runs_and_returns_every_panel(calls):
    scores = scoring.score_all(metric="rmse")
    assert len(scores) == 26  # Munoz 1995 set: 3A-C, 3E + six Scholtz 1991 + two Marqusee 1987 melts (NPY 3D removed)
    assert all(v == v for v in scores.values())  # no NaN from an empty panel


def test_lacroix_uses_recorded_conditions(calls):
    # Lacroix et al. 1998: 278 K, 2.5 mM sodium phosphate made up at pH 7.0 and titrated with HCl/NaOH, not 0 C / 0.1 M
    scoring.score_lacroix_figure_4(metric="rmse")
    buffer = {"sodium_phosphate_M": 0.0025, "prepared_pH": 7.0}
    assert {T for _, (T, _, _) in calls} == {5.0}
    assert all(M == pytest.approx(titrated_ionic_strength(0.005, pH, buffer), rel=1e-12) for _, (_, M, pH) in calls)
    assert max(M for _, (_, M, _) in calls) > 0.015  # pH 12: the NaOH roughly quadruples the buffer's 4.4 mM


def test_munoz_1997_panels_and_mixed_point_use_their_own_conditions(calls):
    scoring.score_munoz_1997_figure_4(metric="rmse")
    cond = {seq: c for seq, c in calls}
    # recorded salt and buffer + the HCl/NaOH that set the pH (validation.titrated_ionic_strength)
    assert cond["AAKAAAAKAAAAKAAY"] == (5.0, titrated_ionic_strength(1.0, 2.5), 2.5)      # 4B, Rohl et al. 1992
    assert cond["AAQAAAAQAA"] == (5.0, titrated_ionic_strength(0.0044, 7.0), 7.0)        # 4A, Munoz & Serrano 1995-II
    assert cond["AAQAAAAQAAAAQAAY"] == (0.0, titrated_ionic_strength(0.102, 7.0), 7.0)   # 4A's point from Scholtz 1991
