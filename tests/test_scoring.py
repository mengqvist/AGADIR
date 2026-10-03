"""pyagadir.scoring scores every dataset at the experimental conditions recorded in its data file.

The model is replaced by a stub that records the conditions it is constructed with, so these
tests check the bookkeeping (which conditions reach the model) and not helicity values.
"""
import pytest

import pyagadir.scoring as scoring


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
    assert len(scores) == 20
    assert all(v == v for v in scores.values())  # no NaN from an empty panel


def test_lacroix_uses_recorded_conditions(calls):
    # Lacroix et al. 1998: 278 K, 2.5 mM sodium phosphate (I ~ 5 mM), not 0 C / 0.1 M
    scoring.score_lacroix_figure_4(metric="rmse")
    assert {(T, M) for _, (T, M, _) in calls} == {(5.0, 0.005)}


def test_munoz_1997_panels_and_mixed_point_use_their_own_conditions(calls):
    scoring.score_munoz_1997_figure_4(metric="rmse")
    cond = {seq: c for seq, c in calls}
    assert cond["AAKAAAAKAAAAKAAY"] == (5.0, 1.0, 2.5)          # 4B, Rohl et al. 1992
    assert cond["AAQAAAAQAA"] == (5.0, 0.0044, 7.0)              # 4A, Munoz & Serrano 1995-II
    assert cond["AAQAAAAQAAAAQAAY"] == (0.0, 0.102, 7.0)         # 4A's point from Scholtz 1991
