import json
import numpy as np
from importlib import resources
from pyagadir.models import AGADIR

METRICS = ("mse", "rmse", "mae", "pearsonr")


def get_package_data_dir():
    with resources.path('pyagadir', 'data') as data_path:
        return data_path


def _score(measured, predicted, metric):
    m = np.array(measured, dtype=float)
    p = np.array(predicted, dtype=float)
    if metric == "mse":
        return float(np.mean((m - p) ** 2))
    elif metric == "rmse":
        return float(np.sqrt(np.mean((m - p) ** 2)))
    elif metric == "mae":
        return float(np.mean(np.abs(m - p)))
    elif metric == "pearsonr":
        if np.std(m) == 0 or np.std(p) == 0:
            return float("nan")
        return float(np.corrcoef(m, p)[0, 1])
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {METRICS}")


def score_lacroix_figure_3b(method="1s", metric="mse"):
    """Score PyAGADIR vs measured helix content vs pH (Lacroix 1998, Fig 3b)."""
    data_dir = get_package_data_dir()
    with open(data_dir / 'validation' / 'lacroix_figure_3_data.json') as f:
        data = json.load(f)

    fig_data = data["figure3b"]
    peptide = fig_data["peptide"]
    ncap = fig_data["ncap"]
    ccap = fig_data["ccap"]
    measured_ph = fig_data["measured_data_ph"]
    measured_helix = fig_data["measured_data_helix"]

    predicted_helix = []
    for ph in measured_ph:
        model = AGADIR(method=method, T=0.0, M=0.1, pH=ph)
        result = model.predict(peptide, ncap=ncap, ccap=ccap)
        predicted_helix.append(result.get_percent_helix())

    return {"lacroix_figure_3b": _score(measured_helix, predicted_helix, metric)}


def score_lacroix_figure_4(method="1s", metric="mse"):
    """Score per subplot vs measured helix content vs pH (Lacroix 1998, Fig 4)."""
    data_dir = get_package_data_dir()
    with open(data_dir / 'validation' / 'lacroix_figure_4_data.json') as f:
        data = json.load(f)

    scores = {}
    for figname, fig_data in data.items():
        peptide = fig_data["peptide"]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]
        measured_ph = fig_data["measured_data_ph"]
        measured_helix = fig_data["measured_data_helix"]

        predicted_helix = []
        for ph in measured_ph:
            model = AGADIR(method=method, T=0.0, M=0.1, pH=ph)
            result = model.predict(peptide, ncap=ncap, ccap=ccap)
            predicted_helix.append(result.get_percent_helix())

        scores[f"lacroix_figure_4_{figname}"] = _score(measured_helix, predicted_helix, metric)

    return scores


def score_huygues_despointes_figure_1(method="1s", metric="mse"):
    """Score per pH condition vs measured helix content (Huygues-Despointes 1993, Fig 1ab)."""
    data_dir = get_package_data_dir()
    with open(data_dir / 'validation' / 'huygues_despointes_figure_1_data.json') as f:
        data = json.load(f)

    fig_data = data["figure1ab"]
    peptides = fig_data["peptides"]
    ncap = fig_data["ncap"]
    ccap = fig_data["ccap"]

    scores = {}
    for ph in [2, 7]:
        measured = [val * 100 for val in fig_data["measured_ph_" + str(ph)]]
        predicted = []
        for pept in peptides:
            model = AGADIR(method=method, T=0.0, M=0.01, pH=ph)
            result = model.predict(pept, ncap=ncap, ccap=ccap)
            predicted.append(result.get_percent_helix())
        scores[f"huygues_despointes_figure_1_ph{ph}"] = _score(measured, predicted, metric)

    return scores


def score_munoz_1997_figure_4(method="1s", metric="mse"):
    """Score per peptide series vs measured helix content (Munoz 1997, Fig 4)."""
    data_dir = get_package_data_dir()
    with open(data_dir / 'validation' / 'munoz_1997_figure_4_data.json') as f:
        data = json.load(f)

    scores = {}
    for figname, fig_data in data.items():
        peptides = fig_data["peptides"]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]
        measured = fig_data["helicity"]

        predicted = []
        for pept in peptides:
            model = AGADIR(method=method, T=0.0, M=0.1, pH=7.0)
            result = model.predict(pept, ncap=ncap, ccap=ccap)
            predicted.append(result.get_percent_helix())

        scores[f"munoz_1997_figure_4_{figname}"] = _score(measured, predicted, metric)

    return scores


def score_munoz_1995_figure_3(method="1s", metric="mse"):
    """Score per peptide vs measured helix content vs temperature (Munoz 1995, Fig 3)."""
    data_dir = get_package_data_dir()
    with open(data_dir / 'validation' / 'munoz_1995_figure_3.json') as f:
        data = json.load(f)

    scores = {}
    for figname, fig_data in data.items():
        peptide = fig_data["peptide"]
        temperatures = fig_data["temperatures"]
        measured = fig_data["helicity"]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]

        predicted = []
        for temp in temperatures:
            model = AGADIR(method=method, T=temp, M=0.025, pH=7.0)
            result = model.predict(peptide, ncap=ncap, ccap=ccap)
            predicted.append(result.get_percent_helix())

        scores[f"munoz_1995_figure_3_{figname}"] = _score(measured, predicted, metric)

    return scores


def score_all(method="1s", metric="mse"):
    """Run all scoring functions and return a flat dict keyed by plot name.

    Args:
        method: AGADIR method (default "1s").
        metric: One of "mse", "rmse", "mae", "pearsonr" (default "mse").

    Returns:
        dict mapping plot name to score.
    """
    scores = {}
    scores.update(score_lacroix_figure_3b(method=method, metric=metric))
    scores.update(score_lacroix_figure_4(method=method, metric=metric))
    scores.update(score_huygues_despointes_figure_1(method=method, metric=metric))
    scores.update(score_munoz_1997_figure_4(method=method, metric=metric))
    scores.update(score_munoz_1995_figure_3(method=method, metric=metric))
    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score PyAGADIR predictions against measured data.")
    parser.add_argument("--method", default="1s", help="AGADIR method (default: 1s)")
    parser.add_argument("--metric", default="mse", choices=METRICS,
                        help="Scoring metric (default: mse)")
    args = parser.parse_args()

    scores = score_all(method=args.method, metric=args.metric)
    for name, value in scores.items():
        print(f"{name}: {args.metric.upper()} = {value:.4f}")
