import numpy as np
from pyagadir.models import AGADIR
from pyagadir.validation import get_package_data_dir, load_panel_conditions, load_validation, point_conditions

METRICS = ("mse", "rmse", "mae", "pearsonr")


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


def _panel_conditions(filename, panel, temp_C, ionic_M, pH=7.0):
    """Conditions for one panel: its ``per_panel`` provenance entry if it has one, else the file's."""
    cond = load_panel_conditions(filename).get(panel, {})
    return (cond.get("temperature_C", temp_C), cond.get("ionic_strength_M", ionic_M),
            cond.get("pH", pH))


def _predict(peptide, ncap, ccap, method, T, M, pH):
    model = AGADIR(method=method, T=T, M=M, pH=pH)
    return model.predict(peptide, ncap=ncap, ccap=ccap).get_percent_helix()


# Every function below scores at the experimental conditions recorded in the dataset's
# ``_provenance`` block (the same ones the validation figures use), at the measured x-values.

def score_lacroix_figure_3b(method="1s", metric="mse"):
    """Score PyAGADIR vs measured helix content vs pH (Lacroix 1998, Fig 3b)."""
    filename = 'lacroix_figure_3_data.json'
    data, temp_C, ionic_M = load_validation(filename)
    fig_data = data["figure3b"]
    T, M, _ = _panel_conditions(filename, "figure3b", temp_C, ionic_M)
    predicted = [_predict(fig_data["peptide"], fig_data["ncap"], fig_data["ccap"], method, T, M, ph)
                 for ph in fig_data["measured_data_ph"]]
    return {"lacroix_figure_3b": _score(fig_data["measured_data_helix"], predicted, metric)}


def score_lacroix_figure_4(method="1s", metric="mse"):
    """Score per subplot vs measured helix content vs pH (Lacroix 1998, Fig 4)."""
    filename = 'lacroix_figure_4_data.json'
    data, temp_C, ionic_M = load_validation(filename)
    scores = {}
    for figname, fig_data in data.items():
        T, M, _ = _panel_conditions(filename, figname, temp_C, ionic_M)
        predicted = [_predict(fig_data["peptide"], fig_data["ncap"], fig_data["ccap"], method, T, M, ph)
                     for ph in fig_data["measured_data_ph"]]
        scores[f"lacroix_figure_4_{figname}"] = _score(fig_data["measured_data_helix"], predicted, metric)
    return scores


def score_huygues_despointes_figure_1(method="1s", metric="mse"):
    """Score per pH condition vs measured helix content (Huygues-Despointes 1993, Fig 1ab)."""
    filename = 'huygues_despointes_figure_1_data.json'
    data, temp_C, ionic_M = load_validation(filename)
    fig_data = data["figure1ab"]
    T, M, _ = _panel_conditions(filename, "figure1ab", temp_C, ionic_M)
    scores = {}
    for ph in [2, 7]:
        measured = [val * 100 for val in fig_data["measured_ph_" + str(ph)]]
        predicted = [_predict(pept, fig_data["ncap"], fig_data["ccap"], method, T, M, ph)
                     for pept in fig_data["peptides"]]
        scores[f"huygues_despointes_figure_1_ph{ph}"] = _score(measured, predicted, metric)
    return scores


def score_munoz_1997_figure_4(method="1s", metric="mse"):
    """Score per peptide series vs measured helix content (Munoz 1997, Fig 4).

    The three panels replot different experiments, so each uses its own recorded conditions,
    and a point taken from yet another experiment uses its ``per_point`` entry.
    """
    filename = 'munoz_1997_figure_4_data.json'
    data, temp_C, ionic_M = load_validation(filename)
    scores = {}
    panel_cond = load_panel_conditions(filename)
    for figname, fig_data in data.items():
        T, M, pH = _panel_conditions(filename, figname, temp_C, ionic_M)
        measured, predicted = [], []
        for pept, y in zip(fig_data["peptides"], fig_data["helicity"]):
            T_pt, M_pt, pH_pt = point_conditions(panel_cond.get(figname, {}), pept, T, M, pH)
            try:
                predicted.append(_predict(pept, fig_data["ncap"], fig_data["ccap"], method, T_pt, M_pt, pH_pt))
            except ValueError:
                continue  # shorter than the model's minimum length; the figure drops it too
            measured.append(y)
        scores[f"munoz_1997_figure_4_{figname}"] = _score(measured, predicted, metric)
    return scores


def score_munoz_1995_figure_3(method="1s", metric="mse"):
    """Score per peptide vs measured helix content vs temperature (Munoz 1995, Fig 3)."""
    data, _temp_unused, ionic_M = load_validation('munoz_1995_figure_3.json')
    scores = {}
    for figname, fig_data in data.items():
        predicted = [_predict(fig_data["peptide"], fig_data["ncap"], fig_data["ccap"], method,
                              temp, ionic_M, fig_data.get("pH", 7.0))
                     for temp in fig_data["temperatures"]]
        scores[f"munoz_1995_figure_3_{figname}"] = _score(fig_data["helicity"], predicted, metric)
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
    import io
    from contextlib import redirect_stdout

    parser = argparse.ArgumentParser(description="Score PyAGADIR predictions against measured data.")
    parser.add_argument("--method", default="1s", help="AGADIR method (default: 1s)")
    parser.add_argument("--metric", default="mse", choices=METRICS,
                        help="Scoring metric (default: mse)")
    args = parser.parse_args()

    with redirect_stdout(io.StringIO()):  # the model prints one line per prediction
        scores = score_all(method=args.method, metric=args.metric)
    for name, value in scores.items():
        print(f"{name}: {args.metric.upper()} = {value:.4f}")
