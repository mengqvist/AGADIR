import matplotlib.pyplot as plt
from pyagadir.models import AGADIR
import json
from importlib import resources
from pathlib import Path

def get_package_data_dir():
    """Get the path to the data directory in the installed package."""
    with resources.path('pyagadir', 'data') as data_path:
        return data_path

def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    data_dir = get_package_data_dir()
    figures_dir = data_dir.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    return figures_dir

def reproduce_figure_3b():
    """
    Reproduce figure 3b from the AGADIR paper.
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'figure_3_data.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    peptide = data["figure3b"]["peptide"]
    ncap = data["figure3b"]["ncap"]
    ccap = data["figure3b"]["ccap"]

    paper_measured_data_ph = data["figure3b"]["measured_data_ph"]
    paper_measured_data_helix = data["figure3b"]["measured_data_helix"]
    paper_predicted_data_ph = data["figure3b"]["predicted_data_ph"]
    paper_predicted_data_helix = data["figure3b"]["predicted_data_helix"]

    # AGADIR results
    pyagadir_predicted_data_helix = []
    for ph in paper_predicted_data_ph:
        model = AGADIR(method="1s", T=0.0, M=0.1, pH=ph)
        result = model.predict(peptide, ncap=ncap, ccap=ccap)
        pyagadir_predicted_data_helix.append(result.get_percent_helix())

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(
        paper_measured_data_ph,
        paper_measured_data_helix,
        "o",
        color="black",
        label="Paper measured",
    )
    ax.plot(
        paper_predicted_data_ph,
        paper_predicted_data_helix,
        "o",
        color="white",
        markeredgecolor="black",
        markeredgewidth=1,
        label="Paper predicted",
    )
    ax.plot(
        paper_predicted_data_ph,
        pyagadir_predicted_data_helix,
        "o",
        color="orange",
        label="PyAGADIR",
    )

    # Configure plot
    ax.set_xlim(2.8, 12)
    ax.set_ylim(0, 65)
    ax.set_xlabel("pH")
    ax.set_ylabel("Helix content (%)")
    ax.legend()

    # Set title with caps
    title = ""
    if ncap == "Z":
        title += "Ac-"
    elif ncap == "X":
        title += "Succ-"
    title += peptide
    if ccap == "B":
        title += "-Am"
    ax.set_title(title, fontsize=12)

    # Save figure
    fig.savefig(figures_dir / "figure_3b.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def reproduce_figure_4():
    """
    Reproduce figure 4 from the AGADIR paper.
    """
    pass

if __name__ == "__main__":
    reproduce_figure_3b()
    reproduce_figure_4()
