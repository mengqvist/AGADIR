import matplotlib.pyplot as plt
from pyagadir.models import AGADIR
import json
from importlib import resources
from pathlib import Path


TITLE_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8

def get_package_data_dir():
    """Get the path to the data directory in the installed package."""
    with resources.path('pyagadir', 'data') as data_path:
        return data_path

def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    data_dir = get_package_data_dir()
    figures_dir = data_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    return figures_dir

def plot_ph_helix_content(paper_measured_data_ph, paper_measured_data_helix,
                        paper_predicted_data_ph, paper_predicted_data_helix,
                        pyagadir_predicted_data_helix, peptide, ncap, ccap,
                        ax=None, fig=None):
    """Create a plot comparing measured and predicted helix content vs pH.
    
    Args:
        paper_measured_data_ph: List of pH values from paper measurements
        paper_measured_data_helix: List of helix content values from paper measurements
        paper_predicted_data_ph: List of pH values from paper predictions
        paper_predicted_data_helix: List of helix content values from paper predictions
        pyagadir_predicted_data_helix: List of helix content values from PyAGADIR
        peptide: Peptide sequence
        ncap: N-terminal capping
        ccap: C-terminal capping
        ax: Optional matplotlib axis to plot on. If None, creates new figure.
        fig: Optional matplotlib figure to plot on. If None, creates new figure.

    Returns:
        matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(
        paper_predicted_data_ph,
        paper_predicted_data_helix,
        "o",
        color="white",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        label="Paper predicted",
    )
    ax.plot(
        paper_measured_data_ph,
        paper_measured_data_helix,
        "o",
        color="black", 
        markersize=5,
        label="Paper measured",
    )
    ax.plot(
        paper_predicted_data_ph,
        pyagadir_predicted_data_helix,
        "o",
        color="orange",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        label="PyAGADIR",
    )

    # Configure plot
    ax.set_xlim(2.8, 12)
    ax.set_ylim(0, 65)
    ax.set_xlabel("pH", fontsize=LEGEND_FONT_SIZE)
    ax.set_ylabel("Helix content (%)", fontsize=LEGEND_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)

    # Set title with caps
    title = ""
    if ncap == "Ac":
        title += "Ac-"
    elif ncap == "Sc":
        title += "Sc-"
    title += peptide
    if ccap == "Am":
        title += "-Am"
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    
    return fig, ax

def plot_peptides_helix_content(paper_measured_data_helix,
                                pyagadir_predicted_data_helix, 
                                xvals, 
                                title,
                                xlabel,
                                ncap, 
                                ccap,
                                ax=None, 
                                fig=None):
    """Create a plot of the helix content of a list of peptides at a given pH.
    
    Args:
        paper_measured_data_helix: List of helix content values from paper measurements
        pyagadir_predicted_data_helix: List of helix content values from PyAGADIR
        xvals: List of x-values for the plot
        title: Title of the plot
        xlabel: Label of the x-axis
        ncap: N-terminal capping
        ccap: C-terminal capping
        ax: Optional matplotlib axis to plot on. If None, creates new figure.
        fig: Optional matplotlib figure to plot on. If None, creates new figure.

    Returns:
        matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(
        xvals,
        paper_measured_data_helix,
        "o",
        color="black", 
        markersize=5,
        label="Paper measured",
    )
    ax.plot(
        xvals,
        pyagadir_predicted_data_helix,
        "o",
        color="orange",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=5,
        label="PyAGADIR",
    )

    # Configure plot
    xval_range = max(xvals)-min(xvals)
    ax.set_xlim(min(xvals)-0.1*xval_range, max(xvals)+0.1*xval_range)
    ax.set_ylim(0, max(max(paper_measured_data_helix), max(pyagadir_predicted_data_helix)) + 10)
    ax.set_xlabel(xlabel, fontsize=LEGEND_FONT_SIZE)
    ax.set_ylabel("Helix content (%)", fontsize=LEGEND_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=LEGEND_FONT_SIZE)
    
    return fig, ax


def reproduce_lacroix_figure_3b(method="1s"):
    """
    Reproduce figure 3b from the Lacroix et al. (1998) AGADIR paper.
    https://doi.org/10.1006/jmbi.1998.2145
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'lacroix_figure_3_data.json'
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
        model = AGADIR(method=method, T=0.0, M=0.1, pH=ph)
        result = model.predict(peptide, ncap=ncap, ccap=ccap)
        pyagadir_predicted_data_helix.append(result.get_percent_helix())
        
    # Plot
    fig, ax = plot_ph_helix_content(paper_measured_data_ph, paper_measured_data_helix,
                                    paper_predicted_data_ph, paper_predicted_data_helix,
                                    pyagadir_predicted_data_helix, peptide, ncap, ccap)

    # Save figure
    fig.savefig(figures_dir / "lacroix_figure_3b.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def reproduce_lacroix_figure_4(method="1s"):
    """
    Reproduce figure 4 from the AGADIR paper.
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'lacroix_figure_4_data.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    # Create figure
    fig, axs = plt.subplots(4, 2, figsize=(10, 16))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot each figure
    for i, (figname, ax) in enumerate(zip(data.keys(), axs.flatten())):
        fig_data = data[figname]
        peptide = fig_data["peptide"]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]

        paper_measured_data_ph = fig_data["measured_data_ph"]
        paper_measured_data_helix = fig_data["measured_data_helix"]
        paper_predicted_data_ph = fig_data["predicted_data_ph"]
        paper_predicted_data_helix = fig_data["predicted_data_helix"]

        pyagadir_predicted_data_helix = []
        for ph in paper_predicted_data_ph:
            model = AGADIR(method=method, T=0.0, M=0.1, pH=ph)
            result = model.predict(peptide, ncap=ncap, ccap=ccap)
            pyagadir_predicted_data_helix.append(result.get_percent_helix())
            

        _, ax = plot_ph_helix_content(paper_measured_data_ph, paper_measured_data_helix,
                                        paper_predicted_data_ph, paper_predicted_data_helix,
                                        pyagadir_predicted_data_helix, peptide, ncap, ccap,
                                        ax=ax)


    fig.savefig(figures_dir / "lacroix_figure_4.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def reproduce_huygues_despointes_figure_1(method="1s"):
    """
    Reproduce Figure 1A and 1B from the Huygues-Despointes et al. (1993) paper.
    10 mM NaCl, 0.0 C
    https://onlinelibrary.wiley.com/doi/10.1002/pro.5560021006
    """
        # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'huygues_despointes_figure_1_data.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot each figure
    fig_data = data["figure1ab"]
    for ph, ax in zip([2, 7], axs.flatten()):
        
        peptides = fig_data["peptides"]
        xvals = fig_data["asp_pos"]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]
        paper_measured_data = [val * 100 for val in fig_data["measured_ph_" + str(ph)]]

        pyagadir_predicted_data_helix = []
        for pept in peptides:
            model = AGADIR(method=method, T=0.0, M=0.01, pH=ph)
            result = model.predict(pept, ncap=ncap, ccap=ccap)
            pyagadir_predicted_data_helix.append(result.get_percent_helix())
            
        title = f"Ac-DAQAAAAQAAAAQAAY-Am\nwith varying Asp positions, at pH " + str(ph)
        xlabel = "Asp position"
        _, ax = plot_peptides_helix_content(paper_measured_data,
                                            pyagadir_predicted_data_helix, 
                                            xvals,
                                            title,
                                            xlabel,
                                            ncap, 
                                            ccap,
                                            ax=ax)


    fig.savefig(figures_dir / "huygues_despointes_figure_1a.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def reproduce_munoz_1997_figure_4(method="1s"):
    """
    Reproduce Figure 4 from the Munoz et al. (1997) paper.
    Original values from Scholtz et al. (1991) and Rohl et al. (1992).
    https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5%3C495::AID-BIP2%3E3.0.CO;2-H
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'munoz_1997_figure_4_data.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(4, 12))
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot each figure
    for figname, repeat, ax in zip(data.keys(), ['AAQAA', 'AAKAA', 'AEAAKA'], axs.flatten()):
        fig_data = data[figname]
        peptides = fig_data["peptides"]
        xvals = [len(pept) for pept in peptides]
        ncap = fig_data["ncap"]
        ccap = fig_data["ccap"]
        paper_measured_data = fig_data["helicity"]

        pyagadir_predicted_data_helix = []
        for pept in peptides:
            model = AGADIR(method=method, T=0.0, M=0.1, pH=7.0)
            result = model.predict(pept, ncap=ncap, ccap=ccap)
            pyagadir_predicted_data_helix.append(result.get_percent_helix())
            
        title = f'{ncap}-Y[{repeat}](n)F-{ccap}'
        xlabel = "Peptide length"
        _, ax = plot_peptides_helix_content(paper_measured_data,
                                            pyagadir_predicted_data_helix, 
                                            xvals,
                                            title,
                                            xlabel,
                                            ncap, 
                                            ccap,
                                            ax=ax)

    fig.savefig(figures_dir / "munoz_1997_figure_4.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def reproduce_munoz_1995_figure_3(method="1s"):
    """
    Reproduce Figure 3 from the Munoz et al. (1995) paper.
    Testing various peptides under different temperatures.
    https://doi.org/10.1006/jmbi.1994.0024
    """
    # Get paths
    data_dir = get_package_data_dir()
    validation_file = data_dir / 'validation' / 'munoz_1995_figure_3.json'
    figures_dir = ensure_figures_dir()

    # Load validation data
    with open(validation_file, "r") as f:
        data = json.load(f)

    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    





    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)


    fig.savefig(figures_dir / "munoz_1995_figure_3.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def predict(method="1s"):
    agadir = AGADIR(method=method, pH=4.0, M=0.05, T=0.0)
    result = agadir.predict("YGGSAAAAAAAKRAAA", ncap=None, ccap='Am', debug=True)
    print(result.get_percent_helix())


if __name__ == "__main__":
    method = "1s"
    reproduce_lacroix_figure_3b(method=method)
    reproduce_lacroix_figure_4(method=method)
    reproduce_huygues_despointes_figure_1(method=method)
    reproduce_munoz_1997_figure_4(method=method)
    reproduce_munoz_1995_figure_3(method=method)

    # predict(method=method) # I typically direct the output from this funcion to a file: python validation.py > output.txt
