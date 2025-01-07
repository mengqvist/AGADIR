import math
from importlib.resources import files
from itertools import combinations

import numpy as np
import pandas as pd

from pyagadir.utils import is_valid_index, is_valid_peptide_sequence

# get params
datapath = files("pyagadir.data")

# load energy contributions for intrinsic propensities, capping, etc.
table_1_lacroix = pd.read_csv(
    datapath.joinpath("table_1_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load the hydrophobic staple motif energy contributions
table_2_lacroix = pd.read_csv(
    datapath.joinpath("table_2_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load the schellman motif energy contributions
table_3_lacroix = pd.read_csv(
    datapath.joinpath("table_3_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load energy contributions for interactions between i and i+3
table_4a_lacroix = pd.read_csv(
    datapath.joinpath("table_4a_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load energy contributions for interactions between i and i+4
table_4b_lacroix = pd.read_csv(
    datapath.joinpath("table_4b_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load sidechain distances for helices
table_6_helix_lacroix = pd.read_csv(
    datapath.joinpath("table_6_helix_lacroix.tsv"),
    index_col="Pos",
    sep="\t",
).astype(float)

# load sidechain distances for coils
table_6_coil_lacroix = pd.read_csv(
    datapath.joinpath("table_6_coil_lacroix.tsv"),
    index_col="Pos",
    sep="\t",
).astype(float)

# load N-terminal distances between charged amino acids and the half charge from the helix macrodipole
table_7_ccap_lacroix = pd.read_csv(
    datapath.joinpath("table_7_Ccap_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load C-terminal distances between charged amino acids and the half charge from the helix macrodipole
table_7_ncap_lacroix = pd.read_csv(
    datapath.joinpath("table_7_Ncap_lacroix.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)

# load pKa values for for side chain ionization and the N- and C-terminal capping groups
pka_values = pd.read_csv(
    datapath.joinpath("pka_values.tsv"),
    index_col="AA",
    sep="\t",
).astype(float)


# # load energy contributions between amino acids and the helix macrodipole, focusing on the C-terminal
# table3a = pd.read_csv(
#     datapath.joinpath('table3a.csv'),
#     index_col='AA',
# ).astype(float)
# table3a.columns = table3a.columns.astype(int)

# # load energy contributions between amino acids and the helix macrodipole, focusing on the N-terminal
# table3b = pd.read_csv(
#     datapath.joinpath('table3b.csv'),
#     index_col='AA',
# ).astype(float)
# table3b.columns = table3b.columns.astype(int)


def get_helix(pept: str, i: int, j: int) -> str:
    """
    Get the helix region of a peptide sequence, including the N- and C-caps.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        str: The helix region of the peptide sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    return pept[i : i + j]


def get_capping_status(
    pept: str, i: int, j: int, has_acetyl: bool, has_succinyl: bool, has_amide: bool
) -> tuple[bool, bool]:
    """
    Determine whether the first and last residues of a helical segment should be treated as capping residues,
    due to modifications. Internal helical segments should be unaffected by modifications and
    the capping status should be determined by the helix start and end indices when those are at
    the peptide termini.

    Args:
        pept (str): The complete peptide sequence
        i (int): Starting index of the helical segment
        j (int): Length of the helical segment
        has_acetyl (bool): Whether the peptide has N-terminal acetylation
        has_succinyl (bool): Whether the peptide has N-terminal succinylation
        has_amide (bool): Whether the peptide has C-terminal amidation

    Returns:
        tuple[bool, bool]: (has_ncap, has_ccap) indicating whether the segment should be treated
                          as having N and C capping residues
    """
    # By default, every helical segment has capping residues
    has_ncap = True
    has_ccap = True

    # Check if this segment includes the N-terminus of the peptide
    if i == 0:
        # If we have N-terminal modification, this segment doesn't have an N-cap
        if has_acetyl or has_succinyl:
            has_ncap = False

    # Check if this segment includes the C-terminus of the peptide
    if i + j == len(pept):
        # If we have C-terminal modification, this segment doesn't have a C-cap
        if has_amide:
            has_ccap = False

    return has_ncap, has_ccap


def get_dG_Int(
    pept: str,
    i: int,
    j: int,
    pH: float,
    T: float,
    has_acetyl: bool = False,
    has_succinyl: bool = False,
    has_amide: bool = False,
) -> np.ndarray:
    """
    Get the intrinsic free energy contributions for a sequence.
    The first and last residues are considered to be caps unless they are
    terminal residues with modifications.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        pH (float): The pH value.
        T (float): Temperature in Kelvin.
        has_acetyl (bool): Whether the peptide has N-terminal acetylation
        has_succinyl (bool): Whether the peptide has N-terminal succinylation
        has_amide (bool): Whether the peptide has C-terminal amidation

    Returns:
        np.ndarray: The intrinsic free energy contributions for each amino acid in the sequence.
    """
    helix = get_helix(pept, i, j)
    has_ncap, has_ccap = get_capping_status(
        pept, i, j, has_acetyl, has_succinyl, has_amide
    )

    # initialize energy array
    energy = np.zeros(len(helix))

    # iterate over the helix and get the intrinsic energy for each residue
    for idx, AA in enumerate(helix):
        # Skip caps only if they exist for this segment
        if (idx == 0 and has_ncap) or (idx == len(helix) - 1 and has_ccap):
            continue

        # Handle N-terminal region specially (note: these positions are relative to helix start)
        if idx == 1 or (idx == 0 and not has_ncap):
            energy[idx] = table_1_lacroix.loc[AA, "N1"]
        elif idx == 2 or (idx == 1 and not has_ncap):
            energy[idx] = table_1_lacroix.loc[AA, "N2"]
        elif idx == 3 or (idx == 2 and not has_ncap):
            energy[idx] = table_1_lacroix.loc[AA, "N3"]
        elif idx == 4 or (idx == 3 and not has_ncap):
            energy[idx] = table_1_lacroix.loc[AA, "N4"]
        else:
            if AA not in ["C", "D", "E", "H", "K", "R", "Y"]:
                energy[idx] = table_1_lacroix.loc[AA, "Ncen"]
            else:
                # pH-dependent selection between 'Ncen' and 'Neutral'
                pKa = pka_values.loc[AA, "pKa"]
                basic_energy_ncen = table_1_lacroix.loc[AA, "Ncen"]
                basic_energy_neutral = table_1_lacroix.loc[AA, "Neutral"]
                if AA in ["C", "D", "E"]:
                    q = acidic_residue_ionization(pH, pKa, 0, T)
                    energy[idx] = q * basic_energy_ncen + (1 - q) * basic_energy_neutral
                elif AA in ["H", "K", "R", "Y"]:
                    q = basic_residue_ionization(pH, pKa, 0, T)
                    energy[idx] = q * basic_energy_ncen + (1 - q) * basic_energy_neutral
                else:
                    raise ValueError(
                        f"Invalid logic in internal energy selection: {AA}"
                    )

    return energy


def get_dG_Ncap(
    pept: str,
    i: int,
    j: int,
    has_acetyl: bool = False,
    has_succinyl: bool = False,
    has_amide: bool = False,
) -> np.ndarray:
    """
    Get the free energy contribution for N-terminal capping.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        has_acetyl (bool): Whether the peptide has N-terminal acetylation
        has_succinyl (bool): Whether the peptide has N-terminal succinylation
        has_amide (bool): Whether the peptide has C-terminal amidation
    Returns:
        np.ndarray: The free energy contribution.
    """
    helix = get_helix(pept, i, j)
    has_ncap, has_ccap = get_capping_status(
        pept, i, j, has_acetyl, has_succinyl, has_amide
    )

    # fix the blocking group names to match the table
    Ncap_AA = helix[0]
    if not has_ncap:
        Ncap_AA = "Ac"

    energy = np.zeros(len(helix))

    # Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.
    N1_AA = helix[1]
    N3_AA = helix[3]
    if N1_AA == "P" and N3_AA in ["E", "D", "Q"]:
        energy[0] = table_1_lacroix.loc[Ncap_AA, "Nc-4"]

    # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
    elif N3_AA in ["E", "D", "Q"]:
        energy[0] = table_1_lacroix.loc[Ncap_AA, "Nc-3"]

    # Nc-2 	N-cap values when there is a Pro at position N1.
    elif N1_AA == "P":
        energy[0] = table_1_lacroix.loc[Ncap_AA, "Nc-2"]

    # Nc-1 	Normal N-cap values.
    else:
        energy[0] = table_1_lacroix.loc[Ncap_AA, "Nc-1"]

    return energy


def get_dG_Ccap(
    pept: str,
    i: int,
    j: int,
    has_acetyl: bool = False,
    has_succinyl: bool = False,
    has_amide: bool = False,
) -> np.ndarray:
    """
    Get the free energy contribution for C-terminal capping.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        has_acetyl (bool): Whether the peptide has N-terminal acetylation
        has_succinyl (bool): Whether the peptide has N-terminal succinylation
        has_amide (bool): Whether the peptide has C-terminal amidation

    Returns:
        np.ndarray: The free energy contribution.
    """
    helix = get_helix(pept, i, j)
    has_ncap, has_ccap = get_capping_status(
        pept, i, j, has_acetyl, has_succinyl, has_amide
    )

    # fix the blocking group names to match the table
    Ccap_AA = helix[-1]
    if not has_ccap:
        Ccap_AA = "Am"

    energy = np.zeros(len(helix))

    # Cc-2 	C-cap values when there is a Pro residue at position C'
    c_prime_idx = i + j
    if (len(pept) > c_prime_idx) and (pept[c_prime_idx] == "P"):
        energy[-1] = table_1_lacroix.loc[Ccap_AA, "Cc-2"]

    # Cc-1 	Normal C-cap values
    else:
        energy[-1] = table_1_lacroix.loc[Ccap_AA, "Cc-1"]

    return energy


def get_dG_staple(pept: str, i: int, j: int) -> float:
    """
    Get the free energy contribution for the hydrophobic staple motif.
    The hydrophobic interaction is between the N' and N4 residues of the helix.
    See https://doi.org/10.1038/nsb0595-380 for more details.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        float: The free energy contribution.
    """
    helix = get_helix(pept, i, j)

    energy = np.zeros(len(helix))

    # get the amino acids governing the staple motif
    N_prime_AA = pept[i - 1]
    Ncap_AA = helix[0]
    N3_AA = helix[3]
    N4_AA = helix[4]
    energy = 0.0

    # staple motif requires the N' residue before the Ncap, so the first residue of the helix cannot be the first residue of the peptide
    if i == 0:
        return energy

    # TODO: verify that the code below is correct

    # The hydrophobic staple motif is only considered whenever the N-cap residue is Asn, Asp, Ser, Pro or Thr.
    if Ncap_AA in ["N", "D", "S", "P", "T"]:
        energy = table_2_lacroix.loc[N_prime_AA, N4_AA]

        # whenever the N-cap residue is Asn, Asp, Ser, or Thr and the N3 residue is Glu, Asp or Gln, multiply by 1.0
        if Ncap_AA in ["N", "D", "S", "T"] and N3_AA in ["E", "D", "Q"]:
            print("staple case i")
            energy *= 1.0

        # whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr
        elif Ncap_AA in ["N", "D"] and N3_AA in ["S", "T"]:
            print("staple case ii")
            energy *= 1.0

        # other cases they are multiplied by 0.5
        else:
            print("staple case iii")
            energy *= 0.5

    else:
        print("no staple motif")

    return energy


def get_dG_schellman(pept: str, i: int, j: int) -> float:
    """
    Get the free energy contribution for the Schellman motif.
    The Schellman motif is only considered whenever Gly is the C-cap residue,
    where the interaction happens between the C' and C3 residues of the helix.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        float: The free energy contribution.
    """
    # TODO: is this affected by acylation, succinylation, or amidation? Find out!
    helix = get_helix(pept, i, j)
    energy = 0.0

    # TODO verify that the code below is correct

    # C-cap residue has to be Gly
    if helix[-1] != "G":
        print("no G cap for schellman")
        return energy

    # there has to be a C' residue after the helix
    if i + j >= len(pept):
        print("no C prime for schellman")
        return energy

    # get the amino acids governing the Schellman motif and extract the energy
    print("detected schellman case")
    C3_AA = helix[3]
    C_prime_AA = pept[i + j]
    energy = table_3_lacroix.loc[C3_AA, C_prime_AA] / 100

    return energy


def get_dG_Hbond(pept: str, i: int, j: int) -> float:
    """
    Get the free energy contribution for hydrogen bonding for a sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.

    Returns:
        float: The total free energy contribution for hydrogen bonding in the sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # The first 4 helical amino acids are considered to have zero net enthalpy
    # since they are nucleating residues and caps don't count,
    # for a total of 6.
    energy = -0.895 * max(
        (j - 6), 0
    )  # value from discussion section of the 1998 lacroix paper

    return energy


def get_dG_Hbond(
    pept: str,
    i: int,
    j: int,
    has_acetyl: bool = False,
    has_succinyl: bool = False,
    has_amide: bool = False,
) -> float:
    """
    Get the free energy contribution for hydrogen bonding for a sequence.

    Always subtract 4 residues for nucleation.
    Add 1 additional non-contributing residue for each cap, as determined by get_capping_status().

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        has_acetyl (bool): Whether N-terminal is acetylated
        has_succinyl (bool): Whether N-terminal is succinylated
        has_amide (bool): Whether C-terminal is amidated

    Returns:
        float: The total free energy contribution for hydrogen bonding in the sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j)

    # Start with nucleating residues
    non_contributing = 4

    # Add caps according to get_capping_status
    has_ncap, has_ccap = get_capping_status(
        pept, i, j, has_acetyl, has_succinyl, has_amide
    )

    if has_ncap:
        non_contributing += 1
    if has_ccap:
        non_contributing += 1

    # Calculate H-bond energy for remaining residues
    energy = -0.895 * max((j - non_contributing), 0)

    return energy


def get_dG_i3(pept: str, i: int, j: int, pH: float, T: float) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        pH (float): The pH value.
        T (float): The temperature in Kelvin.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    helix = get_helix(pept, i, j)

    energy = np.zeros(len(helix))

    # Get interaction free energies between non-charged residues
    for idx in range(len(helix) - 3):
        AAi = helix[idx]
        AAi3 = helix[idx + 3]
        base_energy = table_4a_lacroix.loc[AAi, AAi3] / 100

        # Scale energy based on ionization state
        if AAi in ["K", "R", "H", "D", "E"] and AAi3 in ["K", "R", "H", "D", "E"]:
            q_i = (
                basic_residue_ionization(pH, pka_values.loc[AAi, "pKa"], 0, T)
                if AAi in ["K", "R", "H"]
                else acidic_residue_ionization(pH, pka_values.loc[AAi, "pKa"], 0, T)
            )
            q_i3 = (
                basic_residue_ionization(pH, pka_values.loc[AAi3, "pKa"], 0, T)
                if AAi3 in ["K", "R", "H"]
                else acidic_residue_ionization(pH, pka_values.loc[AAi3, "pKa"], 0, T)
            )
            energy[idx] = base_energy * abs(q_i * q_i3)
        else:
            energy[idx] = base_energy

    return energy


def get_dG_i4(pept: str, i: int, j: int, pH: float, T: float) -> np.ndarray:
    """
    Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        pH (float): The pH of the solution.
        T (float): Temperature in Kelvin.

    Returns:
        np.ndarray: The free energy contributions for each interaction.
    """
    helix = get_helix(pept, i, j)

    energy = np.zeros(len(helix))

    # Get interaction free energies between non-charged residues
    for idx in range(len(helix) - 4):
        AAi = helix[idx]
        AAi4 = helix[idx + 4]
        base_energy = table_4b_lacroix.loc[AAi, AAi4] / 100

        # Scale energy based on ionization state
        if AAi in ["K", "R", "H", "D", "E"] and AAi4 in ["K", "R", "H", "D", "E"]:
            q_i = (
                basic_residue_ionization(pH, pka_values.loc[AAi, "pKa"], 0, T)
                if AAi in ["K", "R", "H"]
                else acidic_residue_ionization(pH, pka_values.loc[AAi, "pKa"], 0, T)
            )
            q_i4 = (
                basic_residue_ionization(pH, pka_values.loc[AAi4, "pKa"], 0, T)
                if AAi4 in ["K", "R", "H"]
                else acidic_residue_ionization(pH, pka_values.loc[AAi4, "pKa"], 0, T)
            )
            energy[idx] = base_energy * abs(q_i * q_i4)
        else:
            energy[idx] = base_energy

    return energy


def acidic_residue_ionization(pH: float, pKa: float, deltaG: float, T: float) -> float:
    """Degree of ionization indicates the fraction of molecules that
    are protonated (neutral) vs. deprotonated (negatively charged).
    Uses the Henderson-Hasselbalch equation to calculate the degree of ionization.

    Args:
        pH (float): The pH of the solution.
        pKa (float): The pKa value of the basic residue.
        deltaG (float): The free energy of Helix or coil
        T (float): The temperature in Kelvin.

    Returns:
        float: The degree of ionization.
    """
    R = 1.987e-3  # kcal/(mol K)
    pKa = pKa + deltaG / (2.3 * R * T)  # Eq. 8-9 Lacroix 1998
    q_acid = -1 / (1 + 10 ** (pKa - pH))
    return q_acid


def basic_residue_ionization(pH: float, pKa: float, deltaG: float, T: float) -> float:
    """Degree of ionization indicates the fraction of molecules that
    are protonated (positively charged) vs. deprotonated (neutral).
    Uses the Henderson-Hasselbalch equation to calculate the degree of ionization.

    Args:
        pH (float): The pH of the solution.
        pKa (float): The pKa value of the basic residue.
        deltaG (float): The free energy of Helix or coil
        T (float): The temperature in Kelvin.

    Returns:
        float: The degree of ionization.
    """
    R = 1.987e-3  # kcal/(mol K)
    pKa = pKa - deltaG / (2.3 * R * T)  # Eq. 8-9 Lacroix 1998
    q_base = 1 / (1 + 10 ** (pH - pKa))
    return q_base


def calculate_r(N: int) -> float:
    """Function to calculate the distance r from the terminal to the helix
    where N is the number of residues between the terminal and the helix.
    p. 177 of Lacroix, 1998. Distances as 2.1, 4.1, 6.1...

    Args:
        N (int): The number of residues between the terminal and the helix.

    Returns:
        float: The calculated distance r in Ångströms.
    """
    r = 0.1 + (N + 1) * 2
    return r


def calculate_permittivity(T: int) -> float:
    """Calculate the relative permittivity of water at a given temperature.
    From J. Am. Chem. Soc. 1950, 72, 7, 2844-2847
    https://doi.org/10.1021/ja01163a006

    Args:
        T (int): Temperature in Kelvin.

    Returns:
        float: The relative permittivity of water.
    """
    epsilon_r = (
        5321 / T + 233.76 - 0.9297 * T + 0.1417 * 1e-2 * T**2 - 0.8292 * 1e-6 * T**3
    )
    return epsilon_r


def debye_huckel_screening_parameter(ionic_strength: float, T: int) -> float:
    """Calculate the Debye-Huckel screening parameter K.
    Equation 7 from Lacroix, 1998.

    Args:
        ionic_strength (float): Ionic strength of the solution in mol/L.
        T (int): Temperature in Kelvin.

    Returns:
        float: The Debye-Huckel screening parameter K.
    """
    # TODO: Is this function actually needed anywhere???

    N_A = 6.022e23  # Avogadro's number in mol^-1
    e = 1.602e-19  # Elementary charge in Coulombs
    k_B = 1.38e-23  # Boltzmann constant in J/K

    # Convert ionic strength to mol/m**3
    ionic_strength = ionic_strength * 1000

    # Calculate Debye screening parameter kappa
    kappa = math.sqrt((8 * math.pi * N_A * e**2 * ionic_strength) / (1000 * k_B * T))

    return kappa


def debue_screening_length(ionic_strength: float, T: int) -> float:
    """Calculate the Debye screening length in electrolyte.
    ISBN 978-0-444-63908-0

    Args:
        ionic_strength (float): Ionic strength of the solution in mol/L.
        T (int): Temperature in Kelvin.

    Returns:
        float: The Debye length kappa.
    """
    # Constants
    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)

    # Temperature dependent relative permittivity of water
    epsilon_r = calculate_permittivity(T)

    N_A = 6.022e23  # Avogadro's number in mol^-1
    e = 1.602e-19  # Elementary charge in Coulombs
    k_B = 1.38e-23  # Boltzmann constant in J/K

    # Convert ionic strength to mol/m**3
    ionic_strength = ionic_strength * 1000

    # Calculate Debye length
    kappa = math.sqrt(
        (2 * N_A * e**2 * ionic_strength) / (epsilon_0 * epsilon_r * k_B * T)
    )

    return kappa


def calculate_term_dipole_interaction_energy(
    mu_helix: float, distance_r: float, screening_factor: float, T: float
) -> float:
    """Calculate the interaction energy between charged termini and the helix dipole.
    Uses equation 10.62 from DOI 10.1007/978-1-4419-6351-2_10 as a base.

    Args:
        mu_helix (float): Helix dipole moment.
        distance_r (float): Distance from the terminal to the helix in Ångströms.
        screening_factor (float): Debye-Huckel screening factor.
        T (float): Temperature in Kelvin.

    Returns:
        float: The interaction energy.
    """
    B_kappa = 332.0  # in kcal Å / (mol e^2)
    epsilon_r = calculate_permittivity(T)  # Relative permittivity of water

    # In the form of equation (10.62) from DOI 10.1007/978-1-4419-6351-2_10
    coloumb_potential = mu_helix / (epsilon_r * distance_r)
    energy = B_kappa * screening_factor * coloumb_potential

    return energy


def get_dG_terminals_macrodipole(
    pept: str,
    i: int,
    j: int,
    ionic_strength: float,
    pH: float,
    T: int,
    has_acetyl: bool,
    has_succinyl: bool,
    has_amide: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the interaction energies for N- and C-terminal capping residues with the helix macrodipole.

    Args:
        pept (str): Peptide sequence.
        i (int): Starting index of the helix segment.
        j (int): Length of the helix segment.
        ionic_strength (float): Ionic strength of the solution.
        pH (float): pH of the solution.
        T (int): Temperature in Kelvin.
        has_acetyl (bool): Whether the N-terminal has an acetyl group.
        has_succinyl (bool): Whether the N-terminal has a succinyl group.
        has_amide (bool): Whether the C-terminal has an amide group.

    Returns:
        tuple[np.ndarray, np.ndarray]: Interaction energies for N and C terminals.
    """
    mu_helix = 0.5
    helix = get_helix(pept, i, j)
    N_term = np.zeros(len(helix))
    C_term = np.zeros(len(helix))

    # N terminal
    residue = helix[0]
    if (
        not has_acetyl and not has_succinyl
    ):  # guard against N-termenal capping residues (no charge, no interaction)
        # TODO: Add pKa values for each aa
        qKaN = pka_values.loc["Nterm", "pKa"]
        distance_r_angstrom = calculate_r(i)  # Distance to N terminal
        distance_r_meter = (
            distance_r_angstrom * 1e-10
        )  # Convert distance from Ångströms to meters
        kappa = debue_screening_length(ionic_strength, T)
        screening_factor = math.exp(
            -kappa * distance_r_meter
        )  # Second half of equation 6 from Lacroix, 1998.
        N_term_energy = calculate_term_dipole_interaction_energy(
            mu_helix, distance_r_angstrom, screening_factor, T
        )
        q = basic_residue_ionization(pH, qKaN, N_term_energy, T)
        N_term_energy *= q
        N_term[0] = N_term_energy

    # C terminal
    residue = helix[-1]
    if (
        not has_amide
    ):  # guard against C-terminal capping residues (no charge, no interaction)
        # TODO: Add pKa values for each aa
        qKaC = pka_values.loc["Cterm", "pKa"]
        distance_r_angstrom = calculate_r(len(pept) - (i + j))  # Distance to C terminal
        distance_r_meter = (
            distance_r_angstrom * 1e-10
        )  # Convert distance from Ångströms to meters
        kappa = debue_screening_length(ionic_strength, T)
        screening_factor = math.exp(
            -kappa * distance_r_meter
        )  # Second half of equation 6 from Lacroix, 1998.
        C_term_energy = calculate_term_dipole_interaction_energy(
            mu_helix, distance_r_angstrom, screening_factor, T
        )
        q = acidic_residue_ionization(pH, qKaC, C_term_energy, T)
        C_term_energy *= -q
        C_term[-1] = C_term_energy

    return N_term, C_term


def electrostatic_interaction_energy(
    qi: float, qp: float, r: float, ionic_strength: float, T: float
) -> float:
    """Calculate the interaction energy between two charges by
    applying equation 6 from Lacroix, 1998.

    Args:
        qi (float): Charge of the first residue.
        qp (float): Charge of the second residue.
        r (float): Distance between the residues in Ångströms.
        ionic_strength (float): Ionic strength of the solution in mol/L.
        T (float): Temperature in Kelvin.

    Returns:
        float: The interaction energy in kcal/mol.
    """
    # Constants
    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)
    epsilon_r = calculate_permittivity(
        T
    )  # Relative permittivity (dielectric constant) of water at 273 K
    N_A = 6.022e23  # Avogadro's number in mol^-1
    e = 1.602e-19  # Elementary charge in Coulombs

    r = r * 1e-10  # Convert distance from Ångströms to meters
    coulomb_term = (e**2 * qi * qp) / (4 * math.pi * epsilon_0 * epsilon_r * r)
    kappa = debue_screening_length(ionic_strength, T)
    energy_joules = coulomb_term * math.exp(-kappa * r)
    energy_kcal_mol = N_A * energy_joules / 4184
    return energy_kcal_mol


def find_charged_pairs(seq: str) -> list[tuple[str, int]]:
    """Find all pairs of charged residues in a sequence and their distance.

    Args:
        seq (str): The peptide sequence.

    Returns:
        list[tuple[str, int]]: List of pairs of charged residues and their distance.
    """
    charged_amino_acids = {"K", "R", "H", "D", "E"}
    positions = [(i + 1, aa) for i, aa in enumerate(seq) if aa in charged_amino_acids]
    result = []
    for i in range(len(positions)):
        pos_i, aa_i = positions[i]
        for j in range(i + 1, len(positions)):
            pos_j, aa_j = positions[j]
            pair = aa_i + aa_j
            distance = pos_j - pos_i
            result.append((pair, distance))
    return result


# def get_dG_electrost(
#     pept: str, i: int, j: int, ionic_strength: float, pH: float, T: float
# ) -> float:
#     """From Lecroix et al. 1998:
#     'This new term includes all electrostatic interactions between two charged residues
#     inside and outside the helical segment'
#     Use equations (5) - (11) from the 1998 lacroix paper.

#     Args:
#         pept (str): The peptide sequence.
#         i (int): The helix start index, python 0-indexed.
#         j (int): The helix length.
#         ionic_strength (float): Ionic strength of the solution.
#         pH (float): pH of the solution.
#         T (float): Temperature in Kelvin.
#     """
#     # TODO: Should we add the N- and C-term backbone charges here as well? Or better separated into a different function?

#     helix = get_helix(pept, i, j)
#     charged_pairs = find_charged_pairs(helix)
#     energy_sum = 0.0
#     for p in charged_pairs:
#         pair, distance = p[0], f"i+{p[1]}"
#         helix_dist = table_6_helix_lacroix.loc[pair, distance]
#         coil_dist = table_6_coil_lacroix.loc[pair, distance]

#         # Lacroix Eq (6). First assume qp = qi = 1
#         qi, qp = 1, 1
#         G_hel = electrostatic_interaction_energy(qi, qp, helix_dist, ionic_strength, T)
#         G_rc = electrostatic_interaction_energy(qi, qp, coil_dist, ionic_strength, T)

#         # Lacroix Eq (8)
#         res1, res2 = pair
#         pKa_ref_1 = pka_values.loc[res1, "pKa"]
#         pKa_ref_2 = pka_values.loc[res2, "pKa"]
#         pKa_rc_1, pKa_rc_2 = pKa_ref_1 + G_rc / (
#             2.3 * 1.987e-3 * T
#         ), pKa_ref_2 + G_rc / (2.3 * 1.987e-3 * T)

#         # Lacroix Eq (9)
#         pKa_hel_1, pKa_hel_2 = pKa_ref_1 + G_hel / (
#             2.3 * 1.987e-3 * T
#         ), pKa_ref_2 + G_hel / (2.3 * 1.987e-3 * T)

#         # Lacroix Eq (10)
#         if res1 in ["D", "E"]:
#             q1_hel = acidic_residue_ionization(pH, pKa_hel_1, G_hel, T)
#             q1_rc = acidic_residue_ionization(pH, pKa_rc_1, G_rc, T)
#         else:
#             q1_hel = basic_residue_ionization(pH, pKa_hel_1, G_hel, T)
#             q1_rc = basic_residue_ionization(pH, pKa_rc_1, G_rc, T)

#         if res2 in ["D", "E"]:
#             q2_hel = acidic_residue_ionization(pH, pKa_hel_2, G_hel, T)
#             q2_rc = acidic_residue_ionization(pH, pKa_rc_2, G_rc, T)
#         else:
#             q2_hel = basic_residue_ionization(pH, pKa_hel_2, G_hel, T)
#             q2_rc = basic_residue_ionization(pH, pKa_rc_2, G_rc, T)

#         # Lacroix Eq (6) again, with the updated values
#         G_hel = electrostatic_interaction_energy(
#             q1_hel, q2_hel, helix_dist, ionic_strength, T
#         )
#         G_rc = electrostatic_interaction_energy(
#             q1_rc, q2_rc, coil_dist, ionic_strength, T
#         )
#         energy_sum += G_hel - G_rc

#     return energy_sum

def get_dG_electrost(
    pept: str, i: int, j: int, ionic_strength: float, pH: float, T: float
) -> float:
    """From Lacroix et al. 1998:
    Calculate all electrostatic interactions between charged residues with
    iterative pKa convergence.
    """
    helix = get_helix(pept, i, j)
    charged_pairs = find_charged_pairs(helix)
    energy_sum = 0.0
    
    for p in charged_pairs:
        pair, distance = p[0], f"i+{p[1]}"
        helix_dist = table_6_helix_lacroix.loc[pair, distance]
        coil_dist = table_6_coil_lacroix.loc[pair, distance]
        
        # Get residue info
        res1, res2 = pair
        pKa_ref_1 = pka_values.loc[res1, "pKa"]
        pKa_ref_2 = pka_values.loc[res2, "pKa"]

        # Initial calculation with qi = qp = 1
        G_hel = electrostatic_interaction_energy(1, 1, helix_dist, ionic_strength, T)
        G_rc = electrostatic_interaction_energy(1, 1, coil_dist, ionic_strength, T)
        
        # Iterative convergence
        max_iter = 10  # Maximum iterations 
        tolerance = 0.01  # Convergence tolerance for pKa change
        
        old_pKa_hel_1, old_pKa_hel_2 = 0, 0
        pKa_hel_1 = pKa_ref_1 + G_hel / (2.3 * 1.987e-3 * T)
        pKa_hel_2 = pKa_ref_2 + G_hel / (2.3 * 1.987e-3 * T)
        
        for _ in range(max_iter):
            # Store old values
            old_pKa_hel_1, old_pKa_hel_2 = pKa_hel_1, pKa_hel_2
            
            # Calculate ionization
            if res1 in ["D", "E"]:
                q1_hel = acidic_residue_ionization(pH, pKa_hel_1, G_hel, T)
                q1_rc = acidic_residue_ionization(pH, pKa_ref_1, G_rc, T)
            else:
                q1_hel = basic_residue_ionization(pH, pKa_hel_1, G_hel, T)
                q1_rc = basic_residue_ionization(pH, pKa_ref_1, G_rc, T)
                
            if res2 in ["D", "E"]:
                q2_hel = acidic_residue_ionization(pH, pKa_hel_2, G_hel, T)
                q2_rc = acidic_residue_ionization(pH, pKa_ref_2, G_rc, T)
            else:
                q2_hel = basic_residue_ionization(pH, pKa_hel_2, G_hel, T)
                q2_rc = basic_residue_ionization(pH, pKa_ref_2, G_rc, T)

            # Recalculate energies with new charges
            G_hel = electrostatic_interaction_energy(
                q1_hel, q2_hel, helix_dist, ionic_strength, T
            )
            G_rc = electrostatic_interaction_energy(
                q1_rc, q2_rc, coil_dist, ionic_strength, T
            )
            
            # Update pKas
            pKa_hel_1 = pKa_ref_1 + G_hel / (2.3 * 1.987e-3 * T)
            pKa_hel_2 = pKa_ref_2 + G_hel / (2.3 * 1.987e-3 * T)
            
            # Check convergence
            if (abs(pKa_hel_1 - old_pKa_hel_1) < tolerance and 
                abs(pKa_hel_2 - old_pKa_hel_2) < tolerance):
                break
                
        energy_sum += G_hel - G_rc
        
    return energy_sum


def get_dG_sidechain_macrodipole(
    pept: str, i: int, j: int, ionic_strength: float, pH: float, T: float
) -> np.ndarray:
    """
    Calculate the interaction energy between charged side-chains and the helix macrodipole.
    The energy should be unaffected by N- and C-terminal modifications.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        ionic_strength (float): Ionic strength of the solution in mol/L.
        pH (float): The pH of the solution.
        T (float): Temperature in Kelvin.

    Returns:
        np.ndarray: The free energy contribution for each residue in the helix.
    """
    helix = get_helix(pept, i, j)
    energy = np.zeros(len(helix))
    q_dipole = 0.5  # Half-charge for the helix macrodipole

    for idx, aa in enumerate(helix):
        # The distance tables only contain values up to C13 from the C-terminus
        # and N13 from the N-terminus, so we need to limit the calculations
        # to avoid indexing errors
        if len(helix) - idx - 1 > 13 or idx > 13:
            continue

        if aa in ["K", "R", "H"]:  # Positively charged residues
            # N-terminal interaction
            position = "Ncap" if idx == 0 else f"N{idx}"
            distance = table_7_ncap_lacroix.loc[aa, position]
            energy[idx] += electrostatic_interaction_energy(
                q_dipole, 1, distance, ionic_strength, T
            )

            # C-terminal interaction
            position = "Ccap" if idx == len(helix) - 1 else f"C{len(helix)-idx-1}"
            distance = table_7_ccap_lacroix.loc[aa, position]
            energy[idx] -= electrostatic_interaction_energy(
                q_dipole, 1, distance, ionic_strength, T
            )

        elif aa in ["D", "E", "C", "Y"]:  # Negatively charged residues
            # N-terminal interaction
            position = "Ncap" if idx == 0 else f"N{idx}"
            distance = table_7_ncap_lacroix.loc[aa, position]
            energy[idx] -= electrostatic_interaction_energy(
                q_dipole, 1, distance, ionic_strength, T
            )

            # C-terminal interaction
            position = "Ccap" if idx == len(helix) - 1 else f"C{len(helix)-idx-1}"
            distance = table_7_ccap_lacroix.loc[aa, position]
            energy[idx] += electrostatic_interaction_energy(
                q_dipole, 1, distance, ionic_strength, T
            )

    return energy
