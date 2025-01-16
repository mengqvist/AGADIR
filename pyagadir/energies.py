import math
from importlib.resources import files
from itertools import combinations

import numpy as np
import pandas as pd
from typing import Dict

from pyagadir.utils import is_valid_index, is_valid_peptide_sequence

# get params
datapath = files("pyagadir.data.params")

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


def get_helix(pept: str, i: int, j: int, min_helix_length: int, has_acetyl: bool, has_succinyl: bool, has_amide: bool) -> str:
    """
    Get the helix region of a peptide sequence, including the N- and C-caps.

    Args:
        pept (str): The peptide sequence.
        i (int): The helix start index, python 0-indexed.
        j (int): The helix length.
        min_helix_length (int): The minimum helix length.
        has_acetyl (bool): Whether the peptide has an N-terminal acetyl modification.
        has_succinyl (bool): Whether the peptide has a C-terminal succinyl modification.
        has_amide (bool): Whether the peptide has a C-terminal amide modification.

    Returns:
        str: The helix region of the peptide sequence.
    """
    is_valid_peptide_sequence(pept)
    is_valid_index(pept, i, j, min_helix_length, has_acetyl, has_succinyl, has_amide)

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


def adjust_pKa(T: float, pKa_ref: float, deltaG: float) -> float:
    """
    Adjust the pKa of a residue based on its free energy difference.
    Eq. 8-9 in Lacroix 1998

    Args:
        T (float): Temperature in Kelvin.
        pKa_ref (float): Reference pKa value.
        deltaG (float): Free energy difference (kcal/mol).

    Returns:
        float: Adjusted pKa value.
    """
    R = 1.987e-3  # kcal/(mol K)
    return pKa_ref + deltaG / (2.3 * R * T)


def acidic_residue_ionization(pH: float, pKa: float) -> float:
    """
    Calculate the degree of ionization for acidic residues.
    Uses the Henderson-Hasselbalch equation.

    Args:
        pH (float): The pH of the solution.
        pKa (float): Adjusted pKa value.

    Returns:
        float: Ionization state (-1 for fully deprotonated, 0 for neutral).
    """
    return -1 / (1 + 10 ** (pKa - pH))


def basic_residue_ionization(pH: float, pKa: float) -> float:
    """
    Calculate the degree of ionization for basic residues.
    Uses the Henderson-Hasselbalch equation.

    Args:
        pH (float): The pH of the solution.
        pKa (float): Adjusted pKa value.

    Returns:
        float: Ionization state (1 for fully protonated, 0 for neutral).
    """
    return 1 / (1 + 10 ** (pH - pKa))


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


class EnergyCalculator:
    """
    Class to calculate the free energy contributions for a peptide sequence.
    """
    def __init__(self, pept: str, pH: float, T: float, ionic_strength: float, min_helix_length: int, has_acetyl=False, has_succinyl=False, has_amide=False):
        """
        Initialize the EnergyCalculator for a peptide sequence.

        Args:
            pept (str): Peptide sequence.
            pH (float): Solution pH.
            T (float): Temperature in Kelvin.
            ionic_strength (float): Ionic strength of the solution in mol/L.
            min_helix_length (int): The minimum helix length.
            has_acetyl (bool): N-terminal acetylation flag.
            has_succinyl (bool): N-terminal succinylation flag.
            has_amide (bool): C-terminal amidation flag.
        """
        self.pept = pept
        self.pH = pH
        self.T = T
        self.ionic_strength = ionic_strength
        self.min_helix_length = min_helix_length
        self.has_acetyl = has_acetyl
        self.has_succinyl = has_succinyl
        self.has_amide = has_amide

        # Precompute ionization states for helical and random-coil states
        self.q_global_hel, self.q_global_rc = self._precompute_ionization_states()
        print("****************")
        print(pH)
        print(self.pept)
        print(self.q_global_hel)
        print(self.q_global_rc)
        print("****************")

    def _precompute_ionization_states(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute ionization states for all residues in the peptide
        for both helical and random-coil states.

        Returns:
            tuple[np.ndarray, np.ndarray]: Ionization states for helical and random-coil states.
        """
        q_hel = np.zeros(len(self.pept))
        q_rc = np.zeros(len(self.pept))

        for idx, AA in enumerate(self.pept):

            if AA in ["C", "D", "E", "Y"]:
                pKa_ref = pka_values.loc[AA, "pKa"]
                pKa_rc = adjust_pKa(self.T, pKa_ref, deltaG=0.0)  # Coil state
                pKa_hel = adjust_pKa(self.T, pKa_ref, deltaG=-0.5)  # Helix state (example deltaG)
                q_rc[idx] = acidic_residue_ionization(self.pH, pKa_rc)
                q_hel[idx] = acidic_residue_ionization(self.pH, pKa_hel)

            elif AA in ["K", "R", "H"]:
                pKa_ref = pka_values.loc[AA, "pKa"]
                pKa_rc = adjust_pKa(self.T, pKa_ref, deltaG=0.0)  # Coil state
                pKa_hel = adjust_pKa(self.T, pKa_ref, deltaG=-0.5)  # Helix state (example deltaG)
                q_rc[idx] = basic_residue_ionization(self.pH, pKa_rc)
                q_hel[idx] = basic_residue_ionization(self.pH, pKa_hel)

        return q_hel, q_rc

    def _find_charged_pairs(self, seq: str, start_idx: int = 0) -> list[tuple[str, int, int]]:
        """
        Find all pairs of charged residues in a sequence and their global positions.

        Args:
            seq (str): The peptide sequence.
            start_idx (int): The global start index of the sequence in the full peptide.

        Returns:
            list[tuple[str, int, int]]: List of tuples containing:
                - Pair of charged residues as a string (e.g., "KR").
                - Global index of the first residue.
                - Global index of the second residue.
        """
        charged_amino_acids = {"K", "R", "H", "D", "E", "Y"}
        positions = [(start_idx + i, aa) for i, aa in enumerate(seq) if aa in charged_amino_acids]
        result = []

        # Iterate over all pairs of charged residues
        for i in range(len(positions)):
            pos_i, aa_i = positions[i]
            for j in range(i + 1, len(positions)):
                pos_j, aa_j = positions[j]
                pair = aa_i + aa_j
                result.append((pair, pos_i, pos_j))  # Include global positions

        return result
    
    def _electrostatic_interaction_energy(self, qi: float, qp: float, r: float) -> float:
        """Calculate the interaction energy between two charges by
        applying equation 6 from Lacroix, 1998.

        Args:
            qi (float): Charge of the first residue.
            qp (float): Charge of the second residue.
            r (float): Distance between the residues in Ångströms.

        Returns:
            float: The interaction energy in kcal/mol.
        """
        # Constants
        epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)
        epsilon_r = calculate_permittivity(
            self.T
        )  # Relative permittivity (dielectric constant) of water at 273 K
        N_A = 6.022e23  # Avogadro's number in mol^-1
        e = 1.602e-19  # Elementary charge in Coulombs

        r = r * 1e-10  # Convert distance from Ångströms to meters
        coulomb_term = (e**2 * qi * qp) / (4 * math.pi * epsilon_0 * epsilon_r * r)
        kappa = debue_screening_length(self.ionic_strength, self.T)
        energy_joules = coulomb_term * math.exp(-kappa * r)
        energy_kcal_mol = N_A * energy_joules / 4184
        return energy_kcal_mol

    def _calculate_term_dipole_interaction_energy(
        self, mu_helix: float, distance_r: float, screening_factor: float
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
        epsilon_r = calculate_permittivity(self.T)  # Relative permittivity of water

        # In the form of equation (10.62) from DOI 10.1007/978-1-4419-6351-2_10
        coloumb_potential = mu_helix / (epsilon_r * distance_r)
        energy = B_kappa * screening_factor * coloumb_potential

        return energy

    def get_dG_Int(self, i: int, j: int) -> np.ndarray:
        """
        Get the intrinsic free energy contributions for a sequence.
        The first and last residues are considered to be caps unless they are
        terminal residues with modifications.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The intrinsic free energy contributions for each amino acid in the sequence.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        has_ncap, has_ccap = get_capping_status(
            self.pept, i, j, self.has_acetyl, self.has_succinyl, self.has_amide
        )

        # Initialize energy array
        energy = np.zeros(len(helix))

        # Iterate over the helix and get the intrinsic energy for each residue
        for idx, AA in enumerate(helix):
            # Skip caps only if they exist for this segment
            if (idx == 0 and has_ncap) or (idx == len(helix) - 1 and has_ccap):
                continue

            # Handle N-terminal region specially
            if idx == 1 or (idx == 0 and not has_ncap):
                energy[idx] = table_1_lacroix.loc[AA, "N1"]
            elif idx == 2 or (idx == 1 and not has_ncap):
                energy[idx] = table_1_lacroix.loc[AA, "N2"]
            elif idx == 3 or (idx == 2 and not has_ncap):
                energy[idx] = table_1_lacroix.loc[AA, "N3"]
            elif idx == 4 or (idx == 3 and not has_ncap):
                energy[idx] = table_1_lacroix.loc[AA, "N4"]
            else:
                # For central residues
                if AA in ["C", "D", "E", "H", "K", "R", "Y"]:
                    # Charged residues: use precomputed ionization state
                    q = self.q_global_hel[i + idx]
                    basic_energy_ncen = table_1_lacroix.loc[AA, "Ncen"]
                    basic_energy_neutral = table_1_lacroix.loc[AA, "Neutral"]
                    energy[idx] = q * basic_energy_ncen + (1 - q) * basic_energy_neutral
                else:
                    # Uncharged residues: directly use Ncen
                    energy[idx] = table_1_lacroix.loc[AA, "Ncen"]

        return energy

    def get_dG_Ncap(
        self,
        i: int,
        j: int,
    ) -> np.ndarray:
        """
        Get the free energy contribution for N-terminal capping.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The free energy contribution.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        has_ncap, has_ccap = get_capping_status(
            self.pept, i, j, self.has_acetyl, self.has_succinyl, self.has_amide
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
        self,
        i: int,
        j: int,
    ) -> np.ndarray:
        """
        Get the free energy contribution for C-terminal capping.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The free energy contribution.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        has_ncap, has_ccap = get_capping_status(
            self.pept, i, j, self.has_acetyl, self.has_succinyl, self.has_amide
        )

        # fix the blocking group names to match the table
        Ccap_AA = helix[-1]
        if not has_ccap:
            Ccap_AA = "Am"

        energy = np.zeros(len(helix))

        # Cc-2 	C-cap values when there is a Pro residue at position C'
        c_prime_idx = i + j
        if (len(self.pept) > c_prime_idx) and (self.pept[c_prime_idx] == "P"):
            energy[-1] = table_1_lacroix.loc[Ccap_AA, "Cc-2"]

        # Cc-1 	Normal C-cap values
        else:
            energy[-1] = table_1_lacroix.loc[Ccap_AA, "Cc-1"]

        return energy

    def get_dG_staple(self, i: int, j: int) -> float:
        """
        Get the free energy contribution for the hydrophobic staple motif.
        The hydrophobic interaction is between the N' and N4 residues of the helix.
        See https://doi.org/10.1038/nsb0595-380 for more details.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            float: The free energy contribution.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)

        energy = np.zeros(len(helix))

        # get the amino acids governing the staple motif
        N_prime_AA = self.pept[i - 1]
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
                # print("staple case i")
                energy *= 1.0

            # whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr
            elif Ncap_AA in ["N", "D"] and N3_AA in ["S", "T"]:
                # print("staple case ii")
                energy *= 1.0

            # other cases they are multiplied by 0.5
            else:
                # print("staple case iii")
                energy *= 0.5

        else:
            pass
            # print("no staple motif")

        return energy

    def get_dG_schellman(self, i: int, j: int) -> float:
        """
        Get the free energy contribution for the Schellman motif.
        The Schellman motif is only considered whenever Gly is the C-cap residue,
        where the interaction happens between the C' and C3 residues of the helix.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            float: The free energy contribution.
        """
        # TODO: is this affected by acylation, succinylation, or amidation? Find out!
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        energy = 0.0

        # TODO verify that the code below is correct

        # C-cap residue has to be Gly
        if helix[-1] != "G":
            # print("no G cap for schellman")
            return energy

        # there has to be a C' residue after the helix
        if i + j >= len(self.pept):
            # print("no C prime for schellman")
            return energy

        # get the amino acids governing the Schellman motif and extract the energy
        # print("detected schellman case")
        C3_AA = helix[3]
        C_prime_AA = self.pept[i + j]
        energy = table_3_lacroix.loc[C3_AA, C_prime_AA] / 100

        return energy

    def get_dG_Hbond(self, i: int, j: int) -> float:
        """
        Get the free energy contribution for hydrogen bonding for a sequence.

        Always subtract 4 residues for nucleation.
        Add 1 additional non-contributing residue for each cap, as determined by get_capping_status().

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            float: The total free energy contribution for hydrogen bonding in the sequence.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)

        # Start with nucleating residues
        non_contributing = 4

        # Add caps according to get_capping_status
        has_ncap, has_ccap = get_capping_status(
            helix, i, j, self.has_acetyl, self.has_succinyl, self.has_amide
        )

        if has_ncap:
            non_contributing += 1
        if has_ccap:
            non_contributing += 1

        # Calculate H-bond energy for remaining residues
        energy = -0.895 * max((j - non_contributing), 0)

        return energy

    def get_dG_i3(self, i: int, j: int) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The free energy contributions for each interaction.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        energy = np.zeros(len(helix))

        # Get interaction free energies for charged residues
        for idx in range(len(helix) - 3):
            AAi = helix[idx]
            AAi3 = helix[idx + 3]
            base_energy = table_4a_lacroix.loc[AAi, AAi3] / 100

            if AAi in ["K", "R", "H", "D", "E"] and AAi3 in ["K", "R", "H", "D", "E"]:
                # Use precomputed ionization states for the helical state
                q_i = self.q_global_hel[i + idx]
                q_i3 = self.q_global_hel[i + idx + 3]
                energy[idx] = base_energy * abs(q_i * q_i3) # TODO: Does this calculation make sense?
            else:
                energy[idx] = base_energy

        return energy

    def get_dG_i4(self, i: int, j: int) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The free energy contributions for each interaction.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        energy = np.zeros(len(helix))

        # Get interaction free energies for charged residues
        for idx in range(len(helix) - 4):
            AAi = helix[idx]
            AAi4 = helix[idx + 4]
            base_energy = table_4b_lacroix.loc[AAi, AAi4] / 100

            if AAi in ["K", "R", "H", "D", "E"] and AAi4 in ["K", "R", "H", "D", "E"]:
                # Use precomputed ionization states for the helical state
                q_i = self.q_global_hel[i + idx]
                q_i4 = self.q_global_hel[i + idx + 4]
                energy[idx] = base_energy * abs(q_i * q_i4) # TODO: Does this calculation make sense?
            else:
                energy[idx] = base_energy

        return energy

    def get_dG_terminals_macrodipole(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate interaction energies between N- and C-terminal residues and the helix macrodipole.

        Args:
            i (int): Starting index of the helix segment.
            j (int): Length of the helix segment.

        Returns:
            tuple[np.ndarray, np.ndarray]: Interaction energies for N-terminal and C-terminal residues.
        """
        mu_helix = 0.5  # Helix dipole moment
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        N_term = np.zeros(len(helix))
        C_term = np.zeros(len(helix))

        def _calculate_terminal_energy(
            distance_r_angstrom: float, pKa_ref: float, residue_type: str, terminal: str
        ) -> float:
            """
            Calculate interaction energy for terminal residues with the helix dipole.

            Args:
                distance_r_angstrom (float): Distance from terminal to helix.
                pKa_ref (float): Reference pKa value.
                residue_type (str): 'acidic' or 'basic'.
                terminal (str): 'N' or 'C'.

            Returns:
                float: The calculated terminal energy.
            """
            # Convert distance to meters and compute screening factor
            distance_r_meter = distance_r_angstrom * 1e-10
            kappa = debue_screening_length(self.ionic_strength, self.T)
            screening_factor = math.exp(-kappa * distance_r_meter)

            # Calculate terminal interaction energy
            energy = self._calculate_term_dipole_interaction_energy(
                mu_helix, distance_r_angstrom, screening_factor, self.T
            )

            # Adjust ionization state using precomputed pKa
            if residue_type == "basic":
                q = basic_residue_ionization(self.pH, pKa_ref)
            elif residue_type == "acidic":
                q = acidic_residue_ionization(self.pH, pKa_ref)
            else:
                raise ValueError(f"Invalid residue type: {residue_type}")

            # Adjust energy by ionization state
            if terminal == "C":
                energy *= -q  # C-terminal interaction is negative
            else:
                energy *= q  # N-terminal interaction

            return energy

        # N-terminal calculation
        if not self.has_acetyl and not self.has_succinyl:  # N-terminal not capped
            pKa_Nterm = pka_values.loc["Nterm", "pKa"]
            distance_r_N = calculate_r(i)
            N_term_energy = _calculate_terminal_energy(
                distance_r_N, pKa_Nterm, "basic", terminal="N"
            )
            N_term[0] = N_term_energy

        # C-terminal calculation
        if not self.has_amide:  # C-terminal not capped
            pKa_Cterm = pka_values.loc["Cterm", "pKa"]
            distance_r_C = calculate_r(len(self.pept) - (i + j))
            C_term_energy = _calculate_terminal_energy(
                distance_r_C, pKa_Cterm, "acidic", terminal="C"
            )
            C_term[-1] = C_term_energy

        return N_term, C_term


    # def get_dG_terminals_macrodipole(
    #     self,
    #     i: int,
    #     j: int,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     """Get the interaction energies for N- and C-terminal capping residues with the helix macrodipole.

    #     Args:
    #         i (int): Starting index of the helix segment.
    #         j (int): Length of the helix segment.

    #     Returns:
    #         tuple[np.ndarray, np.ndarray]: Interaction energies for N and C terminals.
    #     """
    #     mu_helix = 0.5
    #     helix = get_helix(self.pept, i, j)
    #     N_term = np.zeros(len(helix))
    #     C_term = np.zeros(len(helix))

    #     # N terminal
    #     residue = helix[0]
    #     if (
    #         not self.has_acetyl and not self.has_succinyl
    #     ):  # guard against N-termenal capping residues (no charge, no interaction)
    #         # TODO: Add pKa values for each aa
    #         qKaN = pka_values.loc["Nterm", "pKa"]
    #         distance_r_angstrom = calculate_r(i)  # Distance to N terminal
    #         distance_r_meter = (
    #             distance_r_angstrom * 1e-10
    #         )  # Convert distance from Ångströms to meters
    #         kappa = debue_screening_length(self.ionic_strength, self.T)
    #         screening_factor = math.exp(
    #             -kappa * distance_r_meter
    #         )  # Second half of equation 6 from Lacroix, 1998.
    #         N_term_energy = calculate_term_dipole_interaction_energy(
    #             mu_helix, distance_r_angstrom, screening_factor, self.T
    #         )
    #         q = basic_residue_ionization(self.pH, qKaN, N_term_energy, self.T)
    #         N_term_energy *= q
    #         N_term[0] = N_term_energy

    #     # C terminal
    #     residue = helix[-1]
    #     if (
    #         not self.has_amide
    #     ):  # guard against C-terminal capping residues (no charge, no interaction)
    #         # TODO: Add pKa values for each aa
    #         qKaC = pka_values.loc["Cterm", "pKa"]
    #         distance_r_angstrom = calculate_r(len(self.pept) - (i + j))  # Distance to C terminal
    #         distance_r_meter = (
    #             distance_r_angstrom * 1e-10
    #         )  # Convert distance from Ångströms to meters
    #         kappa = debue_screening_length(self.ionic_strength, self.T)
    #         screening_factor = math.exp(
    #             -kappa * distance_r_meter
    #         )  # Second half of equation 6 from Lacroix, 1998.
    #         C_term_energy = calculate_term_dipole_interaction_energy(
    #             mu_helix, distance_r_angstrom, screening_factor, self.T
    #         )
    #         q = acidic_residue_ionization(self.pH, qKaC, C_term_energy, self.T)
    #         C_term_energy *= -q
    #         C_term[-1] = C_term_energy

    #     return N_term, C_term

    def get_dG_sidechain_macrodipole(
        self, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the interaction energy between charged side-chains and the helix macrodipole.
        The helix macrodipole is positively charged at the N-terminus and negatively charged at the C-terminus.
        The energy should be unaffected by N- and C-terminal modifications except for changing which residues are part of the helix.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            tuple[np.ndarray, np.ndarray]: The free energy contribution for each residue in the helix, 
            N-terminal and C-terminal contributions.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)
        energy_N = np.zeros(len(helix))
        energy_C = np.zeros(len(helix))
        q_dipole = 0.5  # Half-charge for the helix macrodipole

        for idx, aa in enumerate(helix):
            # Skip if amino acid is not charged
            if aa not in ["K", "R", "H", "D", "E", "C", "Y"]:
                continue

            # The distance tables only contain values up to C13 from the C-terminus and N13 from the N-terminus
            # Try to fetch the distance here
            N_position = "Ncap" if idx == 0 else f"N{idx}"
            try:
                N_distance = table_7_ncap_lacroix.loc[aa, N_position]
            except (KeyError, ValueError):
                print(f'{aa} with position {N_position} not found in table_7_ncap_lacroix')
                N_distance = None

            C_position = "Ccap" if idx == len(helix) - 1 else f"C{len(helix)-idx-1}"
            try:
                C_distance = table_7_ccap_lacroix.loc[aa, C_position]
            except (KeyError, ValueError):
                print(f'{aa} with position {C_position} not found in table_7_ccap_lacroix')
                C_distance = None

            # Fetch the precomputed ionization state
            q_sidechain = self.q_global_hel[i + idx]

            # N-terminal interaction
            if N_distance is not None:
                energy_N[idx] += self._electrostatic_interaction_energy(
                    q_dipole, q_sidechain, N_distance
                )

            # C-terminal interaction
            if C_distance is not None:
                energy_C[idx] += self._electrostatic_interaction_energy(
                    -q_dipole, q_sidechain, C_distance
                )

        return energy_N, energy_C
        
    def get_dG_electrost(self, i: int, j: int) -> np.ndarray:
        """
        Calculate the electrostatic free energy contribution for charged residues inside and outside
        the helical segment, using Lacroix et al. (1998) equations.

        Args:
            i (int): The helix start index (0-indexed).
            j (int): The helix length.

        Returns:
            np.ndarray: n x n symmetric matrix of pairwise electrostatic free energy contributions,
                       with each interaction energy split between upper and lower triangles.
        """
        helix = get_helix(self.pept, i, j, self.min_helix_length, self.has_acetyl, self.has_succinyl, self.has_amide)  # Extract helical segment
        charged_pairs = self._find_charged_pairs(helix, start_idx=i)  # Identify charged pairs
        energy_matrix = np.zeros((len(helix), len(helix)))

        # Iterate over all charged residue pairs
        for pair, idx1, idx2 in charged_pairs:
            # Skip if not in upper triangle
            if idx2 - i <= idx1 - i:
                continue
                
            res1, res2 = pair  # Amino acid types (e.g., "K", "E")
            distance_key = f"i+{abs(idx2 - idx1)}"

            # Fetch distances from tables
            # TODO: There are more special cases to handle here, e.g. in the case of capping glycines, need to fix!
            # TODO: The distance only goes to 12 residues, so we need to handle the case where the distance is too long.
            try:
                helix_dist = table_6_helix_lacroix.loc[pair, distance_key]
            except (KeyError, ValueError):
                helix_dist = table_6_helix_lacroix.loc['HelixRest', distance_key]

            try:
                coil_dist = table_6_coil_lacroix.loc[pair, distance_key]
            except (KeyError, ValueError):
                coil_dist = table_6_coil_lacroix.loc['RcoilRest', distance_key]

            # Step 1: Get reference pKa values for both residues
            pKa_ref_1 = pka_values.loc[res1, "pKa"]
            pKa_ref_2 = pka_values.loc[res2, "pKa"]

            # Step 2: Assume full unit charges and calculate initial G_hel and G_rc, Lacroix Eq 6
            G_hel = self._electrostatic_interaction_energy(1, 1, helix_dist)
            G_rc = self._electrostatic_interaction_energy(1, 1, coil_dist)

            # Step 3: Adjust pKa values based on the local free energy contributions, Lacroix Eq 8 and 9
            pKa_hel_1 = adjust_pKa(self.T, pKa_ref_1, G_hel)
            pKa_hel_2 = adjust_pKa(self.T, pKa_ref_2, G_hel)
            pKa_rc_1 = adjust_pKa(self.T, pKa_ref_1, G_rc)
            pKa_rc_2 = adjust_pKa(self.T, pKa_ref_2, G_rc)

            # Step 4: Recalculate ionization states using adjusted pKa (Equations 10-11)
            q1_hel = acidic_residue_ionization(self.pH, pKa_hel_1) if res1 in ["D", "E"] else basic_residue_ionization(self.pH, pKa_hel_1)
            q2_hel = acidic_residue_ionization(self.pH, pKa_hel_2) if res2 in ["D", "E"] else basic_residue_ionization(self.pH, pKa_hel_2)
            q1_rc = acidic_residue_ionization(self.pH, pKa_rc_1) if res1 in ["D", "E"] else basic_residue_ionization(self.pH, pKa_rc_1)
            q2_rc = acidic_residue_ionization(self.pH, pKa_rc_2) if res2 in ["D", "E"] else basic_residue_ionization(self.pH, pKa_rc_2)

            # Step 5: Recalculate electrostatic interaction energies with adjusted ionization states, Lacroix Eq 6
            G_hel = self._electrostatic_interaction_energy(q1_hel, q2_hel, helix_dist)
            G_rc = self._electrostatic_interaction_energy(q1_rc, q2_rc, coil_dist)

            # Step 6: Store half the energy difference in both triangles of the matrix
            energy_diff = (G_hel - G_rc) / 2
            energy_matrix[idx1 - i, idx2 - i] = energy_diff  # Upper triangle
            energy_matrix[idx2 - i, idx1 - i] = energy_diff  # Lower triangle

        return energy_matrix
    