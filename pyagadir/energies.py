import math
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import calculate_ionic_strength, adjust_pKa, acidic_residue_ionization, basic_residue_ionization, calculate_permittivity, debye_screening_kappa
from pyagadir.utils import is_valid_index, is_valid_peptide_sequence, is_valid_ncap_ccap, is_valid_conditions
import warnings
import itertools
import copy




class PrecomputeParams:
    """
    Class to load parameters for the AGADIR model and
    for pre-computing distances, and ionization states.
    """

    _params = None # class variable to store the params

    @classmethod
    def load_params(cls):
        """
        Load the parameters for the AGADIR model.
        Only load once, and store in class variable to 
        save time and memory when many instances are created.
        """
        if cls._params is None:

            cls._params = {}

            # get params
            datapath = files("pyagadir.data.params")

            # load energy contributions for intrinsic propensities, capping, etc.
            cls._params["table_1_lacroix"] = pd.read_csv(
                datapath.joinpath("table_1_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load the hydrophobic staple motif energy contributions
            cls._params["table_2_lacroix"] = pd.read_csv(
                datapath.joinpath("table_2_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load the schellman motif energy contributions
            cls._params["table_3_lacroix"] = pd.read_csv(
                datapath.joinpath("table_3_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load energy contributions for interactions between i and i+3
            cls._params["table_4a_lacroix"] = pd.read_csv(
                datapath.joinpath("table_4a_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load energy contributions for interactions between i and i+4
            cls._params["table_4b_lacroix"] = pd.read_csv(
                datapath.joinpath("table_4b_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load sidechain distances for helices
            cls._params["table_6_helix_lacroix"] = pd.read_csv(
                datapath.joinpath("table_6_helix_lacroix.tsv"),
                index_col="Pos",
                sep="\t",
            ).astype(float)

            # load sidechain distances for coils
            cls._params["table_6_coil_lacroix"] = pd.read_csv(
                datapath.joinpath("table_6_coil_lacroix.tsv"),
                index_col="Pos",
                sep="\t",
            ).astype(float)

            # load N-terminal distances between charged amino acids and the half charge from the helix macrodipole
            cls._params["table_7_ccap_lacroix"] = pd.read_csv(
                datapath.joinpath("table_7_Ccap_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load C-terminal distances between charged amino acids and the half charge from the helix macrodipole
            cls._params["table_7_ncap_lacroix"] = pd.read_csv(
                datapath.joinpath("table_7_Ncap_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load pKa values for for side chain ionization and the N- and C-terminal capping groups
            cls._params["pka_values"] = pd.read_csv(
                datapath.joinpath("pka_values.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

        return cls._params

    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None):
        """
        Initialize the PrecomputedParams for a peptide sequence.

        Args:
            seq (str): Peptide sequence.
            i (int): Helix start index, python 0-indexed.
            j (int): Helix length.
            pH (float): Solution pH.
            T (float): Temperature in Celsius.
            ionic_strength (float): Ionic strength of the solution in mol/L.
            ncap (str): N-terminal capping modification (acetylation='Ac', succinylation='Sc').
            ccap (str): C-terminal capping modification (amidation='Am').
        """
        # load params
        params = self.load_params()
        self.table_1_lacroix = params["table_1_lacroix"]
        self.table_2_lacroix = params["table_2_lacroix"]
        self.table_3_lacroix = params["table_3_lacroix"]
        self.table_4a_lacroix = params["table_4a_lacroix"]
        self.table_4b_lacroix = params["table_4b_lacroix"]
        self.table_6_helix_lacroix = params["table_6_helix_lacroix"]
        self.table_6_coil_lacroix = params["table_6_coil_lacroix"]
        self.table_7_ccap_lacroix = params["table_7_ccap_lacroix"]
        self.table_7_ncap_lacroix = params["table_7_ncap_lacroix"]
        self.table_pka_values = params["pka_values"]

        is_valid_peptide_sequence(seq)
        is_valid_ncap_ccap(ncap, ccap)
        # is_valid_index(seq, i, j, ncap, ccap)
        is_valid_conditions(pH, T, ionic_strength)

        self.seq = seq
        self.seq_list = list(seq)
        if ncap is not None:
            self.seq_list.insert(0, ncap)
        if ccap is not None:
            self.seq_list.append(ccap)
        self.helix = self.seq_list[i:i+j]
        self.i = i
        self.j = j
        self.pH = pH
        self.T_celsius = T
        self.T_kelvin = T + 273.15
        self.ionic_strength = ionic_strength
        self.ncap = ncap
        self.ccap = ccap

        # pre-compute indices for the helix and get some key residues
        self.helix_indices = list(range(i, i+j))

        self.ncap_idx = self.helix_indices[0]
        self.Ncap_AA = self.seq_list[self.ncap_idx]
        self.N1_AA = self.seq_list[self.ncap_idx + 1]
        self.N3_AA = self.seq_list[self.ncap_idx + 3]
        self.N4_AA = self.seq_list[self.ncap_idx + 4]

        self.ccap_idx = self.helix_indices[-1]
        self.cprime_idx = self.ccap_idx + 1
        self.Ccap_AA = self.seq_list[self.ccap_idx]
        if self.cprime_idx < len(self.seq_list):
            self.Cprime_AA = self.seq_list[self.cprime_idx]
        else:
            self.Cprime_AA = None
        self.C3_AA = self.seq_list[self.ccap_idx - 3]

        # pre-compute some flags
        self.has_acetyl = True if ncap == "Ac" else False
        self.has_succinyl = True if ncap == "Sc" else False
        self.has_amide = True if ccap == "Am" else False

        # assign some constants
        self.min_helix_length = 6
        self.mu_helix = 0.5
        self.neg_charge_aa = ["C", "D", "E", "Y"] # residues that get a negative charge when deprotonated
        self.pos_charge_aa = ["K", "R", "H"] # residues that get a positive charge when protonated

        # assign some chemistry constants
        self.kappa = debye_screening_kappa(self.ionic_strength, self.T_kelvin)
        self.epsilon_r = calculate_permittivity(self.T_kelvin) # Relative permittivity of water
        self.epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)
        self.N_A = 6.022e23  # Avogadro's number in mol^-1
        self.e = 1.602e-19  # Elementary charge in Coulombs

        # assign None to all variables
        self.charged_pairs = None
        self.seq_pka = None
        self.nterm_pka = None
        self.cterm_pka = None

        self.distances_hel = None
        self.distances_rc = None

        self.seq_ionization = None
        self.nterm_ionization = None
        self.cterm_ionization = None

        self.terminal_macrodipole_distance_nterm = None
        self.terminal_macrodipole_distance_cterm = None

        # modified ionization states    
        self.modified_seq_ionization_hel = None
        self.modified_nterm_ionization_hel = None
        self.modified_cterm_ionization_hel = None
        self.modified_seq_ionization_rc = None
        self.modified_nterm_ionization_rc = None
        self.modified_cterm_ionization_rc = None

        # for printing
        self.category_pad = 15
        self.value_pad = 5

        # find charged pairs
        self._find_charged_pairs()

        # assign pKa values, distances, and ionization states
        self._assign_pka_values()
        self._assign_ionization_states()
        self._assign_terminal_macrodipole_distances()
        self._assign_sidechain_macrodipole_distances()
        self._assign_terminal_sidechain_distances()
        self._assign_sidechain_sidechain_distances()
        self._assign_modified_ionization_states()

    def _find_charged_pairs(self) -> list[tuple[str, int, int]]:
        """
        Find all pairs of charged residues in a sequence and their global positions.
        """
        charged_amino_acids = self.neg_charge_aa + self.pos_charge_aa

        # Iterate over all pairs of charged residues
        result = []
        for idx1 in range(len(self.seq_list)):
            for idx2 in range(idx1 + 1, len(self.seq_list)):
                AA1 = self.seq_list[idx1]
                AA2 = self.seq_list[idx2]
                if not all(aa in charged_amino_acids for aa in (AA1, AA2)):
                    continue
                result.append((AA1, AA2, idx1, idx2))  # Include global positions
        self.charged_pairs = result

    def _calculate_r(self, N: int) -> float:
        """Function to calculate the distance r from the peptide terminal to the helix
        start, where N is the number of residues between the terminal and the helix.
        p. 177 of Lacroix, 1998. Distances in Ångströms as 2.1, 4.1, 6.1... The function is 
        needed because we ignore sidechains here. We only calculate distances between charged
        termini (which are located on the backbone) and the helix macrodipole, which is located
        at the N- and C-terminal capping residues. The capping residues are not included in the helix
        macrodipole, so the distance can never be shorter than 2.1 Ångströms.

        Args:
            N (int): The number of residues between the peptide terminal and the helix start.

        Returns:
            float: The calculated distance r in Ångströms.
        """
        r = 0.1 + (N + 1) * 2
        return r

    def _electrostatic_interaction_energy(self, qi: float, qj: float, r: float) -> float:
        """Calculate the interaction energy between two charges by
        applying equation 6 from Lacroix, 1998.

        Args:
            qi (float): Charge of the first residue.
            qj (float): Charge of the second residue.
            r (float): Distance between the residues in Ångströms.

        Returns:
            float: The interaction energy in kcal/mol.
        """
        distance_r_meter = r * 1e-10  # Convert distance from Ångströms to meters
        screening_factor = math.exp(-self.kappa * distance_r_meter)
        coulomb_term = (self.e**2 * qi * qj) / (4 * math.pi * self.epsilon_0 * self.epsilon_r * distance_r_meter)
        energy_joules = coulomb_term * screening_factor
        energy_kcal_mol = self.N_A * energy_joules / 4184
        return energy_kcal_mol

    def _make_box(self, title: str):
        """
        Make a box with a title in the middle.
        For making nice looking output when printing.
        """
        box_lines = []
        box_lines.append('+' + '-' * (len(title) + 2) + '+')
        box_lines.append('|' + f'{title.center(len(title) + 2)}' + '|')
        box_lines.append('+' + '-' * (len(title) + 2) + '+')
        return '\n'.join(box_lines)

    def show_inputs(self):
        """
        Print out the inputs for the AGADIR model in a nicely formatted way.
        """
        print(self._make_box("Inputs"))
        print(f'seq = {self.seq}, i = {self.i}, j = {self.j}, pH = {self.pH}, T(celcius) = {self.T_celsius}, ionic_strength = {self.ionic_strength}, ncap = {self.ncap}, ccap = {self.ccap}')
        print("")

    def show_helix(self):
        """
        Print out the helix in a nicely formatted way.
        """
        print(self._make_box("Helix"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"structure:".ljust(self.category_pad)} {"".join(["He".ljust(self.value_pad) if self.i <= idx < self.i + self.j else "STC".ljust(self.value_pad) for idx, aa in enumerate(self.seq_list)])}')
        print("")

    def _assign_pka_values(self):
        """
        Assign pKa values to the sequence and the terminal residues.
        """
        self.seq_pka = np.array([float(self.table_pka_values.loc[aa]["pKa"]) if aa in self.neg_charge_aa + self.pos_charge_aa else np.nan 
                                  for aa in self.seq_list])
        self.nterm_pka = float(self.table_pka_values.loc["Nterm"]["pKa"]) if self.ncap is None else float(self.table_pka_values.loc["Sc"]["pKa"]) if self.ncap == "Sc" else np.nan
        self.cterm_pka = float(self.table_pka_values.loc["Cterm"]["pKa"]) if self.ccap is None else np.nan

    def get_pka_values(self):
        """
        Get pKa values for the sequence and the terminal residues.

        Returns:
            np.ndarray: pKa values for the sequence.
            float: pKa value for the N-terminal capping group.
            float: pKa value for the C-terminal capping group.
        """
        return self.seq_pka, self.nterm_pka, self.cterm_pka
    
    def show_pka_values(self):
        """
        Print out pKa values for the sequence in a nicely formatted way.
        """
        print(self._make_box("pKa values"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"pKa:".ljust(self.category_pad)} {"".join([f"{pka:.1f}".ljust(self.value_pad) for pka in self.seq_pka])}')
        print(f'{"nterm_pka:".ljust(self.category_pad)} {self.nterm_pka:.1f}')
        print(f'{"cterm_pka:".ljust(self.category_pad)} {self.cterm_pka:.1f}')
        print("")

    def _assign_ionization_states(self):
        """
        Compute ionization states for charged residues in the sequence and the terminal residues.

        The ionization states are computed using the pKa values and the pH of the solution.
        """
        self.seq_ionization = np.array([
            acidic_residue_ionization(self.pH, pKa) if aa in self.neg_charge_aa else 
            basic_residue_ionization(self.pH, pKa) if aa in self.pos_charge_aa else 0.0 
            for aa, pKa in zip(self.seq_list, self.seq_pka)
        ])
        
        self.nterm_ionization = (
            basic_residue_ionization(self.pH, self.nterm_pka) if self.ncap is None else
            acidic_residue_ionization(self.pH, self.nterm_pka) if self.ncap == "Sc" else 0.0
        )
        
        self.cterm_ionization = (
            acidic_residue_ionization(self.pH, self.cterm_pka) if self.ccap is None else 0.0
        )
    
    def get_ionization_states(self):
        """
        Get the ionization states for the sequence and the terminal residues.

        Returns:
            np.ndarray: Ionization states for the sequence.
            float: Ionization state for the N-terminal capping group.
            float: Ionization state for the C-terminal capping group.
        """
        return self.seq_ionization, self.nterm_ionization, self.cterm_ionization

    def show_ionization_states(self):
        """
        Print out ionization states for the sequence in a nicely formatted way.
        """
        print(self._make_box("Ionization states"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.1f}".ljust(self.value_pad) for q in self.seq_ionization])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.nterm_ionization:.1f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.cterm_ionization:.1f}')
        print("")

    def _assign_sidechain_macrodipole_distances(self):
        """
        Assign all distances between charged sidechains for the sequence,
        and the helix macrodipole. This is only relevant for peptides in the helical state,
        since otherwise there is no helix macrodipole. Which is always located at the N-
        and C-terminal helix capping residues. The distances for residues outside of the actual helix
        are also accounted for.

        This function makes use of the supplementary table 7 from Lacroix, 1998. For residues
        that are inside the helix. There is one table for the N-terminal macrodipole 
        and one for the C-terminal macrodipole. The table only contains distances up to 
        13 residues apart, so distances greater than 13 are assigned a large distance (99 Å).
        The exact number for large distances does not matter much, since the effect will be screened
        by the solvent. Furthermore, for residues outside of the helix, the distances are calculated
        using the function _calculate_r, which is based on the number of residues between the terminal
        and the helix start.
        """
        self.sidechain_macrodipole_distances_nterm = np.full((len(self.seq_list)), np.nan)
        self.sidechain_macrodipole_distances_cterm = np.full((len(self.seq_list)), np.nan)

        for idx, AA in enumerate(self.seq_list):
            if AA not in self.neg_charge_aa + self.pos_charge_aa:
                continue

            # If amino acid is in the helical region
            if idx in self.helix_indices:
                n_residues_N_separation = idx - self.ncap_idx
                n_residues_C_separation = self.ccap_idx - idx

                # Get the distance to the N-terminal dipole using table 7 from Lacroix, 1998
                if n_residues_N_separation > 13:
                    N_distance_angstrom = 99
                else: 
                    N_key = f"N{n_residues_N_separation}" if n_residues_N_separation != 0 else "Ncap"
                    N_distance_angstrom = self.table_7_ncap_lacroix.loc[AA, N_key]

                # Get the distance to the C-terminal dipole using table 7 from Lacroix, 1998
                if n_residues_C_separation > 13:
                    C_distance_angstrom = 99
                else:
                    C_key = f"C{n_residues_C_separation}" if n_residues_C_separation != 0 else "Ccap"
                    C_distance_angstrom = self.table_7_ccap_lacroix.loc[AA, C_key]
    
            # If amino acid is in the coil region, calculate the distance to the helix start and end
            else:
                n_residues_N_separation = abs(self.ncap_idx - idx)  # Distance to helix start
                N_distance_angstrom = self._calculate_r(n_residues_N_separation)

                n_residues_C_separation = abs(self.ccap_idx - idx)  # Distance to helix end
                C_distance_angstrom = self._calculate_r(n_residues_C_separation)

            # Assign the distances to the arrays
            self.sidechain_macrodipole_distances_nterm[idx] = N_distance_angstrom
            self.sidechain_macrodipole_distances_cterm[idx] = C_distance_angstrom

    def get_sidechain_macrodipole_distances(self):
        """
        Get the distances for the sequence, both for helical and random-coil states.

        Returns:
            np.ndarray: Distances for charged sidechains to the N-terminal helix dipole.
            np.ndarray: Distances for charged sidechains to the C-terminal helix dipole.
        """
        return self.sidechain_macrodipole_distances_nterm, self.sidechain_macrodipole_distances_cterm
    
    def show_sidechain_macrodipole_distances(self):
        """
        Print out the pairwise distances for the sequence in a nicely formatted way.
        """
        print(self._make_box("Sidechain macrodipole distances (Å)"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"nterm:".ljust(self.category_pad)} {"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.sidechain_macrodipole_distances_nterm])}')
        print(f'{"cterm:".ljust(self.category_pad)} {"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.sidechain_macrodipole_distances_cterm])}')
        print("")

    def _assign_terminal_sidechain_distances(self):
        """
        Assign the distance between the peptide terminal residues and the charged sidechains.
        """
        self.terminal_sidechain_distances_nterm = np.full(len(self.seq_list), np.nan)
        self.terminal_sidechain_distances_cterm = np.full(len(self.seq_list), np.nan)

        for idx, AA in enumerate(self.seq_list):
            if AA not in self.neg_charge_aa + self.pos_charge_aa:
                continue

            n_residues_separation = idx
            c_residues_separation = len(self.seq_list) - 1 - idx
            
            self.terminal_sidechain_distances_nterm[idx] = self._calculate_r(n_residues_separation) # TODO: This is overly simplistic, since it does not account for the helix boundary, need a better solution
            self.terminal_sidechain_distances_cterm[idx] = self._calculate_r(c_residues_separation) # TODO: This is overly simplistic, since it does not account for the helix boundary, need a better solution

    def get_terminal_sidechain_distances(self):
        """
        Get the distances for the peptide terminal residues and the charged sidechains.
        """
        return self.terminal_sidechain_distances_nterm, self.terminal_sidechain_distances_cterm
    
    def show_terminal_sidechain_distances(self):
        """
        Print out the distances for the peptide terminal residues and the charged sidechains in a nicely formatted way.
        """
        print(self._make_box("Terminal sidechain distances (Å)"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"nterm:".ljust(self.category_pad)} {"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.terminal_sidechain_distances_nterm])}')
        print(f'{"cterm:".ljust(self.category_pad)} {"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.terminal_sidechain_distances_cterm])}')
        print("")

    def _assign_terminal_macrodipole_distances(self):
        """
        Assign the distance between the peptide terminal residues and the helix macrodipole.
        """
        self.terminal_macrodipole_distance_nterm = self._calculate_r(self.ncap_idx)
        self.terminal_macrodipole_distance_cterm = self._calculate_r(len(self.seq_list) - 1 - self.ccap_idx)

    def get_terminal_macrodipole_distances(self) -> tuple[float, float]:
        """
        Get the distances for the peptide terminal residues and the helix macrodipole.
        Only computes N-terminal distance to the N-terminal helix dipole and 
        C-terminal distance to the C-terminal helix dipole.

        Returns:
            tuple[float, float]: Distances between the peptide N-terminal and C-terminal residues and the helix macrodipole
        """
        return self.terminal_macrodipole_distance_nterm, self.terminal_macrodipole_distance_cterm
    
    def show_terminal_macrodipole_distances(self):
        """
        Print out the distances for the peptide terminal residues and the helix macrodipole in a nicely formatted way.
        """
        print(self._make_box("Terminal macrodipole distances (Å)"))
        print(f'{"nterm:".ljust(self.category_pad)} {self.terminal_macrodipole_distance_nterm:.1f}')
        print(f'{"cterm:".ljust(self.category_pad)} {self.terminal_macrodipole_distance_cterm:.1f}')
        print("")

    def _assign_sidechain_sidechain_distances(self):
        """
        Assign the distance between two charged sidechains.

        This function makes use of the supplementary table 6 from Lacroix, 1998.
        There is one table for helical states and one for random-coil states.
        However, the table only contains distances up to 12 residues apart, so
        distances greater than 12 are assigned a large distance (99 Å). The exact
        number for large distances does not matter much, since the effect will be screened
        by the solvent.

        The function handles the following cases:
        1. Both residues in helix: use table 6 helix distances
        2. Both residues in coil: use table 6 coil distances 
        3. One in helix, one in coil: combine distances through the helix boundary
        Special case: Use HelixRest for pairs containing Tyrosine and Cysteine since they're missing from table
        """
        self.sidechain_sidechain_distances_hel = np.full((len(self.seq_list), len(self.seq_list)), np.nan)
        self.charged_sidechain_distances_rc = np.full((len(self.seq_list), len(self.seq_list)), np.nan)

        ### Assign distances to coil state ###
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
            n_residues_separation = idx2 - idx1
            distance_key = f"i+{n_residues_separation}"
                        
            if n_residues_separation >= 13: # table 6 only contains distances up to 12, so assign a large distance to things that are further apart
                distance_angstrom = 99
            else:
                pair = AA1 + AA2
                if ('Y' in pair) or ('C' in pair): # Handle Cysteine and Tyrosine special case
                    pair = 'RcoilRest'
                distance_angstrom = self.table_6_coil_lacroix.loc[pair, distance_key]

            self.charged_sidechain_distances_rc[idx1, idx2] = distance_angstrom
            self.charged_sidechain_distances_rc[idx2, idx1] = distance_angstrom
            
        ### Assign distances to helix state ###
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
            n_residues_separation = idx2 - idx1
            distance_key = f"i+{n_residues_separation}"          

            if n_residues_separation >= 13: # table 6 only contains distances up to 12, so assign a large distance to things that are further apart
                distance_angstrom = 99

            else:
                # Both residues in helix
                if idx1 in self.helix_indices and idx2 in self.helix_indices:
                    pair = AA1 + AA2
                    if ('Y' in pair) or ('C' in pair): # Handle Cysteine and Tyrosine special case
                        pair = 'HelixRest'
                    distance_angstrom = self.table_6_helix_lacroix.loc[pair, distance_key]
                
                # Both residues in coil part of a peptide that (at a different position) contains the helix
                # TODO: this is only correct if both are in the coil on the same side of the helix, not if they are on either side
                elif idx1 not in self.helix_indices and idx2 not in self.helix_indices:
                    pair = AA1 + AA2
                    if ('Y' in pair) or ('C' in pair): # Handle Cysteine and Tyrosine special case
                        pair = 'RcoilRest'
                        distance_angstrom = self.table_6_coil_lacroix.loc[pair, distance_key]
                
                # One residue in helix, one in coil
                else:
                    # Figure out which residue is in helix and which in coil
                    if idx1 in self.helix_indices:
                        helix_idx = idx1
                        coil_idx = idx2
                    else:
                        helix_idx = idx2
                        coil_idx = idx1
                        
                    # Calculate distance through helix boundary
                    if coil_idx < self.ncap_idx:
                        # Coil residue is before helix
                        coil_separation = self.ncap_idx - coil_idx
                        helix_separation = helix_idx - self.ncap_idx
                        # print('coil, helix idx1 idx2', coil_separation, helix_separation, idx1, idx2)
                        d_coil = 0.0 if coil_separation == 0 else self.table_6_coil_lacroix.loc['RcoilRest', f"i+{coil_separation}"]  # Distance to helix start
                        d_helix = 0.0 if helix_separation == 0 else self.table_6_helix_lacroix.loc['HelixRest', f"i+{helix_separation}"]  # Distance from helix start to helix residue
                    else:
                        # Coil residue is after helix
                        coil_separation = coil_idx - self.ccap_idx
                        helix_separation = self.ccap_idx - helix_idx
                        # print('coil, helix idx1 idx2', coil_separation, helix_separation, idx1, idx2)
                        d_coil = 0.0 if coil_separation == 0 else self.table_6_coil_lacroix.loc['RcoilRest', f"i+{coil_separation}"]  # Distance from helix end
                        d_helix = 0.0 if helix_separation == 0 else self.table_6_helix_lacroix.loc['HelixRest', f"i+{helix_separation}"]  # Distance from helix residue to helix end
                        
                    distance_angstrom = d_coil + d_helix
                
            self.sidechain_sidechain_distances_hel[idx1, idx2] = distance_angstrom
            self.sidechain_sidechain_distances_hel[idx2, idx1] = distance_angstrom

    def get_sidechain_sidechain_distances(self):
        """
        Get the charged sidechain distances for the sequence, both for helical and random-coil states.
        """
        return self.sidechain_sidechain_distances_hel, self.charged_sidechain_distances_rc

    def show_sidechain_sidechain_distances(self):
        """
        Print out the charged sidechain distances for the sequence in a nicely formatted way.
        """
        print(self._make_box("Charged sidechain distances, helix (Å)"))
        print(f'{"".ljust(self.category_pad)}{"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        for i in range(len(self.seq_list)):
            print(f'{self.seq_list[i].ljust(self.category_pad)}{"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.sidechain_sidechain_distances_hel[i]])}')
        print("")

        print(self._make_box("Charged sidechain distances, coil (Å)"))
        print(f'{"".ljust(self.category_pad)}{"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        for i in range(len(self.seq_list)):
            print(f'{self.seq_list[i].ljust(self.category_pad)}{"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.charged_sidechain_distances_rc[i]])}')
        print("")

    def _assign_modified_ionization_states(self):
        """
        Assign modified ionization states for residues in the sequence.

        The actual ionization of a residue is not just determined by the 
        pKa value and the pH of the solution. It is also affected by the 
        electrostatic interactions with the helix macrodipole and the 
        charged sidechains. This function accounts for these interactions
        by iterating on the ionization states until they converge.
        """
        MAX_ITERATIONS = 10
        CONVERGENCE_THRESHOLD = 0.01
        
        # Initialize arrays with unmodified values for iteration
        current_seq_ionization = self.seq_ionization.copy()
        current_nterm_ionization = self.nterm_ionization
        current_cterm_ionization = self.cterm_ionization
        
        # Iterative solution for updating the ionization states
        for iteration in range(MAX_ITERATIONS):
            old_seq_ionization_hel = current_seq_ionization.copy()
            old_nterm_ionization = copy.copy(current_nterm_ionization)
            old_cterm_ionization = copy.copy(current_cterm_ionization)

            # For each residue
            for idx1, AA1 in list(enumerate(self.seq_list)) + [(-1, 'Nterm'), (len(self.seq_list), 'Cterm')]:
                deltaG_total = 0.0

                # 1. Get helix macrodipole contribution to the free energy using current charge
                # Use pre-computed distances and current ionization state
                if AA1 == 'Nterm':
                    if self.seq_list[0] == 'Ac':
                        continue
                    N_distance_angstrom = self.terminal_macrodipole_distance_nterm
                    C_distance_angstrom = 99
                    q1 = current_nterm_ionization
                    pKa_intrinsic = self.nterm_pka

                elif AA1 == 'Cterm':
                    if self.seq_list[-1] == 'Am':
                        continue
                    N_distance_angstrom = 99
                    C_distance_angstrom = self.terminal_macrodipole_distance_cterm
                    q1 = current_cterm_ionization
                    pKa_intrinsic = self.cterm_pka

                elif AA1 in self.neg_charge_aa + self.pos_charge_aa:     
                    N_distance_angstrom = self.sidechain_macrodipole_distances_nterm[idx1]
                    C_distance_angstrom = self.sidechain_macrodipole_distances_cterm[idx1]
                    q1 = current_seq_ionization[idx1]
                    pKa_intrinsic = self.seq_pka[idx1]

                else:
                    continue
                   
                # Calculate deltaG for N-terminal dipole, but only if the distance is small enough, don't waste compute on large distances
                if N_distance_angstrom < 40:
                    deltaG_N = self._electrostatic_interaction_energy(
                        qi=self.mu_helix, 
                        qj=q1, 
                        r=N_distance_angstrom
                    )
                    deltaG_total += deltaG_N
                
                # Calculate deltaG for C-terminal dipole, but only if the distance is small enough, don't waste compute on large distances
                if C_distance_angstrom < 40:
                    deltaG_C = self._electrostatic_interaction_energy(
                        qi=-self.mu_helix, 
                        qj=q1, 
                        r=C_distance_angstrom
                    )
                    deltaG_total += deltaG_C
                
                # 2. Get charged sidechain interactions using current charges
                # Only consider interactions with other charged sidechains
                for idx2, AA2 in list(enumerate(self.seq_list)) + [(-1, 'Nterm'), (len(self.seq_list), 'Cterm')]:
                    if idx2 <= idx1 or AA2 not in self.neg_charge_aa + self.pos_charge_aa + ['Nterm', 'Cterm']:
                        continue

                    if AA1 == 'Nterm' and AA2 == 'Cterm':
                        sidechain_distance_angstrom = 99
                        q2 = current_cterm_ionization
                    elif AA1 == 'Nterm' and AA2 in self.neg_charge_aa + self.pos_charge_aa:
                        sidechain_distance_angstrom = self.terminal_sidechain_distances_nterm[idx2]
                        q2 = current_seq_ionization[idx2]
                    elif AA1 in self.neg_charge_aa + self.pos_charge_aa and AA2 == 'Cterm':
                        sidechain_distance_angstrom = self.terminal_sidechain_distances_cterm[idx1]
                        q2 = current_cterm_ionization
                    else:
                        sidechain_distance_angstrom = self.sidechain_sidechain_distances_hel[idx1, idx2]
                        q2 = current_seq_ionization[idx2]
                        
                    # Use current ionization states to calculate deltaG, but only 
                    if sidechain_distance_angstrom < 40:
                        deltaG = self._electrostatic_interaction_energy(qi=q1, 
                                                                        qj=q2, 
                                                                        r=sidechain_distance_angstrom)
                        deltaG_total += deltaG
               
                # 3. Calculate new pKa relative to intrinsic value
                if np.isnan(deltaG_total):
                    raise ValueError(f"deltaG_total is nan, this should not happen. There was probably a problem retrieving precomed distances or ionization states")
                
                # Calculate new pKa relative to intrinsic value
                modified_pKa = adjust_pKa(T=self.T_kelvin, 
                                          pKa_ref=pKa_intrinsic, 
                                          deltaG=deltaG_total,
                                          is_basic=True if AA1 in self.pos_charge_aa else False)

                if AA1 == 'Nterm':
                    if self.seq_list[0] == 'Sc':
                        current_nterm_ionization = acidic_residue_ionization(
                            pH=self.pH, 
                            pKa=modified_pKa
                        )
                    else:
                        current_nterm_ionization = basic_residue_ionization(
                            pH=self.pH, 
                            pKa=modified_pKa
                        )
                elif AA1 == 'Cterm':
                    current_cterm_ionization = acidic_residue_ionization(
                        pH=self.pH, 
                        pKa=modified_pKa
                    )
                elif AA1 in self.neg_charge_aa:
                    current_seq_ionization[idx1] = acidic_residue_ionization(
                        pH=self.pH, 
                        pKa=modified_pKa
                    )
                elif AA1 in self.pos_charge_aa:
                    current_seq_ionization[idx1] = basic_residue_ionization(
                        pH=self.pH, 
                        pKa=modified_pKa
                    )               
          
            # Filter out nan values for convergence check
            valid_old = np.concatenate([old_seq_ionization_hel[~np.isnan(old_seq_ionization_hel)], 
                                        [old_nterm_ionization],
                                        [old_cterm_ionization]])
            valid_current = np.concatenate([current_seq_ionization[~np.isnan(current_seq_ionization)], 
                                             [current_nterm_ionization],
                                             [current_cterm_ionization]])
            
            if len(valid_old) > 0 and len(valid_current) > 0:
                max_change = np.max(np.abs(valid_current - valid_old))
                
                # Check for convergence
                if max_change < CONVERGENCE_THRESHOLD:
                    break
            else:
                raise ValueError("No valid ionization states to compare")
        
        # Save final converged values for helical state
        self.modified_seq_ionization_hel = current_seq_ionization
        self.modified_nterm_ionization_hel = current_nterm_ionization
        self.modified_cterm_ionization_hel = current_cterm_ionization

        # Save final converged values for coil state  TODO: should I actually calculate these? instead of just copying
        self.modified_seq_ionization_rc = self.seq_ionization.copy()
        self.modified_nterm_ionization_rc = self.nterm_ionization
        self.modified_cterm_ionization_rc = self.cterm_ionization

    def get_modified_ionization_states(self):
        """
        Get the modified ionization states for the sequence.
        """
        return self.modified_seq_ionization_hel, self.modified_nterm_ionization_hel, self.modified_cterm_ionization_hel

    def show_modified_ionization_states(self):
        """
        Print out the modified ionization states for the sequence in a nicely formatted way.
        """
        print(self._make_box("Modified ionization states, helix"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.1f}".ljust(self.value_pad) for q in self.modified_seq_ionization_hel])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.modified_nterm_ionization_hel:.1f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.modified_cterm_ionization_hel:.1f}')
        print("")
        print(self._make_box("Modified ionization states, coil"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.1f}".ljust(self.value_pad) for q in self.modified_seq_ionization_rc])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.modified_nterm_ionization_rc:.1f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.modified_cterm_ionization_rc:.1f}')
        print("")

    def show_all(self):
        """
        Print out all the inputs, helix, pKa values, and ionization states in a nicely formatted way.
        """
        self.show_inputs()
        self.show_helix()
        self.show_pka_values()
        self.show_ionization_states()
        self.show_modified_ionization_states()
        self.show_terminal_macrodipole_distances()
        self.show_sidechain_macrodipole_distances()
        self.show_terminal_sidechain_distances()
        self.show_sidechain_sidechain_distances()


class EnergyCalculator(PrecomputeParams):
    """
    Class to calculate the free energy contributions for a peptide sequence.
    """
    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None):
        """
        Initialize the EnergyCalculator for a peptide sequence.

        Args:
            seq (str): Peptide sequence.
            i (int): Helix start index, python 0-indexed.
            j (int): Helix length.
            pH (float): Solution pH.
            T (float): Temperature in Celsius.
            ionic_strength (float): Ionic strength of the solution in mol/L.
            ncap (str): N-terminal capping modification (acetylation='Ac', succinylation='Sc').
            ccap (str): C-terminal capping modification (amidation='Am').
        """
        super().__init__(seq, i, j, pH, T, ionic_strength, ncap, ccap)
    
    def get_dG_Int(self) -> np.ndarray:
        """
        Get the intrinsic free energy contributions for a helical segment.
        This accounts for the loss of entropy due to the helix formation.
        The first and last residues are considered to be caps unless they are
        the peptide terminal residues with modifications.
        Equation (7) from Muñoz & Serrano (1995) is used to correct for temperature effects.

        Returns:
            np.ndarray: The intrinsic free energy contributions for each amino acid in the helical segment.
        """
        # Initialize energy array
        energy = np.zeros(len(self.seq_list))
        Tref = 273.15  # 0°C reference temperature
        dCp = 0.0015  # kcal/(mol*K)

        # Iterate over the helix and get the intrinsic energy for each residue, 
        # not including residues that are capping for the helical segment
        for idx in self.helix_indices:
            if idx == self.ncap_idx or idx == self.ccap_idx:
                continue

            AA = self.seq_list[idx]

            # Handle N-terminal region specially
            if idx == self.ncap_idx + 1:
                energy[idx] = self.table_1_lacroix.loc[AA, "N1"]
            elif idx == self.ncap_idx + 2:
                energy[idx] = self.table_1_lacroix.loc[AA, "N2"]
            elif idx == self.ncap_idx + 3:
                energy[idx] = self.table_1_lacroix.loc[AA, "N3"]
            elif idx == self.ncap_idx + 4:
                energy[idx] = self.table_1_lacroix.loc[AA, "N4"]
            else:
                energy[idx] = self.table_1_lacroix.loc[AA, "Ncen"]

            if AA in self.neg_charge_aa + self.pos_charge_aa:
                # Charged residues: use precomputed ionization state and balance energy based on ionization state
                # If they are completely ionized, use base value, if they are completely neutral, use Neutral, 
                # if they are partially ionized, use a weighted average of base value and Neutral
                q = abs(self.modified_seq_ionization_hel[idx]) # abs because I only want to kno the fraction ionized, I don't care about the sign
                basic_energy = energy[idx]
                basic_energy_neutral = self.table_1_lacroix.loc[AA, "Neutral"]
                energy[idx] = q * basic_energy + (1 - q) * basic_energy_neutral

        # # Apply Equation (7) correction
        # energy += self.T_kelvin * (dCp * np.log(self.T_kelvin / Tref)) # TODO: This is not correct. How to implement this?

        return energy
    
    def get_dG_Hbond(self) -> float:
        """
        Get the free energy contribution for hydrogen bonding for a sequence.

        Capping residues, don't count toward hydrogen bonding, which gives 2.
        Additionally, the first 4 helical amino acids are considered to have 
        zero net enthalpy since they are nucleating residues. 
        This gives a total of 6 residues that don't count toward hydrogen bonding.
        Hbond value is -0.895 kcal/mol per residue, which is the value from the 
        discussion section of the 1998 lacroix paper. Uses equation 6 from Munoz et al. 1994.
        to adjust for temperature (https://doi.org/10.1006/jmbi.1994.0024).

        Returns:
            float: The total free energy contribution for hydrogen bonding in the sequence.
        """
        # Base number of H-bonds excluding caps and nucleating residues
        n_hbonds = max((self.j - 6), 0)
        
        # Temperature correction
        Tref = 273.15  # 0°C reference temperature
        Href = -0.895  # kcal/mol
        dCp = 0.0015  # kcal/(mol*K)
        dG_per_hbond = Href + dCp * (self.T_kelvin - Tref) # uses equation 6 from Munoz et al. 1994

        return dG_per_hbond * n_hbonds

    def get_dG_Ncap(self) -> np.ndarray:
        """
        Get the free energy contribution for N-terminal capping.
        This accounts only for residue capping effects.

        Returns:
            np.ndarray: The free energy contribution.
        """
        energy = np.zeros(len(self.seq_list))

        # Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.
        if self.N1_AA == "P" and self.N3_AA in ["E", "D", "Q"]:
            energy[self.ncap_idx] = self.table_1_lacroix.loc[self.Ncap_AA, "Nc-4"]

        # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
        elif self.N3_AA in ["E", "D", "Q"]:
            energy[self.ncap_idx] = self.table_1_lacroix.loc[self.Ncap_AA, "Nc-3"]

        # Nc-2 	N-cap values when there is a Pro at position N1.
        elif self.N1_AA == "P":
            energy[self.ncap_idx] = self.table_1_lacroix.loc[self.Ncap_AA, "Nc-2"]

        # Nc-1 	Normal N-cap values.
        else:
            energy[self.ncap_idx] = self.table_1_lacroix.loc[self.Ncap_AA, "Nc-1"]

        return energy

    def get_dG_Ccap(self) -> np.ndarray:
        """
        Get the free energy contribution for C-terminal capping.

        Returns:
            np.ndarray: The free energy contribution.
        """
        energy = np.zeros(len(self.seq_list))

        # Cc-2 	C-cap values when there is a Pro residue at position C'
        if self.Cprime_AA == "P":
            energy[self.ccap_idx] = self.table_1_lacroix.loc[self.Ccap_AA, "Cc-2"]

        # Cc-1 	Normal C-cap values
        else:
            energy[self.ccap_idx] = self.table_1_lacroix.loc[self.Ccap_AA, "Cc-1"]

        return energy

    def get_dG_staple(self) -> float:
        """
        Get the free energy contribution for the hydrophobic staple motif.
        The hydrophobic interaction is between the N' and N4 residues of the helix.
        The terminology of Richardson & Richardson (1988) is used.
        See https://doi.org/10.1038/nsb0595-380 for more details.

        Returns:
            float: The free energy contribution.
        """
        # Staple motif requires the N' residue before the Ncap, so the first residue of the helix cannot be the first residue of the peptide
        # This should be true regardless of whether there is an N-terminal modification or not
        energy = 0.0
        if self.ncap_idx == 0:
            return energy

        # The hydrophobic staple motif is only considered whenever the N-cap residue is Asn, Asp, Ser, Pro or Thr.
        if self.Ncap_AA in ["N", "D", "S", "P", "T"]:
            energy = self.table_2_lacroix.loc[self.Ncap_AA, self.N4_AA]

            # whenever the N-cap residue is Asn, Asp, Ser, or Thr and the N3 residue is Glu, Asp or Gln, multiply by 1.0
            if self.Ncap_AA in ["N", "D", "S", "T"] and self.N3_AA in ["E", "D", "Q"]:
                # print("staple case i")
                energy *= 1.0

            # whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr
            elif self.Ncap_AA in ["N", "D"] and self.N3_AA in ["S", "T"]:
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

    def get_dG_schellman(self) -> float:
        """
        Get the free energy contribution for the Schellman motif.
        The Schellman motif is only considered whenever Gly is the C-cap residue,
        where the interaction happens between the C' and C3 residues of the helix.
        The terminology of Richardson & Richardson (1988) is used.

        Returns:
            float: The free energy contribution.
        """
        # The Schellman motif is only considered whenever Gly is the C-cap residue,
        # and there has to be a C' residue after the helix
        energy = 0.0
        if self.Cprime_AA in ["Am", None] or self.Ccap_AA != "G":
            return energy
    
        # get the amino acids governing the Schellman motif and extract the energy
        energy = self.table_3_lacroix.loc[self.C3_AA, self.Cprime_AA] / 100

        return energy

    def get_dG_i3(self) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

        Returns:
            np.ndarray: The free energy contributions for each interaction.
        """
        energy = np.zeros(len(self.seq_list))

        # Get interaction free energies for charged residues
        for idx in self.helix_indices[:-3]:
            AAi = self.seq_list[idx]
            AAi3 = self.seq_list[idx + 3]

            # Skip if N- and C-terminal modifications
            if AAi in ["Ac", "Am", "Sc"] or AAi3 in ["Ac", "Am", "Sc"]:
                continue

            base_energy = self.table_4a_lacroix.loc[AAi, AAi3] / 100

            if AAi in self.pos_charge_aa + self.neg_charge_aa and AAi3 in self.neg_charge_aa + self.pos_charge_aa:
                # Use precomputed ionization states for the helical state
                q_i = self.modified_seq_ionization_hel[idx]
                q_i3 = self.modified_seq_ionization_hel[idx + 3]
                energy[idx] = base_energy * abs(q_i * q_i3) # TODO: This scaling is not correct, it should use the values from table 5
            else:
                energy[idx] = base_energy

        return energy
    
    def get_dG_i4(self) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

        Returns:
            np.ndarray: The free energy contributions for each interaction.
        """
        energy = np.zeros(len(self.seq_list))

        # Get interaction free energies for charged residues
        for idx in self.helix_indices[:-4]:
            AAi = self.seq_list[idx]
            AAi4 = self.seq_list[idx + 4]

            # Skip if N- and C-terminal modifications
            if AAi in ["Ac", "Am", "Sc"] or AAi4 in ["Ac", "Am", "Sc"]:
                continue

            base_energy = self.table_4b_lacroix.loc[AAi, AAi4] / 100

            if AAi in self.pos_charge_aa + self.neg_charge_aa and AAi4 in self.pos_charge_aa + self.neg_charge_aa:
                # Use precomputed ionization states for the helical state
                q_i = self.modified_seq_ionization_hel[idx]
                q_i4 = self.modified_seq_ionization_hel[idx + 4]
                energy[idx] = base_energy * abs(q_i * q_i4) # TODO: This scaling is not correct, it should use the values from table 5
            else:
                energy[idx] = base_energy

        return energy

    def get_dG_terminals_macrodipole(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate interaction energies between N- and C-terminal backbone charges and the helix macrodipole.
        The energy is added to the residue carrying the macrodipole charge.

        Returns:
            tuple[np.ndarray, np.ndarray]: Interaction energies for N-terminal and C-terminal residues.
        """
        N_term = np.zeros(len(self.seq_list))
        C_term = np.zeros(len(self.seq_list))

        # Calculate the interaction energy between the N-terminal and the helix macrodipole
        N_term[self.ncap_idx] = self._electrostatic_interaction_energy(
            qi=self.mu_helix,
            qj=self.modified_nterm_ionization_hel,
            r=self.terminal_macrodipole_distance_nterm
        )

        # Calculate the interaction energy between the C-terminal and the helix macrodipole
        C_term[self.ccap_idx] = self._electrostatic_interaction_energy(
            qi=-self.mu_helix,
            qj=self.modified_cterm_ionization_hel,
            r=self.terminal_macrodipole_distance_cterm
        )

        return N_term, C_term

    def get_dG_sidechain_macrodipole(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the interaction energy between charged side-chains and the helix macrodipole.
        The helix macrodipole is positively charged at the N-terminus and negatively charged at the C-terminus.
        The interaction could be either with side chains inside the helix or outside the helix.
        The energy should be unaffected by N- and C-terminal modifications except for changing which 
        residues are part of the helix. The energy is added to the residue carrying the macrodipole charge.

        Returns:
            tuple[np.ndarray, np.ndarray]: The free energy contribution for each residue in the helix, 
            N-terminal and C-terminal contributions.
        """
        energy_N = np.zeros(len(self.seq_list))
        energy_C = np.zeros(len(self.seq_list))

        # Get the interaction energies for the side chains outside the helix, i.e. in the coils
        for idx, aa in enumerate(self.seq_list):

            # Skip if amino acid is not charged
            if aa not in self.neg_charge_aa + self.pos_charge_aa:
                continue

            # N-terminal interaction
            energy_N[self.ncap_idx] += self._electrostatic_interaction_energy(
                qi=self.mu_helix, 
                qj=self.modified_seq_ionization_hel[idx], 
                r=self.sidechain_macrodipole_distances_nterm[idx]
            )

            # C-terminal interaction
            energy_C[self.ccap_idx] += self._electrostatic_interaction_energy(
                qi=-self.mu_helix, 
                qj=self.modified_seq_ionization_hel[idx], 
                r=self.sidechain_macrodipole_distances_cterm[idx]
            )

        return energy_N, energy_C
        
    def get_dG_terminals_sidechain_electrost(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate electrostatic interaction energies between terminal backbone charges
        and charged sidechains in the sequence. The energy is added to the charged sidechain residue.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays containing the interaction energies for 
            N-terminal and C-terminal interactions respectively. The energy is added
            to the charged sidechain residue.
        """
        energy_N = np.zeros(len(self.seq_list))
        energy_C = np.zeros(len(self.seq_list))
        
        # Skip if terminals are uncharged
        if self.seq_list[0] == 'Ac' and self.seq_list[-1] == "Am":
            return energy_N, energy_C
        
        # Iterate through sequence checking for charged sidechains
        for idx, AA1 in enumerate(self.seq_list):
            if AA1 not in self.neg_charge_aa + self.pos_charge_aa:
                continue
                
            # Get distance from N-terminal to this residue
            n_distance = self.terminal_sidechain_distances_nterm[idx]
            
            # Get distance from C-terminal to this residue  
            c_distance = self.terminal_sidechain_distances_cterm[idx]

            # N-terminal interaction
            energy_N[idx] = self._electrostatic_interaction_energy(
                qi=self.modified_nterm_ionization_hel,
                qj=self.modified_seq_ionization_hel[idx],
                r=n_distance
            )
                
            # C-terminal interaction
            energy_C[idx] = self._electrostatic_interaction_energy(
                qi=self.modified_cterm_ionization_hel,
                qj=self.modified_seq_ionization_hel[idx],
                r=c_distance
            )

        return energy_N, energy_C

    def get_dG_sidechain_sidechain_electrost(self) -> np.ndarray:
        """
        Calculate the electrostatic free energy contribution for charged residue sidechains
        inside and outside the helical segment, using Lacroix et al. (1998) equations.
        Half of the energy is added to each of the charged sidechain residues.

        Returns:
            np.ndarray: n x n symmetric matrix of pairwise electrostatic free energy contributions,
                       with each interaction energy split between upper and lower triangles.
        """
        energy_matrix = np.zeros((len(self.seq_list), len(self.seq_list)))

        # Iterate over all charged residue pairs
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
            # Skip if not in upper triangle
            if idx2 - self.i <= idx1 - self.i:
                continue

            # Get the distances between the charged sidechains
            helix_dist = self.sidechain_sidechain_distances_hel[idx1, idx2]
            coil_dist = self.charged_sidechain_distances_rc[idx2, idx1]
                
            # Get the ionization states of the charged sidechains
            q1_hel = self.modified_seq_ionization_hel[idx1]
            q2_hel = self.modified_seq_ionization_hel[idx2]
            q1_rc = self.modified_seq_ionization_rc[idx1]
            q2_rc = self.modified_seq_ionization_rc[idx2]

            # Valculate electrostatic interaction energies with adjusted ionization states, Lacroix Eq 6
            G_hel = self._electrostatic_interaction_energy(qi=q1_hel, qj=q2_hel, r=helix_dist)
            G_rc = self._electrostatic_interaction_energy(qi=q1_rc, qj=q2_rc, r=coil_dist)

            # Store half the energy difference in both triangles of the matrix (give half of the energy to each sidechain)
            energy_diff = (G_hel - G_rc) / 2
            energy_matrix[idx1, idx2] = energy_diff  # Upper triangle
            energy_matrix[idx2, idx1] = energy_diff  # Lower triangle

        return energy_matrix
    