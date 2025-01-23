import math
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import calculate_ionic_strength, adjust_pKa, acidic_residue_ionization, basic_residue_ionization, calculate_permittivity, debye_screening_length
from pyagadir.utils import is_valid_index, is_valid_peptide_sequence, is_valid_ncap_ccap, is_valid_conditions
import warnings




class PrecomputeParams:
    """
    Class to load parameters for the AGADIR model and
    for pre-computing distances, and ionization states.
    """
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
        is_valid_peptide_sequence(seq)
        is_valid_ncap_ccap(ncap, ccap)
        is_valid_index(seq, i, j, ncap, ccap)
        is_valid_conditions(pH, T, ionic_strength)

        self.seq = seq
        self.seq_list = list(seq)
        if ncap is not None:
            self.seq_list = [ncap] + self.seq_list
        if ccap is not None:
            self.seq_list = self.seq_list + [ccap]
        self.i = i
        self.j = j
        self.pH = pH
        self.T_celsius = T
        self.T_kelvin = T + 273.15
        self.ionic_strength = ionic_strength
        self.ncap = ncap
        self.ccap = ccap

        self.has_acetyl = True if ncap == "Ac" else False
        self.has_succinyl = True if ncap == "Sc" else False
        self.has_amide = True if ccap == "Am" else False

        self.min_helix_length = 6
        self.mu_helix = 0.5
        self.neg_charge_aa = ["C", "D", "E", "Y"] # residues that get a negative charge when deprotonated
        self.pos_charge_aa = ["K", "R", "H"] # residues that get a positive charge when protonated

        # assign some chemistry constants
        self.kappa = debye_screening_length(self.ionic_strength, self.T_kelvin)
        self.epsilon_r = calculate_permittivity(self.T_kelvin) # Relative permittivity of water
        self.epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)
        self.N_A = 6.022e23  # Avogadro's number in mol^-1
        self.e = 1.602e-19  # Elementary charge in Coulombs

        # assign None to all variables
        self.table_1_lacroix = None
        self.table_2_lacroix = None
        self.table_3_lacroix = None
        self.table_4a_lacroix = None
        self.table_4b_lacroix = None
        self.table_6_helix_lacroix = None
        self.table_6_coil_lacroix = None
        self.table_7_ccap_lacroix = None
        self.table_7_ncap_lacroix = None

        self.pka_values = None
        self.distances_hel = None
        self.distances_rc = None
        self.ionization_states = None
        self.nterm_ionization = None
        self.cterm_ionization = None
        self.modified_ionization_states = None
        self.modified_nterm_ionization = None
        self.modified_cterm_ionization = None

        self.category_pad = 15
        self.value_pad = 5

        # load params
        self._load_params()

        # assign pKa values, distances, and ionization states
        self._assign_pka_values()
        # self._assign_distances()
        self._assign_ionization_states()
        # self._assign_modified_ionization_states()

    def _load_params(self):
        """
        Load the parameters for the AGADIR model.
        """
        # get params
        datapath = files("pyagadir.data.params")

        # load energy contributions for intrinsic propensities, capping, etc.
        self.table_1_lacroix = pd.read_csv(
            datapath.joinpath("table_1_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load the hydrophobic staple motif energy contributions
        self.table_2_lacroix = pd.read_csv(
            datapath.joinpath("table_2_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load the schellman motif energy contributions
        self.table_3_lacroix = pd.read_csv(
            datapath.joinpath("table_3_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load energy contributions for interactions between i and i+3
        self.table_4a_lacroix = pd.read_csv(
            datapath.joinpath("table_4a_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load energy contributions for interactions between i and i+4
        self.table_4b_lacroix = pd.read_csv(
            datapath.joinpath("table_4b_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load sidechain distances for helices
        self.table_6_helix_lacroix = pd.read_csv(
            datapath.joinpath("table_6_helix_lacroix.tsv"),
            index_col="Pos",
            sep="\t",
        ).astype(float)

        # load sidechain distances for coils
        self.table_6_coil_lacroix = pd.read_csv(
            datapath.joinpath("table_6_coil_lacroix.tsv"),
            index_col="Pos",
            sep="\t",
        ).astype(float)

        # load N-terminal distances between charged amino acids and the half charge from the helix macrodipole
        self.table_7_ccap_lacroix = pd.read_csv(
            datapath.joinpath("table_7_Ccap_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load C-terminal distances between charged amino acids and the half charge from the helix macrodipole
        self.table_7_ncap_lacroix = pd.read_csv(
            datapath.joinpath("table_7_Ncap_lacroix.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

        # load pKa values for for side chain ionization and the N- and C-terminal capping groups
        self.pka_values = pd.read_csv(
            datapath.joinpath("pka_values.tsv"),
            index_col="AA",
            sep="\t",
        ).astype(float)

    def _make_box(self, title: str):
        """
        Make a box with a title in the middle.
        """
        box_lines = []
        box_lines.append('+' + '-' * (len(title) + 2) + '+')
        box_lines.append('|' + f'{title.center(len(title) + 2)}' + '|')
        box_lines.append('+' + '-' * (len(title) + 2) + '+')
        return '\n'.join(box_lines)

    def show_inputs(self):
        """
        Print out the inputs in a nicely formatted way.
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
        Assign pKa values to the sequence.
        """
        self.seq_pka_values = np.array([float(self.pka_values.loc[aa]["pKa"]) if aa in self.neg_charge_aa + self.pos_charge_aa else np.nan 
                                  for aa in self.seq_list])
        self.nterm_pka = float(self.pka_values.loc["Nterm"]["pKa"]) if self.ncap is None else float(self.pka_values.loc["Sc"]["pKa"]) if self.ncap == "Sc" else np.nan
        self.cterm_pka = float(self.pka_values.loc["Cterm"]["pKa"]) if self.ccap is None else np.nan

    def get_pka_values(self):
        """
        Get pKa values for the sequence and the terminal residues.

        Returns:
            np.ndarray: pKa values for the sequence.
            float: pKa value for the N-terminal capping group.
            float: pKa value for the C-terminal capping group.
        """
        return self.seq_pka_values, self.nterm_pka, self.cterm_pka
    
    def show_pka_values(self):
        """
        Print out pKa values for the sequence in a nicely formatted way.
        """
        print(self._make_box("pKa values"))
        print(f'{"seq:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"pKa:".ljust(self.category_pad)} {"".join([f"{pka:.1f}".ljust(self.value_pad) for pka in self.seq_pka_values])}')
        print(f'{"nterm_pka:".ljust(self.category_pad)} {self.nterm_pka:.1f}')
        print(f'{"cterm_pka:".ljust(self.category_pad)} {self.cterm_pka:.1f}')
        print("")

    def _assign_ionization_states(self):
        """
        Compute ionization states for charged residues in the sequence.
        """
        self.ionization_states = np.array([
            acidic_residue_ionization(self.pH, pKa) if aa in self.neg_charge_aa else 
            basic_residue_ionization(self.pH, pKa) if aa in self.pos_charge_aa else 0.0 
            for aa, pKa in zip(self.seq_list, self.seq_pka_values)
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
        return self.ionization_states, self.nterm_ionization, self.cterm_ionization

    def show_ionization_states(self):
        """
        Print out ionization states for the sequence in a nicely formatted way.
        """
        print(self._make_box("Ionization states"))
        print(f'{"seq:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.1f}".ljust(self.value_pad) for q in self.ionization_states])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.nterm_ionization:.1f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.cterm_ionization:.1f}')
        print("")

    def _assign_distances(self):
        """
        Assign all pairwise distances for the sequence,
        both for helical and random-coil states.
        """
        # self.distances_hel = np.array([self.table_6_helix_lacroix.loc[idx, "dist"] if self.i <= idx < self.i + self.j else self.table_6_coil_lacroix.loc[idx, "dist"] for idx in range(len(self.seq_list))])
        # self.distances_rc = np.array([self.table_6_coil_lacroix.loc[idx, "dist"] for idx in range(len(self.seq_list))])
        raise NotImplementedError("Not implemented")

    def get_distances(self):
        """
        Get the distances for the sequence, both for helical and random-coil states.

        Returns:
            np.ndarray: Pairwise distances for the sequence helical state.
            np.ndarray: Pairwise distances for the sequence in random-coil state.
        """
        return self.distances_hel, self.distances_rc
    
    def show_distances(self):
        """
        Print out the pairwise distances for the sequence in a nicely formatted way.
        """
        # print(self._make_box("Pairwise distances, coiled state (Å)"))
        # print(f'{"distances_rc:".ljust(self.category_pad)} {"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.distances_rc])}')
        # print("")
        # print(self._make_box("Pairwise distances, helical state (Å)"))
        # print(f'{"distances_hel:".ljust(self.category_pad)} {"".join([f"{d:.1f}".ljust(self.value_pad) for d in self.distances_hel])}')
        # print("")
        raise NotImplementedError("Not implemented")

    def show_all(self):
        """
        Print out all the inputs, helix, pKa values, and ionization states in a nicely formatted way.
        """
        self.show_inputs()
        self.show_helix()
        self.show_pka_values()
        self.show_ionization_states()

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

        # Precompute ionization states for helical and random-coil states
        self.q_global_hel, self.q_global_rc = self._precompute_ionization_states()
        # print("****************")
        # print(pH)
        # print(self.seq)
        # print(self.q_global_hel)
        # print(self.q_global_rc)
        # print("****************")


    # def calculate_ionization_state(self, residue_idx: int, helix_start: int, helix_length: int, is_helical: bool) -> float:
    #     """
    #     Calculate ionization state for a specific residue in a specific helix context.
        
    #     Args:
    #         residue_idx: Position of the residue in sequence
    #         helix_start: Start position of helix
    #         helix_length: Length of helix
    #         is_helical: Whether calculating for helical or coil state
    #     """
    #     AA = self.seq[residue_idx]
    #     if AA not in ["C", "D", "E", "Y", "K", "R", "H"]:
    #         return 0.0
            
    #     pKa_ref = pka_values.loc[AA, "pKa"]
        
    #     if not is_helical:
    #         deltaG = 0.0
    #     else:
    #         # Calculate position-dependent deltaG
    #         rel_pos = residue_idx - helix_start  # Position relative to helix start
            
    #         # Calculate dipole contribution based on position
    #         if rel_pos < helix_length/2:
    #             # N-terminal half
    #             dipole_effect = -0.5 * (1 - rel_pos/(helix_length/2))
    #         else:
    #             # C-terminal half
    #             dipole_effect = 0.5 * ((rel_pos - helix_length/2)/(helix_length/2))
                
    #         # Could also factor in:
    #         # - Distance-dependent interactions with other charged residues
    #         # - Specific backbone conformational effects
    #         # - Local environmental factors
            
    #         deltaG = dipole_effect
            
    #     pKa_adjusted = adjust_pKa(self.T_kelvin, pKa_ref, deltaG)
        
    #     if AA in ["C", "D", "E", "Y"]:
    #         return acidic_residue_ionization(self.pH, pKa_adjusted)
    #     else:
    #         return basic_residue_ionization(self.pH, pKa_adjusted)

    def _precompute_ionization_states(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute ionization states for all residues in the peptide
        for both helical and random-coil states.

        Returns:
            tuple[np.ndarray, np.ndarray]: Ionization states for helical and random-coil states.
        """
        q_hel = np.zeros(len(self.seq))
        q_rc = np.zeros(len(self.seq))

        for idx, AA in enumerate(self.seq):

            if AA in self.neg_charge_aa:
                pKa_ref = self.pka_values.loc[AA, "pKa"]
                pKa_rc = adjust_pKa(self.T_kelvin, pKa_ref, deltaG=0.0)  # Coil state
                pKa_hel = adjust_pKa(self.T_kelvin, pKa_ref, deltaG=-0.5)  # Helix state
                q_rc[idx] = acidic_residue_ionization(self.pH, pKa_rc)
                q_hel[idx] = acidic_residue_ionization(self.pH, pKa_hel)

            elif AA in self.pos_charge_aa:
                pKa_ref = self.pka_values.loc[AA, "pKa"]
                pKa_rc = adjust_pKa(self.T_kelvin, pKa_ref, deltaG=0.0)  # Coil state
                pKa_hel = adjust_pKa(self.T_kelvin, pKa_ref, deltaG=-0.5)  # Helix state
                q_rc[idx] = basic_residue_ionization(self.pH, pKa_rc)
                q_hel[idx] = basic_residue_ionization(self.pH, pKa_hel)

        # # Handle terminal ionization states
        # if not (self.has_acetyl or self.has_succinyl):  # Free N-terminus
        #     pKa_ref = pka_values.loc["Nterm", "pKa"]
        #     pKa_nterm = adjust_pKa(self.T_kelvin, pKa_ref, deltaG=0.0) # or -0.5?, or both coil and helix?
        #     q_nterm = basic_residue_ionization(self.pH, pKa_nterm)

        # if not self.has_amide:  # Free C-terminus
        #     pKa_ref = pka_values.loc["Cterm", "pKa"]
        #     pKa_cterm = adjust_pKa(self.T_kelvin, pKa_ref, deltaG=0.0)  # or -0.5?, or both coil and helix?
        #     q_cterm = acidic_residue_ionization(self.pH, pKa_cterm)

        return q_hel, q_rc

    def _get_helix(self, i: int, j: int) -> str:
        """
        Get the helical segment of a peptide sequence, including 
        the residues forming the N- and C-caps of the helix.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            str: The helix region of the peptide sequence.
        """
        is_valid_index(self.seq, i, j, self.ncap, self.ccap)

        return self.seq[i : i + j]

    def _are_terminal_residues_capping(self, pept_len: int, i: int, j: int) -> tuple[bool, bool]:
        """
        Determine whether the first and last residues of a helical segment should be treated as capping residues. 
        
        Without N- and C- terminal modifications, the first and last residues of the peptide are treated as capping residues.
        With N- and C- terminal modifications, the first and last residues of the peptide are not treated as capping residues, 
        since the modifactions are treated as capping residues.
        For internal helical segments, the first and last residues are always capping residues since they are not
        affected by N- and C- terminal modifications.

        Args:
            pept_len (int): The length of the entire peptide sequence, regardless of which helical segment is being considered.
            i (int): Starting index of the helical segment
            j (int): Length of the helical segment

        Returns:
            tuple[bool, bool]: (res_is_ncap, res_is_ccap) indicating whether the two terminal residues should be treated
                            as N- and C- capping residues
        """
        # By default, every helical segment has capping residues
        n_term_is_cap = True
        c_term_is_cap = True

        # Check if this segment includes the N-terminus of the peptide
        if i == 0:
            # If we have N-terminal modification, this segment doesn't have an N-cap
            if self.has_acetyl or self.has_succinyl:
                n_term_is_cap = False

        # Check if this segment includes the C-terminus of the peptide
        if i + j == pept_len:
            # If we have C-terminal modification, this segment doesn't have a C-cap
            if self.has_amide:
                c_term_is_cap = False

        return n_term_is_cap, c_term_is_cap

    def _calculate_r(self, N: int) -> float:
        """Function to calculate the distance r from the peptide terminal to the helix
        start, where N is the number of residues between the terminal and the helix.
        p. 177 of Lacroix, 1998. Distances in Ångströms as 2.1, 4.1, 6.1...

        Args:
            N (int): The number of residues between the peptide terminal and the helix start.

        Returns:
            float: The calculated distance r in Ångströms.
        """
        r = 0.1 + (N + 1) * 2
        return r

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
        charged_amino_acids = set(self.neg_charge_aa + self.pos_charge_aa)
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
        r = r * 1e-10  # Convert distance from Ångströms to meters
        coulomb_term = (self.e**2 * qi * qj) / (4 * math.pi * self.epsilon_0 * self.epsilon_r * r)
        energy_joules = coulomb_term * math.exp(-self.kappa * r)
        energy_kcal_mol = self.N_A * energy_joules / 4184
        return energy_kcal_mol

    def _calculate_terminal_energy(self,
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
                screening_factor = math.exp(-self.kappa * distance_r_meter)

                # Calculate terminal interaction energy
                B_kappa = 332.0  # in kcal Å / (mol e^2)

                # In the form of equation (10.62) from DOI 10.1007/978-1-4419-6351-2_10
                coloumb_potential = self.mu_helix / (self.epsilon_r * distance_r_angstrom)
                energy = B_kappa * screening_factor * coloumb_potential

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

    def get_dG_Int(self, i: int, j: int) -> np.ndarray:
        """
        Get the intrinsic free energy contributions for a helical segment.
        This accounts for the loss of entropy due to the helix formation.
        The first and last residues are considered to be caps unless they are
        the peptide terminal residues with modifications.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The intrinsic free energy contributions for each amino acid in the helical segment.
        """
        helix = self._get_helix(i, j)
        res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(len(self.seq), i, j)

        # Initialize energy array
        energy = np.zeros(len(helix))

        # Iterate over the helix and get the intrinsic energy for each residue, 
        # not including residues that are capping for the helical segment
        for idx, AA in enumerate(helix):
            # Skip caps only if they exist for this segment
            if (idx == 0 and res_is_ncap) or (idx == len(helix) - 1 and res_is_ccap):
                continue

            # Handle N-terminal region specially
            if idx == 1 or (idx == 0 and not res_is_ncap):
                energy[idx] = self.table_1_lacroix.loc[AA, "N1"]
            elif idx == 2 or (idx == 1 and not res_is_ncap):
                energy[idx] = self.table_1_lacroix.loc[AA, "N2"]
            elif idx == 3 or (idx == 2 and not res_is_ncap):
                energy[idx] = self.table_1_lacroix.loc[AA, "N3"]
            elif idx == 4 or (idx == 3 and not res_is_ncap):
                energy[idx] = self.table_1_lacroix.loc[AA, "N4"]
            else:
                energy[idx] = self.table_1_lacroix.loc[AA, "Ncen"]

            if AA in self.neg_charge_aa + self.pos_charge_aa:
                # Charged residues: use precomputed ionization state and balance energy based on ionization state
                # If they are completely ionized, use base value, if they are completely neutral, use Neutral, 
                # if they are partially ionized, use a weighted average of base value and Neutral
                q = self.q_global_hel[i + idx]
                basic_energy = energy[idx]
                basic_energy_neutral = self.table_1_lacroix.loc[AA, "Neutral"]
                energy[idx] = q * basic_energy + (1 - q) * basic_energy_neutral

        return energy

    def get_dG_Ncap(
        self,
        i: int,
        j: int,
    ) -> np.ndarray:
        """
        Get the free energy contribution for N-terminal capping.
        This accounts only for residue capping effects.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            np.ndarray: The free energy contribution.
        """
        helix = self._get_helix(i, j)
        res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(helix, i, j)

        # If it's the peptide N-terminal residue and it's not capping in the helical segment, 
        # then change the name to match an N-terminal modification
        Ncap_AA = helix[0]
        if i == 0 and not res_is_ncap:
            Ncap_AA = "Ac"

        energy = np.zeros(len(helix))

        # Get the N1 and N3 residues, which are the first and third residues of the helical segment.
        # If the helical segment is the peptide N-terminal residue, and it's not capping, then we have to shift the indices
        if i == 0 and not res_is_ncap:
            N1_AA = helix[0]
        else:   
            N1_AA = helix[1]

        if i == 0 and not res_is_ncap:
            N3_AA = helix[2]
        else:
            N3_AA = helix[3]

        # Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.
        if N1_AA == "P" and N3_AA in ["E", "D", "Q"]:
            energy[0] = self.table_1_lacroix.loc[Ncap_AA, "Nc-4"]

        # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
        elif N3_AA in ["E", "D", "Q"]:
            energy[0] = self.table_1_lacroix.loc[Ncap_AA, "Nc-3"]

        # Nc-2 	N-cap values when there is a Pro at position N1.
        elif N1_AA == "P":
            energy[0] = self.table_1_lacroix.loc[Ncap_AA, "Nc-2"]

        # Nc-1 	Normal N-cap values.
        else:
            energy[0] = self.table_1_lacroix.loc[Ncap_AA, "Nc-1"]

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
        helix = self._get_helix(i, j)
        res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(len(self.seq), i, j)

        # fix the blocking group names to match the table
        if i + j == len(self.seq) and not res_is_ccap:
            Ccap_AA = "Am"
        else:
            Ccap_AA = helix[-1]

        energy = np.zeros(len(helix))

        # Cc-2 	C-cap values when there is a Pro residue at position C'
        c_prime_idx = i + j
        if (len(self.seq) > c_prime_idx) and (self.seq[c_prime_idx] == "P"):
            energy[-1] = self.table_1_lacroix.loc[Ccap_AA, "Cc-2"]

        # Cc-1 	Normal C-cap values
        else:
            energy[-1] = self.table_1_lacroix.loc[Ccap_AA, "Cc-1"]

        return energy

    def get_dG_staple(self, i: int, j: int) -> float:
        """
        Get the free energy contribution for the hydrophobic staple motif.
        The hydrophobic interaction is between the N' and N4 residues of the helix.
        The terminology of Richardson & Richardson (1988) is used.
        See https://doi.org/10.1038/nsb0595-380 for more details.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            float: The free energy contribution.
        """
        helix = self._get_helix(i, j)
        res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(len(self.seq), i, j)

        energy = np.zeros(len(helix))

        # Staple motif requires the N' residue before the Ncap, so the first residue of the helix cannot be the first residue of the peptide
        # This should be true regardless of whether there is an N-terminal modification or not
        energy = 0.0
        if i == 0:
            return energy

        # get the amino acids governing the staple motif, but account for any terminal modifications
        Ncap_AA = helix[0]
        N_prime_AA = self.seq[i - 1]
        N3_AA = helix[3]
        N4_AA = helix[4]

        # TODO: verify that the code below is correct

        # The hydrophobic staple motif is only considered whenever the N-cap residue is Asn, Asp, Ser, Pro or Thr.
        if Ncap_AA in ["N", "D", "S", "P", "T"]:
            energy = self.table_2_lacroix.loc[N_prime_AA, N4_AA]

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
        The terminology of Richardson & Richardson (1988) is used.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            float: The free energy contribution.
        """
        helix = self._get_helix(i, j)
        res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(len(self.seq), i, j)

        # The Schellman motif is only considered whenever Gly is the C-cap residue,
        # and there has to be a C' residue after the helix
        energy = 0.0
        if i + j == len(self.seq) and not res_is_ccap:
            return energy
        
        C_cap_AA = helix[-1]
        if C_cap_AA != "G":
            return energy

        # TODO verify that the code below is correct

        # get the amino acids governing the Schellman motif and extract the energy
        C_prime_AA = self.seq[i + j]
        C3_AA = helix[-4]
        energy = self.table_3_lacroix.loc[C3_AA, C_prime_AA] / 100

        return energy

    def get_dG_Hbond(self, i: int, j: int) -> float:
        """
        Get the free energy contribution for hydrogen bonding for a sequence.

        Always subtract 4 residues for nucleation.
        Add 1 additional non-contributing residue for each residue that is capping, 
        as determined by get_capping_status().

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            float: The total free energy contribution for hydrogen bonding in the sequence.
        """
        # Start with nucleating residues
        non_contributing = 4

        # Figure out whether the first and last residues are capping for the helical segment
        helix = self._get_helix(i, j)
        res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(len(helix), i, j) # it's a bit strange to use len(helix) here, instead of len(self.seq), but it's consistent with the expected behavior of the function

        if res_is_ncap:
            non_contributing += 1
        if res_is_ccap:
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
        helix = self._get_helix(i, j)
        energy = np.zeros(len(helix))

        # Get interaction free energies for charged residues
        for idx in range(len(helix) - 3):
            AAi = helix[idx]
            AAi3 = helix[idx + 3]
            base_energy = self.table_4a_lacroix.loc[AAi, AAi3] / 100

            if AAi in self.pos_charge_aa + self.neg_charge_aa and AAi3 in self.neg_charge_aa + self.pos_charge_aa:
                # Use precomputed ionization states for the helical state
                q_i = self.q_global_hel[i + idx]
                q_i3 = self.q_global_hel[i + idx + 3]
                energy[idx] = base_energy * abs(q_i * q_i3)
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
        helix = self._get_helix(i, j)
        energy = np.zeros(len(helix))

        # Get interaction free energies for charged residues
        for idx in range(len(helix) - 4):
            AAi = helix[idx]
            AAi4 = helix[idx + 4]
            base_energy = self.table_4b_lacroix.loc[AAi, AAi4] / 100

            if AAi in self.pos_charge_aa + self.neg_charge_aa and AAi4 in self.pos_charge_aa + self.neg_charge_aa:
                # Use precomputed ionization states for the helical state
                q_i = self.q_global_hel[i + idx]
                q_i4 = self.q_global_hel[i + idx + 4]
                energy[idx] = base_energy * abs(q_i * q_i4)
            else:
                energy[idx] = base_energy

        return energy

    def get_dG_terminals_macrodipole(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate interaction energies between N- and C-terminal backbone charges and the helix macrodipole.

        Args:
            i (int): Starting index of the helix segment.
            j (int): Length of the helix segment.

        Returns:
            tuple[np.ndarray, np.ndarray]: Interaction energies for N-terminal and C-terminal residues.
        """
        helix = self._get_helix(i, j)
        N_term = np.zeros(len(helix))
        C_term = np.zeros(len(helix))

        # N-terminal calculation, not capped by acetyl nor succinyl, so it is positively charged
        if not self.has_acetyl and not self.has_succinyl:
            pKa_Nterm = self.pka_values.loc["Nterm", "pKa"]
            distance_r_N = self._calculate_r(i)
            N_term_energy = self._calculate_terminal_energy(
                distance_r_N, pKa_Nterm, "basic", terminal="N"
            )
            N_term[0] = N_term_energy
            
        # N-terminal calculation, here we account for the negative charge of succinyl, 
        # which is a special case since the N-terminal is normally positively charged
        # and CAN interact with the helix macrodipole (page 177 in Lacroix, 1998)
        elif self.has_succinyl: 
            pKa_Nterm = self.pka_values.loc["Succinyl", "pKa"]
            distance_r_N = self._calculate_r(i)
            N_term_energy = self._calculate_terminal_energy(
                distance_r_N, pKa_Nterm, "acidic", terminal="N"
            )
            N_term[0] = N_term_energy

        # C-terminal calculation, only if not capped by amidation
        if not self.has_amide:
            pKa_Cterm = self.pka_values.loc["Cterm", "pKa"]
            distance_r_C = self._calculate_r(len(self.seq) - (i + j))
            C_term_energy = self._calculate_terminal_energy(
                distance_r_C, pKa_Cterm, "acidic", terminal="C"
            )
            C_term[-1] = C_term_energy

        return N_term, C_term

    # def get_dG_terminals_macrodipole(
    #     self,
    #     i: int,
    #     j: int,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     """Get the interaction energies for the peptides N- and C-terminal residues with the helix macrodipole of
    #     the helical segment under consideration.

    #     Args:
    #         i (int): Starting index of the helix segment.
    #         j (int): Length of the helix segment.

    #     Returns:
    #         tuple[np.ndarray, np.ndarray]: Interaction energies for N and C terminals.
    #     """
    #     mu_helix = 0.5
    #     helix = self._get_helix(i, j)
    #     res_is_ncap, res_is_ccap = self._are_terminal_residues_capping(len(self.seq), i, j)
    #     N_term = np.zeros(len(helix))
    #     C_term = np.zeros(len(helix))

    #     # N terminal, but only if there is no N-terminal modification, since this removes the charge
    #     if (
    #         not self.has_acetyl
    #     ):  # guard against N-termenal capping residues (no charge, no interaction)
    #         # but beware that succinyl is treated as charged and CAN interact with the helix macrodipole (page 177 in Lacroix, 1998)
    #         distance_r_angstrom = self._calculate_r(i)  # Distance to N terminal
    #         distance_r_meter = (
    #             distance_r_angstrom * 1e-10
    #         )  # Convert distance from Ångströms to meters
    #         kappa = debye_screening_length(self.ionic_strength, self.T_kelvin)
    #         screening_factor = math.exp(
    #             -kappa * distance_r_meter
    #         )  # Second half of equation 6 from Lacroix, 1998.
    #         N_term_energy = self._electrostatic_interaction_energy(
    #             mu_helix, distance_r_angstrom, screening_factor
    #         )
    #         # TODO: Add backbone pKa values for each aa
    #         qKaN = pka_values.loc["Nterm", "pKa"]
    #         q = basic_residue_ionization(self.pH, qKaN)
    #         N_term_energy *= q
    #         N_term[0] = N_term_energy

    #     # C terminal, but only if there is no C-terminal modification, since this removes the charge
    #     if (
    #         not self.has_amide
    #     ):  # guard against C-terminal capping residues (no charge, no interaction)
    #         distance_r_angstrom = self._calculate_r(len(self.seq) - (i + j))  # Distance to C terminal
    #         distance_r_meter = (
    #             distance_r_angstrom * 1e-10
    #         )  # Convert distance from Ångströms to meters
    #         kappa = debye_screening_length(self.ionic_strength, self.T_kelvin)
    #         screening_factor = math.exp(
    #             -kappa * distance_r_meter
    #         )  # Second half of equation 6 from Lacroix, 1998.
    #         C_term_energy = self._electrostatic_interaction_energy(
    #             mu_helix, distance_r_angstrom, screening_factor
    #         )
    #         # TODO: Add pKa values for each aa
    #         qKaC = pka_values.loc["Cterm", "pKa"]
    #         q = acidic_residue_ionization(self.pH, qKaC)
    #         C_term_energy *= -q
    #         C_term[-1] = C_term_energy

    #     return N_term, C_term

    def get_dG_sidechain_macrodipole(
        self, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the interaction energy between charged side-chains and the helix macrodipole.
        The helix macrodipole is positively charged at the N-terminus and negatively charged at the C-terminus.
        The interaction could be either with side chains inside the helix or outside the helix.
        The energy should be unaffected by N- and C-terminal modifications except for changing which residues are part of the helix.

        Args:
            i (int): The helix start index, python 0-indexed.
            j (int): The helix length.

        Returns:
            tuple[np.ndarray, np.ndarray]: The free energy contribution for each residue in the helix, 
            N-terminal and C-terminal contributions.
        """
        helix = self._get_helix(i, j)
        energy_N = np.zeros(len(self.seq))
        energy_C = np.zeros(len(self.seq))

        # Get the interaction energies for the side chains outside the helix, i.e. in the coils
        for idx, aa in enumerate(self.seq):
            # Skip if amino acid is in the helical region
            if i <= idx < i + j:
                continue

            # Skip if amino acid is not charged
            if aa not in self.neg_charge_aa + self.pos_charge_aa:
                continue

            # Get the distance to the helix macrodipole
            if idx < i: # N-terminal
                num_resi_between_helix_and_charged_residue = i - idx
                N_distance_angstrom = self._calculate_r(num_resi_between_helix_and_charged_residue)
                q_sidechain = self.q_global_rc[idx] # use random coil ionization state
                energy = self._electrostatic_interaction_energy(qi=self.mu_helix, qj=q_sidechain, r=N_distance_angstrom)
                energy_N[idx] = energy

            else: # C-terminal
                num_resi_between_helix_and_charged_residue = idx - (i + j)
                C_distance_angstrom = self._calculate_r(num_resi_between_helix_and_charged_residue)
                q_sidechain = self.q_global_rc[idx] # use random coil ionization state
                energy = self._electrostatic_interaction_energy(qi=-self.mu_helix, qj=q_sidechain, r=C_distance_angstrom)
                energy_C[idx] = energy

        # Get the macrodipole interaction energies for the side chains inside the helix
        for idx, aa in enumerate(helix):
            # Skip if amino acid is not charged
            if aa not in self.neg_charge_aa + self.pos_charge_aa:
                continue

            # The distance tables only contain values up to C13 from the C-terminus and N13 from the N-terminus
            # Try to fetch the distance here
            N_position = "Ncap" if idx == 0 else f"N{idx}"
            try:
                N_distance = self.table_7_ncap_lacroix.loc[aa, N_position]
            except (KeyError, ValueError):
                warnings.warn(f'{aa} with position {N_position} not found in table_7_ncap_lacroix')
                N_distance = None

            C_position = "Ccap" if idx == len(helix) - 1 else f"C{len(helix)-idx-1}"
            try:
                C_distance = self.table_7_ccap_lacroix.loc[aa, C_position]
            except (KeyError, ValueError):
                warnings.warn(f'{aa} with position {C_position} not found in table_7_ccap_lacroix')
                C_distance = None

            # Fetch the precomputed ionization state
            q_sidechain = self.q_global_hel[i + idx]

            # N-terminal interaction
            if N_distance is not None:
                energy_N[i + idx] += self._electrostatic_interaction_energy(
                    qi=self.mu_helix, qj=q_sidechain, r=N_distance
                )

            # C-terminal interaction
            if C_distance is not None:
                energy_C[i + idx] += self._electrostatic_interaction_energy(
                    qi=-self.mu_helix, qj=q_sidechain, r=C_distance
                )

        return energy_N, energy_C
        
    def get_dG_sidechain_sidechain_electrost(self, i: int, j: int) -> np.ndarray:
        """
        Calculate the electrostatic free energy contribution for charged residue sidechains
        inside and outside the helical segment, using Lacroix et al. (1998) equations.

        Args:
            i (int): The helix start index (0-indexed).
            j (int): The helix length.

        Returns:
            np.ndarray: n x n symmetric matrix of pairwise electrostatic free energy contributions,
                       with each interaction energy split between upper and lower triangles.
        """
        helix = self._get_helix(i, j)  # Extract helical segment
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
                helix_dist = self.table_6_helix_lacroix.loc[pair, distance_key]
            except (KeyError, ValueError):
                helix_dist = self.table_6_helix_lacroix.loc['HelixRest', distance_key]

            try:
                coil_dist = self.table_6_coil_lacroix.loc[pair, distance_key]
            except (KeyError, ValueError):
                coil_dist = self.table_6_coil_lacroix.loc['RcoilRest', distance_key]

            # Step 1: Get reference pKa values for both residues
            pKa_ref_1 = self.pka_values.loc[res1, "pKa"]
            pKa_ref_2 = self.pka_values.loc[res2, "pKa"]

            # Step 2: Assume full unit charges and calculate initial G_hel and G_rc, Lacroix Eq 6
            G_hel = self._electrostatic_interaction_energy(qi=1, qj=1, r=helix_dist)
            G_rc = self._electrostatic_interaction_energy(qi=1, qj=1, r=coil_dist)

            # Step 3: Adjust pKa values based on the local free energy contributions, Lacroix Eq 8 and 9
            pKa_hel_1 = adjust_pKa(self.T_kelvin, pKa_ref_1, G_hel)
            pKa_hel_2 = adjust_pKa(self.T_kelvin, pKa_ref_2, G_hel)
            pKa_rc_1 = adjust_pKa(self.T_kelvin, pKa_ref_1, G_rc)
            pKa_rc_2 = adjust_pKa(self.T_kelvin, pKa_ref_2, G_rc)

            # Step 4: Recalculate ionization states using adjusted pKa (Equations 10-11)
            q1_hel = acidic_residue_ionization(self.pH, pKa_hel_1) if res1 in self.neg_charge_aa else basic_residue_ionization(self.pH, pKa_hel_1)
            q2_hel = acidic_residue_ionization(self.pH, pKa_hel_2) if res2 in self.neg_charge_aa else basic_residue_ionization(self.pH, pKa_hel_2)
            q1_rc = acidic_residue_ionization(self.pH, pKa_rc_1) if res1 in self.neg_charge_aa else basic_residue_ionization(self.pH, pKa_rc_1)
            q2_rc = acidic_residue_ionization(self.pH, pKa_rc_2) if res2 in self.neg_charge_aa else basic_residue_ionization(self.pH, pKa_rc_2)

            # Step 5: Recalculate electrostatic interaction energies with adjusted ionization states, Lacroix Eq 6
            G_hel = self._electrostatic_interaction_energy(qi=q1_hel, qj=q2_hel, r=helix_dist)
            G_rc = self._electrostatic_interaction_energy(qi=q1_rc, qj=q2_rc, r=coil_dist)

            # Step 6: Store half the energy difference in both triangles of the matrix
            energy_diff = (G_hel - G_rc) / 2
            energy_matrix[idx1 - i, idx2 - i] = energy_diff  # Upper triangle
            energy_matrix[idx2 - i, idx1 - i] = energy_diff  # Lower triangle

        return energy_matrix
    