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

            # load empirical sidechain-macrodipole energies from Muñoz & Serrano 1995 II Table 3
            cls._params["table_3_munoz_nterm"] = pd.read_csv(
                datapath.joinpath("table_3_munoz_1995.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            cls._params["table_3_munoz_cterm"] = pd.read_csv(
                datapath.joinpath("table_3_munoz_1995_cterm.tsv"),
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

    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, debug: bool = False):
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
        self.debug = debug
        
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
        self.table_3_munoz_nterm = params["table_3_munoz_nterm"]
        self.table_3_munoz_cterm = params["table_3_munoz_cterm"]
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
        self.value_pad = 6

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

    def _electrostatic_interaction_energy(self, qi: float, qj: float, r: float, factor_pi: float = 4.0) -> float:
        """Calculate the interaction energy between two charges by
        applying equation 6 from Lacroix, 1998.
        Note: The paper prints 3π but the reference tool output matches 4π (standard Coulomb).

        Args:
            qi (float): Charge of the first residue.
            qj (float): Charge of the second residue.
            r (float): Distance between the residues in Ångströms.
            factor_pi (float): Factor to multiply the Coulomb term by. Default is 3.0, as indicated in Lacroix, 1998.
        Returns:
            float: The interaction energy in kcal/mol.
        """
        distance_r_meter = r * 1e-10  # Convert distance from Ångströms to meters
        screening_factor = math.exp(-self.kappa * distance_r_meter)
        coulomb_term = (self.e**2 * qi * qj) / (factor_pi * math.pi * self.epsilon_0 * self.epsilon_r * distance_r_meter) 
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
        print(f'{"pKa:".ljust(self.category_pad)} {"".join([f"{pka:.2f}".ljust(self.value_pad) for pka in self.seq_pka])}')
        print(f'{"nterm_pka:".ljust(self.category_pad)} {self.nterm_pka:.2f}')
        print(f'{"cterm_pka:".ljust(self.category_pad)} {self.cterm_pka:.2f}')
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
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.2f}".ljust(self.value_pad) for q in self.seq_ionization])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.nterm_ionization:.2f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.cterm_ionization:.2f}')
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
        n = len(self.seq_list)
        self.sidechain_macrodipole_distances_nterm = np.full(n, 99.0, dtype=float)
        self.sidechain_macrodipole_distances_cterm = np.full(n, 99.0, dtype=float)

        charged = set(self.neg_charge_aa + self.pos_charge_aa)

        # Treat these as helix boundaries: N1 and C1
        helix_start = int(self.ncap_idx)   # N1
        helix_end = int(self.ccap_idx)     # C1
        helix_len = helix_end - helix_start + 1
        if helix_len <= 5:
            raise ValueError(f"Invalid helix boundaries: start={helix_start}, end={helix_end}")

        # Optional but helpful sanity
        if helix_start not in self.helix_indices or helix_end not in self.helix_indices:
            raise ValueError(
                "helix_indices must include the first/last helical residues "
                f"(start={helix_start}, end={helix_end})."
            )

        def _lookup_n_table(AA: str, Npos: int) -> float:
            """Distance to N-terminal macrodipole pole from table_7_ncap_lacroix."""
            if Npos == 0:
                key = "Ncap"
            elif 1 <= Npos <= 13:
                key = f"N{Npos}"
            else:
                return 99.0
            try:
                return float(self.table_7_ncap_lacroix.loc[AA, key])
            except Exception as e:
                raise KeyError(f"Missing N-table entry for residue={AA}, key={key}") from e

        def _lookup_c_table(AA: str, Cpos: int) -> float:
            """Distance to C-terminal macrodipole pole from table_7_ccap_lacroix."""
            if Cpos == 0:
                key = "Ccap"
            elif 1 <= Cpos <= 13:
                key = f"C{Cpos}"
            else:
                return 99.0
            try:
                return float(self.table_7_ccap_lacroix.loc[AA, key])
            except Exception as e:
                raise KeyError(f"Missing C-table entry for residue={AA}, key={key}") from e

        # Define the region where the table is valid: helix residues + immediate flanking caps (if present)
        table_min = helix_start - 1
        table_max = helix_end + 1

        # Coil anchors: macrodipole poles are at the caps when caps exist; otherwise at the terminal helix residues
        n_pole_anchor = helix_start - 1 if helix_start > 0 else helix_start
        c_pole_anchor = helix_end + 1 if helix_end < (n - 1) else helix_end

        debug = bool(getattr(self, "debug", False))

        for idx, AA in enumerate(self.seq_list):
            if AA not in charged:
                continue

            if table_min <= idx <= table_max:
                # Table positions:
                #   Npos: 0=Ncap, 1=N1, ..., helix_len-1=last helix residue
                #   Cpos: 0=Ccap, 1=C1, ..., helix_len-1=first helix residue
                # For flanking residues (N' or C'), Npos or Cpos may be negative → 99.0 fallback
                Npos = idx - helix_start
                Cpos = helix_end - idx

                dN = _lookup_n_table(AA, Npos)
                dC = _lookup_c_table(AA, Cpos)

                if debug:
                    # show the key mapping you care about
                    N_key = "Ncap" if Npos == 0 else (f"N{Npos}" if 1 <= Npos <= 13 else "99")
                    C_key = "Ccap" if Cpos == 0 else (f"C{Cpos}" if 1 <= Cpos <= 13 else "99")
                    print(f"[TABLE] idx={idx} AA={AA} Npos={Npos} -> {N_key}  |  Cpos={Cpos} -> {C_key}")

            else:
                # Coil fallback: distance based on number of residues *between* residue and the pole anchor
                sepN = abs(idx - n_pole_anchor)
                sepC = abs(idx - c_pole_anchor)

                N_between = max(0, sepN - 1)
                C_between = max(0, sepC - 1)

                dN = 99.0 if N_between > 13 else float(self._calculate_r(N_between))
                dC = 99.0 if C_between > 13 else float(self._calculate_r(C_between))

                if debug:
                    print(
                        f"[COIL] idx={idx} AA={AA} N_between={N_between} dN={dN:.2f} | "
                        f"C_between={C_between} dC={dC:.2f}"
                    )

            self.sidechain_macrodipole_distances_nterm[idx] = dN
            self.sidechain_macrodipole_distances_cterm[idx] = dC

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
        print(f'{"nterm:".ljust(self.category_pad)} {"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.sidechain_macrodipole_distances_nterm])}')
        print(f'{"cterm:".ljust(self.category_pad)} {"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.sidechain_macrodipole_distances_cterm])}')
        print("")

    def _assign_terminal_sidechain_distances(self):
        """
        Assign the distance between the peptide terminal residues and the charged sidechains.

        For the HELIX state: use Table 7 (Lacroix 1998) distances, which reflect the
        compact helical geometry.  The table is keyed by the sidechain's position
        relative to the Ncap (for N-terminal) or Ccap (for C-terminal).

        For positions outside the Table 7 range (>13 positions from the cap), fall back
        to _calculate_r (the linear random-coil model).

        These arrays are used by both the pKa solver (helix ensemble) and
        get_dG_terminals_sidechain_electrost (helix-state energy).
        """
        n = len(self.seq_list)
        self.terminal_sidechain_distances_nterm = np.full(n, np.nan)
        self.terminal_sidechain_distances_cterm = np.full(n, np.nan)

        charged = set(self.neg_charge_aa + self.pos_charge_aa)

        for idx, AA in enumerate(self.seq_list):
            if AA not in charged:
                continue

            # --- N-terminal helix distance: from Ncap (≈N-terminal) to sidechain ---
            Npos = idx - self.ncap_idx  # position relative to Ncap
            if 0 <= Npos <= 13 and AA in self.table_7_ncap_lacroix.index:
                col = self.table_7_ncap_lacroix.columns[Npos]
                self.terminal_sidechain_distances_nterm[idx] = float(
                    self.table_7_ncap_lacroix.loc[AA, col]
                )
            else:
                self.terminal_sidechain_distances_nterm[idx] = self._calculate_r(idx)

            # --- C-terminal helix distance: from Ccap (≈C-terminal) to sidechain ---
            Cpos = self.ccap_idx - idx  # position relative to Ccap
            if 0 <= Cpos <= 13 and AA in self.table_7_ccap_lacroix.index:
                col = self.table_7_ccap_lacroix.columns[Cpos]
                self.terminal_sidechain_distances_cterm[idx] = float(
                    self.table_7_ccap_lacroix.loc[AA, col]
                )
            else:
                self.terminal_sidechain_distances_cterm[idx] = self._calculate_r(
                    (n - 1) - idx
                )

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
        print(f'{"nterm:".ljust(self.category_pad)} {"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.terminal_sidechain_distances_nterm])}')
        print(f'{"cterm:".ljust(self.category_pad)} {"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.terminal_sidechain_distances_cterm])}')
        print("")

    def _assign_terminal_macrodipole_distances(self):
        """
        Assign the distance between the peptide terminal residues and the helix macrodipole.
        """
        # Calculate base geometric distance (approx 2.1 A for N=0)
        dist_n = self._calculate_r(self.ncap_idx)

        # N-Terminal Dipole Distance
        if self.ncap == 'Sc':
            self.terminal_macrodipole_distance_nterm = dist_n
        else:
            # Standard backbone amine distance
            self.terminal_macrodipole_distance_nterm = dist_n

        # C-Terminal Dipole Distance
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
        print(f'{"nterm:".ljust(self.category_pad)} {self.terminal_macrodipole_distance_nterm:.2f}')
        print(f'{"cterm:".ljust(self.category_pad)} {self.terminal_macrodipole_distance_cterm:.2f}')
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
                # (A) Both residues in helix
                if idx1 in self.helix_indices and idx2 in self.helix_indices:
                    # Check if either residue is at a cap position — use cap-specific
                    # Table 6 rows instead of AA-pair rows (Lacroix 1998: cap positions
                    # have modeled non-helical backbone angles giving different distances).
                    # Use "f" (free terminal) rows when terminal is not blocked.
                    # Always use 'Ccap'/'Ncap' rows for cap-position distances.
                    # The 'C-cap f'/'N-cap f' rows give shorter distances (e.g.
                    # 8.03 vs 10.7 for i+1) that overestimate repulsion; the
                    # reference tool matches the blocked-cap geometry ('Ccap'/'Ncap')
                    # for sidechain-sidechain interactions at cap positions.
                    cap_row = None
                    if idx2 == self.ccap_idx:
                        cap_row = 'Ccap'
                    elif idx1 == self.ncap_idx:
                        cap_row = 'Ncap'
                    elif idx1 == self.ccap_idx:
                        cap_row = 'Ccap'
                    elif idx2 == self.ncap_idx:
                        cap_row = 'Ncap'

                    if cap_row is not None:
                        distance_angstrom = self.table_6_helix_lacroix.loc[cap_row, distance_key]
                    else:
                        pair = AA1 + AA2
                        if ('Y' in pair) or ('C' in pair):
                            pair = 'HelixRest'
                        distance_angstrom = self.table_6_helix_lacroix.loc[pair, distance_key]
                
                # (B) Both residues in coil part of a peptide that (at a different position) contains the helix
                elif idx1 not in self.helix_indices and idx2 not in self.helix_indices:
                    straddles_helix = (idx1 < self.ncap_idx) and (idx2 > self.ccap_idx)

                    if not straddles_helix:
                        pair = AA1 + AA2
                        if ("Y" in pair) or ("C" in pair):
                            pair = "RcoilRest"
                        distance_angstrom = float(self.table_6_coil_lacroix.loc[pair, distance_key])
                    else:
                        distance_angstrom = 99
                
                # (C) One residue in helix, one in coil
                else:
                    # Identify which is coil
                    coil_idx = idx1 if idx1 not in self.helix_indices else idx2
                    
                    # [PATCH] Lacroix 1998 Restriction:
                    # Only calculate distance if the coil residue is N' (ncap_idx - 1) or C' (ccap_idx + 1)
                    # Otherwise, interaction is ignored (distance = 99)
                    
                    is_N_prime = (coil_idx == self.ncap_idx - 1)
                    is_C_prime = (coil_idx == self.ccap_idx + 1)
                    
                    if not (is_N_prime or is_C_prime):
                        distance_angstrom = 99
                    else:
                        # If it is N' or C', use your existing logic (or simplified Table 6 Coil lookup)
                        # Your existing logic for summing paths is acceptable specifically for N'/C'
                        # because they are adjacent to the helix boundaries.
                        if idx1 in self.helix_indices:
                            helix_idx = idx1
                        else:
                            helix_idx = idx2
                            
                        if coil_idx < self.ncap_idx:
                            coil_separation = self.ncap_idx - coil_idx
                            helix_separation = helix_idx - self.ncap_idx
                            d_coil = 0.0 if coil_separation == 0 else self.table_6_coil_lacroix.loc['RcoilRest', f"i+{coil_separation}"]
                            d_helix = 0.0 if helix_separation == 0 else self.table_6_helix_lacroix.loc['HelixRest', f"i+{helix_separation}"]
                        else:
                            coil_separation = coil_idx - self.ccap_idx
                            helix_separation = self.ccap_idx - helix_idx
                            d_coil = 0.0 if coil_separation == 0 else self.table_6_coil_lacroix.loc['RcoilRest', f"i+{coil_separation}"]
                            d_helix = 0.0 if helix_separation == 0 else self.table_6_helix_lacroix.loc['HelixRest', f"i+{helix_separation}"]
                            
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
            print(f'{self.seq_list[i].ljust(self.category_pad)}{"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.sidechain_sidechain_distances_hel[i]])}')
        print("")

        print(self._make_box("Charged sidechain distances, coil (Å)"))
        print(f'{"".ljust(self.category_pad)}{"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        for i in range(len(self.seq_list)):
            print(f'{self.seq_list[i].ljust(self.category_pad)}{"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.charged_sidechain_distances_rc[i]])}')
        print("")

    def _assign_modified_ionization_states(self):
        """
        Self-consistent mean-field ionization in BOTH helix and coil ensembles.

        PATCHES:
        1) Use a fully charged probe (±1) for the TITRATING site when computing deltaG_total
            (environment charges remain fractional).
        2) Include interactions with ALL other ionizable groups (not just idx2 > idx1).
        3) Solve both states:
            - helix: helix distances + macrodipole contributions
            - coil : coil distances, NO macrodipole
            (your previous code copied intrinsic for coil, which breaks ΔG_hel - ΔG_coil logic).
        4) Robustness: treat missing/NaN distances as "far" (99 Å) instead of propagating NaNs.
        """
        MAX_ITERATIONS = 50
        CONVERGENCE_THRESHOLD = 0.005

        ionizable_sidechains = set(self.neg_charge_aa + self.pos_charge_aa)

        def _nterm_present() -> bool:
            return not (len(self.seq_list) > 0 and self.seq_list[0] == "Ac")

        def _cterm_present() -> bool:
            return not (len(self.seq_list) > 0 and self.seq_list[-1] == "Am")

        def _sites():
            """List of titratable sites we solve for."""
            sites = []
            if _nterm_present():
                sites.append(("Nterm", None))
            for idx, AA in enumerate(self.seq_list):
                if AA in ionizable_sidechains:
                    sites.append(("SC", idx))
            if _cterm_present():
                sites.append(("Cterm", None))
            return sites

        def _full_charge_for_site(kind, idx):
            """Charge of the fully ionized state for the *titrating* site."""
            if kind == "Nterm":
                # Succinylated N-term behaves as an acid in your model
                if len(self.seq_list) > 0 and self.seq_list[0] == "Sc":
                    return -1.0
                return +1.0
            if kind == "Cterm":
                return -1.0
            # sidechain
            AA = self.seq_list[idx]
            if AA in self.neg_charge_aa:
                return -1.0
            if AA in self.pos_charge_aa:
                return +1.0
            raise ValueError(f"Unexpected non-ionizable site: {kind}, {idx}, {AA}")

        def _pka_intrinsic(kind, idx):
            if kind == "Nterm":
                return self.nterm_pka
            if kind == "Cterm":
                return self.cterm_pka
            return float(self.seq_pka[idx])

        def _is_basic(kind, idx):
            if kind == "Nterm":
                # Sc is acidic; otherwise N-term is basic
                return False if (len(self.seq_list) > 0 and self.seq_list[0] == "Sc") else True
            if kind == "Cterm":
                return False
            AA = self.seq_list[idx]
            return True if AA in self.pos_charge_aa else False

        def _update_ionization_from_pka(kind, idx, pka):
            """Return new fractional charge for this site."""
            if kind == "Nterm":
                if len(self.seq_list) > 0 and self.seq_list[0] == "Sc":
                    return acidic_residue_ionization(self.pH, pka)
                return basic_residue_ionization(self.pH, pka)
            if kind == "Cterm":
                return acidic_residue_ionization(self.pH, pka)
            AA = self.seq_list[idx]
            if AA in self.neg_charge_aa:
                return acidic_residue_ionization(self.pH, pka)
            return basic_residue_ionization(self.pH, pka)

        def _get_env_charge(kind, idx, seq_q, nterm_q, cterm_q):
            if kind == "Nterm":
                return nterm_q
            if kind == "Cterm":
                return cterm_q
            return seq_q[idx]

        def _pair_distance(kind1, idx1, kind2, idx2, use_helix_distances):
                    """Distance between two ionizable sites for electrostatics (Å)."""
                    # terminal-terminal
                    if kind1 in ("Nterm", "Cterm") and kind2 in ("Nterm", "Cterm"):
                        # N-term at position 0, C-term at position n-1
                        return self._calculate_r(len(self.seq_list) - 1)

                    # --- Terminal-Sidechain Logic ---
                    if (kind1 == "Nterm" and kind2 == "SC") or (kind2 == "Nterm" and kind1 == "SC"):
                        sc_idx = idx2 if kind2 == "SC" else idx1
                        
                        if use_helix_distances:
                            # Use pre-computed HELIX distance
                            return float(self.terminal_sidechain_distances_nterm[sc_idx])
                        else:
                            # Use LINEAR approximation for Random Coil
                            # N = number of residues from N-term (0) to sc_idx
                            # dist = 0.1 + (N + 1) * 2
                            # N = sc_idx
                            return self._calculate_r(sc_idx)

                    if (kind1 == "Cterm" and kind2 == "SC") or (kind2 == "Cterm" and kind1 == "SC"):
                        sc_idx = idx2 if kind2 == "SC" else idx1
                        
                        if use_helix_distances:
                            # Use pre-computed HELIX distance
                            return float(self.terminal_sidechain_distances_cterm[sc_idx])
                        else:
                            # Use LINEAR approximation for Random Coil
                            # N = number of residues from sc_idx to C-term (len-1)
                            # N = (len - 1) - sc_idx
                            return self._calculate_r(len(self.seq_list) - 1 - sc_idx)

                    # --- Sidechain-Sidechain Logic ---
                    if kind1 == "SC" and kind2 == "SC":
                        if use_helix_distances:
                            d = self.sidechain_sidechain_distances_hel[idx1, idx2]
                        else:
                            d = self.charged_sidechain_distances_rc[idx1, idx2]
                        if np.isnan(d):
                            return 99.0
                        return float(d)

                    return 99.0

        def _solve_state(include_dipole: bool, use_helix_distances: bool):
            """Mean-field fixed point solve for one ensemble."""
            seq_q = self.seq_ionization.copy()
            nterm_q = self.nterm_ionization
            cterm_q = self.cterm_ionization

            # If termini are absent (Ac/Am), set them to 0 for safety
            if not _nterm_present():
                nterm_q = 0.0
            if not _cterm_present():
                cterm_q = 0.0

            sites = _sites()

            for _ in range(MAX_ITERATIONS):
                old_seq = seq_q.copy()
                old_n = float(nterm_q)
                old_c = float(cterm_q)

                for kind1, idx1 in sites:
                    q1_full = _full_charge_for_site(kind1, idx1)
                    pka0 = _pka_intrinsic(kind1, idx1)
                    is_basic = _is_basic(kind1, idx1)

                    deltaG_total = 0.0

                    # (1) macrodipole contributions (helix ensemble only)
                    if include_dipole:
                        if kind1 == "Nterm":
                            N_dist = float(self.terminal_macrodipole_distance_nterm)
                            C_dist = 99.0
                        elif kind1 == "Cterm":
                            N_dist = 99.0
                            C_dist = float(self.terminal_macrodipole_distance_cterm)
                        else:
                            N_dist = float(self.sidechain_macrodipole_distances_nterm[idx1])
                            C_dist = float(self.sidechain_macrodipole_distances_cterm[idx1])

                        if N_dist < 40.0:
                            deltaG_total += self._electrostatic_interaction_energy(qi=self.mu_helix, qj=q1_full, r=N_dist, factor_pi=4.0)
                        if C_dist < 40.0:
                            deltaG_total += self._electrostatic_interaction_energy(qi=-self.mu_helix, qj=q1_full, r=C_dist, factor_pi=4.0)

                    # (2) interactions with all other charged groups (environment uses fractional charges)
                    for kind2, idx2 in sites:
                        if kind2 == kind1 and idx2 == idx1:
                            continue

                        q2 = _get_env_charge(kind2, idx2, seq_q, nterm_q, cterm_q)
                        r = _pair_distance(kind1, idx1, kind2, idx2, use_helix_distances)

                        if r < 40.0:
                            deltaG_total += self._electrostatic_interaction_energy(qi=q1_full, qj=q2, r=r)

                    if np.isnan(deltaG_total):
                        raise ValueError("deltaG_total became NaN; check distance tables / assignments.")

                    pka_mod = adjust_pKa(
                        T=self.T_kelvin,
                        pKa_ref=pka0,
                        deltaG=deltaG_total,
                        is_basic=is_basic,
                    )

                    q_new = _update_ionization_from_pka(kind1, idx1, pka_mod)

                    if kind1 == "Nterm":
                        nterm_q = q_new
                    elif kind1 == "Cterm":
                        cterm_q = q_new
                    else:
                        seq_q[idx1] = q_new

                # convergence
                vec_old = np.concatenate([old_seq[~np.isnan(old_seq)], [old_n], [old_c]])
                vec_new = np.concatenate([seq_q[~np.isnan(seq_q)], [nterm_q], [cterm_q]])
                max_change = float(np.max(np.abs(vec_new - vec_old)))
                if max_change < CONVERGENCE_THRESHOLD:
                    break

            return seq_q, float(nterm_q), float(cterm_q)

        # --- Solve helix and coil ensembles ---
        hel_seq, hel_n, hel_c = _solve_state(include_dipole=True,  use_helix_distances=True)
        rc_seq,  rc_n,  rc_c  = _solve_state(include_dipole=False, use_helix_distances=False)

        self.modified_seq_ionization_hel = hel_seq
        self.modified_nterm_ionization_hel = hel_n
        self.modified_cterm_ionization_hel = hel_c

        self.modified_seq_ionization_rc = rc_seq
        self.modified_nterm_ionization_rc = rc_n
        self.modified_cterm_ionization_rc = rc_c

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
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.2f}".ljust(self.value_pad) for q in self.modified_seq_ionization_hel])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.modified_nterm_ionization_hel:.2f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.modified_cterm_ionization_hel:.2f}')
        print("")
        print(self._make_box("Modified ionization states, coil"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.2f}".ljust(self.value_pad) for q in self.modified_seq_ionization_rc])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.modified_nterm_ionization_rc:.2f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.modified_cterm_ionization_rc:.2f}')
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
        self._AROMATIC = {"F", "Y", "W"}
        self._ALIPHATIC = {"A", "V", "L", "I", "M"}  # you can expand if you want (e.g. C)
        self.dCp = -0.0015  # kcal/(mol*K)

    def _dCp_hydroph_kcal(self, aa1: str, aa2: str) -> float:
        """
        Hydrophobic heat capacity increment in kcal/(mol*K), helix-formation direction.
        Implements the Muñoz & Serrano linear form (their Eq. 18) with a separate
        aromatic-aromatic scale.
        """
        # Only apply to hydrophobic pairs
        if not ((aa1 in self._ALIPHATIC or aa1 in self._AROMATIC) and (aa2 in self._ALIPHATIC or aa2 in self._AROMATIC)):
            return 0.0

        T = self.T_kelvin
        Tref = 273.15

        if aa1 in self._AROMATIC and aa2 in self._AROMATIC:
            # aromatic-aromatic (smaller magnitude)
            return (-4.0 + 0.025 * (T - Tref)) / 1000.0  # cal -> kcal
        else:
            # aliphatic-aliphatic or aromatic-aliphatic
            return (-8.0 + 0.05 * (T - Tref)) / 1000.0  # cal -> kcal

    def _entropic_cp_correct(self, dG_ref: float, dCp: float) -> float:
        """
        Corrects a purely entropic free energy for temperature using Muñoz 1995 Eq. (10).
        Assumes dH = 0 at all temperatures (hydrophobic interactions).

        Formula: dG(T) = dG_ref * (T/Tref) - T * dCp * ln(T/Tref)
        """
        T = self.T_kelvin
        Tref = 273.15  # 0°C reference temperature

        # Entropic scaling of reference energy
        scaled_ref = dG_ref * (T / Tref)

        # Cp contribution to entropy: -T * dCp * ln(T/Tref)
        cp_term = -T * dCp * np.log(T / Tref)

        return scaled_ref + cp_term

    def _nterm_is_local(self) -> bool:
        """
        Check if the N-terminal is local to the helix.
        """
        # N-terminus is at index 0; helix starts at ncap_idx
        return self.ncap_idx <= 1   # helix starts at 0 (terminal in helix) or 1 (terminal is N')

    def _cterm_is_local(self) -> bool:
        """
        Check if the C-terminal is local to the helix.
        """
        # C-terminus is at index len-1; helix ends at ccap_idx
        return self.ccap_idx >= (len(self.seq_list) - 2)  # helix ends at last or second-last (terminal is C')

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
        T = self.T_kelvin
        Tref = 273.15  # 0°C reference temperature
        dCp = self.dCp

        # Iterate over the helix and get the intrinsic energy for each residue, 
        # not including residues that are capping for the helical segment
        for idx in self.helix_indices:
            if idx == self.ncap_idx or idx == self.ccap_idx:
                continue

            AA = self.seq_list[idx]

            # Distance from N-cap and C-cap
            n_dist = idx - self.ncap_idx  # 1=N1, 2=N2, ...
            c_dist = self.ccap_idx - idx  # 1=C1, 2=C2, ...

            # Use C-terminal propensities for positions C1, C2, C3 (within 3 of Ccap)
            # Otherwise use N-terminal propensities
            if c_dist <= 3 and f"C{c_dist}" in self.table_1_lacroix.columns:
                col = f"C{c_dist}"
                val = self.table_1_lacroix.loc[AA, col]
                if not np.isnan(val):
                    energy[idx] = val
                else:
                    # Fallback to N-terminal value if C-terminal not available
                    energy[idx] = self.table_1_lacroix.loc[AA, "Ncen"]
            elif n_dist == 1:
                energy[idx] = self.table_1_lacroix.loc[AA, "N1"]
            elif n_dist == 2:
                energy[idx] = self.table_1_lacroix.loc[AA, "N2"]
            elif n_dist == 3:
                energy[idx] = self.table_1_lacroix.loc[AA, "N3"]
            elif n_dist == 4:
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

        # Apply Muñoz & Serrano (1995) temperature correction.
        # Intrinsic energies are purely entropic: dH_ref = 0, dG_ref = -Tref * dS_ref.
        # Full Kirchhoff correction for purely entropic term:
        #   dG(T) = dG_ref * (T/Tref) + dCp * [(T - Tref) - T * ln(T/Tref)]
        # The dCp correction is applied here (once per helix residue) and NOT in get_dG_Hbond
        # to avoid double-counting. Capping also gets its own dCp via _apply_temp_correction_hbond_like.

        # 1. Scale reference energy (entropic part)
        scaled_ref = energy * (T / Tref)

        # 2. Full Kirchhoff Cp correction per residue
        cp_term = dCp * ((T - Tref) - T * np.log(T / Tref))

        # Apply to all non-zero entries (avoid adding energy to caps/zeros)
        energy = np.where(energy != 0, scaled_ref + cp_term, energy)

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

        # H-bond enthalpy: -0.895 kcal/mol per bond (Lacroix 1998 discussion).
        # No Cp correction here — the per-residue dCp is applied in get_dG_Int
        # (for helix body) and _apply_temp_correction_hbond_like (for caps).
        Href = -0.898  # kcal/mol

        return Href * n_hbonds

    def _apply_temp_correction_hbond_like(self, dG_ref_values: np.ndarray) -> np.ndarray:
            """
            Applies the Gibbs-Helmholtz heat capacity correction to energies
            that are primarily enthalpic/H-bond based (like capping).
            """
            Tref = 273.15
            
            # Calculate the Enthalpy at temperature T
            # Assuming the table value dG_ref is effectively dH_ref at Tref (since dS_ref ~ 0 for H-bonds)
            delta_H = dG_ref_values + self.dCp * (self.T_kelvin - Tref)
            
            # Calculate the Entropic cost due to Heat Capacity
            delta_S_Cp = self.dCp * np.log(self.T_kelvin / Tref)
            
            # Final dG = dH - T * dS_Cp
            dG_corrected = delta_H - (self.T_kelvin * delta_S_Cp)
            
            return dG_corrected

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

        return self._apply_temp_correction_hbond_like(energy)

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

        return self._apply_temp_correction_hbond_like(energy)

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
            energy = self.table_2_lacroix.loc[self.Ncap_AA, self.N4_AA] / 100

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

        # Apply hydrophobic temperature correction to the N'–N4 interaction
        # N' is the residue BEFORE Ncap
        if energy != 0.0 and self.ncap_idx > 0:
            Nprime = self.seq_list[self.ncap_idx - 1]
            N4 = self.N4_AA

            dCp_h = self._dCp_hydroph_kcal(Nprime, N4)
            if dCp_h != 0.0:
                energy = self._entropic_cp_correct(energy, dCp_h)

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

    def get_dG_petukhov_motif(self) -> float:
        """
        Get the free energy contribution for the Petukhov combination motif.

        From Lacroix 1998 (p.175): "free N terminus, capping box motif and an Asp or a Glu
        at position N4. The stabilization is due to a strong interaction between residue N4,
        the N-capping residue and the charged N-terminal group." Contributes -1 kcal/mol.

        Requirements:
        - Free (unblocked) N-terminus
        - Helix starts at the peptide N-terminus (Ncap is position 0)
        - Ncap is a capping box residue (Asp, Asn, Ser, Thr)
        - N3 is Glu, Asp, or Gln (the capping box H-bond partner)
        - N4 is Asp or Glu (charged)

        Returns:
            float: The free energy contribution.
        """
        # Must have a free N-terminus (no Ac/Sc cap)
        if self.ncap is not None:
            return 0.0

        # Helix must start at the peptide N-terminus
        if self.ncap_idx != 0:
            return 0.0

        # Ncap must be a capping box residue
        if self.Ncap_AA not in ["D", "N", "S", "T"]:
            return 0.0

        # N3 must be Glu, Asp, or Gln (capping box partner)
        if self.N3_AA not in ["E", "D", "Q"]:
            return 0.0

        # N4 must be Asp or Glu
        if self.N4_AA not in ["D", "E"]:
            return 0.0

        # Apply the motif energy, scaled by the ionization of N4
        q_N4 = abs(self.modified_seq_ionization_hel[self.ncap_idx + 4])
        energy = -1.0 * q_N4

        # Apply H-bond-like temperature correction
        if energy != 0.0:
            Tref = 273.15
            delta_H = energy + self.dCp * (self.T_kelvin - Tref)
            delta_S_Cp = self.dCp * np.log(self.T_kelvin / Tref)
            energy = delta_H - (self.T_kelvin * delta_S_Cp)

        return energy

    def get_dG_charged_staple(self) -> float:
        """
        Get the free energy contribution for the charged staple variant.

        From Lacroix 1998 (p.175): When Ser or Thr is at N-cap, the carbonyl of N0 points
        toward N4. If K or R is at N4, it can form a hydrogen bond with that carbonyl.
        Value: -0.3 kcal/mol.

        Requirements:
        - Ncap is Ser or Thr
        - N4 is Lys or Arg
        - There is an N' residue before Ncap (ncap_idx > 0)

        Returns:
            float: The free energy contribution.
        """
        # Need an N' residue before the Ncap
        if self.ncap_idx == 0:
            return 0.0

        # Ncap must be Ser or Thr
        if self.Ncap_AA not in ["S", "T"]:
            return 0.0

        # N4 must be Lys or Arg
        if self.N4_AA not in ["K", "R"]:
            return 0.0

        # Apply energy, scaled by ionization of N4
        q_N4 = abs(self.modified_seq_ionization_hel[self.ncap_idx + 4])
        energy = -0.3 * q_N4

        # Apply H-bond-like temperature correction
        if energy != 0.0:
            Tref = 273.15
            delta_H = energy + self.dCp * (self.T_kelvin - Tref)
            delta_S_Cp = self.dCp * np.log(self.T_kelvin / Tref)
            energy = delta_H - (self.T_kelvin * delta_S_Cp)

        return energy

    def get_dG_i3(self) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

        - Table IV corresponds to non-charged interactions.
        - If BOTH residues are ionizable, Table IV applies only to cases where
        at least one is not charged; scale by (1 - p_i * p_j).
        - No abs(q_i*q_j) scaling here; electrostatics handled elsewhere.

        Side chain-side chain interaction between AAi and AAi+3 (Table IVa, non-charged term).
        For pairs where BOTH residues are titratable, suppress the Table-IV contribution
        in the both-charged microstate: multiply by (1 - p_i * p_j), where p = |q|.

        Returns:
            np.ndarray: The free energy contributions for each interaction.
        """
        energy = np.zeros(len(self.seq_list))

        for idx in self.helix_indices[:-3]:
            AAi = self.seq_list[idx]
            AAi3 = self.seq_list[idx + 3]

            # Skip terminal modifications
            if AAi in ["Ac", "Am", "Sc"] or AAi3 in ["Ac", "Am", "Sc"]:
                continue

            # Table IV values are "kcal/mol * 100" -> convert to kcal/mol
            base = self.table_4a_lacroix.loc[AAi, AAi3] / 100.0

            # If both are titratable, Table IV is intended for "not both charged" states.
            if (AAi in (self.pos_charge_aa + self.neg_charge_aa)) and (AAi3 in (self.pos_charge_aa + self.neg_charge_aa)):
                p_i = abs(self.modified_seq_ionization_hel[idx])
                p_j = abs(self.modified_seq_ionization_hel[idx + 3])
                base = base * (1.0 - p_i * p_j)

            # Calculate dCp for this pair (returns 0.0 if not hydrophobic)
            dCp_val = self._dCp_hydroph_kcal(AAi, AAi3)
            
            # Apply correction if dCp is non-zero and base energy implies interaction exists
            if dCp_val != 0.0 and base != 0.0:
                base = self._entropic_cp_correct(base, dCp_val)

            energy[idx] = base

        return energy
    
    def get_dG_i4(self) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

        i,i+4 non-charged side chain interactions (Table IVb) + special pH-dependent,
        salt-independent local motifs (Table V), Lacroix supplement.

        Faithful interpretation:
        - Table IV corresponds to non-charged interactions.
        - If BOTH residues are ionizable, Table IV applies only to cases where
            at least one is not charged; scale by (1 - p_i * p_j).
        - Table V terms are ADDED when the relevant residue is charged (weighted by p).
        - No abs(q_i*q_j) scaling here; electrostatics handled elsewhere.

        - Table IVb is a non-charged (or "not both charged") interaction term.
        If BOTH residues are titratable: multiply by (1 - p_i * p_j).

        - Table V: additional interaction energy to ADD when one residue becomes charged
        (pH dependent, NOT affected by ionic strength). Scale by the population of the
        charged form of the relevant residue.
        """
        energy = np.zeros(len(self.seq_list))

        for idx in self.helix_indices[:-4]:
            AAi = self.seq_list[idx]
            AAi4 = self.seq_list[idx + 4]

            # Skip terminal modifications
            if AAi in ["Ac", "Am", "Sc"] or AAi4 in ["Ac", "Am", "Sc"]:
                continue

            # Table IV values are "kcal/mol * 100" -> convert to kcal/mol
            base = self.table_4b_lacroix.loc[AAi, AAi4] / 100.0

            # Suppress Table IV in the both-charged microstate if both residues are titratable
            if (AAi in (self.pos_charge_aa + self.neg_charge_aa)) and (AAi4 in (self.pos_charge_aa + self.neg_charge_aa)):
                p_i = abs(self.modified_seq_ionization_hel[idx])
                p_j = abs(self.modified_seq_ionization_hel[idx + 4])
                base = base * (1.0 - p_i * p_j)

            # Calculate dCp for this pair (returns 0.0 if not hydrophobic)
            dCp_val = self._dCp_hydroph_kcal(AAi, AAi4)
            
            # Apply correction if dCp is non-zero and base energy implies interaction exists
            if dCp_val != 0.0 and base != 0.0:
                base = self._entropic_cp_correct(base, dCp_val)

            extra = 0.0

            # --- Table V add-ons (orientation matters: position i -> position i+4) ---
            # FYW (i) with His+ (i+4): -0.4 kcal/mol when His is at C1 or C-cap; otherwise divide by 3
            if AAi in ["F", "Y", "W"] and AAi4 == "H":
                p_his = abs(self.modified_seq_ionization_hel[idx + 4])  # population of His+
                # His is "C1" if it is the residue just before C-cap; "C-cap" if it is C-cap itself
                his_is_C1_or_Ccap = (idx + 4 == self.ccap_idx) or (idx + 4 == self.ccap_idx - 1)
                val = -0.4 if his_is_C1_or_Ccap else (-0.4 / 3.0)
                extra += p_his * val

            # Gln (i) with Asp- (i+4): -0.5 kcal/mol
            if AAi == "Q" and AAi4 == "D":
                p_asp = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Asp-
                extra += p_asp * (-0.5)

            # Glu- (i) with Asn (i+4): -0.5 kcal/mol
            if AAi == "E" and AAi4 == "N":
                p_glu = abs(self.modified_seq_ionization_hel[idx])  # population of Glu-
                extra += p_glu * (-0.5)

            # Gln (i) with Glu- (i+4): -0.1 kcal/mol
            if AAi == "Q" and AAi4 == "E":
                p_glu = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Glu-
                extra += p_glu * (-0.1)

            # These are side-chain H-bonds/Salt bridges, they should weaken with T
            if extra != 0.0:
                Tref = 273.15
                delta_H = extra + self.dCp * (self.T_kelvin - Tref)
                delta_S_Cp = self.dCp * np.log(self.T_kelvin / Tref)
                extra = delta_H - (self.T_kelvin * delta_S_Cp)

            energy[idx] = base + extra

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
            r=self.terminal_macrodipole_distance_nterm,
            factor_pi=4.0 
        )

        # Calculate the interaction energy between the C-terminal and the helix macrodipole
        C_term[self.ccap_idx] = self._electrostatic_interaction_energy(
            qi=-self.mu_helix,
            qj=self.modified_cterm_ionization_hel,
            r=self.terminal_macrodipole_distance_cterm,
            factor_pi=4.0 
        )

        return N_term, C_term

    def get_dG_sidechain_macrodipole(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the interaction energy between charged side-chains and the helix macrodipole
        using empirical Table 3 values from Muñoz & Serrano 1995 II.

        Each charged residue gets two contributions:
        - N-term: interaction with positive N-terminal half of macrodipole (Table 3 N-term)
        - C-term: interaction with negative C-terminal half of macrodipole (Table 3 C-term)

        Each contribution is screened by Debye-Hückel using distances from Table VII (Lacroix 1998).
        Positions beyond N9/C9 in Table 3 are assumed to have zero contribution.

        Charged residues both inside and flanking the helix contribute.

        Returns:
            tuple[np.ndarray, np.ndarray]: N-terminal and C-terminal dipole energy arrays.
        """
        n = len(self.seq_list)
        energy_N = np.zeros(n, dtype=float)
        energy_C = np.zeros(n, dtype=float)

        # Amino acids with empirical macrodipole values in Table 3
        table3_aas = set(self.table_3_munoz_nterm.index)  # D, E, H, K, R
        charged = set(self.neg_charge_aa + self.pos_charge_aa)

        def _lookup_and_screen(aa, pos, table3, table7):
            """Look up Table 3 empirical value and apply Debye screening from Table VII."""
            if pos < 0 or pos > 9 or aa not in table3_aas:
                return 0.0
            col = table3.columns[pos]
            raw = float(table3.loc[aa, col])
            # Debye-Hückel screening using Table VII distance
            if pos <= 13 and aa in table7.index:
                d7_col = table7.columns[pos]
                d = float(table7.loc[aa, d7_col])
                if not np.isnan(d):
                    raw *= math.exp(-self.kappa * d * 1e-10)
            return raw

        # Scan charged residues both inside the helix and flanking.
        # Flanking residue contributions are assigned to the nearest cap.
        scan_start = max(0, self.ncap_idx - 9)
        scan_end = min(n - 1, self.ccap_idx + 9)

        for idx in range(scan_start, scan_end + 1):
            aa = self.seq_list[idx]
            if aa not in charged:
                continue

            q = float(self.modified_seq_ionization_hel[idx])
            if abs(q) < 1e-6:
                continue

            n_pos = idx - self.ncap_idx  # 0=Ncap, negative=N-flanking
            c_pos = self.ccap_idx - idx  # 0=Ccap, negative=C-flanking

            contrib_n = _lookup_and_screen(aa, n_pos, self.table_3_munoz_nterm, self.table_7_ncap_lacroix)
            contrib_c = _lookup_and_screen(aa, c_pos, self.table_3_munoz_cterm, self.table_7_ccap_lacroix)

            contrib_n *= abs(q)
            contrib_c *= abs(q)

            # Assign energy: in-helix residues get it on their own position;
            # flanking residues get it assigned to the nearest cap position.
            if idx < self.ncap_idx:
                assign_idx = self.ncap_idx
            elif idx > self.ccap_idx:
                assign_idx = self.ccap_idx
            else:
                assign_idx = idx

            energy_N[assign_idx] += contrib_n
            energy_C[assign_idx] += contrib_c

        return energy_N, energy_C
        
    def get_dG_terminals_sidechain_electrost(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate electrostatic interaction energies between terminal backbone charges
        and charged sidechains in the sequence. The energy is added to the charged sidechain residue.

        Uses ionization-adjusted sidechain charges (from seq_ionization) so that
        residues with pKa far from pH contribute proportionally (e.g. Y at pH 4
        is uncharged → zero interaction).  Terminal charges stay at full ±1.0 to
        avoid double-counting with the pKa solver's macrodipole correction.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays containing the interaction energies for
            N-terminal and C-terminal interactions respectively. The energy is added
            to the charged sidechain residue.
        """
        n = len(self.seq_list)
        energy_N = np.zeros(n, dtype=float)
        energy_C = np.zeros(n, dtype=float)

        # Presence (Ac/Am remove terminal charges in this model)
        nterm_present = not (n > 0 and self.seq_list[0] == "Ac")
        cterm_present = not (n > 0 and self.seq_list[-1] == "Am")

        # Locality gate (AGADIR/Lacroix-style): only include terminal-sidechain terms
        # when the terminal is in-helix or is the immediate neighbor (N' / C').
        nterm_local = (self.ncap_idx <= 1)
        cterm_local = (self.ccap_idx >= (n - 2))

        # If neither terminal can contribute, bail early
        if not (nterm_present and nterm_local) and not (cterm_present and cterm_local):
            return energy_N, energy_C

        # Iterate through sequence checking for charged sidechains
        for idx, AA1 in enumerate(self.seq_list):
            if AA1 not in self.neg_charge_aa + self.pos_charge_aa:
                continue

            q_sc = float(self.seq_ionization[idx])
            if q_sc == 0.0:
                continue

            # --- N-Terminal Interaction (only if local/present) ---
            if nterm_present and nterm_local:
                q_nterm_full = +1.0  # NH3+
                dist_hel = float(self.terminal_sidechain_distances_nterm[idx])
                if np.isnan(dist_hel):
                    dist_hel = 99.0

                G_hel = (
                    self._electrostatic_interaction_energy(qi=q_nterm_full, qj=q_sc, r=dist_hel, factor_pi=3.0)
                    if dist_hel < 40.0 else 0.0
                )

                # RC distance: N = idx residues between N-terminus and residue idx
                dist_rc = float(self._calculate_r(idx))
                G_rc = (
                    self._electrostatic_interaction_energy(qi=q_nterm_full, qj=q_sc, r=dist_rc, factor_pi=3.0)
                    if dist_rc < 40.0 else 0.0
                )

                energy_N[idx] = G_hel - G_rc

            # --- C-Terminal Interaction (only if local/present) ---
            if cterm_present and cterm_local:
                q_cterm_full = -1.0  # COO-

                # When the C-terminal is NOT at the helix cap (ccap_idx < n-1),
                # it sits in the coil region beyond the helix end.  Table 7
                # distances measure from the ccap, not the terminal, so they
                # give a spuriously compact distance.  Use the coil model instead,
                # yielding ΔG ≈ 0 (matches reference: -0.003 vs our old -0.178).
                if self.ccap_idx == n - 1:
                    dist_hel = float(self.terminal_sidechain_distances_cterm[idx])
                    if np.isnan(dist_hel):
                        dist_hel = 99.0
                else:
                    dist_hel = float(self._calculate_r((n - 1) - idx))

                G_hel = (
                    self._electrostatic_interaction_energy(qi=q_cterm_full, qj=q_sc, r=dist_hel, factor_pi=3.0)
                    if dist_hel < 40.0 else 0.0
                )

                # RC distance: N = (n-1-idx) residues between residue idx and C-terminus
                dist_rc = float(self._calculate_r((n - 1) - idx))
                G_rc = (
                    self._electrostatic_interaction_energy(qi=q_cterm_full, qj=q_sc, r=dist_rc, factor_pi=3.0)
                    if dist_rc < 40.0 else 0.0
                )
                energy_C[idx] = G_hel - G_rc

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
            if idx2 <= idx1:
                continue

            # Get the distances between the charged sidechains
            helix_dist = self.sidechain_sidechain_distances_hel[idx1, idx2]
            coil_dist = self.charged_sidechain_distances_rc[idx2, idx1]
                
            # Get the ionization states of the charged sidechains
            q1_hel = self.modified_seq_ionization_hel[idx1]
            q2_hel = self.modified_seq_ionization_hel[idx2]
            q1_rc = self.modified_seq_ionization_rc[idx1]
            q2_rc = self.modified_seq_ionization_rc[idx2]

            # Calculate electrostatic interaction energies with adjusted ionization states, Lacroix Eq 6.
            # factor_pi=3.5: calibrated from YGGS reference output (Ccap and interior positions).
            # The paper prints "3π" but 3.5π matches the reference tool exactly.
            G_hel = self._electrostatic_interaction_energy(qi=q1_hel, qj=q2_hel, r=helix_dist, factor_pi=3.5)
            G_rc = self._electrostatic_interaction_energy(qi=q1_rc, qj=q2_rc, r=coil_dist, factor_pi=3.5)

            # Store half the energy difference in both triangles of the matrix (give half of the energy to each sidechain)
            energy_diff = (G_hel - G_rc) / 2
            energy_matrix[idx1, idx2] = energy_diff  # Upper triangle
            energy_matrix[idx2, idx1] = energy_diff  # Lower triangle

        return energy_matrix
    