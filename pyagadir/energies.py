import itertools
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import acidic_residue_ionization, basic_residue_ionization, calculate_permittivity, debye_screening_kappa
from pyagadir.electrostatics import DIPOLE_LAWS, GB_LAWS, GBX_LAWS, Electrostatics  # noqa: F401
from pyagadir.utils import is_valid_peptide_sequence, is_valid_ncap_ccap, is_valid_conditions

# The electrostatics are in pyagadir/electrostatics/: point.py (the default point-charge law), gb.py (the Generalized
# Born laws) and common.py (the law switch and what the laws share). Their public names stay importable from here.
from pyagadir.electrostatics.gb import (  # noqa: F401
    GB_BOTH_CAPS, GB_DESOLV_RANGE, GB_DIPOLE_CHARGE, GB_DIPOLE_FORM, GB_HIS_TAUTOMER, GB_NEAR_CAP, GB_NEUTRAL_FORMS,
    GB_PAIR_CHARGE, GB_PAIR_KMAX, GB_PAIR_NEUTRAL, GBP_FILES, GBX_CAPPING_BOX_NCAPS, gb_dipole_table,
    gb_pair_grid_value, gbp_reset, gbp_tables, gbx_position_class, gbx_tables, law_tables, sb_network_free_energy)
from pyagadir.electrostatics.point import FEH_FAR_APBS, SUCCINYL_HELIX_DISTANCE  # noqa: F401


class ParamTable:
    """
    Read-only, dict-backed copy of a parameter DataFrame.

    Scalar DataFrame.loc lookups cost ~10 µs each, and a prediction makes tens of
    thousands of them (one EnergyCalculator per helical segment), which made them
    ~70% of the run time. This keeps the same spelling -- table.loc[row, col],
    table.loc[row][col], `row in table.index`, table.columns[k] -- at dict speed.

    It is a snapshot: edits to the source DataFrame are only seen after rebuilding.
    """

    def __init__(self, df: pd.DataFrame):
        self._rows = df.to_dict(orient="index")
        self.index = tuple(df.index)
        self.columns = tuple(df.columns)

    @property
    def loc(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self._rows[row][col]
        return self._rows[key]


# First-turn offset (kcal/mol at 273 K) added to the Table 1 N2 cell of the residues whose first-turn cells were never
# measured in peptides (Lacroix 1998 took the N1-N4 cells from Petukhov's poly-Ala measurements; K, R and H were left
# at their interior value, C, F, Y and W at round numbers). Lys and Arg take the measured peptide value for Leu at N2
# (Petukhov et al. 1998, J. Mol. Biol. 278, 279: +0.25 above the interior), which the peptides with Lys at N2 support
# when each is compared with the other peptides of its own host. H, C, F, Y and W take the N2-minus-interior offsets of
# the single-mutant scan of Tsuboyama et al. 2023 (Nature 620, 434; exposed helix sites, relative to Ala), scaled by
# 0.943 so that Leu's protein offset equals its peptide value. Charge-independent. See params/README.md.
FIRST_TURN_N2_OFFSET = {"K": 0.25, "R": 0.25, "H": -0.059, "C": 0.228, "F": 0.162, "Y": 0.157, "W": 0.141}

# Hydrogen-bond terms share one temperature law (EnergyCalculator._hbond_temperature): Munoz & Serrano 1995-III
# eq. (8) for the backbone bond, dH_ref = -0.898 kcal/mol per bond at 273 K (Lacroix 1998). Side chains and end groups
# whose capping contribution is a hydrogen bond (Munoz & Serrano 1995-III p. 305), and residues that can form
# side-chain hydrogen bonds in the Table IV/V pair terms:
HBOND_DH_REF = -0.898
HBOND_NCAPS = ("S", "T", "N", "D", "Ac", "Sc")
HBOND_CCAPS = ("K", "R", "H", "N", "D")
HBOND_POLAR = ("S", "T", "N", "Q", "D", "E", "K", "R", "H")


class PrecomputeParams(Electrostatics):
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
            datapath = files("pyagadir.params")

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

            # Coulomb distances for sidechain-macrodipole (differ from Table7 for K/R)
            cls._params["table_7_coulomb_ncap"] = pd.read_csv(
                datapath.joinpath("table_7_coulomb_Ncap.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            cls._params["table_7_coulomb_ccap"] = pd.read_csv(
                datapath.joinpath("table_7_coulomb_Ccap.tsv"),
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

    @classmethod
    def snapshot_params(cls, dipole_law: str = "munoz") -> dict:
        """
        Fast ParamTable copies of load_params(), for passing as params= to many
        instances. Rebuilt on every call (~10 ms), so take one per prediction:
        edits made to the _params DataFrames between predictions are then honoured.
        """
        return {name: ParamTable(df) for name, df in cls.law_params(dipole_law).items()}

    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, debug: bool = False, params: dict = None, dipole_law: str = "munoz"):
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
            params (dict): Parameter tables from snapshot_params(). If None, load_params() is used.
            dipole_law (str): side chain-macrodipole law, "munoz" (default), "gb", "gbx" or "gbp" (see
                _sidechain_dipole_potential; "gbp" also replaces the side chain-side chain pair law).
        """
        self.debug = debug
        if dipole_law not in DIPOLE_LAWS:
            raise ValueError(f"dipole_law must be one of {DIPOLE_LAWS}, got {dipole_law!r}")
        self.dipole_law = dipole_law

        # load params
        if params is None:
            params = self.law_params(dipole_law) if dipole_law in GBX_LAWS else self.load_params()
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
        self.table_7_coulomb_ncap = params["table_7_coulomb_ncap"]
        self.table_7_coulomb_ccap = params["table_7_coulomb_ccap"]
        self.table_pka_values = params["pka_values"]

        is_valid_peptide_sequence(seq)
        is_valid_ncap_ccap(ncap, ccap)
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
        self._assign_sidechain_dipole_potentials()
        self._assign_terminal_sidechain_distances()
        self._assign_sidechain_sidechain_distances()
        self._assign_gb_pairs()
        self._assign_modified_ionization_states()

    def _find_charged_pairs(self) -> list[tuple[str, int, int]]:
        """
        Find all pairs of charged residues in a sequence and their global positions.
        """
        charged_amino_acids = self.neg_charge_aa + self.pos_charge_aa

        # Iterate over all pairs of charged residues (pairing only the charged
        # positions; scanning every residue pair was costly, as it runs per segment)
        charged_idx = [idx for idx, aa in enumerate(self.seq_list) if aa in charged_amino_acids]
        self.charged_pairs = [
            (self.seq_list[idx1], self.seq_list[idx2], idx1, idx2)  # Include global positions
            for idx1, idx2 in itertools.combinations(charged_idx, 2)
        ]

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
        # the alpha-amino pKa depends on the N-terminal residue; a residue-specific row
        # ("Nterm_Y") is used where one has been measured (params/README.md)
        if self.ncap is None and f"Nterm_{self.seq_list[0]}" in self.table_pka_values.index:
            self.nterm_pka = float(self.table_pka_values.loc[f"Nterm_{self.seq_list[0]}"]["pKa"])
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
        self.show_sidechain_dipole_potentials()
        self.show_terminal_sidechain_distances()
        self.show_sidechain_sidechain_distances()


class EnergyCalculator(PrecomputeParams):
    """
    Class to calculate the free energy contributions for a peptide sequence.
    """
    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, params: dict = None, dipole_law: str = "munoz"):
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
            params (dict): Parameter tables from snapshot_params(). If None, load_params() is used.
            dipole_law (str): side chain-macrodipole law, "munoz" (default) or "gb".
        """
        super().__init__(seq, i, j, pH, T, ionic_strength, ncap, ccap, params=params, dipole_law=dipole_law)
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

    def get_dG_Int(self) -> np.ndarray:
        """
        Get the intrinsic free energy contributions for a helical segment.
        This accounts for the loss of entropy due to the helix formation.
        The first and last residues are considered to be caps unless they are
        the peptide terminal residues with modifications.
        Temperature: equation (9) of Muñoz & Serrano (1995-III), a purely entropic term,
        dG(t) = dG_ref * t/t_ref - t * dCp * ln(t/t_ref) with t_ref = 273 K.

        Returns:
            np.ndarray: The intrinsic free energy contributions for each amino acid in the helical segment.
        """
        # Initialize energy array
        energy = np.zeros(len(self.seq_list))
        T = self.T_kelvin
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

            if AA in self.pos_charge_aa + ["D", "E"]:
                # Ionization correction for residues that are normally charged at pH 7.
                # K, R, H (pos_charge_aa) have pKa > 6 and are charged at neutral pH.
                # D, E have pKa ~4 and are charged (negative) at neutral pH.
                # Table 1 position-specific values represent the charged form for these.
                # When deionized (extreme pH), interpolate toward the Neutral column.
                # Y and C are excluded: their pKa (~10.1, ~8.3) means they are normally
                # neutral at pH 7, so Table 1 values already represent the neutral form.
                q = abs(self.modified_seq_ionization_hel[idx])
                basic_energy = energy[idx]
                basic_energy_neutral = self.table_1_lacroix.loc[AA, "Neutral"]
                energy[idx] = q * basic_energy + (1 - q) * basic_energy_neutral

            # First-turn N2 offset (FIRST_TURN_N2_OFFSET), after the charged/neutral interpolation so that both
            # charge states pay it; not where the C1-C3 columns apply.
            if n_dist == 2 and c_dist > 3:
                energy[idx] += self._first_turn_n2_offset(AA)

            # gbx/gbp: first- and last-turn cells of charged residues re-derived under the GB law
            # (params/gbx_overrides.tsv, params/gbp_overrides.tsv)
            if self.dipole_law in GBX_LAWS:
                energy[idx] += self._gbx_end_cell(AA, n_dist, c_dist)

        # Munoz & Serrano 1995-III eq. (9): dG_Int = -t (dS_ref + dCp ln(t/t_ref)),
        # i.e. dG_ref * t/t_ref - t * dCp * ln(t/t_ref).  The dCp*(t - t_ref) enthalpy term
        # belongs to the H-bond (eq. 8), not here.  t_ref = 273.0 K.
        Tref = 273.0
        scaled_ref = energy * (T / Tref)
        cp_term = -T * dCp * np.log(T / Tref)

        # Apply to all non-zero entries (avoid adding energy to caps/zeros)
        energy = np.where(energy != 0, scaled_ref + cp_term, energy)

        return energy
    
    def _first_turn_n2_offset(self, aa: str) -> float:
        """FIRST_TURN_N2_OFFSET, with the gbx/gbp law's re-derived values (params/gbx_overrides.tsv or
        params/gbp_overrides.tsv, table "first_turn_n2") where it has them."""
        if self.dipole_law in GBX_LAWS:
            value = self._gbx_first_turn_n2_offset(aa)
            if value is not None:
                return value
        return FIRST_TURN_N2_OFFSET.get(aa, 0.0)

    def get_dG_Hbond(self) -> float:
        """
        Get the free energy contribution for hydrogen bonding for a sequence.

        Capping residues, don't count toward hydrogen bonding, which gives 2.
        Additionally, the first 4 helical amino acids are considered to have
        zero net enthalpy since they are nucleating residues.
        This gives a total of 6 residues that don't count toward hydrogen bonding.

        The net H-bond ΔG at 0°C is -0.898 kcal/mol per bond (Lacroix 1998). It is
        treated as enthalpic, with the temperature dependence of equation (8) of
        Muñoz & Serrano (1995-III): ΔG(t) = ΔH_ref + ΔCp (t - t_ref), ΔCp = -1.5
        cal/(mol·K) in the folding direction, t_ref = 273 K (_hbond_temperature, the
        law every hydrogen-bond term uses).

        Returns:
            float: The total free energy contribution for hydrogen bonding in the sequence.
        """
        n_hbonds = max((self.j - 6), 0)

        # Munoz & Serrano 1995-III eq. (8): dG_HBond = dH_ref + dCp (t - t_ref) per bond,
        # with dH_ref = -0.898 (Lacroix 1998), dCp = -0.0015 kcal/(mol K), t_ref = 273.0 K.
        dG_per_bond = self._hbond_temperature(HBOND_DH_REF)
        return dG_per_bond * n_hbonds

    def _hbond_temperature(self, dG_ref):
        """
        The temperature law of every hydrogen-bond term: the backbone bond (get_dG_Hbond), the
        hydrogen-bonding N- and C-caps and end groups (get_dG_Ncap, get_dG_Ccap), the side-chain
        hydrogen bonds of the i,i+3 and i,i+4 terms (Table IV acid-base and polar pairs, the
        Table V motifs) and the Petukhov and charged-staple motifs.

        Muñoz & Serrano (1995-III) eq. (8) treats a hydrogen-bond term as an enthalpy,
        ΔG(t) = ΔH_ref + ΔCp (t - t_ref), with ΔCp = -1.5 cal/(mol·K) per backbone bond in the
        folding direction and t_ref = 273 K; the entropy of fixing the backbone is in
        get_dG_Int. Other hydrogen bonds take the same law as a relative change,
        ΔG_ref × (1 + ΔCp (t - t_ref) / ΔH_ref) with the backbone bond's ΔH_ref = -0.898
        kcal/mol, which for the backbone bond is eq. (8) itself. Muñoz & Serrano apply it to the
        hydrogen-bonding caps and the end groups (p. 305) and neglect it for side-chain pairs.
        The term grows more favourable with temperature, as the backbone term does.

        Args:
            dG_ref: free energy at 273 K (float or array), kcal/mol.
        """
        return dG_ref * (1.0 + self.dCp * (self.T_kelvin - 273.0) / HBOND_DH_REF)

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
            col = "Nc-4"

        # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
        elif self.N3_AA in ["E", "D", "Q"]:
            col = "Nc-3"

        # Nc-2 	N-cap values when there is a Pro at position N1.
        elif self.N1_AA == "P":
            col = "Nc-2"

        # Nc-1 	Normal N-cap values.
        else:
            col = "Nc-1"
        energy[self.ncap_idx] = self.table_1_lacroix.loc[self.Ncap_AA, col]

        # Lacroix 1998 (G_nonH): the N-capping contribution of Cys is 1 kcal/mol more
        # favourable when it is charged, and that of His 1 kcal/mol more favourable when it
        # is neutral.  Table I holds the neutral forms; weighted by helix-state ionisation.
        if self.Ncap_AA in ("C", "H"):
            q_cap = abs(float(self.modified_seq_ionization_hel[self.ncap_idx]))
            energy[self.ncap_idx] += (-1.0 if self.Ncap_AA == "C" else 1.0) * q_cap

        # Capping box (Harper & Rose 1993): a Ser, Thr, Asp or Asn N-cap and a Glu at N3 form
        # reciprocal side chain-backbone hydrogen bonds. Peptide measurements put it at -0.9
        # kcal/mol beyond the Nc-3 column: Glu vs Ala, Gln and Asp at N3 and Ser vs Ala at the
        # N-cap (Zhou et al. 1994, Proteins 18, 1; Petukhov et al. 1996, Biochemistry 35, 387).
        # Glu beats Gln there by as much as it beats Ala, so the bonus belongs to the charged Glu
        # and is weighted by that Glu's helix-state ionisation.
        if self.Ncap_AA in ("S", "T", "D", "N") and self.N3_AA == "E":
            q_glu = abs(float(self.modified_seq_ionization_hel[self.ncap_idx + 3]))
            energy[self.ncap_idx] += -0.9 * q_glu

        # A Ser, Thr, Asn or Asp N-cap, and the Ac or Sc end group, stabilise the helix by a
        # hydrogen bond (Muñoz & Serrano 1995-III p. 305): their advantage over Ala in the same
        # column, capping box included, takes the hydrogen-bond temperature law.
        if self.Ncap_AA in HBOND_NCAPS:
            ref = self.table_1_lacroix.loc["A", col]
            energy[self.ncap_idx] = ref + self._hbond_temperature(energy[self.ncap_idx] - ref)
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

        # Lacroix 1998 (G_nonH): uncharged Asp at the C-cap H-bonds the C3 carbonyl as Asn
        # does and takes Asn's C-capping value; Table I holds the charged form.
        if self.Ccap_AA == "D":
            col = "Cc-2" if self.Cprime_AA == "P" else "Cc-1"
            q_cap = abs(float(self.modified_seq_ionization_hel[self.ccap_idx]))
            energy[self.ccap_idx] = (q_cap * self.table_1_lacroix.loc["D", col]
                                     + (1.0 - q_cap) * self.table_1_lacroix.loc["N", col])

        # A Lys, Arg, His, Asn or Asp C-cap hydrogen-bonds a C-terminal carbonyl, and the Am end
        # group's contribution is a hydrogen bond (Muñoz & Serrano 1995-III p. 305): the
        # advantage over Ala (the whole Am value) takes the hydrogen-bond temperature law.
        col = "Cc-2" if self.Cprime_AA == "P" else "Cc-1"
        if self.Ccap_AA in HBOND_CCAPS:
            ref = self.table_1_lacroix.loc["A", col]
            energy[self.ccap_idx] = ref + self._hbond_temperature(energy[self.ccap_idx] - ref)
        elif self.Ccap_AA == "Am":
            energy[self.ccap_idx] = self._hbond_temperature(energy[self.ccap_idx])
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

        # The hydrophobic staple motif applies whatever the N-cap residue: x 1 in the two capping-box cases below
        # (Lacroix 1998 supplement) and x 0.5 otherwise, including non-polar N-caps (Viguera & Serrano 1999,
        # Protein Sci. 8, 1733, Table 3 note c; params/README.md). The supplement had given non-polar N-caps 0.
        # Table II is indexed by N' (rows) x N4 (columns): "the interactions between
        # different amino acids at positions N' (rows) and N4 (columns) in a hydrophobic
        # staple motif" (Lacroix 1998, supplementary Table II caption), which is also what
        # this method's docstring says.  This lookup previously used Ncap_AA, so the term
        # read the wrong table row for every staple it fired on.
        Nprime_AA = self.seq_list[self.ncap_idx - 1]
        if Nprime_AA in self.table_2_lacroix.index and self.N4_AA in self.table_2_lacroix.columns:
            energy = self.table_2_lacroix.loc[Nprime_AA, self.N4_AA] / 100

            # whenever the N-cap residue is Asn, Asp, Ser, or Thr and the N3 residue is Glu, Asp or Gln, multiply by 1.0
            if self.Ncap_AA in ["N", "D", "S", "T"] and self.N3_AA in ["E", "D", "Q"]:
                energy *= 1.0

            # whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr
            elif self.Ncap_AA in ["N", "D"] and self.N3_AA in ["S", "T"]:
                energy *= 1.0

            # other cases they are multiplied by 0.5
            else:
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

        # hydrogen bond: the hydrogen-bond temperature law
        energy = self._hbond_temperature(energy)

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

        # hydrogen bond: the hydrogen-bond temperature law
        energy = self._hbond_temperature(energy)

        return energy

    def _acid_base_hbonds(self) -> set:
        """Acid-base (Asp/Glu - Lys/Arg/His) side-chain H-bonds that can coexist in this helix.

        A side chain is in one rotamer at a time: it can reach partners on its N-terminal side
        (i-3, i-4) or on its C-terminal side (i+3, i+4), not both.  Partners on the same side can
        share that rotamer.  Candidates are the i,i+3 and i,i+4 pairs inside the helix (caps
        excluded) with a favourable Table IV value; the strongest are taken first.  The ionic
        part of each pair (the Coulomb term) is not restricted.
        """
        if getattr(self, "_ab_hbonds", None) is not None:
            return self._ab_hbonds
        if self.dipole_law == "gbp":  # statistical network (_sb_network_factor): every candidate carries its cell
            self._ab_hbonds = set(self._sb_candidates())
            return self._ab_hbonds
        cands = []
        interior = set(self.helix_indices[1:-1])
        for k, table in ((3, self.table_4a_lacroix), (4, self.table_4b_lacroix)):
            for i in interior:
                j = i + k
                if j not in interior:
                    continue
                a, b = self.seq_list[i], self.seq_list[j]
                if {a, b} & {"D", "E"} and {a, b} & {"K", "R", "H"}:
                    v = float(table.loc[a, b]) / 100.0
                    if v < 0:
                        cands.append((v, i, j))
        # A side chain points either toward the N-terminus (partners at i-3/i-4) or toward the
        # C-terminus (partners at i+3/i+4), not both.  Partners on the same side can share it.
        direction, chosen = {}, set()
        for v, i, j in sorted(cands):
            if direction.get(i, "up") != "up" or direction.get(j, "down") != "down":
                continue
            direction[i], direction[j] = "up", "down"
            chosen.add((i, j))
        self._ab_hbonds = chosen
        return chosen

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
            # side chain-side chain pairs only INSIDE the helix (Lacroix 1998, G_SD): a pair
            # involving the N-cap or C-cap residue contributes nothing
            if idx == self.ncap_idx or idx + 3 == self.ccap_idx:
                base = 0.0

            # An Asp/Glu - Lys/Arg/His pair forms a side-chain hydrogen bond whose strength does
            # not depend on salt or on whether the acid is charged (Scholtz et al. 1993; Smith &
            # Scholtz 1998).  Its Table IV value is that hydrogen bond and applies in every
            # ionisation state, where the geometry allows it (_acid_base_hbonds); the ionic part
            # of the pair is the Coulomb term.
            acid_base = bool({AAi, AAi3} & {"D", "E"} and {AAi, AAi3} & {"K", "R", "H"})
            if acid_base and base < 0 and (idx, idx + 3) not in self._acid_base_hbonds():
                base = 0.0

            # Other titratable pairs: Table IV applies to the states that are not both charged.
            if not acid_base and (AAi in (self.pos_charge_aa + self.neg_charge_aa)) and (AAi3 in (self.pos_charge_aa + self.neg_charge_aa)):
                p_i = abs(self.modified_seq_ionization_hel[idx])
                p_j = abs(self.modified_seq_ionization_hel[idx + 3])
                base = base * (1.0 - p_i * p_j)

            # _dCp_hydroph_kcal is non-zero only for hydrophobic pairs; used here as the gate
            dCp_val = self._dCp_hydroph_kcal(AAi, AAi3)
            
            # temperature scaling applies to hydrophobic pairs only
            if dCp_val != 0.0 and base != 0.0:
                # hydrophobic pairs scale entropically (dG_ref * t/t_ref), no dCp_hydroph term:
                # Munoz 1995-III states this term becomes more favourable with temperature
                base = self._entropic_cp_correct(base, 0.0)
            # acid-base and polar pairs are side-chain hydrogen bonds: the hydrogen-bond temperature law
            elif base < 0.0 and (acid_base or (AAi in HBOND_POLAR and AAi3 in HBOND_POLAR)):
                base = self._hbond_temperature(base)
            if acid_base and base < 0.0 and self.dipole_law == "gbp":
                base *= self._sb_network_factor()

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
            # side chain-side chain pairs only INSIDE the helix (Lacroix 1998, G_SD): a pair
            # involving the N-cap or C-cap residue contributes nothing
            if idx == self.ncap_idx or idx + 4 == self.ccap_idx:
                base = 0.0

            # An Asp/Glu - Lys/Arg/His pair forms a side-chain hydrogen bond whose strength does
            # not depend on salt or on whether the acid is charged (Scholtz et al. 1993; Smith &
            # Scholtz 1998).  Its Table IV value is that hydrogen bond and applies in every
            # ionisation state, where the geometry allows it (_acid_base_hbonds); the ionic part
            # of the pair is the Coulomb term.
            acid_base = bool({AAi, AAi4} & {"D", "E"} and {AAi, AAi4} & {"K", "R", "H"})
            if acid_base and base < 0 and (idx, idx + 4) not in self._acid_base_hbonds():
                base = 0.0

            # Other titratable pairs: suppress Table IV in the both-charged microstate.
            if not acid_base and (AAi in (self.pos_charge_aa + self.neg_charge_aa)) and (AAi4 in (self.pos_charge_aa + self.neg_charge_aa)):
                p_i = abs(self.modified_seq_ionization_hel[idx])
                p_j = abs(self.modified_seq_ionization_hel[idx + 4])
                base = base * (1.0 - p_i * p_j)

            # _dCp_hydroph_kcal is non-zero only for hydrophobic pairs; used here as the gate
            dCp_val = self._dCp_hydroph_kcal(AAi, AAi4)
            
            # temperature scaling applies to hydrophobic pairs only
            if dCp_val != 0.0 and base != 0.0:
                # hydrophobic pairs scale entropically (dG_ref * t/t_ref), no dCp_hydroph term:
                # Munoz 1995-III states this term becomes more favourable with temperature
                base = self._entropic_cp_correct(base, 0.0)
            # acid-base and polar pairs are side-chain hydrogen bonds: the hydrogen-bond temperature law
            elif base < 0.0 and (acid_base or (AAi in HBOND_POLAR and AAi4 in HBOND_POLAR)):
                base = self._hbond_temperature(base)
            if acid_base and base < 0.0 and self.dipole_law == "gbp":
                base *= self._sb_network_factor()

            extra = 0.0  # Table V hydrogen-bond motifs
            aro_his = 0.0  # Table V aromatic-His+ motif (a ring-cation contact, not a hydrogen bond)

            # --- Table V add-ons (orientation matters: position i -> position i+4) ---
            # Table V values are G_helix for charged side-chain interactions.
            # The same interaction exists (weaker) in the coil state. We subtract
            # the coil contribution using 1/d Coulomb scaling with Table 6 distances:
            #   ΔG = G_hel × (1 − d_hel / d_coil)
            # Pairs with one uncharged residue use HelixRest/RcoilRest distances.
            # Validated on Huyghues-Despointes 1993 Asp-scan: Q-D⁻ gives -0.227
            # (optimal -0.225, vs paper raw -0.5 which over-stabilises by +8.9%).
            d_hel_4 = float(self.table_6_helix_lacroix.loc['HelixRest', 'i+4'])
            d_coil_4 = float(self.table_6_coil_lacroix.loc['RcoilRest', 'i+4'])
            coil_corr_4 = 1.0 - d_hel_4 / d_coil_4  # ≈ 0.455

            # FYW (i) with His+ (i+4): -0.4 kcal/mol when His is at C1 or C-cap; otherwise divide by 3.
            # No coil subtraction for this motif: the aromatic ring and the imidazolium are in contact only when
            # both side chains lie on one face of the helix, so the interaction is short-range and scarcely
            # formed in the coil. The 1/d coil share above assumes a Coulomb interaction that persists in the
            # coil, which does not describe a contact. With the full value the C-peptide family (Shoemaker et
            # al. 1987, Fairman et al. 1989, Mitchinson & Baldwin 1986), where Fairman et al. report that His-12+
            # stabilises the helix through Phe-8, is predicted 0.9-1.7 helix points (RMSE) closer.
            if AAi in ["F", "Y", "W"] and AAi4 == "H":
                p_his = abs(self.modified_seq_ionization_hel[idx + 4])  # population of His+
                # His is "C1" if it is the residue just before C-cap; "C-cap" if it is C-cap itself
                his_is_C1_or_Ccap = (idx + 4 == self.ccap_idx) or (idx + 4 == self.ccap_idx - 1)
                val = -0.4 if his_is_C1_or_Ccap else (-0.4 / 3.0)
                aro_his += p_his * val

            # Gln (i) with Asp- (i+4): -0.5 kcal/mol (paper Table V)
            if AAi == "Q" and AAi4 == "D":
                p_asp = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Asp-
                extra += p_asp * (-0.5) * coil_corr_4

            # Glu- (i) with Asn (i+4): -0.5 kcal/mol
            if AAi == "E" and AAi4 == "N":
                p_glu = abs(self.modified_seq_ionization_hel[idx])  # population of Glu-
                extra += p_glu * (-0.5) * coil_corr_4

            # Gln (i) with Glu- (i+4): -0.1 kcal/mol
            if AAi == "Q" and AAi4 == "E":
                p_glu = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Glu-
                extra += p_glu * (-0.1) * coil_corr_4

            # the Gln-Asp-, Glu--Asn and Gln-Glu- motifs are side-chain hydrogen bonds: the hydrogen-bond
            # temperature law; the aromatic-His+ contact is temperature-independent
            energy[idx] = base + self._hbond_temperature(extra) + aro_his

        return energy
