import itertools
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import (E_CHARGE, EPSILON_0, N_AVOGADRO, T_REF, acidic_residue_ionization,
                                basic_residue_ionization, calculate_permittivity, debye_screening_kappa)
from pyagadir.electrostatics import DIPOLE_LAWS, GB_LAWS, GBX_LAWS, Electrostatics  # noqa: F401
from pyagadir.electrostatics.common import SegmentBatch
from pyagadir.utils import is_valid_peptide_sequence, is_valid_ncap_ccap, is_valid_conditions

# The electrostatics are in pyagadir/electrostatics/: point.py (the default point-charge law), gb.py (the Generalized
# Born law gbx) and common.py (the law switch and what the laws share). Their public names stay importable from here.
from pyagadir.electrostatics.gb import (  # noqa: F401
    GB_BOTH_CAPS, GB_DIPOLE_CHARGE, GB_DIPOLE_FORM, GB_NEAR_CAP, GBX_CAPPING_BOX_NCAPS, gb_dipole_table,
    gbx_position_class, gbx_tables, law_tables)
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


# First-turn offset (kcal/mol at 0 C) added to the Table 1 N2 cell of the residues whose first-turn cells were never
# measured in peptides (Lacroix 1998 took the N1-N4 cells from Petukhov's poly-Ala measurements; K, R and H were left
# at their interior value, C, F, Y and W at round numbers). Lys and Arg take the measured peptide value for Leu at N2
# (Petukhov et al. 1998, J. Mol. Biol. 278, 279: +0.25 above the interior), which the peptides with Lys at N2 support
# when each is compared with the other peptides of its own host. H, C, F, Y and W take the N2-minus-interior offsets of
# the single-mutant scan of Tsuboyama et al. 2023 (Nature 620, 434; exposed helix sites, relative to Ala), scaled by
# 0.943 so that Leu's protein offset equals its peptide value. Charge-independent. See params/README.md.
FIRST_TURN_N2_OFFSET = {"K": 0.25, "R": 0.25, "H": -0.059, "C": 0.228, "F": 0.162, "Y": 0.157, "W": 0.141}

# Hydrogen-bond terms share one temperature law (EnergyCalculator._hbond_temperature): Munoz & Serrano 1995-III
# eq. (8) for the backbone bond, dH_ref = -0.895 kcal/mol per bond at 0 C (Lacroix 1998, Discussion). Side chains and end groups
# whose capping contribution is a hydrogen bond (Munoz & Serrano 1995-III p. 305), and residues that can form
# side-chain hydrogen bonds in the Table IV/V pair terms:
HBOND_DH_REF = -0.895

# Table V Gln(i)-Asp-(i+4) side-chain hydrogen bond (Lacroix 1998 supplementary Table V: -0.5 kcal/mol), re-derived from
# the measurement that isolates it, the Asp walk of Huyghues-Despointes, Scholtz & Baldwin (1993, Protein Sci. 2, 1604):
# with this model its printed tables are best fitted at -0.15 or weaker and its pH 7 figure at -0.30, and -0.5 is
# rejected by both (+1.7 and +1.1 helix points RMSE); -0.225 lies between. The other Table V motifs keep their values.
TABLE_V_GLN_ASP = -0.225
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

    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, debug: bool = False, params: dict = None, dipole_law: str = "munoz", batch=None):
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
            dipole_law (str): side chain-macrodipole law, "munoz" (default) or "gbx" (see
                _sidechain_dipole_potential).
            batch (SegmentBatch): The peptide's shared tables and coil solution
                (electrostatics/common.py): the setup takes them instead of recomputing them, the pair lists cover the
                interacting pairs only, and the helix-state ionisation is left to batch.solve_helix.
        """
        self.debug = debug
        self.segment_batch = batch
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
        self.epsilon_0 = EPSILON_0  # Permittivity of free space in C^2/(Nm^2)
        self.N_A = N_AVOGADRO  # Avogadro's number in mol^-1
        self.e = E_CHARGE  # Elementary charge in Coulombs

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

        if batch is not None:
            # Default law or gbx, segments of one prediction solved together (electrostatics/common.py SegmentBatch). The
            # pair list (only the pairs within PAIR_RANGE), the pKa and the intrinsic ionisation are the peptide's and
            # come from the batch. What depends on the segment (macrodipole distances and potentials, terminal group
            # distances, helix pair distances) is assigned as usual. The ionisation is left to batch.solve_helix,
            # which models.AGADIR calls for many segments at once; the modified_* attributes stay None until then.
            self.charged_pairs = batch.charged_pairs
            self.seq_pka, self.nterm_pka, self.cterm_pka = batch.seq_pka, batch.nterm_pka, batch.cterm_pka
            self.seq_ionization = batch.seq_ionization.copy()  # a copy: the segment owns its arrays
            self.nterm_ionization, self.cterm_ionization = batch.nterm_ionization, batch.cterm_ionization
            self._assign_terminal_macrodipole_distances()
            self._assign_sidechain_dipole_potentials()
            self._assign_terminal_sidechain_distances()
            self._assign_sidechain_sidechain_distances()
            return

        # find charged pairs
        self._find_charged_pairs()

        # assign pKa values, distances, and ionization states
        self._assign_pka_values()
        self._assign_ionization_states()
        self._assign_terminal_macrodipole_distances()
        self._assign_sidechain_dipole_potentials()
        self._assign_terminal_sidechain_distances()
        self._assign_sidechain_sidechain_distances()
        # Every law solves its ionisation in a SegmentBatch (electrostatics/common.py); a segment built on its own is a
        # batch of one. models.AGADIR reuses this segment's batch for all the segments of a prediction.
        self.segment_batch = SegmentBatch(self)
        self.segment_batch.solve_helix([self])

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


class EnergyCalculator(PrecomputeParams):
    """
    Class to calculate the free energy contributions for a peptide sequence.
    """
    # class attributes: the ionisation solver (run from PrecomputeParams.__init__) evaluates the energy terms below
    # through _nonCoulomb_charge_coefficients before EnergyCalculator.__init__ has finished
    _AROMATIC = {"F", "Y", "W"}
    _ALIPHATIC = {"A", "V", "L", "I", "M"}
    dCp = -0.0015  # kcal/(mol*K)

    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, params: dict = None, dipole_law: str = "munoz", batch=None):
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
            dipole_law (str): side chain-macrodipole law, "munoz" (default) or "gbx".
            batch (SegmentBatch): see PrecomputeParams.
        """
        super().__init__(seq, i, j, pH, T, ionic_strength, ncap, ccap, params=params, dipole_law=dipole_law,
                         batch=batch)

    def _charge_dependent_nonCoulomb_energy(self) -> float:
        """Sum of the helix-state terms that depend on charged fractions other than through Coulomb's law: the Table 1
        charged/neutral interpolation, the Cys/His N-cap and Asp C-cap rules, the capping box, the Petukhov and
        charged-staple motifs, and the Table IV (1 - p p) and Table V terms (at the current helix charges)."""
        return float(np.sum(self.get_dG_Int()) + np.sum(self.get_dG_Ncap()) + np.sum(self.get_dG_Ccap())
                     + self.get_dG_petukhov_motif() + self.get_dG_charged_staple()
                     + np.sum(self.get_dG_i3()) + np.sum(self.get_dG_i4()))

    def _nonCoulomb_charge_coefficients(self) -> tuple[dict, dict]:
        """
        To be removed (owner, 2026-10-22) with the per-segment solver (Electrostatics._assign_modified_ionization_states),
        its only user, once the GB law that is kept is ported to SegmentBatch. _nonCoulomb_charge_coefficients_local
        replaces it; tests/test_segment_batch.py compares the two. Before deleting it, keep its explanation of the
        coefficients (below) with the local version.

        The charge-dependent non-Coulomb terms of the helix state as G0 + sum_a c_a p_a + sum_{a<b} c_ab p_a p_b in the
        charged fractions p_a = |q_a^h| of the titratable residues of the segment (N-cap to C-cap). Each term is
        linear in one fraction or, for the Table IV (1 - p_a p_b) rule, bilinear in two, so the coefficients follow
        exactly from the terms themselves evaluated at p in {0, 1}: c_a = G(e_a) - G(0) and, for titratable pairs 3 or
        4 apart, c_ab = G(e_a + e_b) - G(e_a) - G(e_b) + G(0). The ionisation solver adds dG/dp_a = c_a + sum_b c_ab p_b
        to each group's helix-state potential (common.py).

        Returns:
            tuple[dict, dict]: {a: c_a}, {(a, b): c_ab} with sequence indices, zero coefficients left out.
        """
        charged = set(self.neg_charge_aa + self.pos_charge_aa)
        titr = [k for k in range(int(self.ncap_idx), int(self.ccap_idx) + 1) if self.seq_list[k] in charged]
        if not titr:
            return {}, {}
        saved = self.modified_seq_ionization_hel
        n = len(self.seq_list)

        def G(on):
            self.modified_seq_ionization_hel = np.array([1.0 if k in on else 0.0 for k in range(n)])
            return self._charge_dependent_nonCoulomb_energy()

        try:
            g0 = G(())
            g1 = {a: G((a,)) for a in titr}
            c_a = {a: g1[a] - g0 for a in titr if abs(g1[a] - g0) > 1e-12}
            c_ab = {}
            for a, b in itertools.combinations(titr, 2):
                if b - a in (3, 4):
                    c = G((a, b)) - g1[a] - g1[b] + g0
                    if abs(c) > 1e-12:
                        c_ab[(a, b)] = c
        finally:
            self.modified_seq_ionization_hel = saved
        return c_a, c_ab

    def _nonCoulomb_charge_coefficients_local(self) -> tuple[dict, dict]:
        """
        The coefficients of _nonCoulomb_charge_coefficients from seven evaluations of the terms per segment, whatever
        its length, instead of one per titratable residue and one per i+3/i+4 pair.

        Each charge-dependent entry of the terms reads at most two helix charges, at most 4 residues apart:
        get_dG_Int[k] reads p_k; get_dG_Ncap[ncap] p_ncap and p_ncap+3; get_dG_Ccap[ccap] p_ccap; the Petukhov and
        charged-staple motifs p_ncap+4; get_dG_i3[k] p_k and p_k+3; get_dG_i4[k] p_k and p_k+4. So with the titratable
        residues of one class mod 5 switched on, an entry sees at most one of them, and with all of them on, a pair
        entry sees both of its residues. c_a is the sum, over the entries that read a, of their value with a's class on
        minus their value with none on; c_ab = (all on) - (a's class) - (b's class) + (none) on the entries that read
        both. The terms are the same functions; the result differs from the whole-energy differences only in
        rounding (about 1e-14 kcal/mol). Used by the default law's SegmentBatch.

        Returns:
            tuple[dict, dict]: {a: c_a}, {(a, b): c_ab} with sequence indices, zero coefficients left out.
        """
        charged = set(self.neg_charge_aa + self.pos_charge_aa)
        nc, cc = int(self.ncap_idx), int(self.ccap_idx)
        titr = [k for k in range(nc, cc + 1) if self.seq_list[k] in charged]  # the segment's titratable residues
        if not titr:
            return {}, {}
        saved = self.modified_seq_ionization_hel
        n = len(self.seq_list)

        def terms(on):
            """The seven charge-dependent terms, each kept per entry, with the helix charged fraction 1 for the residues
            in on and 0 for all others (the patterns of _nonCoulomb_charge_coefficients, which sums them)."""
            on = set(on)
            self.modified_seq_ionization_hel = np.array([1.0 if k in on else 0.0 for k in range(n)])
            return {"Int": self.get_dG_Int(), "Ncap": self.get_dG_Ncap(), "Ccap": self.get_dG_Ccap(),
                    "pet": self.get_dG_petukhov_motif(), "sta": self.get_dG_charged_staple(),
                    "i3": self.get_dG_i3(), "i4": self.get_dG_i4()}

        # The seven patterns: none on; the titratable residues of one class mod 5 on (two residues of a class are at
        # least 5 apart, more than any entry's reach of 4, so no entry sees two of them); all on.
        try:
            t0 = terms(())
            tc = {r: terms([k for k in titr if k % 5 == r]) for r in range(5) if any(k % 5 == r for k in titr)}
            tall = terms(titr)
        finally:
            self.modified_seq_ionization_hel = saved

        def val(t, term, k):
            """One entry of one pattern's terms: element k of an array term, or a scalar term (k None)."""
            return float(t[term]) if k is None else float(t[term][k])

        def entries(a):
            """The charge-dependent entries that read residue a: (term, index; None for a scalar term)."""
            e = [("Int", a), ("i3", a), ("i4", a)]  # its own Table 1 cell; the i+3 and i+4 pairs it starts
            if a >= 3:
                e.append(("i3", a - 3))  # the i+3 pair it ends
            if a >= 4:
                e.append(("i4", a - 4))  # the i+4 pair it ends
            if a in (nc, nc + 3):
                e.append(("Ncap", nc))  # a Cys/His N-cap, or the Glu of a capping box at N3
            if a == cc:
                e.append(("Ccap", cc))  # an Asp C-cap
            if a == nc + 4:
                e += [("pet", None), ("sta", None)]  # the N4 residue of the Petukhov and charged-staple motifs
            return e

        # c_a: in the pattern of a's class, a is the only "on" residue that the entries reading a can see, so each
        # such entry's change from the all-off pattern is its response to p_a alone.
        c_a, c_ab = {}, {}
        for a in titr:
            ta = tc[a % 5]
            c = sum(val(ta, *key) - val(t0, *key) for key in entries(a))
            if abs(c) > 1e-12:
                c_a[a] = c
        # c_ab, for titratable pairs 3 or 4 apart: the entries that read both a and b (the i+3 or i+4 entry at a, and
        # the N-cap entry for the pair N-cap, N3) see both on in the all-on pattern and one of them in each class
        # pattern, so (both) - (a) - (b) + (none) is their pair coupling.
        tset = set(titr)
        for a in titr:
            for d in (3, 4):
                b = a + d
                if b not in tset:
                    continue
                keys = [("i3" if d == 3 else "i4", a)] + ([("Ncap", nc)] if d == 3 and a == nc else [])
                ta, tb = tc[a % 5], tc[b % 5]
                c = sum(val(tall, *k) - val(ta, *k) - val(tb, *k) + val(t0, *k) for k in keys)
                if abs(c) > 1e-12:
                    c_ab[(a, b)] = c
        return c_a, c_ab

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
        Tref = T_REF

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
        Tref = T_REF  # 0°C reference temperature

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
        dG(t) = dG_ref * t/t_ref - t * dCp * ln(t/t_ref) with t_ref = 0 C (T_REF); the dCp part on the
        residues beyond the N1-N4 nucleus only, so that it is counted once per hydrogen bond, as the
        enthalpy of get_dG_Hbond is.

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

            col = self._int_column(n_dist, c_dist)
            energy[idx] = self.table_1_lacroix.loc[AA, col]

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
            # charge states pay it, wherever the N2 cell applies.
            if col == "N2":
                energy[idx] += self._first_turn_n2_offset(AA)

            # gbx: first- and last-turn cells of charged residues re-derived under the GB law
            # (params/gbx_overrides.tsv)
            if self.dipole_law in GBX_LAWS:
                energy[idx] += self._gbx_end_cell(AA, n_dist, c_dist)

        # Munoz & Serrano 1995-III eq. (9): dG_Int = -t (dS_ref + dCp ln(t/t_ref)),
        # i.e. dG_ref * t/t_ref - t * dCp * ln(t/t_ref).  t_ref = 0 C (T_REF).
        # The heat capacity is counted once per backbone hydrogen bond in both of its terms: the enthalpy
        # dCp (t - t_ref) is in the H-bond term (eq. 8, j - 6 bonds), and the matching entropy -t dCp ln(t/t_ref) is
        # carried here by the j - 6 helical residues beyond the four-residue nucleus N1-N4, which forms no
        # hydrogen bond ("the first four residues in a helical conformation do not have mean-residue enthalpic
        # contribution", Munoz & Serrano 1997). Counting it on all j - 2 residues left the nucleus an unmatched
        # entropy, a temperature-dependent nucleation penalty (+0.16 kcal/mol per segment at 25 C).
        Tref = T_REF
        scaled_ref = energy * (T / Tref)
        cp_term = -T * dCp * np.log(T / Tref)
        beyond_nucleus = np.zeros(len(energy), dtype=bool)
        beyond_nucleus[self.ncap_idx + 5:self.ccap_idx] = True

        # Apply to all non-zero entries (avoid adding energy to caps/zeros)
        energy = np.where(energy != 0, scaled_ref + np.where(beyond_nucleus, cp_term, 0.0), energy)

        return energy
    
    def _int_column(self, n_dist: int, c_dist: int) -> str:
        """
        Table 1 column of a helical residue n_dist from the N-cap and c_dist from the C-cap (Lacroix 1998 supplementary
        Table I): N1-N4 in the first turn, Ncen ("between N4 and C-cap") otherwise. The published table has no
        C-terminal position columns.
        """
        if n_dist <= 4:
            return f"N{n_dist}"
        return "Ncen"

    def _first_turn_n2_offset(self, aa: str) -> float:
        """FIRST_TURN_N2_OFFSET, with the gbx law's re-derived values (params/gbx_overrides.tsv, table
        "first_turn_n2") where it has them."""
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

        The net H-bond ΔG at 0°C is -0.895 kcal/mol per bond (Lacroix 1998, Discussion). It is
        treated as enthalpic, with the temperature dependence of equation (8) of
        Muñoz & Serrano (1995-III): ΔG(t) = ΔH_ref + ΔCp (t - t_ref), ΔCp = -1.5
        cal/(mol·K) in the folding direction, t_ref = 0 C (_hbond_temperature, the
        law every hydrogen-bond term uses).

        Returns:
            float: The total free energy contribution for hydrogen bonding in the sequence.
        """
        n_hbonds = max((self.j - 6), 0)

        # Munoz & Serrano 1995-III eq. (8): dG_HBond = dH_ref + dCp (t - t_ref) per bond,
        # with dH_ref = HBOND_DH_REF, dCp = -0.0015 kcal/(mol K), t_ref = 0 C (T_REF).
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
        folding direction and t_ref = 0 C; the entropy of fixing the backbone is in
        get_dG_Int. Other hydrogen bonds take the same law as a relative change,
        ΔG_ref × (1 + ΔCp (t - t_ref) / ΔH_ref) with the backbone bond's ΔH_ref = -0.895
        kcal/mol, which for the backbone bond is eq. (8) itself. Muñoz & Serrano apply it to the
        hydrogen-bonding caps and the end groups (p. 305) and neglect it for side-chain pairs.
        The term grows more favourable with temperature, as the backbone term does.

        Args:
            dG_ref: free energy at 0 C (T_REF; float or array), kcal/mol.
        """
        return dG_ref * (1.0 + self.dCp * (self.T_kelvin - T_REF) / HBOND_DH_REF)

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

        # Temperature: the law of the hydrophobic i,i+3 / i,i+4 pairs (get_dG_i3, get_dG_i4), entropic
        # dG_ref * T / T_REF without a heat-capacity term. Lacroix 1998 (Discussion) gives the staple "a temperature
        # dependence similar to that for hydrophobic side-chain-side-chain interactions"; Munoz & Serrano 1995-III
        # require those to become more favourable with temperature, which eq. 18 as printed does not do.
        # N' is the residue BEFORE Ncap; _dCp_hydroph_kcal is non-zero only for hydrophobic pairs and is the gate.
        if energy != 0.0 and self.ncap_idx > 0:
            Nprime = self.seq_list[self.ncap_idx - 1]
            if self._dCp_hydroph_kcal(Nprime, self.N4_AA) != 0.0:
                energy = self._entropic_cp_correct(energy, 0.0)

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

            extra = 0.0  # Table V hydrogen-bond motifs
            aro_his = 0.0  # Table V aromatic-His+ motif (a ring-cation contact, not a hydrogen bond)

            # --- Table V add-ons (orientation matters: position i -> position i+4) ---
            # Added to Table IV when the residue is charged, weighted by its helix-state charged fraction, as net
            # (helix-minus-coil) energies like Table IV (Lacroix 1998 supplement). Each is a side-chain contact formed
            # on one face of the helix and scarcely in the coil. Values as published, except Gln-Asp- (TABLE_V_GLN_ASP),
            # re-derived from the Asp walk that isolates it.

            # FYW (i) with His+ (i+4): -0.4 kcal/mol when His is at C1 or C-cap; otherwise divide by 3. This is the
            # Phe8...His12+ interaction of the C-peptide (Fairman et al. 1989); with the full value the C-peptide
            # family (Shoemaker et al. 1987, Fairman et al. 1989, Mitchinson & Baldwin 1986) is predicted 0.9-1.7
            # helix points (RMSE) closer.
            if AAi in ["F", "Y", "W"] and AAi4 == "H":
                p_his = abs(self.modified_seq_ionization_hel[idx + 4])  # population of His+
                # His is "C1" if it is the residue just before C-cap; "C-cap" if it is C-cap itself
                his_is_C1_or_Ccap = (idx + 4 == self.ccap_idx) or (idx + 4 == self.ccap_idx - 1)
                val = -0.4 if his_is_C1_or_Ccap else (-0.4 / 3.0)
                aro_his += p_his * val

            # Gln (i) with Asp- (i+4): TABLE_V_GLN_ASP (published -0.5; re-derived from the Asp walk)
            if AAi == "Q" and AAi4 == "D":
                p_asp = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Asp-
                extra += p_asp * TABLE_V_GLN_ASP

            # Glu- (i) with Asn (i+4): -0.5 kcal/mol (Table V)
            if AAi == "E" and AAi4 == "N":
                p_glu = abs(self.modified_seq_ionization_hel[idx])  # population of Glu-
                extra += p_glu * (-0.5)

            # Gln (i) with Glu- (i+4): -0.1 kcal/mol (Table V)
            if AAi == "Q" and AAi4 == "E":
                p_glu = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Glu-
                extra += p_glu * (-0.1)

            # the Gln-Asp-, Glu--Asn and Gln-Glu- motifs are side-chain hydrogen bonds: the hydrogen-bond
            # temperature law; the aromatic-His+ contact is temperature-independent
            energy[idx] = base + self._hbond_temperature(extra) + aro_his

        return energy
