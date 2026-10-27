from typing import Dict, Tuple

import numpy as np

from pyagadir.chemistry import R_KCAL
from pyagadir.electrostatics.common import SEGMENT_BATCH_SIZE
from pyagadir.electrostatics.side_chain_states import SideChainStates
from pyagadir.energies import EnergyCalculator
from pyagadir.utils import is_valid_peptide_sequence, is_valid_ncap_ccap

# Salt effects on the neutral helix (Scholtz et al. 1991, JACS 113:5102; see params/README.md).
# Salting-in: the form of equation 12 of Lacroix et al. 1998, -a (1 - exp(-b I)), per helical segment, with
# a = 0.30 kcal/mol (the measured amplitude; the paper's 0.15 was fitted without a salting-out term) and
# b = 3 /M (the paper's 6), which keeps the paper's low-salt slope a * b = 0.9 kcal/mol per M.
SALTING_IN_ALPHA = 0.30
SALTING_IN_BETA = 3.0
# Salting-out (NaCl Hofmeister term): linear in ionic strength, per helical residue (j - 2).
K_HOFMEISTER = 0.0085


class ModelResult:
    """
    Class containing the result of a model.
    """

    def __init__(self, seq: str, ncap: str = None, ccap: str = None) -> None:
        """
        Initialize the ModelResult object.

        Args:
            seq (str): The peptide sequence.
        """
        is_valid_peptide_sequence(seq)

        self.seq = seq
        self.ncap = ncap
        self.ccap = ccap
        self.seq_list = list(self.seq)
        if self.ncap is not None:  
            self.seq_list.insert(0, self.ncap)
        if self.ccap is not None:
            self.seq_list.append(self.ccap)
        self.seq_length = len(self.seq_list)
        self.K_tot = 0.0
        self.K_tot_array = np.zeros(self.seq_length)
        self.Z = 0.0
        self.Z_array = np.zeros(self.seq_length)
        self.helical_propensity = None
        self.percent_helix = None

    def __repr__(self) -> str:
        """
        Return a string representation of the helical propensity.

        Returns:
            str: The helical propensity.
        """
        return str(self.helical_propensity)

    def get_sequence(self) -> str:
        """
        Get the peptide sequence.

        Returns:
            str: The peptide sequence.
        """
        return self.seq

    def get_helical_propensity(self) -> np.ndarray:
        """
        Get the helical propensity.

        Returns:
            np.ndarray: The helical propensity for each amino acid.
        """
        return self.helical_propensity

    def get_percent_helix(self) -> float:
        """
        Get the percentage of helix.

        Returns:
            float: The percentage of helix for the peptide.
        """
        return self.percent_helix


class AGADIR(object):
    """
    AGADIR class for predicting helical propensity using AGADIR method.
    """

    def __init__(
        self, method: str = "1s", T: float = 4.0, M: float = 0.15, pH: float = 7.0,
    ):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
            T (float): Temperature in Celsius. Default is 4.0.
            M (float): Ionic strength in Molar (M, mol/L). Default is 0.15.
            pH (float): Solution pH. Default is 7.0.
        """
        self.method_options = ["r", "1s"]
        if method not in self.method_options:
            raise ValueError(
                "Method provided must be one of ['r','1s']; \
                'r' : Residue partition function. \
                '1s': One-sequence approximation. \
            See documentation and AGADIR papers for more information. \
            "
            )

        # check for valid pH
        if not isinstance(pH, (float, int)):
            raise ValueError("pH must be a number")
        if pH < 0 or pH > 14:
            raise ValueError("pH must be between 0 and 14")

        # check for valid temperature
        if not isinstance(T, (float, int)):
            raise ValueError("Temperature must be a number")
        if T < 0.0:
            raise ValueError("Temperature must be above zero degrees Celsius")

        # check for valid molarity
        if not isinstance(M, (float, int)):
            raise ValueError("Molarity must be a number")
        if M < 0:
            raise ValueError("Molarity must be greater than 0")

        self._method = method
        self.T_celsius = T
        self.T_kelvin = T + 273.15
        self.molarity = M  # possibly need to adjust based on ion valency to actually get ionic strength here
        self.pH = pH

        self.n_cap = None
        self.c_cap = None
        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

        self.min_helix_length = 6  # default from Lacroix 1998

    def _calc_dG_Hel(self, i: int, j: int) -> Tuple[np.float64, Dict[str, float]]:
        """
        Calculate free energy for a helical segment.

        Args:
            i (int): The starting position of the helical segment.
            j (int): The length of the helical segment.

        Returns:
            float: The Helix free energy.
        """
        # Intrinsic energies for the helical segment, excluding N- and C-terminal capping residues
        dG_Int = self.energy_calculator.get_dG_Int()

        # "non-hydrogen bond" capping energies, only for the first and last residues of the helix
        # TODO dG_nonH might need further adjustment, see page 175 in lacroix paper
        dG_Ncap = self.energy_calculator.get_dG_Ncap()
        dG_Ccap = self.energy_calculator.get_dG_Ccap()
        dG_nonH = dG_Ncap + dG_Ccap

        # get hydrophobic staple motif energies
        dG_staple = self.energy_calculator.get_dG_staple()

        # get schellman motif energies
        dG_schellman = self.energy_calculator.get_dG_schellman()

        # get Petukhov motif (free N-term + capping box + D/E at N4)
        dG_petukhov = self.energy_calculator.get_dG_petukhov_motif()

        # get charged staple variant (S/T at Ncap + K/R at N4)
        dG_charged_staple = self.energy_calculator.get_dG_charged_staple()

        # calculate hydrogen bond energies for the helical segment here
        dG_Hbond = self.energy_calculator.get_dG_Hbond()

        # side-chain interactions, excluding N- and C-terminal capping residues
        dG_i3_tot = self.energy_calculator.get_dG_i3()
        dG_i4_tot = self.energy_calculator.get_dG_i4()
        dG_SD = dG_i3_tot + dG_i4_tot

        # get the interactions between N- and C-terminal backbone charges and the helix macrodipole
        dG_terminals_dipole_N, dG_terminals_dipole_C = self.energy_calculator.get_dG_terminals_macrodipole()

        # get the interaction between charged side chains and the helix macrodipole
        dG_sidechain_dipole_N, dG_sidechain_dipole_C = self.energy_calculator.get_dG_sidechain_macrodipole()
        dG_sidechain_dipole = dG_sidechain_dipole_N + dG_sidechain_dipole_C

        # get electrostatic energies between N- and C-terminal backbone charges and charged side chains
        dG_electrost_term_N, dG_electrost_term_C = self.energy_calculator.get_dG_terminals_sidechain_electrost()
        dG_electrost_term = dG_electrost_term_N + dG_electrost_term_C

        # get electrostatic energies between pairs of charged side chains
        dG_electrost_sidechain = self.energy_calculator.get_dG_sidechain_sidechain_electrost()

        # get electrostatic energy between N-terminal and C-terminal backbone charges
        dG_electrost_term_term = self.energy_calculator.get_dG_terminal_terminal_electrost()

        # Salt effects on the neutral helix (SALTING_IN_ALPHA, SALTING_IN_BETA, K_HOFMEISTER above): salting-in per
        # segment, salting-out per helical residue (j - 2: the segment without its N- and C-cap).
        dG_ionic = -SALTING_IN_ALPHA * (1 - np.exp(-SALTING_IN_BETA * self.molarity))
        dG_ionic += K_HOFMEISTER * self.molarity * (j - 2)

        # Ionisation free energy of the segment: the part of the mean-field electrostatic free
        # energy that the q-weighted electrostatic terms below leave out (electrostatics/common.py,
        # SegmentBatch).
        dG_ionization = self.energy_calculator.get_dG_ionization()

        # Side-chain states: the charge-dependent free energy summed over the side chains' contact
        # and capping-motif states, minus the mean-field one above (electrostatics/side_chain_states.py; 0 for a
        # peptide without acid-base pairs three or four residues apart)
        dG_contacts = self._contacts.segment_correction(self.energy_calculator)

        # sum all components
        dG_Hel = (
            sum(dG_Int)
            + sum(dG_nonH)
            + sum(dG_SD)
            + dG_staple
            + dG_schellman
            + dG_petukhov
            + dG_charged_staple
            + dG_Hbond
            + dG_ionic
            # every electrostatic term is screened Coulomb in water at eps(T) (calculate_permittivity)
            + sum(dG_terminals_dipole_N)
            + sum(dG_terminals_dipole_C)
            + np.sum(dG_sidechain_dipole)
            + np.sum(dG_electrost_term)
            + np.sum(dG_electrost_sidechain)
            + dG_electrost_term_term
            + dG_ionization
            + dG_contacts
        )

        return dG_Hel

    def _calc_K(self, dG_Hel: float) -> float:
        """
        Calculate the equilibrium constant K.

        Args:
            dG_Hel (float): The Helix free energy.

        Returns:
            float: The equilibrium constant K.
        """
        return np.exp(-dG_Hel / (R_KCAL * self.T_kelvin))

    def _calc_partition_fxn(self) -> None:
        """
        Calculate partition function for helical segments
        by summing over all possible helices.

        The ionisation of SEGMENT_BATCH_SIZE segments is solved at once (electrostatics/common.py SegmentBatch), with
        the same results as solving each segment on its own (identical % helix, per-residue helicity within 1e-12) at
        a cost that grows as n^3 with the length instead of n^4. The segments, their energies and the sums are
        computed in the order j (length) then i (position).
        """
        # one fast snapshot of the parameter tables, shared by every segment
        params = EnergyCalculator.snapshot_params()

        # Every helical segment (i: index of its N-cap, j: its length with both caps), lengths outer and positions
        # inner, so that K_tot and K_tot_array accumulate in that order.
        L = self.result.seq_length
        segments = [(i, j) for j in range(self.min_helix_length, L + 1) for i in range(0, L - j + 1)]

        def calculator(i, j, batch=None):
            return EnergyCalculator(seq=self.result.seq, i=i, j=j, pH=self.pH, T=self.T_celsius,
                                    ionic_strength=self.molarity, ncap=self.n_cap, ccap=self.c_cap, params=params,
                                    batch=batch)

        # The first segment, built on its own, is a batch of one (energies.PrecomputeParams) and is solved with it; its
        # batch holds the peptide's tables and coil state for the other segments.
        first = calculator(*segments[0])
        batch = first.segment_batch

        def solved():
            """The segments in order with their ionisation solved: the first, then the others SEGMENT_BATCH_SIZE at a
            time (their setup without the ionisation, then their helix ionisation in one solve)."""
            yield segments[0], first
            for start in range(1, len(segments), SEGMENT_BATCH_SIZE):
                chunk = segments[start:start + SEGMENT_BATCH_SIZE]
                calcs = [calculator(i, j, batch) for i, j in chunk]
                batch.solve_helix(calcs)
                yield from zip(chunk, calcs)

        # each segment's energy and equilibrium constant
        for (i, j), calc in solved():
            self.energy_calculator = calc
            K = self._calc_K(self._calc_dG_Hel(i=i, j=j))
            # method='r': the helical region excludes the capping residues, hence i+1 and i+j-1
            self.result.K_tot_array[i + 1 : i + j - 1] += K
            self.result.K_tot += K  # method='1s'

    def _calc_helical_propensity(self) -> None:
        """
        Calculate helical propensity based on the selected method.
        """
        # get per residue helical propensity
        if self._method == "r":
            self.result.helical_propensity = (
                100 * self.result.K_tot_array / (1.0 + self.result.K_tot_array)
            )

        elif self._method == "1s":
            self.result.helical_propensity = (
                100 * self.result.K_tot_array / (1.0 + self.result.K_tot)
            )

        # Exclude synthetic cap tokens ("Ac", "Am") from % helix averaging.
        # CD measurements average over ALL residues, so the CD-compatible convention
        # (all real amino acids) is used for measured-data comparisons.
        start = 1 if self.result.ncap is not None else 0
        end = -1 if self.result.ccap is not None else None
        self.result.percent_helix = float(np.round(np.mean(self.result.helical_propensity[start:end]), 2))


    def predict(self, seq: str, ncap: str = None, ccap: str = None) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.
            ncap (str): N-terminal capping modification (acetylation='Ac', succinylation='Sc').
            ccap (str): C-terminal capping modification (amidation='Am').

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        # check for valid sequence
        is_valid_peptide_sequence(seq)
        seq = seq.upper()

        # check for valid ncap and ccap
        is_valid_ncap_ccap(ncap, ccap)

        # reset per-call state: caps were previously only ever SET, so reusing one AGADIR
        # object for an Ac-capped peptide and then a free one silently kept the acetyl
        self.n_cap = None
        self.c_cap = None
        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

        # assign ncap
        if ncap is not None:
            if ncap == "Ac":
                self.has_acetyl = True
                self.n_cap = "Ac"
            elif ncap == "Sc":
                self.has_succinyl = True
                self.n_cap = "Sc"

        # check for valid ccap
        if ccap is not None:
            self.has_amide = True
            self.c_cap = "Am"

        # check for valid sequence length
        # the minimum applies to the cap-extended chain: Ac and Am each occupy a position
        n_positions = len(seq) + (self.n_cap is not None) + (self.c_cap is not None)
        if n_positions < self.min_helix_length:
            raise ValueError(
                f"Input sequence plus terminal caps must span at least {self.min_helix_length} "
                f"positions (got {n_positions})."
            )
        
        print(f"Predicting helical propensity for sequence: {seq}, method: {self._method}, T(C): {self.T_celsius}, M: {self.molarity}, pH: {self.pH}, ncap: {self.n_cap}, ccap: {self.c_cap}")

        # initialize the result object
        self.result = ModelResult(seq, ncap=self.n_cap, ccap=self.c_cap)
        # side-chain states: the coil terms are computed once per prediction
        self._contacts = SideChainStates(seq)

        # calculate the partition function and helical propensity
        self._calc_partition_fxn()
        self._calc_helical_propensity()

        return self.result


if __name__ == "__main__":
    pass
