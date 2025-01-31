from typing import Dict, Tuple

import numpy as np

from pyagadir.energies import EnergyCalculator
from pyagadir.utils import is_valid_peptide_sequence, is_valid_ncap_ccap


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
        self, method: str = "1s", T: float = 4.0, M: float = 0.15, pH: float = 7.0
    ):
        """
        Initialize AGADIR object.

        Args:
            method (str): Method for calculating helical propensity. Must be one of ['r','1s'].
                'r' : Residue partition function.
                '1s': One-sequence approximation.
            T (float): Temperature in Celsius. Default is 4.0.
            M (float): Ionic strength in Molar (M, mol/L). Default is 0.15.
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
        self.debug = False

    def _print_debug_info(self, 
                          dG_nonH: np.ndarray, 
                          dG_Int: np.ndarray,
                          dG_terminals_dipole_N: np.ndarray, 
                          dG_terminals_dipole_C: np.ndarray, 
                          dG_sidechain_dipole_N: np.ndarray, 
                          dG_sidechain_dipole_C: np.ndarray, 
                          dG_sidechain_dipole: np.ndarray, 
                          dG_electrost_term_N: np.ndarray,
                          dG_electrost_term_C: np.ndarray,
                          dG_electrost_sidechain: np.ndarray,
                          dG_SD: np.ndarray, 
                          dG_staple: float, 
                          dG_schellman: float, 
                          dG_Hbond: float, 
                          dG_ionic: float, 
                          dG_Hel: float, 
                          i: int, 
                          j: int) -> None:
        """
        Print debug information for a helical segment.

        Args:
            dG_terminals_dipole_N (np.ndarray): The free energy for the N-terminal capping residue.
            dG_terminals_dipole_C (np.ndarray): The free energy for the C-terminal capping residue.
            dG_nonH (np.ndarray): The free energy for the non-hydrogen bond capping residues.
            dG_Int (np.ndarray): The intrinsic free energy for the helical segment.
            dG_sidechain_dipole_N (np.ndarray): The free energy for the N-terminal side chain dipole.
            dG_sidechain_dipole_C (np.ndarray): The free energy for the C-terminal side chain dipole.
            dG_sidechain_dipole (np.ndarray): The total free energy for the side chain dipoles.
            dG_electrost_term_N (np.ndarray): The free energy for the electrostatic interactions between N-terminal charges and side chains.
            dG_electrost_term_C (np.ndarray): The free energy for the electrostatic interactions between C-terminal charges and side chains.
            dG_electrost_sidechain (np.ndarray): The free energy for the electrostatic interactions between charged side chains.
            dG_SD (np.ndarray): The free energy for the side chain-side chain interactions.
            dG_staple (float): The free energy for the hydrophobic staple motif.
            dG_schellman (float): The free energy for the schellman motif.
            dG_Hbond (float): The free energy for the main chain-main chain H-bonds.
            dG_ionic (float): The free energy for the ionic strength correction.
            dG_Hel (float): The total free energy for the helical segment.
            i (int): The starting position of the helical segment.
            j (int): The length of the helical segment.
        """
        # make fancy printout for debugging and development
        for seq_idx, arr_idx in zip(range(i, i + j), range(j)):
            print(f"Helix: start= {i+1} end= {i+j}  length=  {j}")
            print(f"residue index = {seq_idx+1}")
            print(f"residue = {self.result.seq_list[seq_idx]}")
            print(f"g dipole terminal N = {dG_terminals_dipole_N[seq_idx]}")
            print(f"g dipole terminal C = {dG_terminals_dipole_C[seq_idx]}")
            print(f"g capping =   {dG_nonH[seq_idx]:.4f}")
            print(f"g intrinsic = {dG_Int[seq_idx]:.4f}")
            print(f"g dipole sidechain N = {dG_sidechain_dipole_N[seq_idx]:.4f}")
            print(f"g dipole sidechain C = {dG_sidechain_dipole_C[seq_idx]:.4f}")
            print(f"g dipole sidechain total = {dG_sidechain_dipole[seq_idx]:.4f}")
            print(f"gresidue = {dG_terminals_dipole_N[seq_idx] + dG_terminals_dipole_C[seq_idx] + dG_nonH[seq_idx] + dG_Int[seq_idx] + dG_sidechain_dipole[seq_idx]:.4f}")
            print("****************")
        print("Additional terms for helical segment")
        print(f"i,i+3 and i,i+4 side chain-side chain interaction = {sum(dG_SD):.4f}")
        print(f"g staple = {dG_staple:.4f}")
        print(f"g schellman = {dG_schellman:.4f}")
        print(f"dG_electrost = {(np.sum(dG_electrost_sidechain) + np.sum(dG_electrost_term_N) + np.sum(dG_electrost_term_C)):.4f}")
        print(f"main chain-main chain H-bonds = {dG_Hbond:.4f}")
        print(f"ionic strngth corr. from eq. 12 {dG_ionic:.4f}")
        print(f"total Helix free energy = {dG_Hel:.4f}")
        print("==============================================")

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

        # modify by ionic strength according to equation 12 of the paper
        alpha = 0.15
        beta = 6.0
        dG_ionic = -alpha * (1 - np.exp(-beta * self.molarity))

        # sum all components
        dG_Hel = (
            sum(dG_Int)
            + sum(dG_nonH)
            + sum(dG_SD)
            + dG_staple
            + dG_schellman
            + dG_Hbond
            + dG_ionic
            + sum(dG_terminals_dipole_N)
            + sum(dG_terminals_dipole_C)
            + np.sum(dG_sidechain_dipole)
            + np.sum(dG_electrost_term)
            + np.sum(dG_electrost_sidechain)
        )

        if self.debug:
            self._print_debug_info(
                                dG_nonH=dG_nonH, 
                                dG_Int=dG_Int, 
                                dG_terminals_dipole_N=dG_terminals_dipole_N, 
                                dG_terminals_dipole_C=dG_terminals_dipole_C, 
                                dG_sidechain_dipole_N=dG_sidechain_dipole_N, 
                                dG_sidechain_dipole_C=dG_sidechain_dipole_C, 
                                dG_sidechain_dipole=dG_sidechain_dipole, 
                                dG_electrost_term_N=dG_electrost_term_N,
                                dG_electrost_term_C=dG_electrost_term_C,
                                dG_electrost_sidechain=dG_electrost_sidechain, 
                                dG_SD=dG_SD, 
                                dG_staple=dG_staple, 
                                dG_schellman=dG_schellman, 
                                dG_Hbond=dG_Hbond, 
                                dG_ionic=dG_ionic, 
                                dG_Hel=dG_Hel, 
                                i=i, 
                                j=j)

        return dG_Hel

    def _calc_K(self, dG_Hel: float) -> float:
        """
        Calculate the equilibrium constant K.

        Args:
            dG_Hel (float): The Helix free energy.

        Returns:
            float: The equilibrium constant K.
        """
        R = 1.987204258e-3  # kcal/mol/K
        return np.exp(-dG_Hel / (R * self.T_kelvin))

    def _calc_partition_fxn(self) -> None:
        """
        Calculate partition function for helical segments
        by summing over all possible helices.
        """
        for j in range(
            self.min_helix_length, self.result.seq_length + 1
        ):  # helix lengths (including capping residues)
            for i in range(
                0, self.result.seq_length - j + 1
            ):  # helical segment positions

                # create energy calculator instance
                self.energy_calculator = EnergyCalculator(
                    seq=self.result.seq,
                    i=i,
                    j=j,
                    pH=self.pH,
                    T=self.T_celsius,
                    ionic_strength=self.molarity,
                    ncap=self.n_cap,
                    ccap=self.c_cap
                )

                # calculate dG_Hel and dG_dict
                dG_Hel = self._calc_dG_Hel(i=i, j=j)

                # calculate the partition function K
                K = self._calc_K(dG_Hel)

                # add the partition function to the total partition function
                self.result.K_tot_array[
                    i + 1 : i + j - 1
                ] += K  # method='r', by definition helical region does not include capping residues, hence i+1 and i+j-1
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

        # get overall percentage helix
        self.result.percent_helix = np.round(np.mean(self.result.helical_propensity), 2)

    def predict(self, seq: str, ncap: str = None, ccap: str = None, debug: bool = False) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.
            ncap (str): N-terminal capping modification (acetylation='Ac', succinylation='Sc').
            ccap (str): C-terminal capping modification (amidation='Am').
            debug (bool): Whether to print debug information.

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        # check for valid sequence
        is_valid_peptide_sequence(seq)
        seq = seq.upper()

        if not isinstance(debug, bool):
            raise ValueError("Debug must be a boolean")
        self.debug = debug

        # check for valid ncap and ccap
        is_valid_ncap_ccap(ncap, ccap)

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
        if len(seq) < self.min_helix_length:
            raise ValueError(
                f"Input sequence must be at least {self.min_helix_length} amino acids long."
            )
        
        print(f"Predicting helical propensity for sequence: {seq}, method: {self._method}, T(C): {self.T_celsius}, M: {self.molarity}, pH: {self.pH}, ncap: {self.n_cap}, ccap: {self.c_cap}")

        # initialize the result object
        self.result = ModelResult(seq, ncap=self.n_cap, ccap=self.c_cap)

        # calculate the partition function and helical propensity
        self._calc_partition_fxn()
        self._calc_helical_propensity()

        return self.result


if __name__ == "__main__":
    pass
