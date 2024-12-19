from typing import Dict, List, Tuple

import numpy as np

from pyagadir import energies
from pyagadir.energies import EnergyCalculator
from pyagadir.utils import is_valid_index, is_valid_peptide_sequence


class ModelResult(object):
    """
    Class representing the result of a model.
    """

    def __init__(self, seq: str) -> None:
        """
        Initialize the ModelResult object.

        Args:
            seq (str): The peptide sequence.
        """
        self.seq: str = seq
        n: int = len(seq)
        self.dG_dict_mat: List[List[None]] = [None for j in range(5)] + [
            [None for _ in range(0, n - j)] for j in range(5, n)
        ]  # helix length is at least 6 but we zero-index
        self.K_tot: float = 0.0
        self.K_tot_array: np.ndarray = np.zeros(len(seq))
        self.Z: float = 0.0
        self.Z_array: np.ndarray = np.zeros(len(seq))
        self.helical_propensity: np.ndarray = None
        self.percent_helix: float = None

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
            M (float): Ionic strength in Molar. Default is 0.15.
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
        if not isinstance(pH, float):
            raise ValueError("pH must be a float")
        if pH < 0 or pH > 14:
            raise ValueError("pH must be between 0 and 14")

        # check for valid temperature
        if not isinstance(T, float):
            raise ValueError("Temperature must be a float")
        if T < -273.15:
            raise ValueError("Temperature must be above absolute zero")

        # check for valid molarity
        if not isinstance(M, float):
            raise ValueError("Molarity must be a float")
        if M < 0:
            raise ValueError("Molarity must be greater than 0")

        self._method = method
        self.T = T + 273.15
        self.molarity = M
        self.pH = pH

        self.has_acetyl = False
        self.has_succinyl = False
        self.has_amide = False

        self.min_helix_length = 6
        self.ionization_states = None

    def _calc_dG_Hel(self, i: int, j: int) -> Tuple[np.float64, Dict[str, float]]:
        """
        Calculate the Helix free energy and its components.

        Args:
            i (int): The starting position of the helical segment.
            j (int): The length of the helical segment.

        Returns:
            Tuple[np.float64, Dict[str, float]]: The Helix free energy and its components.
        """
        # intrinsic energies for the helical segment, excluding N- and C-terminal capping residues
        dG_Int = self.energy_calculator.get_dG_Int(i, j)

        # "non-hydrogen bond" capping energies, only for the first and last residues of the helix
        dG_Ncap = self.energy_calculator.get_dG_Ncap(i, j)
        dG_Ccap = self.energy_calculator.get_dG_Ccap(i, j)
        dG_nonH = dG_Ncap + dG_Ccap
        # TODO dG_nonH might need further adjustment, see page 175 in lacroix paper

        # get hydrophobic staple motif energies
        dG_staple = self.energy_calculator.get_dG_staple(i, j)

        # get schellman motif energies
        dG_schellman = self.energy_calculator.get_dG_schellman(i, j)

        # calculate dG_Hbond for the helical segment here
        dG_Hbond = self.energy_calculator.get_dG_Hbond(i, j)

        # side-chain interactions, excluding N- and C-terminal capping residues
        dG_i3_tot = self.energy_calculator.get_dG_i3(i, j)
        dG_i4_tot = self.energy_calculator.get_dG_i4(i, j)
        dG_SD = dG_i3_tot + dG_i4_tot  # dG_i1_tot

        # get the interactions between N- and C-terminal capping charges and the helix macrodipole
        dG_N_term, dG_C_term = self.energy_calculator.get_dG_terminals_macrodipole(i, j)

        # get the interaction between charged side chains and the helix macrodipole
        dG_dipole = self.energy_calculator.get_dG_sidechain_macrodipole(i, j)

        # get electrostatic energies between pairs of charged side chains
        dG_electrost = self.energy_calculator.get_dG_electrost(i, j)

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
            + sum(dG_N_term)
            + sum(dG_C_term)
            + dG_electrost
            + np.sum(dG_dipole)
        )

        # make fancy printout for debugging and development
        for seq_idx, arr_idx in zip(range(i, i + j), range(j)):
            print(f"Helix: start= {i+1} end= {i+j}  length=  {j}")
            print(f"residue index = {seq_idx+1}")
            print(f"residue = {self.result.seq[seq_idx]}")
            print(f"g N term = {dG_N_term[arr_idx]:.4f}")
            print(f"g C term = {dG_C_term[arr_idx]:.4f}")
            print(f"g capping =   {dG_nonH[arr_idx]:.4f}")
            print(f"g intrinsic = {dG_Int[arr_idx]:.4f}")
            print(f"g dipole = {dG_dipole[arr_idx]:.4f}")
            print(f"gresidue = ")
            print("****************")
        print("Additional terms for helical segment")
        print(f"i,i+3 and i,i+4 side chain-side chain interaction = {sum(dG_SD):.4f}")
        print(f"g staple = {dG_staple:.4f}")
        print(f"g schellman = {dG_schellman:.4f}")
        print(f"dG_electrost = {dG_electrost:.4f}")
        print(f"dG_electrost = {dG_electrost:.4f}")
        print(f"main chain-main chain H-bonds = {dG_Hbond:.4f}")
        print(f"ionic strngth corr. from eq. 12 {dG_ionic:.4f}")
        print(f"total Helix free energy = {dG_Hel:.4f}")
        print("==============================================")

        # TODO: do we need to return all these components? It was initally intended for the "ms" partition function calculation

        # dG_dict = {
        #     'dG_Helix': dG_Hel,
        #     'dG_Int': dG_Int,
        #     'dG_Hbond': dG_Hbond,
        #     'dG_SD': dG_SD,
        #     'dG_nonH': dG_nonH,
        #     'dG_dipole': dG_dipole,
        #     'dG_N_dipole': dG_N_dipole,
        #     'dG_C_dipole': dG_C_dipole,
        #     'dG_i1_tot': dG_i1_tot,
        #     'dG_i3_tot': dG_i3_tot,
        #     'dG_i4_tot': dG_i4_tot,
        #     'dG_Ncap': dG_Ncap,
        #     'dG_Ccap': dG_Ccap
        # }

        return dG_Hel, {}

    def _calc_K(self, dG_Hel: float) -> float:
        """
        Calculate the equilibrium constant K.

        Args:
            dG_Hel (float): The Helix free energy.

        Returns:
            float: The equilibrium constant K.
        """
        R = 1.987204258e-3  # kcal/mol/K
        return np.exp(-dG_Hel / (R * self.T))

    def _calc_partition_fxn(self) -> None:
        """
        Calculate partition function for helical segments
        by summing over all possible helices.
        """
        # for i in range(0, len(self.result.seq) - self.min_helix_length + 1):  # for each position i
        #     for j in range(self.min_helix_length, len(self.result.seq) - i + 1):  # for each helix length j

        for j in range(
            self.min_helix_length, len(self.result.seq) + 1
        ):  # helix lengths (including caps)
            for i in range(
                0, len(self.result.seq) - j + 1
            ):  # helical segment positions

                # calculate dG_Hel and dG_dict
                dG_Hel, dG_dict = self._calc_dG_Hel(i=i, j=j)

                # TODO: these shuld be accounted for in the new table 1, verify this!
                # # Add acetylation and amidation effects.
                # # These are only considered for the first and last residues of the helix,
                # # and only if the peptide has been created in a way that they are present.
                # if i == 0 and self.has_acetyl is True:
                #     dG_Hel += -1.275
                #     if self.result.seq[0] == 'A':
                #         dG_Hel += -0.1

                # elif i == 0 and self.has_succinyl is True:
                #     dG_Hel += -1.775
                #     if self.result.seq[0] == 'A':
                #         dG_Hel += -0.1

                # if (i + j == len(self.result.seq)) and (self.has_amide is True):
                #     dG_Hel += -0.81
                #     if self.result.seq[-1] == 'A':
                #         dG_Hel += -0.1

                # calculate the partition function K
                K = self._calc_K(dG_Hel)
                self.result.K_tot_array[
                    i + 1 : i + j - 1
                ] += K  # method='r', by definition helical region does not include caps
                self.result.K_tot += K  # method='1s'

        # if method='ms' (custom calculation here with result.dG_dict_mat)
        ### Not implemented yet ###

    def _calc_helical_propensity(self) -> None:
        """
        Calculate helical propensity based on the selected method.
        """
        # get per residue helical propensity
        if self._method == "r":
            print("r")
            self.result.helical_propensity = (
                100 * self.result.K_tot_array / (1.0 + self.result.K_tot_array)
            )

        elif self._method == "1s":
            print("1s")
            self.result.helical_propensity = (
                100 * self.result.K_tot_array / (1.0 + self.result.K_tot)
            )

        # get overall percentage helix
        self.result.percent_helix = np.round(np.mean(self.result.helical_propensity), 2)

    def predict(self, seq: str, ncap: str = None, ccap: str = None) -> ModelResult:
        """
        Predict helical propensity for a given sequence.

        Args:
            seq (str): Input sequence.
            ncap (str): N-terminal capping modification (acetylation='Z', succinylation='X').
            ccap (str): C-terminal capping modification (amidation='B').

        Returns:
            ModelResult: Object containing the predicted helical propensity.
        """
        seq = seq.upper()

        if len(seq) < self.min_helix_length:
            raise ValueError(
                f"Input sequence must be at least {self.min_helix_length} amino acids long."
            )

        if ncap is not None:
            if ncap not in ["Z", "X"]:
                raise ValueError(
                    f"Invalid N-terminal capping modification: {ncap}, must be None,'Z' or 'X'"
                )

        if ccap is not None:
            if ccap not in ["B"]:
                raise ValueError(
                    f"Invalid C-terminal capping modification: {ccap}, must be None or 'B'"
                )

        # check for acylation and amidation
        if ncap == "Z":
            self.has_acetyl = True

        elif ncap == "X":
            self.has_succinyl = True

        if ccap == "B":
            self.has_amide = True

        # ensure that the sequence is valid
        is_valid_peptide_sequence(seq)

        # create energy calculator instance
        self.energy_calculator = EnergyCalculator(
            seq,
            pH=self.pH,
            T=self.T,
            ionic_strength=self.molarity,
            has_acetyl=self.has_acetyl,
            has_succinyl=self.has_succinyl,
            has_amide=self.has_amide,
        )

        self.result = ModelResult(seq)
        self._calc_partition_fxn()
        self._calc_helical_propensity()
        return self.result


if __name__ == "__main__":
    pass
