from typing import Dict


def is_valid_peptide_sequence(pept: str) -> None:
    """
    Validate that the input is a valid peptide sequence.

    Args:
        seq (str): The input sequence.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the sequence contains invalid amino acids.
    """
    if not isinstance(pept, str):
        raise TypeError("Input must be a string.")

    if not len(pept) >= 6:
        raise ValueError("Sequence must be at least 6 amino acids long.")

    pept = pept.upper()
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

    # check for invalid residues
    invalid_residues = [residue for residue in pept if residue not in valid_amino_acids]
    if invalid_residues:
        raise ValueError(
            f"Invalid residues found in sequence: {', '.join(invalid_residues)}"
        )


def is_valid_index(pept: str, i: int, j: int, ncap: str, ccap: str) -> None:
    """
    Validate that the input indexes are valid.

    Args:
        pept (str): The peptide sequence.
        i (int): The start index.
        j (int): The helix length.
        ncap (str): The N-terminal capping modification.
        ccap (str): The C-terminal capping modification.
    Raises:
        TypeError: If the input indexes are not integers.
        ValueError: If the indexes are out of range.
    """
    min_helix_length = 6

    if not isinstance(i, int) or not isinstance(j, int):
        raise TypeError("Indexes must be integers.")

    if i < 0:
        raise ValueError("Start index must be greater than or equal to zero.")

    if j < min_helix_length:
        raise ValueError(f"Helix length must be greater than or equal to {min_helix_length}.")

    if i + j > len(pept) + (1 if ncap is not None else 0) + (1 if ccap is not None else 0):
        raise ValueError(
            "The sum of the indexes must be less than the length of the sequence (including capping residues)."
        )


def is_valid_ncap_ccap(ncap: str, ccap: str) -> None:
    """
    Validate that the input N- and C-terminal capping modifications are valid.
    """
    if ncap not in ["Ac", "Sc", None]:
        raise ValueError(
            f"Invalid N-terminal capping modification: {ncap}, must be None,'Ac' or 'Sc'"
        )
    if ccap not in ["Am", None]:
        raise ValueError(f"Invalid C-terminal capping modification: {ccap}, must be None or 'Am'.")


def is_valid_conditions(pH: float, T: float, ionic_strength: float) -> None:
    """
    Validate that the input conditions are valid.
    """
    if not isinstance(pH, (int, float)):
        raise ValueError("pH must be a number.")
    if not 0 <= pH <= 14:
        raise ValueError("pH must be between 0 and 14.")
    if not isinstance(T, (int, float)):
        raise ValueError("Temperature must be a number.")
    if not 0 <= T <= 140:
        raise ValueError("Temperature must be between 0 and 140.")
    if not isinstance(ionic_strength, (int, float)):
        raise ValueError("Ionic strength must be a number.")
    if not 0 <= ionic_strength <= 2.0:
        raise ValueError("Ionic strength must be between 0 and 2.0.")


