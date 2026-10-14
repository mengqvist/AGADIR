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

    # The 6 of Lacroix 1998 is the minimum helical SEGMENT length over the cap-extended chain,
    # not a minimum peptide length: Ac-AAQAA-NH2 (7 positions) has valid segments.  The
    # segment-length check lives in AGADIR.predict, where the caps are known.
    if len(pept) < 1:
        raise ValueError("Sequence must not be empty.")

    pept = pept.upper()
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids

    # check for invalid residues
    invalid_residues = [residue for residue in pept if residue not in valid_amino_acids]
    if invalid_residues:
        raise ValueError(
            f"Invalid residues found in sequence: {', '.join(invalid_residues)}"
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
    # Upper bound 10.0 M: measured data reach 4.8 M NaCl (Huyghues-Despointes 1993).  The
    # bound is kept only to catch a millimolar value passed as molar.
    if not 0 <= ionic_strength <= 10.0:
        raise ValueError("Ionic strength must be between 0 and 10.0 M.")


