"""
Split a protein structure into helices, strands, loops and other elements, and score them.

Reads one chain of a PDB file, assigns secondary structure with the DSSP algorithm,
groups the residues into elements and scores each element:

- helix: AGADIR helical propensity of the isolated helix, and the free energy of the
  observed helical segment.
- sheet: edge or middle strand, the surface amino-acid score measured by Tsuboyama et al.
  (2023), and the number of large hydrophobics the strand buries in the core.
- every element: ABEGO sequence-backbone compatibility, its hydrophobic side-chain
  contacts, the nonpolar surface it buries and exposes, and its buried polar atoms left
  without a hydrogen-bond partner.
- the whole chain: the same scores summed or averaged, and the connectivity of the
  hydrophobic core.

The structure-independent scores are in ``pyagadir.element_scores``. This module only
needs NumPy: it parses PDB files and computes torsions, burial and DSSP itself.

Usage::

    python -m pyagadir.pdb_elements protein.pdb [--chain A] [--json out.json]
"""

import argparse
import contextlib
import gzip
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from pyagadir.element_scores import (
    abego_from_torsions,
    abego_profile,
    buried_unsatisfied_polar_atoms,
    core_participation,
    hydrophobic_core_clusters,
    sidechain_neighbors,
    strand_surface_score,
    surface_burial,
)
from pyagadir.energies import EnergyCalculator
from pyagadir.models import AGADIR

RESIDUE_CODES = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # selenomethionine and common force-field names for protonation states
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

# DSSP states grouped into the four element classes. H: alpha helix. E: strand in a
# ladder of at least two bridges. T: H-bonded turn. S: bend. '-': coil. G: 3-10 helix.
# I: pi helix. B: isolated beta bridge. X: residue with an incomplete backbone.
ELEMENT_CLASSES = {
    "H": "helix",
    "E": "sheet",
    "T": "loop", "S": "loop", "-": "loop",
    "G": "other", "I": "other", "B": "other", "X": "other",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
MAX_PEPTIDE_BOND = 2.5  # angstrom; a longer C(i)-N(i+1) distance is a chain break (DSSP)
SURFACE_CUTOFF = 2.0  # side-chain neighbours below which a residue is solvent exposed


@dataclass
class ProteinChain:
    """
    Coordinates of one protein chain.

    Backbone coordinate arrays have shape (n, 3) and hold NaN for atoms missing from the
    file. ``cb`` holds the real CB where present and a virtual CB elsewhere (glycine).

    Attributes:
        chain_id (str): PDB chain identifier.
        residue_ids (List[str]): PDB residue number plus insertion code, e.g. '42A'.
        sequence (str): One-letter sequence, 'X' for non-standard residues.
        n (np.ndarray): Backbone N coordinates.
        ca (np.ndarray): CA coordinates.
        c (np.ndarray): Carbonyl C coordinates.
        o (np.ndarray): Carbonyl O coordinates.
        cb (np.ndarray): CB coordinates, virtual where the residue has none.
        atoms (List[Dict[str, np.ndarray]]): Per residue, the coordinates of all its heavy
            atoms present in the file (hydrogens excluded) keyed by atom name.
    """

    chain_id: str
    residue_ids: List[str]
    sequence: str
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    o: np.ndarray
    cb: np.ndarray
    atoms: List[Dict[str, np.ndarray]]

    def __len__(self) -> int:
        return len(self.sequence)

    @property
    def sidechains(self) -> List[Dict[str, np.ndarray]]:
        """Per residue, the side-chain heavy atoms (CB included) keyed by atom name."""
        return [{atom: xyz for atom, xyz in residue.items() if atom not in BACKBONE_ATOMS} for residue in self.atoms]


@dataclass
class SecondaryStructure:
    """
    DSSP assignment of a chain.

    Attributes:
        dssp (str): One DSSP state per residue (H, G, I, E, B, T, S, '-' or X).
        ladders (List[Tuple[str, Tuple[int, int], Tuple[int, int]]]): Each beta ladder as
            ``(type, (first, last), (first, last))``: 'parallel' or 'antiparallel', then
            the 0-based residue range of each of the two paired stretches.
    """

    dssp: str
    ladders: List[Tuple[str, Tuple[int, int], Tuple[int, int]]]


@dataclass
class Element:
    """
    A run of consecutive residues of one element class.

    Attributes:
        kind (str): 'helix', 'sheet', 'loop' or 'other'.
        start (int): 0-based index of the first residue.
        end (int): 0-based index one past the last residue.
        first_residue (str): PDB residue id of the first residue.
        last_residue (str): PDB residue id of the last residue.
        sequence (str): Sequence of the element.
        dssp (str): DSSP states of the element.
        abego (str): ABEGO bins of the element.
        strand_class (Optional[str]): For strands, 'middle' (paired on both sides),
            'edge' (paired on one side) or None.
        scores (Dict[str, float]): Scores of the element; see ``score_structure``.
        notes (List[str]): Why a score of the element is missing, if one is.
    """

    kind: str
    start: int
    end: int
    first_residue: str
    last_residue: str
    sequence: str
    dssp: str
    abego: str
    strand_class: Optional[str] = None
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.end - self.start


def read_pdb(path: Union[str, Path], chain: Optional[str] = None) -> ProteinChain:
    """
    Read the coordinates of one protein chain from a PDB file.

    Only the first model is read, and the first alternate location of each atom is kept.
    Hydrogens are skipped.
    A residue is kept if it is a standard amino acid (or a common variant such as MSE),
    or if it has N, CA and C atoms; other residues (water, ligands, ions, nucleic acids)
    are skipped. Glycine and residues without a CB get a virtual CB.

    Args:
        path (Union[str, Path]): PDB file, optionally gzipped (.gz).
        chain (Optional[str]): Chain identifier. Default: the first protein chain.

    Returns:
        ProteinChain: Sequence and coordinates of the chain.

    Raises:
        ValueError: If the file has no protein residues, or none in the requested chain.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    residues: Dict[Tuple[str, str], Dict] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            record = line[:6]
            if record.startswith("ENDMDL"):
                break
            if record not in ("ATOM  ", "HETATM"):
                continue
            chain_id = line[21]
            residue_id = line[22:27].strip()
            residue = residues.setdefault(
                (chain_id, residue_id), {"name": line[17:20].strip(), "atoms": {}}
            )
            atom = line[12:16].strip()
            element = line[76:78].strip() or atom.lstrip("0123456789")[:1]
            if element in ("H", "D"):
                continue
            if atom not in residue["atoms"]:
                residue["atoms"][atom] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

    def is_amino_acid(residue: Dict) -> bool:
        return residue["name"] in RESIDUE_CODES or {"N", "CA", "C"} <= residue["atoms"].keys()

    protein = [(key, res) for key, res in residues.items() if is_amino_acid(res)]
    if not protein:
        raise ValueError(f"No protein residues found in {path}.")
    if chain is None:
        chain = protein[0][0][0]
    selected = [(key, res) for key, res in protein if key[0] == chain]
    if not selected:
        found = sorted({key[0] for key, _ in protein})
        raise ValueError(f"Chain {chain!r} not found in {path}; protein chains: {found}.")

    def coordinates(atom: str) -> np.ndarray:
        return np.array([res["atoms"].get(atom, [np.nan] * 3) for _, res in selected], dtype=float)

    n, ca, c, o, cb = (coordinates(atom) for atom in ("N", "CA", "C", "O", "CB"))
    missing_cb = np.isnan(cb).any(axis=1)
    cb[missing_cb] = virtual_cb(n[missing_cb], ca[missing_cb], c[missing_cb])
    return ProteinChain(
        chain_id=chain,
        residue_ids=[key[1] for key, _ in selected],
        sequence="".join(RESIDUE_CODES.get(res["name"], "X") for _, res in selected),
        n=n, ca=ca, c=c, o=o, cb=cb,
        atoms=[{atom: np.array(xyz) for atom, xyz in res["atoms"].items()} for _, res in selected],
    )


def virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Place an ideal CB from the backbone N, CA and C atoms.

    Uses the fixed linear combination for ideal tetrahedral geometry common in protein
    modelling (e.g. trRosetta and ProteinMPNN).

    Args:
        n (np.ndarray): N coordinates, shape (k, 3).
        ca (np.ndarray): CA coordinates, shape (k, 3).
        c (np.ndarray): C coordinates, shape (k, 3).

    Returns:
        np.ndarray: Virtual CB coordinates, shape (k, 3).
    """
    b = ca - n
    v = c - ca
    a = np.cross(b, v)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * v + ca


def chain_breaks(chain: ProteinChain) -> np.ndarray:
    """
    Find the missing peptide bonds of a chain.

    Args:
        chain (ProteinChain): The chain.

    Returns:
        np.ndarray: Boolean array of length n-1; element i is True when residues i and
            i+1 are not bonded (C-N distance above 2.5 angstrom, or an atom missing).
    """
    distance = np.linalg.norm(chain.c[:-1] - chain.n[1:], axis=1)
    return ~(distance <= MAX_PEPTIDE_BOND)


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Dihedral angles in degrees for arrays of four points, shape (k, 3) each."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.sum(b0 * b1, axis=1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=1)[:, None] * b1
    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    return np.degrees(np.arctan2(y, x))


def backbone_torsions(chain: ProteinChain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the backbone torsion angles of a chain.

    ``omega[i]`` is CA(i)-C(i)-N(i+1)-CA(i+1), the peptide bond after residue i.

    Args:
        chain (ProteinChain): The chain.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: phi, psi and omega in degrees, NaN
            where undefined (chain termini, chain breaks, missing atoms).
    """
    size = len(chain)
    phi = np.full(size, np.nan)
    psi = np.full(size, np.nan)
    omega = np.full(size, np.nan)
    if size > 1:
        bonded = ~chain_breaks(chain)
        n, ca, c = chain.n, chain.ca, chain.c
        phi[1:] = np.where(bonded, _dihedral(c[:-1], n[1:], ca[1:], c[1:]), np.nan)
        psi[:-1] = np.where(bonded, _dihedral(n[:-1], ca[:-1], c[:-1], n[1:]), np.nan)
        omega[:-1] = np.where(bonded, _dihedral(ca[:-1], c[:-1], n[1:], ca[1:]), np.nan)
    return phi, psi, omega


def _hbond_matrix(chain: ProteinChain, breaks: np.ndarray) -> np.ndarray:
    """
    Backbone hydrogen bonds by the DSSP electrostatic criterion.

    Returns a boolean matrix ``hbond[i, j]``: the C=O of residue i accepts a hydrogen bond
    from the N-H of residue j. As in DSSP, the amide H lies 1 angstrom from N opposite the
    previous carbonyl, proline and residues after a break have no H, a bond needs an energy
    below -0.5 kcal/mol, and only the two strongest acceptors of each N-H count.
    """
    size = len(chain)
    n, ca, c, o = chain.n, chain.ca, chain.c, chain.o
    h = np.full((size, 3), np.nan)
    if size > 1:
        carbonyl = c[:-1] - o[:-1]
        h[1:] = n[1:] + carbonyl / np.linalg.norm(carbonyl, axis=1)[:, None]
        h[1:][breaks] = np.nan
    h[np.array([aa == "P" for aa in chain.sequence], dtype=bool)] = np.nan

    def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    # rows: acceptor (its O and C); columns: donor (its N and H)
    d_on, d_ch, d_oh, d_cn = distance(o, n), distance(c, h), distance(o, h), distance(c, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = 27.888 * (1 / d_on + 1 / d_ch - 1 / d_oh - 1 / d_cn)  # 0.42 * 0.20 * 332
    too_close = np.minimum.reduce([d_on, d_ch, d_oh, d_cn]) < 0.5
    energy = np.where(too_close, -9.9, np.maximum(np.round(energy, 3), -9.9))
    # DSSP evaluates pairs with CA closer than 9 angstrom, never a residue with itself, and
    # never the C=O of residue i with the N-H of residue i+1
    evaluated = distance(ca, ca) < 9.0
    np.fill_diagonal(evaluated, False)
    evaluated[np.arange(size - 1), np.arange(1, size)] = False
    energy = np.where(evaluated & ~np.isnan(energy), energy, 0.0)

    hbond = np.zeros((size, size), dtype=bool)
    if size > 1:
        strongest = np.argsort(energy, axis=0, kind="stable")[:2]
        for donor in range(size):
            for acceptor in strongest[:, donor]:
                if energy[acceptor, donor] < -0.5:
                    hbond[acceptor, donor] = True
    return hbond


def assign_secondary_structure(chain: ProteinChain) -> SecondaryStructure:
    """
    Assign secondary structure with the DSSP algorithm (Kabsch & Sander 1983).

    Implements the classic DSSP rules, with the priority order of DSSP 2:
    H > E, B > G > I > T > S. Beta bridges are joined into ladders, including across beta
    bulges. Residues missing a backbone atom are assigned 'X' and treated as breaks.

    Args:
        chain (ProteinChain): The chain.

    Returns:
        SecondaryStructure: The DSSP string and the beta ladders.
    """
    size = len(chain)
    complete = ~np.isnan(np.hstack([chain.n, chain.ca, chain.c, chain.o])).any(axis=1)
    breaks = chain_breaks(chain) if size > 1 else np.zeros(0, dtype=bool)
    # DSSP drops residues with an incomplete backbone, which breaks the chain on both sides
    breaks = breaks | ~complete[:-1] | ~complete[1:]
    broken_before = np.concatenate([[0], np.cumsum(breaks)])  # breaks between 0 and i

    def unbroken(first: int, last: int) -> bool:
        return broken_before[last] == broken_before[first]

    hb = _hbond_matrix(chain, breaks)
    ss = ["-"] * size

    # beta bridges between residues i and j >= i + 3, both with bonded neighbours
    ladders: List[Dict] = []
    for i in range(1, size - 4):
        if not unbroken(i - 1, i + 1):
            continue
        for j in range(i + 3, size - 1):
            if not unbroken(j - 1, j + 1):
                continue
            if (hb[i - 1, j] and hb[j, i + 1]) or (hb[j - 1, i] and hb[i, j + 1]):
                kind = "parallel"
            elif (hb[i, j] and hb[j, i]) or (hb[i - 1, j + 1] and hb[j - 1, i + 1]):
                kind = "antiparallel"
            else:
                continue
            for ladder in ladders:
                if ladder["type"] != kind or i != ladder["i"][-1] + 1:
                    continue
                if kind == "parallel" and ladder["j"][-1] + 1 == j:
                    ladder["i"].append(i)
                    ladder["j"].append(j)
                    break
                if kind == "antiparallel" and ladder["j"][0] - 1 == j:
                    ladder["i"].append(i)
                    ladder["j"].insert(0, j)
                    break
            else:
                ladders.append({"type": kind, "i": [i], "j": [j]})

    # join ladders of the same type separated by a beta bulge: a gap of at most one residue
    # on one strand and four on the other (DSSP uses unsigned differences; a negative
    # difference never qualifies)
    def within(difference: int, limit: int) -> bool:
        return 0 <= difference < limit

    ladders.sort(key=lambda ladder: ladder["i"][0])
    a = 0
    while a < len(ladders):
        b = a + 1
        while b < len(ladders):
            first, second = ladders[a], ladders[b]
            ibi, iei, jbi, jei = first["i"][0], first["i"][-1], first["j"][0], first["j"][-1]
            ibj, iej, jbj, jej = second["i"][0], second["i"][-1], second["j"][0], second["j"][-1]
            if (
                first["type"] != second["type"]
                or not unbroken(min(ibi, ibj), max(iei, iej))
                or not unbroken(min(jbi, jbj), max(jei, jej))
                or not within(ibj - iei, 6)
                or (iei >= ibj and ibi <= iej)
            ):
                b += 1
                continue
            if first["type"] == "parallel":
                bulge = (within(jbj - jei, 6) and within(ibj - iei, 3)) or within(jbj - jei, 3)
            else:
                bulge = (within(jbi - jej, 6) and within(ibj - iei, 3)) or within(jbi - jej, 3)
            if bulge:
                first["i"].extend(second["i"])
                if first["type"] == "parallel":
                    first["j"].extend(second["j"])
                else:
                    first["j"] = second["j"] + first["j"]
                del ladders[b]
            else:
                b += 1
        a += 1

    for ladder in ladders:
        state = "E" if len(ladder["i"]) > 1 else "B"
        for side in ("i", "j"):
            for k in range(ladder[side][0], ladder[side][-1] + 1):
                if ss[k] != "E":
                    ss[k] = state

    # n-turns: C=O(i) bonded to N-H(i+n) with no break in between
    turns = {}
    for stride in (3, 4, 5):
        turns[stride] = np.array(
            [i + stride < size and unbroken(i, i + stride) and hb[i, i + stride] for i in range(size)],
            dtype=bool,
        )
    # helices: two consecutive n-turns; alpha first, then 3-10 and pi only on free residues
    for stride, state in ((4, "H"), (3, "G"), (5, "I")):
        for i in range(1, size - stride):
            if turns[stride][i] and turns[stride][i - 1]:
                span = range(i, i + stride)
                if state == "H" or all(ss[k] in ("-", state) for k in span):
                    for k in span:
                        ss[k] = state

    # turns and bends on the remaining residues
    for i in range(1, size - 1):
        if ss[i] != "-":
            continue
        if any(i >= k and turns[stride][i - k] for stride in (3, 4, 5) for k in range(1, stride)):
            ss[i] = "T"
        elif 2 <= i < size - 2 and unbroken(i - 2, i + 2):
            before = chain.ca[i] - chain.ca[i - 2]
            after = chain.ca[i + 2] - chain.ca[i]
            cosine = np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
            if np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > 70.0:
                ss[i] = "S"

    for i in np.flatnonzero(~complete):
        ss[i] = "X"
    return SecondaryStructure(
        dssp="".join(ss),
        ladders=[
            (ladder["type"], (ladder["i"][0], ladder["i"][-1]), (ladder["j"][0], ladder["j"][-1]))
            for ladder in ladders
        ],
    )


def split_elements(chain: ProteinChain, structure: SecondaryStructure, abego: str) -> List[Element]:
    """
    Group consecutive residues of the same element class into elements.

    DSSP states map to classes through ``ELEMENT_CLASSES``. Elements also end at chain
    breaks. Strands get a class from their ladder partners: 'middle' when paired with two
    or more other strands, 'edge' when paired with one.

    Args:
        chain (ProteinChain): The chain.
        structure (SecondaryStructure): Its DSSP assignment.
        abego (str): Its ABEGO string.

    Returns:
        List[Element]: The elements in sequence order.
    """
    labels = [ELEMENT_CLASSES.get(state, "other") for state in structure.dssp]
    breaks = chain_breaks(chain) if len(chain) > 1 else np.zeros(0, dtype=bool)
    elements = []
    start = 0
    for i in range(1, len(chain) + 1):
        if i == len(chain) or labels[i] != labels[start] or breaks[i - 1]:
            elements.append(
                Element(
                    kind=labels[start],
                    start=start,
                    end=i,
                    first_residue=chain.residue_ids[start],
                    last_residue=chain.residue_ids[i - 1],
                    sequence=chain.sequence[start:i],
                    dssp=structure.dssp[start:i],
                    abego=abego[start:i],
                )
            )
            start = i

    element_of = {}
    for index, element in enumerate(elements):
        if element.kind == "sheet":
            element_of.update({k: index for k in range(element.start, element.end)})
    partners: Dict[int, set] = {index: set() for index in set(element_of.values())}
    for _, (i_first, i_last), (j_first, j_last) in structure.ladders:
        side_i = {element_of[k] for k in range(i_first, i_last + 1) if k in element_of}
        side_j = {element_of[k] for k in range(j_first, j_last + 1) if k in element_of}
        for index in side_i:
            partners[index] |= side_j - {index}
        for index in side_j:
            partners[index] |= side_i - {index}
    for index, paired in partners.items():
        if paired:
            elements[index].strand_class = "middle" if len(paired) >= 2 else "edge"
    return elements


def score_helix(
    sequence: str, start: int, end: int, T: float = 25.0, M: float = 0.15, pH: float = 7.0
) -> Dict[str, float]:
    """
    Score one helix of a protein with AGADIR.

    The helix is cut out with two flanking residues on each side: N' and the N-cap before
    it, the C-cap and C' after it. AGADIR needs the real N' residue for the hydrophobic
    staple (N'-N4) and the real C' residue for the Schellman motif (C3-C'). Acetyl and amide
    groups stand in for the rest of the chain (uncharged ends, as inside a protein). Where
    the sequence ends sooner, the window is shorter and an end group takes the missing
    position.

    Args:
        sequence (str): Sequence of the chain, or of the unbroken stretch of it that holds
            the helix.
        start (int): 0-based index of the first helical residue.
        end (int): 0-based index one past the last helical residue.
        T (float): Temperature in Celsius.
        M (float): Ionic strength in mol/L.
        pH (float): pH.

    Returns:
        Dict[str, float]:
            - ``agadir_helix_percent``: mean AGADIR helical propensity (%) of the helical
              residues in the isolated helix peptide.
            - ``agadir_dG_helix``: free energy (kcal/mol) of the observed helical segment,
              caps included; negative values favour the helix.

    Raises:
        ValueError: If the helix window contains a non-standard residue.
        KeyError: If AGADIR lacks a parameter for the window.
    """
    first = max(start - 2, 0)
    last = min(end + 2, len(sequence))
    window = sequence[first:last]
    if "X" in window:
        raise ValueError("The helix contains a non-standard residue.")

    model = AGADIR(method="1s", T=T, M=M, pH=pH)
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.predict(window, ncap="Ac", ccap="Am")
    # positions in the cap-extended chain: Ac, window..., Am
    propensity = result.get_helical_propensity()
    helical = propensity[1 + start - first : 1 + end - first]

    # the observed segment runs from the N-cap (or Ac) to the C-cap (or Am)
    segment_start = start - first if start > 0 else 0
    segment_end = 1 + end - first if end < len(sequence) else len(window) + 1
    length = segment_end - segment_start + 1
    model.energy_calculator = EnergyCalculator(
        seq=window, i=segment_start, j=length, pH=pH, T=T, ionic_strength=M, ncap="Ac", ccap="Am"
    )
    dG = model._calc_dG_Hel(i=segment_start, j=length)
    return {"agadir_helix_percent": float(np.mean(helical)), "agadir_dG_helix": float(dG)}


def score_structure(
    path: Union[str, Path],
    chain: Optional[str] = None,
    T: float = 25.0,
    M: float = 0.15,
    pH: float = 7.0,
    agadir: bool = True,
    surface_cutoff: float = SURFACE_CUTOFF,
) -> Dict[str, object]:
    """
    Split a protein chain into elements and score each one.

    Element scores (in ``Element.scores``):

    - every element: ``abego_total`` and ``abego_mean``, the ABEGO log-odds of the
      element's residues (``element_scores.abego_profile``).
    - every element: ``n_hydrophobic_contacts``, the hydrophobic side-chain contacts
      (``element_scores.hydrophobic_core_clusters``) with at least one residue in it;
      ``buried_npsa`` and ``exposed_npsa`` of its residues (``element_scores.surface_burial``);
      ``n_buried_unsatisfied``, its buried polar atoms left without a hydrogen-bond partner
      (``element_scores.buried_unsatisfied_polar_atoms``).
    - helix: ``agadir_helix_percent`` and ``agadir_dG_helix`` (``score_helix``);
      ``n_core_hydrophobics``.
    - sheet: ``strand_surface_score`` over its exposed residues, ``n_exposed`` and
      ``n_core_hydrophobics``.

    A score that cannot be computed is None, with the reason in ``Element.notes``.

    Args:
        path (Union[str, Path]): PDB file.
        chain (Optional[str]): Chain identifier. Default: the first protein chain.
        T (float): Temperature in Celsius for AGADIR.
        M (float): Ionic strength in mol/L for AGADIR.
        pH (float): pH for AGADIR.
        agadir (bool): Score helices with AGADIR (the slowest step).
        surface_cutoff (float): Side-chain neighbour count below which a residue counts as
            solvent exposed for the strand surface score.

    Returns:
        Dict[str, object]:
            - ``chain``: the ProteinChain.
            - ``dssp`` and ``abego``: per-residue strings.
            - ``burial``: side-chain neighbour count per residue.
            - ``elements``: list of scored Elements.
            - ``hydrophobic_core``: the result of ``hydrophobic_core_clusters``.
            - ``surface``: the result of ``surface_burial``.
            - ``unsatisfied``: the result of ``buried_unsatisfied_polar_atoms``.
            - ``summary``: whole-chain scores (``abego_res_profile``,
              ``abego_res_profile_penalty``, ``one_core_each``, ``two_core_each``,
              ``ss_contributes_core``, ``strand_surface_score``, ``n_hydrophobic``,
              ``hphob_sc_contacts``, ``hphob_sc_degree``, ``largest_hphob_cluster``,
              ``n_hphob_clusters``, ``buried_npsa``, ``buried_npsa_per_residue``,
              ``buried_npsa_afilmvwy_per_residue`` (from hydrophobic residues only, the
              quantity behind the Rocklin 2017 threshold), ``exposed_npsa``, ``buried_psa``,
              ``exposed_psa``, ``buried_unsat_backbone``, ``buried_unsat_sidechain``,
              ``buried_unsat_hydrogens``) and residue counts per element class.
    """
    protein = read_pdb(path, chain=chain)
    phi, psi, omega = backbone_torsions(protein)
    abego = abego_from_torsions(phi, psi, omega)
    structure = assign_secondary_structure(protein)
    elements = split_elements(protein, structure, abego)
    burial = sidechain_neighbors(protein.ca, protein.cb)
    exposed = [bool(value < surface_cutoff) for value in burial]

    sequence = protein.sequence
    whole_chain = abego_profile(sequence, abego)
    ss3 = "".join(
        {"helix": "H", "sheet": "E"}.get(ELEMENT_CLASSES.get(state, "other"), "L")
        for state in structure.dssp
    )
    core = core_participation(sequence, ss3, burial)
    core_counts = {(start, end): n_core for _, start, end, n_core, _ in core["elements"]}
    strand_class = [None] * len(protein)
    for element in elements:
        if element.kind == "sheet":
            strand_class[element.start : element.end] = [element.strand_class] * len(element)
    surface = strand_surface_score(sequence, ss3, exposed, strand_class)
    nonpolar_sidechains = [
        np.array([xyz for atom, xyz in sidechain.items() if atom[0] in "CS"]).reshape(-1, 3)
        for sidechain in protein.sidechains
    ]
    hydrophobic = hydrophobic_core_clusters(sequence, nonpolar_sidechains)
    breaks = chain_breaks(protein) if len(protein) > 1 else np.zeros(0, dtype=bool)
    surface_areas = surface_burial(sequence, protein.atoms, bonded=~breaks)
    buried_nonpolar = surface_areas["reference_nonpolar"] - surface_areas["exposed_nonpolar"]
    unsatisfied = buried_unsatisfied_polar_atoms(
        sequence, protein.atoms, surface_areas["atom_area"], bonded=~breaks
    )

    for element in elements:
        positions = range(element.start, element.end)
        profile = abego_profile(sequence, abego, positions)
        element.scores["abego_total"] = profile["total"]
        element.scores["abego_mean"] = profile["mean"]
        element.scores["n_hydrophobic_contacts"] = sum(
            i in positions or j in positions for i, j in hydrophobic["contacts"]
        )
        element.scores["buried_npsa"] = float(buried_nonpolar[element.start : element.end].sum())
        element.scores["exposed_npsa"] = float(surface_areas["exposed_nonpolar"][element.start : element.end].sum())
        element.scores["n_buried_unsatisfied"] = sum(i in positions for i, _, _ in unsatisfied["unsatisfied"])
        if element.kind in ("helix", "sheet"):
            element.scores["n_core_hydrophobics"] = core_counts.get((element.start, element.end))
        if element.kind == "helix" and agadir:
            # up to two flanking residues on each side, within the unbroken stretch
            first, last = element.start, element.end
            for _ in range(2):
                if first > 0 and not breaks[first - 1]:
                    first -= 1
                if last < len(protein) and not breaks[last - 1]:
                    last += 1
            try:
                element.scores.update(
                    score_helix(sequence[first:last], element.start - first, element.end - first, T=T, M=M, pH=pH)
                )
            except (ValueError, KeyError) as error:
                element.scores.update({"agadir_helix_percent": None, "agadir_dG_helix": None})
                reason = f"missing AGADIR parameter {error}" if isinstance(error, KeyError) else str(error)
                element.notes.append(f"AGADIR not scored: {reason}")
        if element.kind == "sheet":
            element.scores["n_exposed"] = sum(exposed[k] for k in positions)
            element.scores["strand_surface_score"] = sum(
                effect for position, _, _, effect in surface["contributions"] if position in positions
            )

    counts = {kind: 0 for kind in ("helix", "sheet", "loop", "other")}
    for element in elements:
        counts[element.kind] += len(element)
    summary = {
        "n_residues": len(protein),
        **{f"n_{kind}": count for kind, count in counts.items()},
        "abego_res_profile": whole_chain["mean"],
        "abego_res_profile_penalty": whole_chain["penalty"],
        "one_core_each": core["one_core_each"],
        "two_core_each": core["two_core_each"],
        "ss_contributes_core": core["ss_contributes_core"],
        "strand_surface_score": surface["score"],
        "n_hydrophobic": hydrophobic["n_hydrophobic"],
        "hphob_sc_contacts": hydrophobic["n_contacts"],
        "hphob_sc_degree": hydrophobic["contacts_per_residue"],
        "largest_hphob_cluster": hydrophobic["largest_cluster"],
        "n_hphob_clusters": hydrophobic["n_clusters"],
        "buried_npsa": surface_areas["buried_npsa"],
        "buried_npsa_per_residue": surface_areas["buried_npsa"] / len(protein),
        "buried_npsa_afilmvwy_per_residue": float(
            sum(b for aa, b in zip(sequence, buried_nonpolar) if aa in "AFILMVWY") / len(protein)
        ),
        "exposed_npsa": surface_areas["exposed_npsa"],
        "buried_psa": surface_areas["buried_psa"],
        "exposed_psa": surface_areas["exposed_psa"],
        "buried_unsat_backbone": unsatisfied["n_backbone"],
        "buried_unsat_sidechain": unsatisfied["n_sidechain"],
        "buried_unsat_hydrogens": unsatisfied["n_hydrogen"],
    }
    return {
        "chain": protein,
        "dssp": structure.dssp,
        "abego": abego,
        "burial": burial,
        "elements": elements,
        "hydrophobic_core": hydrophobic,
        "surface": surface_areas,
        "unsatisfied": unsatisfied,
        "summary": summary,
    }


def _format_report(path: Union[str, Path], report: Dict[str, object]) -> str:
    """Render the result of ``score_structure`` as a plain-text table."""
    protein = report["chain"]
    lines = [
        f"{path}  chain {protein.chain_id}  {len(protein)} residues",
        f"sequence  {protein.sequence}",
        f"dssp      {report['dssp']}",
        f"abego     {report['abego']}",
        "",
        f"{'#':>3} {'kind':<6} {'residues':<11} {'len':>3}  {'sequence':<20} {'abego':>7}"
        f" {'helix%':>7} {'dG_hel':>7} {'strand':>7} {'surface':>8} {'core':>5} {'contacts':>8}"
        f" {'bNPSA':>6} {'unsat':>5}",
    ]

    def number(value: Optional[float], digits: int) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    for index, element in enumerate(report["elements"], 1):
        scores = element.scores
        residues = f"{element.first_residue}-{element.last_residue}"
        sequence = element.sequence if len(element) <= 20 else element.sequence[:17] + "..."
        core = scores.get("n_core_hydrophobics")
        lines.append(
            f"{index:>3} {element.kind:<6} {residues:<11} {len(element):>3}  {sequence:<20}"
            f" {number(scores['abego_total'], 2):>7}"
            f" {number(scores.get('agadir_helix_percent'), 1):>7}"
            f" {number(scores.get('agadir_dG_helix'), 2):>7}"
            f" {element.strand_class or '':>7}"
            f" {number(scores.get('strand_surface_score'), 3):>8}"
            f" {'' if core is None else core:>5}"
            f" {scores['n_hydrophobic_contacts']:>8}"
            f" {scores['buried_npsa']:>6.0f} {scores['n_buried_unsatisfied']:>5}"
        )
        lines.extend(f"      note: {note}" for note in element.notes)
    lines.extend([
        "",
        "abego: summed ABEGO log-odds of the residues (positive: the sequence suits the backbone)",
        "helix%: AGADIR helicity of the isolated helix; dG_hel: AGADIR free energy of the",
        "  helical segment (kcal/mol); strand: edge or middle strand; surface: amino-acid",
        "  effect of the exposed strand residues (kcal/mol, positive stabilises);",
        "  core: large hydrophobics buried in the core; contacts: hydrophobic side-chain",
        "  contacts involving the element; bNPSA: buried nonpolar surface (square angstrom);",
        "  unsat: buried polar atoms without a hydrogen-bond partner",
        "",
    ])
    width = max(len(key) for key in report["summary"])
    lines.extend(f"{key:<{width}} {value:.4g}" for key, value in report["summary"].items())
    return "\n".join(lines)


def _to_json(report: Dict[str, object]) -> Dict[str, object]:
    """Convert the result of ``score_structure`` to JSON-serialisable types."""
    protein = report["chain"]
    return {
        "chain": protein.chain_id,
        "sequence": protein.sequence,
        "dssp": report["dssp"],
        "abego": report["abego"],
        "residues": [
            {
                "residue": rid, "aa": aa, "dssp": state, "burial": round(float(burial), 3),
                "exposed_npsa": round(float(exposed), 2), "buried_npsa": round(float(reference - exposed), 2),
            }
            for rid, aa, state, burial, exposed, reference in zip(
                protein.residue_ids, protein.sequence, report["dssp"], report["burial"],
                report["surface"]["exposed_nonpolar"], report["surface"]["reference_nonpolar"],
            )
        ],
        "elements": [asdict(element) for element in report["elements"]],
        "buried_unsatisfied": [
            {"residue": protein.residue_ids[i], "atom": atom, "free_hydrogens": free}
            for i, atom, free in report["unsatisfied"]["unsatisfied"]
        ],
        "hydrophobic_clusters": [
            [protein.residue_ids[i] for i in cluster] for cluster in report["hydrophobic_core"]["clusters"]
        ],
        "summary": report["summary"],
    }


def main() -> None:
    """Command-line entry point: print, and optionally save, the element scores of a PDB."""
    parser = argparse.ArgumentParser(description="Split a protein structure into elements and score them.")
    parser.add_argument("pdb", type=Path, help="PDB file (optionally .gz)")
    parser.add_argument("--chain", help="chain identifier (default: first protein chain)")
    parser.add_argument("--T", type=float, default=25.0, help="temperature in Celsius (default 25)")
    parser.add_argument("--M", type=float, default=0.15, help="ionic strength in mol/L (default 0.15)")
    parser.add_argument("--pH", type=float, default=7.0, help="pH (default 7)")
    parser.add_argument("--no-agadir", action="store_true", help="skip the AGADIR helix scores")
    parser.add_argument("--json", type=Path, help="also write the full result to this JSON file")
    args = parser.parse_args()

    try:
        report = score_structure(args.pdb, chain=args.chain, T=args.T, M=args.M, pH=args.pH, agadir=not args.no_agadir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(_format_report(args.pdb, report))
    if args.json:
        args.json.write_text(json.dumps(_to_json(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
