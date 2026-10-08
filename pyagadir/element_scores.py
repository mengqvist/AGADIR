"""
Lightweight scores for the loops and strands of small proteins.

AGADIR scores helices. The functions here score the other elements of a folded
small protein from its sequence, its per-residue secondary structure and a few
backbone-derived quantities. They come from two studies of de novo designed
mini-proteins whose stability was measured by protease resistance:

- Rocklin et al. (2017) Science 357, 168. Global analysis of protein folding
  using massively parallel design, synthesis, and testing.
- Kim et al. (2022) PNAS 119, e2122676119. Dissecting the stability determinants
  of a challenging de novo protein fold using massively parallel design and
  experimentation. Its scoring scripts reuse the Rocklin 2017 tables.

The exposed-strand effects come from a third study by the same laboratory, which
measured the folding free energy of every single mutant of several hundred natural
and designed domains:

- Tsuboyama et al. (2023) Nature 620, 434. Mega-scale experimental analysis of
  protein folding stability in biology and design.

Scores:

- ``abego_profile``: sequence-backbone compatibility, the log-odds of each
  residue's amino acid given the ABEGO bins of it and its two neighbours. This is
  the main loop score, and the whole-chain mean is the published
  ``abego_res_profile`` feature of both studies.
- ``strand_surface_score``: the measured effect on folding free energy of each
  amino acid at solvent-exposed edge and middle strand positions (Tsuboyama 2023).
- ``core_participation``: whether each helix and strand contributes large
  hydrophobic side chains to the core (Rocklin 2017 ``one_core_each``,
  ``two_core_each`` and ``ss_contributes_core``).
- ``hydrophobic_core_clusters``: the contact graph of the large hydrophobic side
  chains and its connected clusters (Kim 2022 ``hphob_sc_contacts`` and related
  features).
- ``sidechain_neighbors``: the cone-weighted side-chain neighbour count that
  defines burial for the strand and core-participation scores.
- ``surface_burial``: solvent-accessible nonpolar and polar surface, exposed in the
  structure and buried relative to an unfolded reference. Buried nonpolar surface
  area (NPSA) was the dominant difference between stable and unstable designs in
  Rocklin 2017.
- ``buried_unsatisfied_polar_atoms``: buried hydrogen-bonding atoms without a
  partner (one of the ten stability determinants of Kim 2022).

The scores are empirical features in the units of their source (log-odds, kcal/mol,
square angstrom). None of them is a folding free energy of the protein.

Positions are 0-based throughout. ``ss`` is a per-residue secondary-structure
string in which only 'H' (helix) and 'E' (strand) are read; every other character
counts as neither. ``pyagadir.pdb_elements`` derives all of the inputs from a PDB
file.
"""

import math
from functools import lru_cache
from importlib.resources import files
from itertools import groupby
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Amino acids the core-participation filters of Rocklin 2017 count as large hydrophobics.
CORE_HYDROPHOBICS = "VILMFYW"

# United-atom radii (angstrom) of the ProtOr set, Tsai, Taylor, Chothia & Gerstein (1999)
# J. Mol. Biol. 290, 253, as distributed with FreeSASA (share/protor.config). Hydrogens are folded
# into their heavy atom, so the areas do not depend on whether a file has hydrogens. Atoms not
# listed below take the radius of their element: sp3 carbon (C4H1-3) 1.88, N 1.64, O (O1H0)
# 1.42, S 1.77.
_SP2_CARBONS = {  # C3H0, 1.61; the backbone carbonyl C is added in atom_radius
    "R": {"CZ"}, "N": {"CG"}, "D": {"CG"}, "Q": {"CD"}, "E": {"CD"}, "H": {"CG"},
    "F": {"CG"}, "W": {"CG", "CD2", "CE2"}, "Y": {"CG", "CZ"},
}
_AROMATIC_CH = {  # C3H1, 1.76
    "H": {"CD2", "CE1"}, "F": {"CD1", "CD2", "CE1", "CE2", "CZ"},
    "W": {"CD1", "CE3", "CZ2", "CZ3", "CH2"}, "Y": {"CD1", "CD2", "CE1", "CE2"},
}
_HYDROXYL_OXYGENS = {  # O2H1, 1.46; OXT is added in atom_radius
    "D": {"OD2"}, "E": {"OE2"}, "S": {"OG"}, "T": {"OG1"}, "Y": {"OH"},
}
_ELEMENT_RADII = {"C": 1.88, "N": 1.64, "O": 1.42, "S": 1.77}

# Hydrogen-bonding heavy atoms: (can donate, can accept, number of polar hydrogens). The
# backbone N of proline has no hydrogen and is not listed. His ring nitrogens can play either
# role depending on the tautomer, so their hydrogen is not counted.
_POLAR_BACKBONE = {"N": (True, False, 1), "O": (False, True, 0), "OXT": (False, True, 0)}
_POLAR_SIDECHAIN = {
    "S": {"OG": (True, True, 1)},
    "T": {"OG1": (True, True, 1)},
    "Y": {"OH": (True, True, 1)},
    "N": {"OD1": (False, True, 0), "ND2": (True, False, 2)},
    "Q": {"OE1": (False, True, 0), "NE2": (True, False, 2)},
    "D": {"OD1": (False, True, 0), "OD2": (False, True, 0)},
    "E": {"OE1": (False, True, 0), "OE2": (False, True, 0)},
    "K": {"NZ": (True, False, 3)},
    "R": {"NE": (True, False, 1), "NH1": (True, False, 2), "NH2": (True, False, 2)},
    "H": {"ND1": (True, True, 0), "NE2": (True, True, 0)},
    "W": {"NE1": (True, False, 1)},
}

# Effect of each amino acid on folding free energy (kcal/mol, positive stabilises) at
# solvent-exposed strand positions, relative to the average of the 19 amino acids at the same
# site. Derived from Tsuboyama et al. (2023), Data_tables_for_figs/dG_site_feature_Fig3.csv
# (SHA-256 79ff1e86...65f5a04aa): per-site ddG of every substitution, measured by cDNA display
# proteolysis at pH 7.4 and room temperature, on AlphaFold models of the domains. Sites are the
# residues this module's pipeline (pdb_elements) calls exposed strand: DSSP E, side-chain
# neighbours below 2.0, middle (paired with two or more strands) or edge (paired with one).
# 1,211 middle sites in 188 domains and 1,158 edge sites in 209 domains; standard errors from
# resampling domains are 0.01-0.06 kcal/mol. Scans of mutant backgrounds of the same domain are
# pooled into one profile per site. Cysteine is omitted: single Cys substitutions read as
# stabilising in every context in this assay (exposed loops included), which the authors
# attribute to disulfide formation, and they leave Cys out of their own analyses.
#
# The values agree with the earlier, figure-read surface-strand effects of Rocklin et al. (2017,
# Figure 4I/J; Spearman 0.94 middle, 0.89 edge), at about 4-5 kcal/mol per stability-score unit.
# The derivation script is kept with the Tsuboyama study material
# (study/Tsuboyama/derive_strand_surface_table.py).
STRAND_SURFACE_EFFECTS: Dict[str, Dict[str, float]] = {
    "middle": {
        "I": 0.61, "Y": 0.51, "V": 0.50, "W": 0.50, "F": 0.47, "L": 0.45, "M": 0.42,
        "T": 0.21, "R": 0.20, "K": 0.09, "H": 0.06, "Q": 0.05, "S": -0.08, "A": -0.20,
        "E": -0.20, "N": -0.35, "G": -0.83, "D": -0.84, "P": -1.55,
    },
    "edge": {
        "I": 0.43, "V": 0.37, "W": 0.37, "F": 0.35, "Y": 0.35, "M": 0.30, "L": 0.27,
        "T": 0.16, "R": 0.12, "H": 0.05, "K": 0.05, "Q": 0.04, "S": -0.01, "A": -0.09,
        "E": -0.12, "N": -0.26, "D": -0.55, "G": -0.67, "P": -1.15,
    },
}


def segments(labels: str) -> List[Tuple[str, int, int]]:
    """
    Split a per-residue label string into runs of identical labels.

    Args:
        labels (str): One label character per residue, e.g. a DSSP string.

    Returns:
        List[Tuple[str, int, int]]: ``(label, start, end)`` for each run, with
            ``start`` inclusive and ``end`` exclusive (0-based).

    Example:
        >>> segments("LLHHHHL")
        [('L', 0, 2), ('H', 2, 6), ('L', 6, 7)]
    """
    runs = []
    start = 0
    for label, group in groupby(labels):
        length = len(list(group))
        runs.append((label, start, start + length))
        start += length
    return runs


def abego_from_torsions(
    phi: Sequence[float], psi: Sequence[float], omega: Sequence[float]
) -> str:
    """
    Classify each residue's backbone torsions into an ABEGO bin.

    The bins and boundaries are those of the Rocklin 2017 and Kim 2022 scripts:

    - O: cis peptide bond, ``|omega| < 90``
    - G: ``phi > 0`` and ``-100 <= psi < 100`` (left-handed helix region)
    - E: ``phi > 0`` otherwise (extended, positive phi)
    - A: ``phi <= 0`` and ``-75 <= psi < 50`` (right-handed helix region)
    - B: ``phi <= 0`` otherwise (beta and polyproline region)

    ``omega[i]`` is the torsion CA(i)-C(i)-N(i+1)-CA(i+1), the peptide bond that follows
    residue i, as in the source scripts. An undefined omega is treated as trans. A residue
    whose phi or psi is undefined (a chain terminus or a chain break) gets 'X'. The source
    scripts instead gave such residues 'O', or a positive phi; this changes only the
    terminal characters, which no score reads.

    Args:
        phi (Sequence[float]): Phi angles in degrees, NaN where undefined.
        psi (Sequence[float]): Psi angles in degrees, NaN where undefined.
        omega (Sequence[float]): Omega angles in degrees, NaN where undefined.

    Returns:
        str: One ABEGO character (A, B, E, G, O or X) per residue.

    Raises:
        ValueError: If the three angle arrays differ in length.
    """
    phi = np.asarray(phi, dtype=float)
    psi = np.asarray(psi, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if not phi.shape == psi.shape == omega.shape:
        raise ValueError("phi, psi and omega must have the same length.")

    abego = []
    for f, s, w in zip(phi, psi, omega):
        if np.isnan(f) or np.isnan(s):
            abego.append("X")
        elif not np.isnan(w) and abs(w) < 90:
            abego.append("O")
        elif f > 0:
            abego.append("G" if -100 <= s < 100 else "E")
        else:
            abego.append("A" if -75 <= s < 50 else "B")
    return "".join(abego)


@lru_cache(maxsize=1)
def load_abego_table() -> Dict[Tuple[str, str], float]:
    """
    Load the Rocklin 2017 ABEGO-triad amino-acid log-odds table.

    The table gives ln(P(aa | triad) / P(aa)) for the central residue of each ABEGO
    triad, from the natural-protein statistics of Rocklin et al. (2017). The scoring
    scripts of Kim et al. (2022) ship the same table.

    Returns:
        Dict[Tuple[str, str], float]: Log-odds keyed by ``(triad, amino_acid)``.
    """
    path = files("pyagadir.data.params").joinpath("abego_res_profile_table")
    table = {}
    lines = path.read_text().splitlines()
    header = lines[0].split()
    column = header.index("log_aa_freq_given_triad_over_aa_freq")
    for line in lines[1:]:
        fields = line.split()
        table[(fields[0], fields[1])] = float(fields[column])
    return table


def abego_profile(
    sequence: str, abego: str, positions: Optional[Sequence[int]] = None
) -> Dict[str, Union[float, int, np.ndarray]]:
    """
    Score how well a sequence suits its local backbone conformation.

    Each residue i is scored with the log-odds of its amino acid given the ABEGO triad
    ``abego[i-1:i+2]``. Positive values mean that natural proteins put this amino acid in
    this local conformation more often than chance. Glycine at positive-phi loop positions
    (G and E bins), for example, scores high. The first two and last two residues of the
    chain are not scored, nor are residues whose triad is absent from the table (triads
    with a cis or undefined bin).

    With ``positions`` left as None the whole chain is scored, and ``mean`` and
    ``penalty`` are the published ``abego_res_profile`` and ``abego_res_profile_penalty``
    of Rocklin et al. (2017) and Kim et al. (2022). Passing the positions of one loop
    scores that loop with its flanking residues as context.

    Args:
        sequence (str): One-letter amino-acid sequence of the chain.
        abego (str): ABEGO string of the chain, aligned with ``sequence``.
        positions (Optional[Sequence[int]]): 0-based positions to score. Default: all.

    Returns:
        Dict[str, Union[float, int, np.ndarray]]:
            - ``total``: sum of the log-odds over the scored positions.
            - ``mean``: ``total`` divided by the number of positions considered, with
              unscored positions counting as zero (the published convention).
            - ``penalty``: as ``mean``, summing only the negative log-odds.
            - ``n_scored``: number of positions with a log-odds value.
            - ``per_residue``: log-odds of every chain position, NaN where unscored.

    Raises:
        ValueError: If the sequence and ABEGO strings are not aligned, or a position is
            out of range.
    """
    if len(sequence) != len(abego):
        raise ValueError("sequence and abego must have the same length.")
    table = load_abego_table()

    per_residue = np.full(len(sequence), np.nan)
    for i in range(2, len(sequence) - 2):
        key = (abego[i - 1 : i + 2], sequence[i])
        if key in table:
            per_residue[i] = table[key]

    if positions is None:
        positions = range(len(sequence))
    positions = list(positions)
    if any(p < 0 or p >= len(sequence) for p in positions):
        raise ValueError("positions must lie within the sequence.")

    values = per_residue[positions]
    scored = values[~np.isnan(values)]
    n_considered = max(len(positions), 1)
    return {
        "total": float(scored.sum()),
        "mean": float(scored.sum() / n_considered),
        "penalty": float(np.minimum(scored, 0).sum() / n_considered),
        "n_scored": int(len(scored)),
        "per_residue": per_residue,
    }


def strand_surface_score(
    sequence: str,
    ss: str,
    exposed: Sequence[bool],
    strand_class: Sequence[Optional[str]],
    effects: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, object]:
    """
    Sum the measured amino-acid effects at solvent-exposed strand positions.

    Tsuboyama et al. (2023) measured the folding free energy of every substitution at about
    2,400 solvent-exposed strand positions; ``STRAND_SURFACE_EFFECTS`` holds the mean effect
    of each amino acid relative to the average amino acid, separately for middle strands
    (paired on both sides) and edge strands (paired on one side). Beta-branched and
    aromatic residues stabilise; Gly, Asp and especially Pro destabilise, more so in middle
    strands. This function adds up those effects over the exposed strand residues of a
    protein, in kcal/mol.

    Only residues with ``ss[i] == 'E'`` and ``exposed[i]`` true are scored. Buried strand
    residues, helices and loops contribute nothing: the effects were measured at surface
    sites and say nothing about the core. A scored residue with no strand class, or an
    amino acid missing from the table (cysteine, whose measured effects are an assay
    artefact, and non-standard residues), is skipped and listed in ``unscored``.

    Args:
        sequence (str): One-letter amino-acid sequence.
        ss (str): Per-residue secondary structure; 'E' marks strand residues.
        exposed (Sequence[bool]): True for solvent-exposed residues.
        strand_class (Sequence[Optional[str]]): 'middle', 'edge' or None per residue.
        effects (Optional[Mapping[str, Mapping[str, float]]]): Replacement effect table
            ``{'middle': {aa: value}, 'edge': {aa: value}}``. Default:
            ``STRAND_SURFACE_EFFECTS``.

    Returns:
        Dict[str, object]:
            - ``score``: sum over the scored residues in kcal/mol (positive is stabilising).
            - ``middle_score`` and ``edge_score``: the sum split by strand class.
            - ``n_middle`` and ``n_edge``: number of scored residues in each class.
            - ``contributions``: ``(position, aa, strand_class, effect)`` per scored residue.
            - ``unscored``: ``(position, reason)`` for exposed strand residues left out.

    Raises:
        ValueError: If the inputs are not aligned, or a strand class is not 'middle',
            'edge' or None.
    """
    if not len(sequence) == len(ss) == len(exposed) == len(strand_class):
        raise ValueError("sequence, ss, exposed and strand_class must be aligned.")
    table = STRAND_SURFACE_EFFECTS if effects is None else effects
    for kind in ("middle", "edge"):
        if kind not in table:
            raise ValueError(f"effects must contain a '{kind}' table.")

    sums = {"middle": 0.0, "edge": 0.0}
    counts = {"middle": 0, "edge": 0}
    contributions = []
    unscored = []
    for i, (aa, state, is_exposed, kind) in enumerate(zip(sequence, ss, exposed, strand_class)):
        if state != "E" or not is_exposed:
            continue
        if kind is None:
            unscored.append((i, "no strand class"))
            continue
        if kind not in sums:
            raise ValueError(f"Invalid strand class {kind!r} at position {i}.")
        if aa not in table[kind]:
            unscored.append((i, f"no {kind}-strand effect for {aa}"))
            continue
        effect = float(table[kind][aa])
        sums[kind] += effect
        counts[kind] += 1
        contributions.append((i, aa, kind, effect))

    return {
        "score": math.fsum(c[3] for c in contributions),
        "middle_score": sums["middle"],
        "edge_score": sums["edge"],
        "n_middle": counts["middle"],
        "n_edge": counts["edge"],
        "contributions": contributions,
        "unscored": unscored,
    }


def sidechain_neighbors(
    ca: np.ndarray,
    cb: np.ndarray,
    distance_midpoint: float = 9.0,
    angle_shift: float = 0.5,
    angle_exponent: float = 2.0,
) -> np.ndarray:
    """
    Count the residues in the cone that each side chain points into.

    For residue i, every other residue j contributes

        1 / (1 + exp(d_ij - distance_midpoint))
        * (max(0, cos(theta_ij) + angle_shift) / (1 + angle_shift)) ** angle_exponent

    where d_ij is the distance from CB(i) to CA(j) and theta_ij the angle between the
    CA(i)->CB(i) vector and the CB(i)->CA(j) vector. A side chain pointing into the protein
    collects many neighbours; one pointing into solvent collects few. This is the burial
    measure of the Kim et al. (2022) scripts, which has the functional form of Rosetta's
    side-chain-neighbour layer selector. With the cutoffs of ``core_participation`` it
    reproduces the Rosetta core-participation filters of Rocklin et al. (2017) on their
    designs. Counts scale with protein size and packing: the core cutoff of 4.0 was set
    for 40-residue designs, and core residues of larger proteins can fall near or below it.

    Supply a virtual CB for glycine (``pdb_elements.virtual_cb``): with CB equal to CA the
    side-chain direction is undefined and the count is NaN.

    Args:
        ca (np.ndarray): CA coordinates, shape (n, 3).
        cb (np.ndarray): CB coordinates, shape (n, 3).
        distance_midpoint (float): Distance (angstrom) at which the distance weight is 0.5.
        angle_shift (float): Shift added to the cosine before clipping at zero.
        angle_exponent (float): Exponent of the angular weight.

    Returns:
        np.ndarray: Neighbour count per residue, shape (n,).

    Raises:
        ValueError: If the coordinate arrays do not both have shape (n, 3).
    """
    ca = np.asarray(ca, dtype=float)
    cb = np.asarray(cb, dtype=float)
    if ca.ndim != 2 or ca.shape[1] != 3 or ca.shape != cb.shape:
        raise ValueError("ca and cb must both have shape (n, 3).")

    axis = cb - ca
    length = np.linalg.norm(axis, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = axis / length[:, None]
        toward = ca[None, :, :] - cb[:, None, :]
        distance = np.linalg.norm(toward, axis=2)
        direction = toward / distance[:, :, None]
        cosine = np.einsum("ijk,ik->ij", direction, unit)
    angular = (np.maximum(0.0, cosine + angle_shift) / (1.0 + angle_shift)) ** angle_exponent
    radial = 1.0 / (1.0 + np.exp(distance - distance_midpoint))
    contribution = radial * angular
    np.fill_diagonal(contribution, 0.0)

    counts = np.nansum(contribution, axis=1)
    counts[~(length > 0)] = np.nan
    return counts


def hydrophobic_core_clusters(
    sequence: str,
    sidechain_atoms: Sequence[np.ndarray],
    cutoff: float = 4.3,
    min_separation: int = 3,
) -> Dict[str, object]:
    """
    Build the contact graph of the large hydrophobic side chains and find its clusters.

    The nodes are the large hydrophobic residues (FILMVWY). Two of them are in contact when
    any pair of their side-chain atoms is closer than ``cutoff`` and they are at least
    ``min_separation`` apart in sequence. Kim et al. (2022) found the number of contacts
    second only to the number of large hydrophobics among their stability determinants.
    A single large cluster means one connected hydrophobic core; several small clusters
    mean a fragmented core.

    The source script considered every side-chain atom whose name lacks the letter H. That
    rule drops hydrogens but also the Trp CH2 carbon, and keeps the polar Trp NE1. Pass the
    nonpolar heavy atoms (C and S) instead, as ``pdb_elements`` does. On the 3,862 round-4
    designs of Rocklin et al. (2017) the two atom sets give identical features for 98% of
    designs, and correlations of at least 0.997 with the authors' values.

    Args:
        sequence (str): One-letter amino-acid sequence.
        sidechain_atoms (Sequence[np.ndarray]): Per residue, the coordinates of the
            side-chain atoms to consider, shape (k, 3); k may be zero.
        cutoff (float): Contact distance in angstrom (strict inequality).
        min_separation (int): Minimum sequence separation of a contact.

    Returns:
        Dict[str, object]:
            - ``n_hydrophobic``: number of large hydrophobic residues (graph nodes).
            - ``n_contacts``: number of contacts (graph edges); ``hphob_sc_contacts``.
            - ``contacts_per_residue``: contacts per hydrophobic residue; ``hphob_sc_degree``.
            - ``largest_cluster``: residues in the largest cluster; ``largest_hphob_cluster``.
            - ``n_clusters``: number of clusters, isolated residues included;
              ``n_hphob_clusters``.
            - ``contacts``: the contacts as 0-based ``(i, j)`` pairs.
            - ``clusters``: the clusters as sorted lists of 0-based positions, largest first.

    Raises:
        ValueError: If ``sidechain_atoms`` is not aligned with the sequence.
    """
    if len(sequence) != len(sidechain_atoms):
        raise ValueError("sequence and sidechain_atoms must be aligned.")
    nodes = [i for i, aa in enumerate(sequence) if aa in CORE_HYDROPHOBICS]
    coordinates = [np.asarray(sidechain_atoms[i], dtype=float).reshape(-1, 3) for i in nodes]
    owner = np.concatenate([[i] * len(xyz) for i, xyz in zip(nodes, coordinates)]).astype(int)

    contacts = set()
    if len(owner):
        atoms = np.concatenate(coordinates)
        close = np.linalg.norm(atoms[:, None, :] - atoms[None, :, :], axis=2) < cutoff
        first, second = np.nonzero(close)
        for i, j in zip(owner[first], owner[second]):
            if j - i >= min_separation:
                contacts.add((int(i), int(j)))
    contacts = sorted(contacts)

    neighbours = {i: set() for i in nodes}
    for i, j in contacts:
        neighbours[i].add(j)
        neighbours[j].add(i)
    clusters = []
    unvisited = set(nodes)
    while unvisited:
        stack = [min(unvisited)]
        cluster = set()
        while stack:
            i = stack.pop()
            if i not in cluster:
                cluster.add(i)
                stack.extend(neighbours[i] - cluster)
        unvisited -= cluster
        clusters.append(sorted(cluster))
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]))

    return {
        "n_hydrophobic": len(nodes),
        "n_contacts": len(contacts),
        "contacts_per_residue": len(contacts) / max(len(nodes), 1),
        "largest_cluster": len(clusters[0]) if clusters else 0,
        "n_clusters": len(clusters),
        "contacts": contacts,
        "clusters": clusters,
    }


def core_participation(
    sequence: str,
    ss: str,
    burial: Sequence[float],
    core_cutoff: float = 4.0,
    boundary_cutoff: float = 2.0,
    min_helix_length: int = 4,
    min_strand_length: int = 3,
) -> Dict[str, object]:
    """
    Check that every helix and strand anchors itself in the hydrophobic core.

    Rocklin et al. (2017) found that designs were more often stable when each secondary
    structure element buried at least one large hydrophobic side chain (VILMFYW). For every
    helix (run of 'H' of at least ``min_helix_length``) and strand (run of 'E' of at least
    ``min_strand_length``) this counts the large hydrophobics whose burial reaches
    ``core_cutoff`` (core) or ``boundary_cutoff`` (core or boundary), and reports the
    fraction of elements that pass. The defaults follow the Rosetta filters behind the
    published features: side-chain-neighbour core at 4.0 for ``one_core_each`` and
    ``two_core_each``, and the default surface boundary of 2.0 for ``ss_contributes_core``.

    Args:
        sequence (str): One-letter amino-acid sequence.
        ss (str): Per-residue secondary structure; 'H' marks helix and 'E' strand residues.
        burial (Sequence[float]): Per-residue burial, e.g. from ``sidechain_neighbors``.
        core_cutoff (float): Minimum burial of a core residue.
        boundary_cutoff (float): Minimum burial of a core-or-boundary residue.
        min_helix_length (int): Shortest helix that counts as an element.
        min_strand_length (int): Shortest strand that counts as an element.

    Returns:
        Dict[str, object]:
            - ``one_core_each``: fraction of elements with at least one core hydrophobic.
            - ``two_core_each``: fraction with at least two.
            - ``ss_contributes_core``: fraction with at least one hydrophobic in the core
              or boundary layer.
            - ``elements``: per element, ``(ss, start, end, n_core, n_core_or_boundary)``.
            All fractions are 0.0 when there are no elements.

    Raises:
        ValueError: If the inputs are not aligned.
    """
    burial = np.asarray(burial, dtype=float)
    if not len(sequence) == len(ss) == len(burial):
        raise ValueError("sequence, ss and burial must be aligned.")
    min_length = {"H": min_helix_length, "E": min_strand_length}

    elements = []
    for state, start, end in segments(ss):
        if state not in min_length or end - start < min_length[state]:
            continue
        hydrophobic = np.array([aa in CORE_HYDROPHOBICS for aa in sequence[start:end]])
        n_core = int(np.sum(hydrophobic & (burial[start:end] >= core_cutoff)))
        n_boundary = int(np.sum(hydrophobic & (burial[start:end] >= boundary_cutoff)))
        elements.append((state, start, end, n_core, n_boundary))

    def fraction(passes: List[bool]) -> float:
        return float(np.mean(passes)) if passes else 0.0

    return {
        "one_core_each": fraction([e[3] >= 1 for e in elements]),
        "two_core_each": fraction([e[3] >= 2 for e in elements]),
        "ss_contributes_core": fraction([e[4] >= 1 for e in elements]),
        "elements": elements,
    }


def atom_radius(aa: str, atom: str) -> float:
    """
    United-atom radius of a heavy atom, from the ProtOr set (Tsai et al. 1999).

    Args:
        aa (str): One-letter code of the residue.
        atom (str): PDB atom name, e.g. 'CB'.

    Returns:
        float: Radius in angstrom. Atoms of non-standard residues take the radius of their
            element (1.80 for an unknown element).
    """
    if atom == "C" or atom in _SP2_CARBONS.get(aa, ()):
        return 1.61
    if atom in _AROMATIC_CH.get(aa, ()):
        return 1.76
    if atom == "OXT" or atom in _HYDROXYL_OXYGENS.get(aa, ()):
        return 1.46
    return _ELEMENT_RADII.get(atom[:1], 1.80)


def _sphere_points(n_points: int) -> np.ndarray:
    """Unit vectors spread evenly over a sphere (golden-section spiral)."""
    k = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * k / n_points
    r = np.sqrt(1.0 - z * z)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * k
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def solvent_accessible_area(
    coords: np.ndarray, radii: Sequence[float], probe: float = 1.4, n_points: int = 200
) -> np.ndarray:
    """
    Solvent-accessible surface area of each atom (Shrake & Rupley 1973).

    Each atom is a sphere of its radius plus the probe radius, sampled with ``n_points``
    evenly spaced points. An atom's area is the fraction of its points that lie inside no other
    sphere, times the sphere's area. Areas resolve to about 0.5 square angstrom per point for
    N and O atoms at the defaults.

    Args:
        coords (np.ndarray): Atom coordinates, shape (n, 3).
        radii (Sequence[float]): Atom radii in angstrom, without the probe.
        probe (float): Probe (water) radius in angstrom.
        n_points (int): Sample points per atom.

    Returns:
        np.ndarray: Area of each atom in square angstrom, shape (n,).
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    extended = np.asarray(radii, dtype=float) + probe
    unit = _sphere_points(n_points)
    area = np.zeros(len(coords))
    for i in range(len(coords)):
        distance = np.linalg.norm(coords - coords[i], axis=1)
        close = distance < extended + extended[i]
        close[i] = False
        neighbours = np.flatnonzero(close)
        points = coords[i] + extended[i] * unit
        if len(neighbours):
            gaps = points[:, None, :] - coords[neighbours][None, :, :]
            buried = (np.einsum("pnk,pnk->pn", gaps, gaps) < extended[neighbours] ** 2).any(axis=1)
            exposed = 1.0 - buried.mean()
        else:
            exposed = 1.0
        area[i] = 4.0 * np.pi * extended[i] ** 2 * exposed
    return area


def surface_burial(
    sequence: str,
    atoms: Sequence[Mapping[str, np.ndarray]],
    bonded: Optional[Sequence[bool]] = None,
    probe: float = 1.4,
    n_points: int = 200,
) -> Dict[str, object]:
    """
    Measure the nonpolar and polar surface each residue exposes and buries.

    Exposed areas are solvent-accessible areas in the structure (``solvent_accessible_area``,
    ProtOr radii). Nonpolar atoms are carbon and sulfur; polar atoms are nitrogen and oxygen.
    The unfolded reference of each residue is the residue alone with the backbone atoms of
    its two neighbours, in their conformation in the structure: a Gly-X-Gly peptide with the
    native backbone and rotamer. Buried area is reference minus exposed area. It counts
    burial by everything beyond the immediate neighbours, from helical i+3/i+4 contacts to
    tertiary packing, and is never negative.

    Rocklin et al. (2017) found buried NPSA the dominant difference between stable and
    unstable designs: none of their designs below 32 square angstrom per residue was stable.
    Their value (buried NPSA of AFILMVWY residues per residue) came from Rosetta with explicit
    hydrogens and a fixed reference table. The same quantity computed from this function
    correlates with Rosetta's at r = 0.98 on 5,618 designs of Kim et al. (2022) and r = 0.86 on
    3,862 round-4 designs of Rocklin et al., with Rosetta's values 16-27% larger. Their
    threshold is about 25 square angstrom per residue on this scale.

    Args:
        sequence (str): One-letter sequence.
        atoms (Sequence[Mapping[str, np.ndarray]]): Per residue, the coordinates of its heavy
            atoms keyed by PDB atom name.
        bonded (Optional[Sequence[bool]]): Length n-1; whether residues i and i+1 are joined
            by a peptide bond. Default: all joined.
        probe (float): Probe radius in angstrom.
        n_points (int): Sample points per atom.

    Returns:
        Dict[str, object]:
            - ``exposed_nonpolar``, ``exposed_polar``: per-residue areas in the structure.
            - ``reference_nonpolar``, ``reference_polar``: per-residue areas in the reference.
            - ``buried_npsa``, ``exposed_npsa``, ``buried_psa``, ``exposed_psa``: chain totals.
            - ``atom_area``: per residue, the area of each atom keyed by name.
            All areas in square angstrom.

    Raises:
        ValueError: If the inputs are not aligned.
    """
    n = len(sequence)
    if len(atoms) != n or (bonded is not None and len(bonded) != max(n - 1, 0)):
        raise ValueError("sequence, atoms and bonded must be aligned.")
    bonded = [True] * max(n - 1, 0) if bonded is None else list(bonded)

    def flatten(residues: Sequence[int], backbone_only: Sequence[int] = ()) -> Tuple[list, np.ndarray, np.ndarray]:
        keys, coords, radii = [], [], []
        for i in residues:
            for atom, xyz in atoms[i].items():
                if i in backbone_only and atom not in ("N", "CA", "C", "O"):
                    continue
                keys.append((i, atom))
                coords.append(xyz)
                radii.append(atom_radius(sequence[i], atom))
        return keys, np.array(coords, dtype=float).reshape(-1, 3), np.array(radii)

    def split(keys: list, area: np.ndarray, residue: int) -> Tuple[float, float]:
        nonpolar = sum(a for (i, atom), a in zip(keys, area) if i == residue and atom[0] in "CS")
        polar = sum(a for (i, atom), a in zip(keys, area) if i == residue and atom[0] in "NO")
        return nonpolar, polar

    keys, coords, radii = flatten(range(n))
    area = solvent_accessible_area(coords, radii, probe, n_points)
    atom_area: List[Dict[str, float]] = [dict() for _ in range(n)]
    for (i, atom), a in zip(keys, area):
        atom_area[i][atom] = float(a)

    exposed = np.zeros((n, 2))
    reference = np.zeros((n, 2))
    for i in range(n):
        exposed[i] = split(keys, area, i)
        flanks = [j for j in (i - 1, i + 1) if 0 <= j < n and bonded[min(i, j)]]
        ref_keys, ref_coords, ref_radii = flatten([i] + flanks, backbone_only=flanks)
        reference[i] = split(ref_keys, solvent_accessible_area(ref_coords, ref_radii, probe, n_points), i)

    return {
        "exposed_nonpolar": exposed[:, 0],
        "exposed_polar": exposed[:, 1],
        "reference_nonpolar": reference[:, 0],
        "reference_polar": reference[:, 1],
        "buried_npsa": float(np.sum(reference[:, 0] - exposed[:, 0])),
        "exposed_npsa": float(np.sum(exposed[:, 0])),
        "buried_psa": float(np.sum(reference[:, 1] - exposed[:, 1])),
        "exposed_psa": float(np.sum(exposed[:, 1])),
        "atom_area": atom_area,
    }


def buried_unsatisfied_polar_atoms(
    sequence: str,
    atoms: Sequence[Mapping[str, np.ndarray]],
    atom_area: Sequence[Mapping[str, float]],
    bonded: Optional[Sequence[bool]] = None,
    hbond_distance: float = 3.5,
    atom_burial_cutoff: float = 0.1,
    residue_surface_cutoff: float = 20.0,
) -> Dict[str, object]:
    """
    Count buried hydrogen-bonding atoms that have no hydrogen-bond partner.

    A buried polar group that loses its hydrogen bonds to water on folding and finds no
    partner in the protein costs stability; Kim et al. (2022) kept buried unsatisfied polar
    atoms among their ten stability determinants. A polar atom (``_POLAR_BACKBONE``,
    ``_POLAR_SIDECHAIN``) is buried when its solvent-accessible area is below
    ``atom_burial_cutoff`` and its residue's total area is at most
    ``residue_surface_cutoff``. The second condition ignores surface residues, whose polar
    atoms are often covered only by a flexible side chain. Both defaults are those of the
    Rosetta BuriedUnsatHbonds filter as Kim et al. ran it, although Rosetta measures burial
    with its own surface method (VSASA). A donor is satisfied by an acceptor of another
    residue within ``hbond_distance`` (heavy atoms), and an acceptor by a donor. Where the
    hydrogen position is fixed by the backbone (the amide N-H, placed as in DSSP), the bond
    must also meet the Baker & Hubbard (1984) geometry: H...A at most 2.5 angstrom and
    N-H...A at least 120 degrees. Side-chain hydrogens are not placed, so side-chain donors use
    distance alone, and a donor's hydrogens count as satisfied one per acceptor partner.
    Peptide-bonded N and O are never partners.

    This is a geometric count, not Rosetta's. On 5,618 designs of Kim et al. (2022) it agrees
    moderately with Rosetta's side-chain counts (r about 0.6) and poorly with its backbone and
    hydrogen counts (r about 0.2); both versions correlate only weakly with measured stability.

    Args:
        sequence (str): One-letter sequence.
        atoms (Sequence[Mapping[str, np.ndarray]]): Per residue, heavy-atom coordinates keyed
            by PDB atom name.
        atom_area (Sequence[Mapping[str, float]]): Per residue, the solvent-accessible area of
            each atom, e.g. ``surface_burial(...)["atom_area"]``.
        bonded (Optional[Sequence[bool]]): Length n-1; whether residues i and i+1 are joined.
            Default: all joined. The first residue's N counts three hydrogens (two for Pro).
        hbond_distance (float): Maximum donor-acceptor distance in angstrom.
        atom_burial_cutoff (float): Area (square angstrom) below which a polar atom is buried.
        residue_surface_cutoff (float): Residue area (square angstrom) above which the residue
            counts as surface and its atoms are not considered.

    Returns:
        Dict[str, object]:
            - ``n_backbone``: unsatisfied buried backbone N and O atoms.
            - ``n_sidechain``: unsatisfied buried side-chain polar atoms.
            - ``n_hydrogen``: polar hydrogens of buried donors left without an acceptor.
            - ``unsatisfied``: ``(position, atom, n_free_hydrogens)`` for every buried polar
              atom with an unsatisfied heavy atom or hydrogen.

    Raises:
        ValueError: If the inputs are not aligned.
    """
    n = len(sequence)
    if not len(atoms) == len(atom_area) == n:
        raise ValueError("sequence, atoms and atom_area must be aligned.")
    bonded = [True] * max(n - 1, 0) if bonded is None else list(bonded)

    polar = []  # (position, atom, donor, acceptor, hydrogens, backbone)
    for i, (aa, residue) in enumerate(zip(sequence, atoms)):
        for atom in residue:
            if atom in _POLAR_BACKBONE:
                donor, acceptor, hydrogens = _POLAR_BACKBONE[atom]
                if atom == "N":
                    terminal = i == 0
                    if aa == "P" and not terminal:
                        continue
                    hydrogens = (2 if aa == "P" else 3) if terminal else 1
                polar.append((i, atom, donor, acceptor, hydrogens, True))
            elif atom in _POLAR_SIDECHAIN.get(aa, {}):
                polar.append((i, atom, *_POLAR_SIDECHAIN[aa][atom], False))
    if not polar:
        return {"n_backbone": 0, "n_sidechain": 0, "n_hydrogen": 0, "unsatisfied": []}

    coords = np.array([atoms[i][atom] for i, atom, *_ in polar], dtype=float)
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    residue_area = [sum(areas.values()) for areas in atom_area]

    # amide hydrogens, 1 angstrom from N opposite the previous carbonyl (as in DSSP)
    amide_h = {}
    for i in range(1, n):
        if bonded[i - 1] and sequence[i] != "P" and "N" in atoms[i] and {"C", "O"} <= atoms[i - 1].keys():
            carbonyl = np.asarray(atoms[i - 1]["C"], float) - np.asarray(atoms[i - 1]["O"], float)
            amide_h[i] = np.asarray(atoms[i]["N"], float) + carbonyl / np.linalg.norm(carbonyl)

    def hydrogen_bond_geometry(donor: Tuple, acceptor_xyz: np.ndarray) -> bool:
        i, atom = donor[:2]
        if atom != "N" or i not in amide_h:
            return True
        to_donor = np.asarray(atoms[i]["N"], float) - amide_h[i]
        to_acceptor = acceptor_xyz - amide_h[i]
        cosine = np.dot(to_donor, to_acceptor) / (np.linalg.norm(to_donor) * np.linalg.norm(to_acceptor))
        return np.linalg.norm(to_acceptor) <= 2.5 and cosine <= np.cos(np.radians(120.0))

    def peptide_bonded(a: Tuple, b: Tuple) -> bool:
        (i, atom_i), (j, atom_j) = a[:2], b[:2]
        if j == i + 1 and atom_i == "O" and atom_j == "N":
            return bonded[i]
        if i == j + 1 and atom_i == "N" and atom_j == "O":
            return bonded[j]
        return False

    counts = {"n_backbone": 0, "n_sidechain": 0, "n_hydrogen": 0}
    unsatisfied = []
    for k, site in enumerate(polar):
        i, atom, donor, acceptor, hydrogens, backbone = site
        if atom_area[i].get(atom, 0.0) >= atom_burial_cutoff or residue_area[i] > residue_surface_cutoff:
            continue
        accepting, donating = 0, 0  # partners that accept from / donate to this atom
        for m, other in enumerate(polar):
            if other[0] == i or distance[k, m] > hbond_distance or peptide_bonded(site, other):
                continue
            accepting += donor and other[3] and hydrogen_bond_geometry(site, coords[m])
            donating += acceptor and other[2] and hydrogen_bond_geometry(other, coords[k])
        free_hydrogens = max(hydrogens - accepting, 0)
        if accepting + donating == 0:
            counts["n_backbone" if backbone else "n_sidechain"] += 1
        counts["n_hydrogen"] += free_hydrogens
        if accepting + donating == 0 or free_hydrogens:
            unsatisfied.append((i, atom, free_hydrogens))
    return {**counts, "unsatisfied": unsatisfied}
