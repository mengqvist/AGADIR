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

Scores:

- ``abego_profile``: sequence-backbone compatibility, the log-odds of each
  residue's amino acid given the ABEGO bins of it and its two neighbours. This is
  the main loop score, and the whole-chain mean is the published
  ``abego_res_profile`` feature of both studies.
- ``strand_surface_score``: the measured effect on stability of each amino acid
  at solvent-exposed edge and middle strand positions (Rocklin 2017, Figure 4I/J).
- ``core_participation``: whether each helix and strand contributes large
  hydrophobic side chains to the core (Rocklin 2017 ``one_core_each``,
  ``two_core_each`` and ``ss_contributes_core``).
- ``hydrophobic_core_clusters``: the contact graph of the large hydrophobic side
  chains and its connected clusters (Kim 2022 ``hphob_sc_contacts`` and related
  features).
- ``sidechain_neighbors``: the cone-weighted side-chain neighbour count that
  defines burial for the strand and core-participation scores.

The scores are empirical features in the units of their source (log-odds, or
protease stability-score units). None of them is a folding free energy.

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

# Mean stability-score effect of each amino acid at solvent-exposed strand positions, from
# Rocklin et al. (2017) Figure 4I (middle strands) and 4J (edge strands). Positive values
# stabilise. The units are the paper's consensus stability score (log10 protease EC50 above
# the unfolded-state prediction), not kcal/mol.
#
# The values were read from the bar heights of the vector PDF of the figure (page 6 of the
# article, SHA-256 66724a9f...c4d19d), calibrated on its printed axis ticks, and are therefore
# approximate. The proline bars run off the axis; their printed labels (-0.48 and -0.28) are
# used. The figure labels 19 amino acids. An unlabeled bar sits at the wild-type marker and
# cannot be assigned to cysteine with certainty, so cysteine has no value and is not scored.
STRAND_SURFACE_EFFECTS: Dict[str, Dict[str, float]] = {
    "middle": {
        "Y": 0.09252, "W": 0.0869, "F": 0.08106, "I": 0.08133, "V": 0.0712,
        "L": 0.04786, "H": 0.04248, "M": 0.03314, "T": 0.03245, "R": 0.02587,
        "E": 0.02407, "Q": 0.02028, "A": 0.0063, "K": -0.00158, "S": -0.01022,
        "N": -0.01367, "D": -0.0982, "G": -0.11079, "P": -0.48,
    },
    "edge": {
        "W": 0.06351, "Y": 0.05213, "F": 0.05272, "V": 0.0505, "I": 0.04463,
        "M": 0.0389, "T": 0.03433, "Q": 0.02319, "A": 0.01745, "K": 0.0155,
        "R": 0.01466, "L": 0.00959, "S": 0.00161, "H": -0.0044, "N": -0.00889,
        "E": -0.01081, "D": -0.07366, "G": -0.08977, "P": -0.28,
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

    Rocklin et al. (2017, Figure 4I/J) mutated surface positions of designed strands and
    averaged the stability effect of each amino acid, separately for middle strands
    (paired on both sides) and edge strands (paired on one side). Beta-branched and
    aromatic residues stabilise; Gly, Asp and especially Pro destabilise. This function
    adds up those effects over the exposed strand residues of a protein.

    Only residues with ``ss[i] == 'E'`` and ``exposed[i]`` true are scored. Buried strand
    residues, helices and loops contribute nothing: the effects were measured at surface
    sites and say nothing about the core. A scored residue with no strand class, or an
    amino acid missing from the table (cysteine, by default), is skipped and listed in
    ``unscored``.

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
            - ``score``: sum over the scored residues (positive is stabilising).
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
