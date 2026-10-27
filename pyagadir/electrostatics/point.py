"""
Point-charge electrostatics of the helix-coil model.

Charges are points at tabulated distances (Lacroix et al. 1998, J. Mol. Biol. 284, 173, supplementary Tables VI and
VII) and interact by screened Coulomb in water at water's eps(T) (Lacroix 1998 eq. 6, with 4 pi). The helix macrodipole
is a +1/2 charge at the N-terminal end and a -1/2 charge at the C-terminal end (Hol, van Duijnen & Berendsen 1978), and
every charge feels it by that same law: side chains inside the helix (both ends, at their Table VII distances), side
chains in the flanks (the nearby end) and the free terminal groups.

PointChargeElectrostatics is a mixin of energies.PrecomputeParams and reads the segment set up there (seq_list,
helix_indices, ncap_idx, ccap_idx, charged_pairs, kappa, epsilon_r, T_kelvin, the parameter tables and the ionisation
states).
"""
from importlib.resources import files
import math

import numpy as np
import pandas as pd

# Helix-state distance (A) between a succinyl N-cap's carboxylate and a charged side chain at N1, N2 or N3, used in
# place of Lacroix 1998 Table VI 'N-cap f' (which places a free amine on the backbone: 9.2 A to N2 in the helix vs 9.6 A
# in the coil). Each value is the distance at which this model's own Coulomb law reproduces the helix-minus-coil pair
# energy from continuum electrostatics (APBS, carboxylate on the N-cap residue as the succinyl proxy, helix minus PPII,
# rotamer-averaged, interior dielectric 16, the value at which the same calculation matches the N1/N2 His titrations of
# Cochran et al. 2001, Protein Sci. 10, 463) at 0.1 M and 3 C: acid N2 +0.50, acid N3 +0.35, acid N1 +0.04, base N2
# -0.34 kcal/mol. Pairs not listed keep Table VI. See params/README.md.
SUCCINYL_HELIX_DISTANCE = {("acid", 1): 9.33, ("acid", 2): 3.95, ("acid", 3): 5.27, ("base", 2): 4.80}

# Capping-box N-caps (a Glu at N3 behind one of these is its own Table 1 class, "N3cb").
CAPPING_BOX_NCAPS = ("S", "T", "D", "N")
_TRANSFER = None


def table1_position_class(aa: str, n_dist: int, c_dist: int, ncap_res: str) -> str:
    """
    Table 1 class of a charged residue n_dist from the N-cap and c_dist from the C-cap, following get_dG_Int's column
    choice: the N-cap; the C-cap; C1-C3 (whose charged-residue cells equal Ncen); N1-N4 for Asp/Glu, per N-cap type
    (acetyl/succinyl "N1", residue "N1r"); Ncen otherwise (Lys/Arg/His N1-N4 cells equal Ncen). "N3cb": Glu at N3 behind
    an S/T/D/N N-cap, the capping box, whose bonus was measured in residue-capped peptides. The classes of the macrodipole
    carry-over (macrodipole_transfer).
    """
    if aa == "E" and n_dist == 3 and c_dist > 3 and ncap_res in CAPPING_BOX_NCAPS:
        return "N3cb"
    if n_dist == 0:
        return "Ncap"
    if c_dist == 0:
        return "Ccap"
    if c_dist <= 3:
        return "Ncen"
    if n_dist <= 4 and aa in ("D", "E"):
        return f"N{n_dist}" + ("" if ncap_res in ("Ac", "Sc") else "r")
    return "Ncen"


def macrodipole_transfer() -> dict:
    """params/macrodipole_transfer.tsv, loaded once: {(residue, class): dphi} (kcal/mol per +1 charge)."""
    global _TRANSFER
    if _TRANSFER is None:
        t = pd.read_csv(files("pyagadir.params").joinpath("macrodipole_transfer.tsv"), sep="\t", comment="#")
        _TRANSFER = {(r.residue, r.cls): float(r.dphi) for r in t.itertuples()}
    return _TRANSFER


class PointChargeElectrostatics:
    """
    The point-charge law: interaction distances, the Coulomb energy, the helix macrodipole potential, and the
    electrostatic energy terms of the terminal groups.
    """

    def _calculate_r(self, N: int) -> float:
        """Function to calculate the distance r from the peptide terminal to the helix
        start, where N is the number of residues between the terminal and the helix.
        p. 177 of Lacroix, 1998. Distances in Ångströms as 2.1, 4.1, 6.1... The function is
        needed because we ignore sidechains here. We only calculate distances between charged
        termini (which are located on the backbone) and the helix macrodipole, which is located
        at the N- and C-terminal capping residues. The capping residues are not included in the helix
        macrodipole, so the distance can never be shorter than 2.1 Ångströms.

        Args:
            N (int): The number of residues between the peptide terminal and the helix start.

        Returns:
            float: The calculated distance r in Ångströms.
        """
        r = 0.1 + (N + 1) * 2
        return r

    def _electrostatic_interaction_energy(self, qi: float, qj: float, r: float, factor_pi: float = 4.0) -> float:
        """Calculate the interaction energy between two charges by
        applying equation 6 from Lacroix, 1998.
        Note: The paper prints 3π; 4π (standard Coulomb) is used.

        Args:
            qi (float): Charge of the first residue.
            qj (float): Charge of the second residue.
            r (float): Distance between the residues in Ångströms.
            factor_pi (float): The pi multiple of the Coulomb denominator, 4 (standard Coulomb; Lacroix 1998 prints 3).
        Returns:
            float: The interaction energy in kcal/mol.
        """
        distance_r_meter = r * 1e-10  # Convert distance from Ångströms to meters
        screening_factor = math.exp(-self.kappa * distance_r_meter)
        coulomb_term = (self.e**2 * qi * qj) / (factor_pi * math.pi * self.epsilon_0 * self.epsilon_r * distance_r_meter) 
        energy_joules = coulomb_term * screening_factor
        energy_kcal_mol = self.N_A * energy_joules / 4184
        return energy_kcal_mol

    def _macrodipole_potential(self, idx: int) -> tuple[float, float]:
        """
        Energy (kcal/mol) of a +1 charge on residue idx in the field of the helix macrodipole, split into the N-terminal
        and C-terminal ends' contributions.

        The ends are a +1/2 and a -1/2 charge (Hol, van Duijnen & Berendsen 1978), felt by screened Coulomb in water at
        water's eps(T), the law of every other charge pair:
        - residues N-cap..C-cap feel both ends, at their Lacroix 1998 Table VII distances to each end (extended past
          13 positions by the helix rise of 1.5 A per residue), with no cutoff; to that potential the residue's Table 1
          carry-over is added (params/macrodipole_transfer.tsv: the Table 1 values of charged residues were measured
          with Muñoz & Serrano's eq. 11 near-end law in the loop; the offset keeps each at its reference measurement);
        - flanking residues (up to nine positions outside a cap) feel the nearby end at the Lacroix 1998 flank
          distance, 6 A at N'/C' and 3 A more per further position;
        - residues farther away return (0, 0).
        """
        n = len(self.seq_list)
        ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)
        aa = self.seq_list[idx]

        def half(pole, r):
            return self._electrostatic_interaction_energy(qi=pole * self.mu_helix, qj=1.0, r=r)

        if ncap_i <= idx <= ccap_i:
            n_pos, c_pos = idx - ncap_i, ccap_i - idx

            def dist(table, prefix, p):
                if aa not in table.index:
                    return 99.0
                if p == 0:
                    return float(table.loc[aa, prefix + "cap"])
                if p <= 13:
                    return float(table.loc[aa, f"{prefix}{p}"])
                return float(table.loc[aa, f"{prefix}13"]) + 1.5 * (p - 13)

            phi_N = half(1.0, dist(self.table_7_ncap_lacroix, "N", n_pos))
            phi_C = half(-1.0, dist(self.table_7_ccap_lacroix, "C", c_pos))
            d = macrodipole_transfer().get((aa, table1_position_class(aa, n_pos, c_pos, self.seq_list[ncap_i])), 0.0)
            if n_pos <= c_pos:  # the carry-over is booked on the nearer end
                phi_N += d
            else:
                phi_C += d
            return phi_N, phi_C
        if ccap_i < idx < min(n, ccap_i + 10):
            return 0.0, half(-1.0, 6.0 + 3.0 * (idx - ccap_i - 1))
        if max(0, ncap_i - 9) <= idx < ncap_i:
            return half(1.0, 6.0 + 3.0 * (ncap_i - idx - 1)), 0.0
        return 0.0, 0.0

    def _terminal_group_distance(self, row: str, x: int) -> float:
        """Helix-state distance (A) between a free terminal group and a helical charged
        residue x positions away, from Lacroix 1998 supplementary Table VI (rows 'N-cap f',
        'N’ f', 'C-cap f', 'C’ f').  Beyond the table (x > 12) the pair is not
        modelled and 99.0 is returned."""
        if 1 <= x <= 12:
            return float(self.table_6_helix_lacroix.loc[row, f"i+{x}"])
        return 99.0

    def _assign_terminal_sidechain_distances(self):
        """
        Distances (A) between a free terminal group and each charged side chain in the helix and
        coil states. The ionisation solver and get_dG_terminals_sidechain_electrost both read
        these arrays, so the pKa shifts and the energy come from the same pairs and geometry
        (Lacroix 1998 supplementary Table VI):

          coil   row RcoilRest, x = 1..12 residues apart; not modelled (99 A) beyond.
          helix  when the terminus is local (at the cap, or at N'/C') and the side chain is a
                 helix-interior residue, row 'N-cap f'/'N’ f' ('C-cap f'/'C’ f'); otherwise the
                 pair has no modelled change between the states and keeps its coil distance.
          x = 0  the side chain and the terminal group of the same residue: 2.1 A
                 (_calculate_r(0)) in both states. This local geometry shifts the residue's pKa
                 alike in both states (the N-terminal Asp side chain, for example; Doig & Baldwin
                 1995 Table 2); its energy changes between the states through the charges only.

        terminal_sidechain_modelled_nterm / _cterm mark the pairs with a helix geometry of their own. The energy term
        and the solver count every pair; a pair without its own helix geometry keeps its coil distance.
        """
        n = len(self.seq_list)
        charged = set(self.neg_charge_aa + self.pos_charge_aa)
        interior = set(self.helix_indices[1:-1]) if len(self.helix_indices) > 2 else set()
        nterm_local = self.ncap_idx <= 1
        cterm_local = self.ccap_idx >= n - 2
        n_row = "N-cap f" if self.ncap_idx == 0 else "N’ f"
        c_row = "C-cap f" if self.ccap_idx == n - 1 else "C’ f"

        self.terminal_sidechain_distances_nterm = np.full(n, np.nan)
        self.terminal_sidechain_distances_cterm = np.full(n, np.nan)
        self.terminal_sidechain_distances_nterm_rc = np.full(n, np.nan)
        self.terminal_sidechain_distances_cterm_rc = np.full(n, np.nan)
        self.terminal_sidechain_modelled_nterm = np.zeros(n, dtype=bool)
        self.terminal_sidechain_modelled_cterm = np.zeros(n, dtype=bool)

        def coil(x):
            if x == 0:
                return self._calculate_r(0)
            if 1 <= x <= 12:
                return float(self.table_6_coil_lacroix.loc["RcoilRest", f"i+{x}"])
            return 99.0

        for idx, aa in enumerate(self.seq_list):
            if aa not in charged:
                continue
            for x, local, row, hel, rc, modelled in (
                (idx, nterm_local, n_row, self.terminal_sidechain_distances_nterm,
                 self.terminal_sidechain_distances_nterm_rc, self.terminal_sidechain_modelled_nterm),
                ((n - 1) - idx, cterm_local, c_row, self.terminal_sidechain_distances_cterm,
                 self.terminal_sidechain_distances_cterm_rc, self.terminal_sidechain_modelled_cterm),
            ):
                d_rc = coil(x)
                d_hel = d_rc
                if local and idx in interior:
                    d_tab = self._terminal_group_distance(row, x)
                    if d_tab < 99.0:
                        d_hel = d_tab
                        modelled[idx] = True
                    # A succinyl N-cap reaches further than a free amine: its carboxylate sits on a flexible
                    # -CH2-CH2- arm, and an N2/N3 side chain leans back toward the N-cap in the helix.
                    if row == n_row and self.ncap == "Sc" and self.ncap_idx == 0:
                        kind = "acid" if aa in ("D", "E") else ("base" if aa in ("K", "R") else None)
                        d_sc = SUCCINYL_HELIX_DISTANCE.get((kind, x))
                        if d_sc is not None:
                            d_hel = d_sc
                            modelled[idx] = True
                hel[idx], rc[idx] = d_hel, d_rc

    def _assign_terminal_macrodipole_distances(self):
        """
        Assign the distance between the peptide terminal groups and the helix macrodipole: Lacroix 1998's linear
        rule, 2.1 A when the cap carries the terminal group and 2 A more per residue between them (_calculate_r),
        for the free amine, the succinyl carboxylate and the free carboxylate, with no cutoff (the paper states
        none). The rule reproduces the paper's quoted energies (0.81, 0.36, 0.21, 0.14 kcal/mol) at a constant ratio.
        """
        self.terminal_macrodipole_distance_nterm = self._calculate_r(self.ncap_idx)
        self.terminal_macrodipole_distance_cterm = self._calculate_r(len(self.seq_list) - 1 - self.ccap_idx)

    def _assign_sidechain_sidechain_distances(self):
        """
        Assign the distance between two charged sidechains.

        This function makes use of the supplementary table 6 from Lacroix, 1998.
        There is one table for helical states and one for random-coil states.
        However, the table only contains distances up to 12 residues apart, so
        distances greater than 12 are assigned a large distance (99 Å). The exact
        number for large distances does not matter much, since the effect will be screened
        by the solvent.

        The function handles the following cases:
        1. Both residues in helix: use table 6 helix distances
        2. Both residues in coil: use table 6 coil distances 
        3. One in helix, one in coil: only N' or C' interacts with helical residues, at the
           table 6 N' / C' (C'G-cap when the C-cap is Gly) distances; other such pairs have no
           helix geometry
        A pair without helix geometry (case 3 other than N'/C', or both outside and straddling the
        helix) keeps its coil distance in the helix state: its interaction changes between the states
        through the charges only.
        Pairs without a row of their own (His-His, and any pair with Tyr or Cys) use the
        HelixRest / RcoilRest rows, which the table caption defines as "all possible charged
        pairs not included before" (helix) and the pairs "not included in Rcoil" (coil).
        """
        batch = getattr(self, "segment_batch", None)
        if batch is not None:
            # default law, many segments (common.SegmentBatch): the coil distances are the peptide's, and the pairs
            # beyond PAIR_RANGE (99 A in both states) are in the template; self.charged_pairs holds the others
            self.charged_sidechain_distances_rc = batch.coil_distances
            self.sidechain_sidechain_distances_hel = batch.helix_distance_template.copy()
        else:
            self.sidechain_sidechain_distances_hel = np.full((len(self.seq_list), len(self.seq_list)), np.nan)
            self.charged_sidechain_distances_rc = np.full((len(self.seq_list), len(self.seq_list)), np.nan)

        ### Assign distances to coil state ###
        # (with a batch the coil matrix is already the peptide's, assigned once by the reference segment: skipped)
        for AA1, AA2, idx1, idx2 in (self.charged_pairs if batch is None else ()):
            n_residues_separation = idx2 - idx1
            distance_key = f"i+{n_residues_separation}"
                        
            if n_residues_separation >= 13: # table 6 only contains distances up to 12, so assign a large distance to things that are further apart
                distance_angstrom = 99
            else:
                pair = AA1 + AA2
                if pair not in self.table_6_coil_lacroix.index:  # His-His, Tyr, Cys: no own row
                    pair = 'RcoilRest'
                distance_angstrom = self.table_6_coil_lacroix.loc[pair, distance_key]

            self.charged_sidechain_distances_rc[idx1, idx2] = distance_angstrom
            self.charged_sidechain_distances_rc[idx2, idx1] = distance_angstrom
            
        ### Assign distances to helix state ###
        hel = set(self.helix_indices)  # the segment's residues, N-cap to C-cap (a set for the membership tests)
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
            n_residues_separation = idx2 - idx1
            distance_key = f"i+{n_residues_separation}"          

            if n_residues_separation >= 13: # table 6 only contains distances up to 12, so assign a large distance to things that are further apart
                distance_angstrom = 99

            else:
                # (A) Both residues in helix
                if idx1 in hel and idx2 in hel:
                    # Check if either residue is at a cap position — use cap-specific
                    # Table 6 rows instead of AA-pair rows (Lacroix 1998: cap positions
                    # have modeled non-helical backbone angles giving different distances).
                    # Use "f" (free terminal) rows when terminal is not blocked.
                    # Always use 'Ccap'/'Ncap' rows for cap-position distances.
                    # The 'C-cap f'/'N-cap f' rows give shorter distances (e.g.
                    # 8.03 vs 10.7 for i+1) that overestimate repulsion; the
                    # blocked-cap geometry ('Ccap'/'Ncap') is used for
                    # sidechain-sidechain interactions at cap positions.
                    cap_row = None
                    if idx2 == self.ccap_idx:
                        cap_row = 'Ccap'
                    elif idx1 == self.ncap_idx:
                        cap_row = 'Ncap'
                    elif idx1 == self.ccap_idx:
                        cap_row = 'Ccap'
                    elif idx2 == self.ncap_idx:
                        cap_row = 'Ncap'

                    if cap_row is not None:
                        distance_angstrom = self.table_6_helix_lacroix.loc[cap_row, distance_key]
                    else:
                        pair = AA1 + AA2
                        if pair not in self.table_6_helix_lacroix.index:  # His-His, Tyr, Cys: no own row
                            pair = 'HelixRest'
                        distance_angstrom = self.table_6_helix_lacroix.loc[pair, distance_key]
                
                # (B) Both residues in coil part of a peptide that (at a different position) contains the helix
                elif idx1 not in hel and idx2 not in hel:
                    straddles_helix = (idx1 < self.ncap_idx) and (idx2 > self.ccap_idx)

                    if not straddles_helix:
                        pair = AA1 + AA2
                        if pair not in self.table_6_coil_lacroix.index:  # His-His, Tyr, Cys: no own row
                            pair = "RcoilRest"
                        distance_angstrom = float(self.table_6_coil_lacroix.loc[pair, distance_key])
                    else:
                        distance_angstrom = 99
                
                # (C) One residue in helix, one in coil
                else:
                    # Identify which is coil
                    coil_idx = idx1 if idx1 not in hel else idx2
                    
                    # [PATCH] Lacroix 1998 Restriction:
                    # Only calculate distance if the coil residue is N' (ncap_idx - 1) or C' (ccap_idx + 1)
                    # Otherwise, interaction is ignored (distance = 99)
                    
                    is_N_prime = (coil_idx == self.ncap_idx - 1)
                    is_C_prime = (coil_idx == self.ccap_idx + 1)
                    
                    if not (is_N_prime or is_C_prime):
                        distance_angstrom = 99
                    else:
                        # Lacroix 1998 supplementary Table VI gives these distances directly:
                        # row N' (residue N' to a helical residue i+x), row C' (residue C' to a
                        # helical residue i-x), and row C'G-cap for C' when the C-cap is a Gly
                        # (the row's name; its printed caption, "when C' is a Gly", cannot apply
                        # to a charged C').  Beyond the table the pair is not modelled.
                        helix_idx = idx1 if idx1 in hel else idx2
                        x = abs(coil_idx - helix_idx)
                        if is_N_prime:
                            row = "N’"
                        else:
                            row = "C’G-cap" if self.seq_list[self.ccap_idx] == "G" else "C’"
                        if row is None or not 1 <= x <= 12:
                            distance_angstrom = 99
                        else:
                            distance_angstrom = float(self.table_6_helix_lacroix.loc[row, f"i+{x}"])
                
            # A pair within range that has no helix geometry of its own keeps its coil geometry in the helix state:
            # its interaction is unchanged by the helix apart from the charges.
            if distance_angstrom >= 99 and n_residues_separation < 13:
                distance_angstrom = self.charged_sidechain_distances_rc[idx1, idx2]

            self.sidechain_sidechain_distances_hel[idx1, idx2] = distance_angstrom
            self.sidechain_sidechain_distances_hel[idx2, idx1] = distance_angstrom

    def _point_pair_energy(self, idx1: int, idx2: int):
        """
        Helix-minus-coil Coulomb energy (kcal/mol, Lacroix 1998 eq. 6) of the charged side chains idx1 < idx2 at
        their converged ionisation in each state, or None for a pair whose interaction does not change between the
        states. get_dG_sidechain_sidechain_electrost gives half of it to each residue.
        """
        # Get the distances between the charged sidechains
        helix_dist = self.sidechain_sidechain_distances_hel[idx1, idx2]
        coil_dist = self.charged_sidechain_distances_rc[idx2, idx1]

        # 99 A: pairs >= 13 apart, out of range in both states. A pair without helix geometry of its own (of the
        # residues outside the helix only the caps and N'/C' have helix distances to helical residues, Lacroix 1998)
        # keeps its coil distance in the helix state and contributes only through the change of the charges.
        if helix_dist >= 99:
            return None

        # Get the ionization states of the charged sidechains
        q1_hel = self.modified_seq_ionization_hel[idx1]
        q2_hel = self.modified_seq_ionization_hel[idx2]
        q1_rc = self.modified_seq_ionization_rc[idx1]
        q2_rc = self.modified_seq_ionization_rc[idx2]

        # Calculate electrostatic interaction energies with adjusted ionization states, Lacroix Eq 6.
        # factor_pi=4.0: calibrated from YGGS AA reference (scsc-only segments).
        G_hel = self._electrostatic_interaction_energy(qi=q1_hel, qj=q2_hel, r=helix_dist, factor_pi=4.0)
        batch = getattr(self, "segment_batch", None)
        if batch is None:
            G_rc = self._electrostatic_interaction_energy(qi=q1_rc, qj=q2_rc, r=coil_dist, factor_pi=4.0)
        else:
            # The coil-state energy depends on the coil charges and coil distance only, both the peptide's
            # (common.SegmentBatch): computed for the first segment that needs it and reused by the others.
            G_rc = batch.coil_pair_energy.get((idx1, idx2))
            if G_rc is None:
                G_rc = batch.coil_pair_energy[(idx1, idx2)] = self._electrostatic_interaction_energy(
                    qi=q1_rc, qj=q2_rc, r=coil_dist, factor_pi=4.0)
        return G_hel - G_rc

    def get_dG_terminals_macrodipole(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate interaction energies between N- and C-terminal backbone charges and the helix macrodipole.
        The energy is added to the residue carrying the macrodipole charge.

        Returns:
            tuple[np.ndarray, np.ndarray]: Interaction energies for N-terminal and C-terminal residues.
        """
        N_term = np.zeros(len(self.seq_list))
        C_term = np.zeros(len(self.seq_list))

        # Calculate the interaction energy between the N-terminal and the helix macrodipole (no cutoff)
        N_term[self.ncap_idx] = self._electrostatic_interaction_energy(
            qi=self.mu_helix,
            qj=self.modified_nterm_ionization_hel,
            r=self.terminal_macrodipole_distance_nterm,
            factor_pi=4.0
        )

        # Calculate the interaction energy between the C-terminal and the helix macrodipole
        C_term[self.ccap_idx] = self._electrostatic_interaction_energy(
            qi=-self.mu_helix,
            qj=self.modified_cterm_ionization_hel,
            r=self.terminal_macrodipole_distance_cterm,
            factor_pi=4.0
        )

        return N_term, C_term

    def get_dG_terminals_sidechain_electrost(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Helix-minus-coil electrostatic energy between the free terminal groups (alpha-NH3+, succinyl carboxylate,
        alpha-COO-) and the charged side chains, at the distances of _assign_terminal_sidechain_distances (Lacroix
        1998 supplementary Table VI 'f' rows for a local terminus and a helix-interior side chain, RcoilRest
        otherwise). The energy is booked on the side chain.

        Every terminus-side chain pair the ionisation solver counts, q_X^h q_k^h W(r^h) - q_X^c q_k^c W(r^c) with each
        state's converged charges, so that the terms plus dG_ionization are the mean-field free energy of the
        solver's own model (common.py). Pairs at the same distance in both states (a non-local terminus, a side chain
        outside the helix interior, a residue's own terminal group at 2.1 A) contribute through the change of the
        charges only.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays containing the interaction energies for
            N-terminal and C-terminal interactions respectively. The energy is added
            to the charged sidechain residue.
        """
        n = len(self.seq_list)
        energy_N = np.zeros(n, dtype=float)
        energy_C = np.zeros(n, dtype=float)

        # Presence (Ac/Am remove terminal charges in this model)
        nterm_present = not (n > 0 and self.seq_list[0] == "Ac")
        cterm_present = not (n > 0 and self.seq_list[-1] == "Am")
        charged = set(self.neg_charge_aa + self.pos_charge_aa)

        def pair(q_t, q_sc, r):
            return self._electrostatic_interaction_energy(qi=q_t, qj=q_sc, r=r) if r < 40.0 else 0.0

        for idx, aa in enumerate(self.seq_list):
            if aa not in charged:
                continue
            q_h = float(self.modified_seq_ionization_hel[idx])
            q_c = float(self.modified_seq_ionization_rc[idx])
            for present, q_th, q_tc, d_hel, d_rc, out in (
                (nterm_present, self.modified_nterm_ionization_hel, self.modified_nterm_ionization_rc,
                 self.terminal_sidechain_distances_nterm[idx], self.terminal_sidechain_distances_nterm_rc[idx],
                 energy_N),
                (cterm_present, self.modified_cterm_ionization_hel, self.modified_cterm_ionization_rc,
                 self.terminal_sidechain_distances_cterm[idx], self.terminal_sidechain_distances_cterm_rc[idx],
                 energy_C),
            ):
                if present:
                    out[idx] = pair(q_th, q_h, float(d_hel)) - pair(q_tc, q_c, float(d_rc))
        return energy_N, energy_C

    def get_dG_terminal_terminal_electrost(self) -> float:
        """
        Electrostatic interaction between the free N-terminal group (NH3+, or the succinyl carboxylate) and the free
        C-terminal carboxylate, helix minus coil (kcal/mol).

        Lacroix 1998 Table VI has no terminus-terminus geometry. The ionisation solver places the two at
        _calculate_r(L - 1) in both states and drops the pair beyond 40 A (chains of 20 or more positions).

        That pair at each state's converged charges, [q_N^h q_C^h - q_N^c q_C^c] W(r); the macrodipole shifts both
        termini in the helix, so it is not zero.

        Returns:
            float: The terminal-terminal electrostatic free energy in kcal/mol.
        """
        n = len(self.seq_list)
        if self.seq_list[0] == "Ac" or self.seq_list[-1] == "Am":
            return 0.0
        r = self._calculate_r(n - 1)
        if r >= 40.0:
            return 0.0
        return (self._electrostatic_interaction_energy(qi=self.modified_nterm_ionization_hel,
                                                       qj=self.modified_cterm_ionization_hel, r=r)
                - self._electrostatic_interaction_energy(qi=self.modified_nterm_ionization_rc,
                                                         qj=self.modified_cterm_ionization_rc, r=r))
