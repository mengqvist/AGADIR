"""
Point-charge electrostatics: dipole_law="munoz", the default law.

Charges are points at tabulated distances (Lacroix et al. 1998, J. Mol. Biol. 284, 173, supplementary Tables VI and
VII) and interact by screened Coulomb in water (Lacroix 1998 eq. 6); a charged side chain feels the nearest helix end
by Muñoz & Serrano 1995-II eq. 11 and the far end as a half charge. Every law takes the terminal groups from here (the
free amine's and carboxylate's interactions with the macrodipole and with side chains), and the Generalized Born law
(gb.py) also falls back on this one wherever its tables do not reach.

PointChargeElectrostatics is a mixin of energies.PrecomputeParams and reads the segment set up there (seq_list,
helix_indices, ncap_idx, ccap_idx, charged_pairs, kappa, epsilon_r, T_kelvin, the parameter tables and the ionisation
states).
"""
import math

import numpy as np

from pyagadir.chemistry import T_REF, calculate_permittivity

POINT_LAW = "munoz"

# Helix-state distance (A) between a succinyl N-cap's carboxylate and a charged side chain at N1, N2 or N3, used in
# place of Lacroix 1998 Table VI 'N-cap f' (which places a free amine on the backbone: 9.2 A to N2 in the helix vs 9.6 A
# in the coil). Each value is the distance at which this model's own Coulomb law reproduces the helix-minus-coil pair
# energy from continuum electrostatics (APBS, carboxylate on the N-cap residue as the succinyl proxy, helix minus PPII,
# rotamer-averaged, interior dielectric 16, the value at which the same calculation matches the N1/N2 His titrations of
# Cochran et al. 2001, Protein Sci. 10, 463) at 0.1 M and 3 C: acid N2 +0.50, acid N3 +0.35, acid N1 +0.04, base N2
# -0.34 kcal/mol. Pairs not listed keep Table VI. See params/README.md.
SUCCINYL_HELIX_DISTANCE = {("acid", 1): 9.33, ("acid", 2): 3.95, ("acid", 3): 5.27, ("base", 2): 4.80}

# Far end of the helix macrodipole: a charged residue inside the helix feels the nearest end by Munoz & Serrano 1995-II
# eq. 11 and the other end as a half charge (Hol, van Duijnen & Berendsen 1978), screened Coulomb in water. Poisson-
# Boltzmann calculations on built helices (interior dielectric 4; Asp/Glu at N1-N3 and Lys/Arg at C1-C3, helix lengths
# 6-28) give 0.82 times that half-charge law (least squares through the origin over 72 cases), so the far end is scaled
# by this factor. The flank half charges are not (params/README.md: the same calculation on flanking residues gives 0.63,
# but the measured data prefer the unscaled law).
FEH_FAR_APBS = 0.82


class PointChargeElectrostatics:
    """
    The point-charge law: interaction distances, the Coulomb energy, the side chain-macrodipole potential of
    Muñoz & Serrano, and the electrostatic energy terms that only this law provides (terminal groups).
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
            factor_pi (float): Factor to multiply the Coulomb term by. Default is 3.0, as indicated in Lacroix, 1998.
        Returns:
            float: The interaction energy in kcal/mol.
        """
        distance_r_meter = r * 1e-10  # Convert distance from Ångströms to meters
        screening_factor = math.exp(-self.kappa * distance_r_meter)
        coulomb_term = (self.e**2 * qi * qj) / (factor_pi * math.pi * self.epsilon_0 * self.epsilon_r * distance_r_meter) 
        energy_joules = coulomb_term * screening_factor
        energy_kcal_mol = self.N_A * energy_joules / 4184
        return energy_kcal_mol

    def _munoz_dipole_potential_scaled(self, idx: int) -> tuple[float, float]:
        """Muñoz's law with its eps(T) factor applied, for the positions the GB law leaves to it."""
        factor = math.exp(0.004314 * (self.T_kelvin - T_REF))
        phi_N, phi_C = self._munoz_dipole_potential(idx)
        return factor * phi_N, factor * phi_C

    def _munoz_dipole_potential(self, idx: int) -> tuple[float, float]:
        """
        The default law (dipole_law="munoz"), before the eps(T) factor.

        Inside the helix (N-cap to C-cap): the nearest end by Muñoz 1995-II eq. 11,

            N-terminal end:  g =  K / d_N² × exp(−κ × d_N)
            C-terminal end:  g = −K / d_C² × exp(−κ × d_C)

        with K = 0.6 × 4.9² kcal Å² mol⁻¹ and d_N / d_C from the Coulomb distance tables (after
        Lacroix 1998 Table VII), zero beyond nine positions from the cap. The other (far) end adds
        the field of its half charge (Hol, van Duijnen & Berendsen 1978): screened Coulomb in water
        at the Table VII distance to that end, extended past 13 positions by the helix rise of
        1.5 A per residue (reasoning/nodes/N083.md), scaled by FEH_FAR_APBS (0.82, Poisson-Boltzmann
        calculations on built helices).

        Flanking residues (up to nine positions outside a cap) feel the nearby end's half charge by
        the same screened Coulomb law in water, at the Lacroix 1998 flank distance (6 A at N'/C',
        +3 A per further position).

        Residues farther away return (0, 0).
        """
        n = len(self.seq_list)
        ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)
        aa = self.seq_list[idx]
        kappa_01A = self.kappa * 1e-11  # Debye-Hückel parameter in 0.1 A units
        K_DIPOLE = 0.6 * 49.0**2  # eq. 11 with r in 0.1 A units
        # the half-charge terms are evaluated at the 0 C dielectric because the whole side
        # chain-macrodipole term is multiplied by dipole_temperature_factor()
        eps_to_0C = self.epsilon_r / calculate_permittivity(T_REF)

        def dist_n(p):
            key = "Ncap" if p == 0 else (f"N{p}" if 1 <= p <= 13 else None)
            if key is None or aa not in self.table_7_coulomb_ncap.index:
                return 99.0
            return float(self.table_7_coulomb_ncap.loc[aa, key])

        def dist_c(p):
            key = "Ccap" if p == 0 else (f"C{p}" if 1 <= p <= 13 else None)
            if key is None or aa not in self.table_7_coulomb_ccap.index:
                return 99.0
            return float(self.table_7_coulomb_ccap.loc[aa, key])

        def half_charge(pole, r):
            return self._electrostatic_interaction_energy(qi=pole * self.mu_helix, qj=1.0, r=r) * eps_to_0C

        phi_N = phi_C = 0.0
        if ncap_i <= idx <= ccap_i:
            n_pos, c_pos = idx - ncap_i, ccap_i - idx
            d_N, d_C = dist_n(n_pos), dist_c(c_pos)
            if d_N * 10.0 < 1.0 or d_C * 10.0 < 1.0:
                return 0.0, 0.0
            # the nearest end (ties go to the N-terminus); the other end as a half charge
            if n_pos <= c_pos:
                if n_pos <= 9:
                    phi_N += K_DIPOLE / (d_N * 10.0) ** 2 * math.exp(-kappa_01A * d_N * 10.0)
                d_far = d_C if c_pos <= 13 else dist_c(13) + 1.5 * (c_pos - 13)
                phi_C += FEH_FAR_APBS * half_charge(-1.0, d_far)
            else:
                if c_pos <= 9:
                    phi_C += -K_DIPOLE / (d_C * 10.0) ** 2 * math.exp(-kappa_01A * d_C * 10.0)
                d_far = d_N if n_pos <= 13 else dist_n(13) + 1.5 * (n_pos - 13)
                phi_N += FEH_FAR_APBS * half_charge(1.0, d_far)
        elif ccap_i < idx < min(n, ccap_i + 10):
            phi_C += half_charge(-1.0, 6.0 + 3.0 * (idx - ccap_i - 1))
        elif max(0, ncap_i - 9) <= idx < ncap_i:
            phi_N += half_charge(1.0, 6.0 + 3.0 * (ncap_i - idx - 1))
        return phi_N, phi_C

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

    def get_terminal_sidechain_distances(self):
        """
        Get the helix-state distances between the peptide terminal groups and the charged sidechains.
        """
        return self.terminal_sidechain_distances_nterm, self.terminal_sidechain_distances_cterm

    def show_terminal_sidechain_distances(self):
        """
        Print out the distances for the peptide terminal residues and the charged sidechains in a nicely formatted way.
        """
        print(self._make_box("Terminal sidechain distances (Å)"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        for label, arr in (("nterm hel:", self.terminal_sidechain_distances_nterm),
                           ("nterm rc:", self.terminal_sidechain_distances_nterm_rc),
                           ("cterm hel:", self.terminal_sidechain_distances_cterm),
                           ("cterm rc:", self.terminal_sidechain_distances_cterm_rc)):
            print(f'{label.ljust(self.category_pad)} {"".join([f"{d:.2f}".ljust(self.value_pad) for d in arr])}')
        print("")

    def _assign_terminal_macrodipole_distances(self):
        """
        Assign the distance between the peptide terminal groups and the helix macrodipole: Lacroix 1998's linear
        rule, 2.1 A when the cap carries the terminal group and 2 A more per residue between them (_calculate_r),
        for the free amine, the succinyl carboxylate and the free carboxylate, with no cutoff (the paper states
        none). The rule reproduces the paper's quoted energies (0.81, 0.36, 0.21, 0.14 kcal/mol) at a constant ratio.
        """
        self.terminal_macrodipole_distance_nterm = self._calculate_r(self.ncap_idx)
        self.terminal_macrodipole_distance_cterm = self._calculate_r(len(self.seq_list) - 1 - self.ccap_idx)

    def get_terminal_macrodipole_distances(self) -> tuple[float, float]:
        """
        Get the distances for the peptide terminal residues and the helix macrodipole.
        Only computes N-terminal distance to the N-terminal helix dipole and 
        C-terminal distance to the C-terminal helix dipole.

        Returns:
            tuple[float, float]: Distances between the peptide N-terminal and C-terminal residues and the helix macrodipole
        """
        return self.terminal_macrodipole_distance_nterm, self.terminal_macrodipole_distance_cterm
    
    def show_terminal_macrodipole_distances(self):
        """
        Print out the distances for the peptide terminal residues and the helix macrodipole in a nicely formatted way.
        """
        print(self._make_box("Terminal macrodipole distances (Å)"))
        print(f'{"nterm:".ljust(self.category_pad)} {self.terminal_macrodipole_distance_nterm:.2f}')
        print(f'{"cterm:".ljust(self.category_pad)} {self.terminal_macrodipole_distance_cterm:.2f}')
        print("")

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
        self.sidechain_sidechain_distances_hel = np.full((len(self.seq_list), len(self.seq_list)), np.nan)
        self.charged_sidechain_distances_rc = np.full((len(self.seq_list), len(self.seq_list)), np.nan)

        ### Assign distances to coil state ###
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
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
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
            n_residues_separation = idx2 - idx1
            distance_key = f"i+{n_residues_separation}"          

            if n_residues_separation >= 13: # table 6 only contains distances up to 12, so assign a large distance to things that are further apart
                distance_angstrom = 99

            else:
                # (A) Both residues in helix
                if idx1 in self.helix_indices and idx2 in self.helix_indices:
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
                elif idx1 not in self.helix_indices and idx2 not in self.helix_indices:
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
                    coil_idx = idx1 if idx1 not in self.helix_indices else idx2
                    
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
                        helix_idx = idx1 if idx1 in self.helix_indices else idx2
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

    def get_sidechain_sidechain_distances(self):
        """
        Get the charged sidechain distances for the sequence, both for helical and random-coil states.
        """
        return self.sidechain_sidechain_distances_hel, self.charged_sidechain_distances_rc

    def show_sidechain_sidechain_distances(self):
        """
        Print out the charged sidechain distances for the sequence in a nicely formatted way.
        """
        print(self._make_box("Charged sidechain distances, helix (Å)"))
        print(f'{"".ljust(self.category_pad)}{"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        for i in range(len(self.seq_list)):
            print(f'{self.seq_list[i].ljust(self.category_pad)}{"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.sidechain_sidechain_distances_hel[i]])}')
        print("")

        print(self._make_box("Charged sidechain distances, coil (Å)"))
        print(f'{"".ljust(self.category_pad)}{"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        for i in range(len(self.seq_list)):
            print(f'{self.seq_list[i].ljust(self.category_pad)}{"".join([f"{d:.2f}".ljust(self.value_pad) for d in self.charged_sidechain_distances_rc[i]])}')
        print("")

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
        G_rc = self._electrostatic_interaction_energy(qi=q1_rc, qj=q2_rc, r=coil_dist, factor_pi=4.0)
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
