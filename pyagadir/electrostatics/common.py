"""
What the electrostatic laws share, and the switch between them (dipole_law).

The side chain-macrodipole potential and the side chain-side chain pair energy come from the selected law
(point.py or gb.py); the self-consistent ionisation of every group in the helix and coil states, and the energy terms
that collect both laws, are the same for all of them.
"""
import math

import numpy as np

from pyagadir.chemistry import adjust_pKa, acidic_residue_ionization, basic_residue_ionization, ionization_free_energy
from pyagadir.electrostatics.gb import GB_LAWS, GBElectrostatics
from pyagadir.electrostatics.point import POINT_LAW, PointChargeElectrostatics

DIPOLE_LAWS = (POINT_LAW, *GB_LAWS)


class Electrostatics(GBElectrostatics, PointChargeElectrostatics):
    """
    The electrostatics of a helical segment under dipole_law: the law switch, the ionisation solver and the
    energy terms shared by the laws. Mixin of energies.PrecomputeParams.
    """

    def dipole_temperature_factor(self) -> float:
        """Muñoz 1995-III eq. (12): the side chain-macrodipole term uses a fixed dielectric and is
        scaled by eps(0 C)/eps(T) = exp(+0.004314 (T - 273.15)) (stronger at higher T).
        The GB laws (dipole_law="gb" or "gbx") carry their temperature dependence in eps_w(T) itself: 1."""
        if self.dipole_law in GB_LAWS:
            return 1.0
        return math.exp(0.004314 * (self.T_kelvin - 273.15))

    def _sidechain_dipole_potential(self, idx: int) -> tuple[float, float]:
        """
        Energy (kcal/mol) of a +1 charge on residue idx in the field of the helix macrodipole,
        split into the contributions of the N-terminal and the C-terminal end, before the eps(T)
        factor (dipole_temperature_factor). get_dG_sidechain_macrodipole multiplies these by the
        residue's charge, and the ionisation solver uses them as the helix-state field, so both see
        one electrostatic model. The law is _munoz_dipole_potential (default) or, with
        dipole_law="gb" or "gbx", _gb_dipole_potential.
        """
        if self.dipole_law in GB_LAWS:
            return self._gb_dipole_potential(idx)
        return self._munoz_dipole_potential(idx)

    def _assign_sidechain_dipole_potentials(self):
        """
        Helix-state macrodipole field on every charged residue (kcal/mol per +1 charge, including
        the eps(T) factor): the potential the ionisation solver uses for the helix state. It is
        the same law the side chain-macrodipole energy applies (_sidechain_dipole_potential).
        Also the gbp law's neutral-form energies (_assign_gb_neutral_energies; zero for the other laws).
        """
        n = len(self.seq_list)
        if int(self.ccap_idx) - int(self.ncap_idx) + 1 <= 5:
            raise ValueError(f"Invalid helix boundaries: start={self.ncap_idx}, end={self.ccap_idx}")
        charged = set(self.neg_charge_aa + self.pos_charge_aa)
        factor = self.dipole_temperature_factor()
        self.sidechain_dipole_potential = np.zeros(n, dtype=float)
        for idx, aa in enumerate(self.seq_list):
            if aa in charged:
                self.sidechain_dipole_potential[idx] = factor * sum(self._sidechain_dipole_potential(idx))
        self._assign_gb_neutral_energies()

    def get_sidechain_dipole_potentials(self):
        """
        Get the helix-state macrodipole potential on each residue (kcal/mol per +1 charge).
        """
        return self.sidechain_dipole_potential

    def show_sidechain_dipole_potentials(self):
        """
        Print out the helix-state macrodipole potentials in a nicely formatted way.
        """
        print(self._make_box("Sidechain macrodipole potential (kcal/mol per +1)"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"potential:".ljust(self.category_pad)} {"".join([f"{p:.2f}".ljust(self.value_pad) for p in self.sidechain_dipole_potential])}')
        print("")

    def _assign_modified_ionization_states(self):
        """
        Self-consistent mean-field ionisation in the helix and coil ensembles, and the ionisation
        free energy of the helical segment.

        Each titratable group's pKa is shifted by the electrostatic potential psi it feels as a
        fully charged probe (Lacroix 1998, eqs 8-11): the macrodipole (helix state only) and every
        other ionisable group at its current fractional charge, at helix or coil distances. The
        interaction model is the one the energy terms use: the side chain-macrodipole law
        (_sidechain_dipole_potential), the terminal-macrodipole term with its locality gate, the
        terminal-side chain geometry of _assign_terminal_sidechain_distances and the side
        chain-side chain distances of _assign_sidechain_sidechain_distances.

        The energy terms evaluate sum q_i q_j W_ij + sum q_i phi_i at the converged charges. That is
        the mean-field energy, not the free energy: it leaves out the cost of moving each group's
        ionisation away from its intrinsic value. At the self-consistent point the mean-field free
        energy is that energy plus, per group, ionization_free_energy(ln x_i, psi_i, q_i) >= 0
        (chemistry.py). dG_ionization = sum(helix) - sum(coil) supplies it; it is zero for groups
        that stay fully charged or fully neutral. Checked against exact enumeration of all
        protonation microstates (reasoning/nodes/N096.md).
        """
        MAX_ITERATIONS = 50
        CONVERGENCE_THRESHOLD = 0.005
        RT = 1.9865e-3 * self.T_kelvin  # same R as adjust_pKa

        n = len(self.seq_list)
        ionizable_sidechains = set(self.neg_charge_aa + self.pos_charge_aa)
        nterm_present = not (n > 0 and self.seq_list[0] == "Ac")
        cterm_present = not (n > 0 and self.seq_list[-1] == "Am")
        succinyl = n > 0 and self.seq_list[0] == "Sc"

        sites = []
        if nterm_present:
            sites.append(("Nterm", None))
        for idx, aa in enumerate(self.seq_list):
            if aa in ionizable_sidechains:
                sites.append(("SC", idx))
        if cterm_present:
            sites.append(("Cterm", None))

        def full_charge(kind, idx):
            """Charge of the fully ionised state of a site."""
            if kind == "Nterm":
                return -1.0 if succinyl else 1.0
            if kind == "Cterm":
                return -1.0
            return -1.0 if self.seq_list[idx] in self.neg_charge_aa else 1.0

        def base_pka(kind, idx):
            if kind == "Nterm":
                return self.nterm_pka
            if kind == "Cterm":
                return self.cterm_pka
            return float(self.seq_pka[idx])

        def is_basic(kind, idx):
            return full_charge(kind, idx) > 0

        def ionization(kind, idx, pka):
            if is_basic(kind, idx):
                return basic_residue_ionization(self.pH, pka)
            return acidic_residue_ionization(self.pH, pka)

        def env_charge(kind, idx, seq_q, nterm_q, cterm_q):
            if kind == "Nterm":
                return nterm_q
            if kind == "Cterm":
                return cterm_q
            return seq_q[idx]

        def pair_distance(kind1, idx1, kind2, idx2, helix):
            """Distance (A) between two ionisable sites; 99 for an unmodelled pair."""
            if kind1 != "SC" and kind2 != "SC":
                # terminus-terminus: the same distance in both states (_calculate_r over the whole
                # chain), so it shifts both pKas alike and adds no helix-coil energy
                # (get_dG_terminal_terminal_electrost models none)
                return self._calculate_r(n - 1)
            if kind1 != "SC" or kind2 != "SC":
                term = kind1 if kind1 != "SC" else kind2
                sc = idx2 if kind2 == "SC" else idx1
                if term == "Nterm":
                    arr = self.terminal_sidechain_distances_nterm if helix else self.terminal_sidechain_distances_nterm_rc
                else:
                    arr = self.terminal_sidechain_distances_cterm if helix else self.terminal_sidechain_distances_cterm_rc
                d = arr[sc]
            else:
                d = self.sidechain_sidechain_distances_hel[idx1, idx2] if helix else self.charged_sidechain_distances_rc[idx1, idx2]
            return 99.0 if np.isnan(d) else float(d)

        def site_potential(kind1, idx1, seq_q, nterm_q, cterm_q, helix):
            """Energy psi (kcal/mol) of the fully charged state of a site in its environment."""
            q1 = full_charge(kind1, idx1)
            psi = 0.0
            if helix:
                # macrodipole: the same terms as get_dG_terminals_macrodipole and
                # get_dG_sidechain_macrodipole
                if kind1 == "Nterm":
                    if self.ncap_idx < 6:
                        psi += self._electrostatic_interaction_energy(
                            qi=self.mu_helix, qj=q1, r=self.terminal_macrodipole_distance_nterm, factor_pi=4.0)
                elif kind1 == "Cterm":
                    if n - 1 - self.ccap_idx < 6:
                        psi += self._electrostatic_interaction_energy(
                            qi=-self.mu_helix, qj=q1, r=self.terminal_macrodipole_distance_cterm, factor_pi=4.0)
                else:
                    psi += q1 * self.sidechain_dipole_potential[idx1]
            for kind2, idx2 in sites:
                if kind2 == kind1 and idx2 == idx1:
                    continue
                if kind1 == "SC" and kind2 == "SC":  # gbp: the exact two-state GB pair coupling
                    c = self.gb_pairs[helix].get((min(idx1, idx2), max(idx1, idx2)))
                    if c is not None:
                        q2 = env_charge(kind2, idx2, seq_q, nterm_q, cterm_q)
                        psi += q1 * ((c[1] if idx1 < idx2 else c[2]) + q2 * c[0])
                        continue
                r = pair_distance(kind1, idx1, kind2, idx2, helix)
                if r < 40.0:
                    q2 = env_charge(kind2, idx2, seq_q, nterm_q, cterm_q)
                    psi += self._electrostatic_interaction_energy(qi=q1, qj=q2, r=r)
            if np.isnan(psi):
                raise ValueError("psi became NaN; check distance tables / assignments.")
            return psi

        def solve_state(helix: bool):
            """Mean-field fixed point for one ensemble; returns charges and the ionisation free energy."""
            seq_q = self.seq_ionization.copy()
            nterm_q = self.nterm_ionization if nterm_present else 0.0
            cterm_q = self.cterm_ionization if cterm_present else 0.0

            for _ in range(MAX_ITERATIONS):
                old = np.concatenate([seq_q[~np.isnan(seq_q)], [nterm_q, cterm_q]])
                for kind1, idx1 in sites:
                    psi = site_potential(kind1, idx1, seq_q, nterm_q, cterm_q, helix)
                    pka = adjust_pKa(T=self.T_kelvin, pKa_ref=base_pka(kind1, idx1), deltaG=psi,
                                     is_basic=is_basic(kind1, idx1))
                    q_new = ionization(kind1, idx1, pka)
                    if kind1 == "Nterm":
                        nterm_q = q_new
                    elif kind1 == "Cterm":
                        cterm_q = q_new
                    else:
                        seq_q[idx1] = q_new
                new = np.concatenate([seq_q[~np.isnan(seq_q)], [nterm_q, cterm_q]])
                if float(np.max(np.abs(new - old))) < CONVERGENCE_THRESHOLD:
                    break

            g_ion = 0.0
            psis = np.full(n, np.nan)
            for kind1, idx1 in sites:
                psi = site_potential(kind1, idx1, seq_q, nterm_q, cterm_q, helix)
                if kind1 == "SC":
                    psis[idx1] = psi
                pka = base_pka(kind1, idx1)
                ln_x = (pka - self.pH if is_basic(kind1, idx1) else self.pH - pka) * math.log(10.0)
                q = abs(float(env_charge(kind1, idx1, seq_q, nterm_q, cterm_q)))
                g_ion += ionization_free_energy(ln_x, psi, q, RT)
            return seq_q, float(nterm_q), float(cterm_q), g_ion, psis

        hel_seq, hel_n, hel_c, g_hel, self.site_potential_hel = solve_state(helix=True)
        rc_seq, rc_n, rc_c, g_rc, self.site_potential_rc = solve_state(helix=False)

        self.modified_seq_ionization_hel = hel_seq
        self.modified_nterm_ionization_hel = hel_n
        self.modified_cterm_ionization_hel = hel_c

        self.modified_seq_ionization_rc = rc_seq
        self.modified_nterm_ionization_rc = rc_n
        self.modified_cterm_ionization_rc = rc_c

        self.dG_ionization = g_hel - g_rc

    def get_modified_ionization_states(self):
        """
        Get the modified ionization states for the sequence.
        """
        return self.modified_seq_ionization_hel, self.modified_nterm_ionization_hel, self.modified_cterm_ionization_hel

    def show_modified_ionization_states(self):
        """
        Print out the modified ionization states for the sequence in a nicely formatted way.
        """
        print(self._make_box("Modified ionization states, helix"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.2f}".ljust(self.value_pad) for q in self.modified_seq_ionization_hel])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.modified_nterm_ionization_hel:.2f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.modified_cterm_ionization_hel:.2f}')
        print("")
        print(self._make_box("Modified ionization states, coil"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.2f}".ljust(self.value_pad) for q in self.modified_seq_ionization_rc])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.modified_nterm_ionization_rc:.2f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.modified_cterm_ionization_rc:.2f}')
        print("")

    def get_dG_sidechain_macrodipole(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the interaction energy between charged side-chains and the helix macrodipole.

        The macrodipole is treated as local: the field of a helix end comes from the
        unpaired amides of the first turn or carbonyls of the last turn, so each charged
        residue (Ncap to Ccap inclusive) interacts with the NEAREST helix end only (ties go
        to the N-terminus), with the law of Munoz 1995-II eq. 11:

            N-terminal end:  g =  q × K / d_N² × exp(−κ × d_N)
            C-terminal end:  g = −q × K / d_C² × exp(−κ × d_C)

        K = 0.6 × 4.9² kcal Å² mol⁻¹, d_N / d_C are distances from the charged group to that
        end (Coulomb distance tables, after Lacroix 1998 Table VII), and residues more than
        nine positions from the cap contribute nothing.

        The other (far) end adds the field of its half charge (Hol et al. 1978): screened
        Coulomb in water at the Table VII distance to that end, extended past 13 positions by
        1.5 A per residue (FEH; reasoning/nodes/N083.md), scaled by FEH_FAR_APBS.

        Flanking charged residues outside the helix interact with the nearby end's half charge
        by the same screened Coulomb law in water, at the Lacroix 1998 flank distance (6 A at
        N'/C', +3 A per further position), and the energy is assigned to the cap.

        The law itself is _sidechain_dipole_potential (shared with the ionisation solver); with
        dipole_law="gb" it is the Generalized Born field (_gb_dipole_potential) and the text above
        describes only its flanks.

        Returns:
            tuple[np.ndarray, np.ndarray]: N-terminal and C-terminal dipole energy arrays.
        """
        n = len(self.seq_list)
        energy_N = np.zeros(n, dtype=float)
        energy_C = np.zeros(n, dtype=float)
        charged = set(self.neg_charge_aa + self.pos_charge_aa)
        ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)

        for idx in range(max(0, ncap_i - 9), min(n, ccap_i + 10)):
            if self.seq_list[idx] not in charged:
                continue
            if self.gb_neutral_energy[idx] != 0.0:  # gbp: the neutral form's energy, at every charge (N-cap..C-cap)
                k0, c0 = idx - ncap_i, ccap_i - idx
                (energy_N if k0 <= c0 else energy_C)[idx] += self.gb_neutral_energy[idx]
            q = float(self.modified_seq_ionization_hel[idx])
            if abs(q) < 1e-6:
                continue
            phi_N, phi_C = self._sidechain_dipole_potential(idx)
            if idx < ncap_i:  # N-terminal flank: assigned to the N-cap
                energy_N[ncap_i] += q * phi_N
            elif idx > ccap_i:  # C-terminal flank: assigned to the C-cap
                energy_C[ccap_i] += q * phi_C
            else:  # += : a charged cap also collects its flanking residues' energy
                energy_N[idx] += q * phi_N
                energy_C[idx] += q * phi_C

        return energy_N, energy_C
        
    def get_dG_ionization(self) -> float:
        """
        Ionisation free energy of the helical segment relative to the coil (kcal/mol): the part of
        the mean-field electrostatic free energy that the q-weighted energy terms leave out (see
        _assign_modified_ionization_states). Not scaled by temperature beyond RT.
        """
        return float(self.dG_ionization)

    def get_dG_sidechain_sidechain_electrost(self) -> np.ndarray:
        """
        Calculate the electrostatic free energy contribution for charged residue sidechains
        inside and outside the helical segment, using Lacroix et al. (1998) equations.
        Half of the energy is added to each of the charged sidechain residues.

        Each pair takes the GB pair law where it covers the pair (gbp, _gb_pair_energy) and the
        point-charge law otherwise (_point_pair_energy).

        Returns:
            np.ndarray: n x n symmetric matrix of pairwise electrostatic free energy contributions,
                       with each interaction energy split between upper and lower triangles.
        """
        energy_matrix = np.zeros((len(self.seq_list), len(self.seq_list)))

        # Iterate over all charged residue pairs
        for AA1, AA2, idx1, idx2 in self.charged_pairs:
            # Skip if not in upper triangle
            if idx2 <= idx1:
                continue
            dG = self._gb_pair_energy(idx1, idx2)
            if dG is None:
                dG = self._point_pair_energy(idx1, idx2)
            if dG is None:  # no modelled change between the states
                continue

            # Store half the energy difference in both triangles of the matrix (give half of the energy to each sidechain)
            energy_matrix[idx1, idx2] = energy_matrix[idx2, idx1] = dG / 2

        return energy_matrix
