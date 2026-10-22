"""
What the electrostatic laws share, and the switch between them (dipole_law).

The side chain-macrodipole potential and the side chain-side chain pair energy come from the selected law
(point.py or gb.py); the self-consistent ionisation of every group in the helix and coil states, and the energy terms
that collect both laws, are the same for all of them.
"""
import math
import warnings

import numpy as np

from pyagadir.chemistry import (LN10, R_KCAL, T_REF, adjust_pKa, acidic_residue_ionization, basic_residue_ionization,
                                ionization_free_energy)
from pyagadir.electrostatics.gb import GB_LAWS, GBElectrostatics
from pyagadir.electrostatics.point import POINT_LAW, PointChargeElectrostatics

DIPOLE_LAWS = (POINT_LAW, *GB_LAWS)


class IonizationConvergenceWarning(RuntimeWarning):
    """The helix- or coil-state ionisation did not converge within the sweep limit."""


class Electrostatics(GBElectrostatics, PointChargeElectrostatics):
    """
    The electrostatics of a helical segment under dipole_law: the law switch, the ionisation solver and the
    energy terms shared by the laws. Mixin of energies.PrecomputeParams.
    """

    def _nonCoulomb_charge_coefficients(self) -> tuple[dict, dict]:
        """
        Charge-dependent non-Coulomb terms of the helix state as G0 + sum_a c_a p_a + sum_{a<b} c_ab p_a p_b in the
        charged fractions p = |q^h| (EnergyCalculator provides them; none here). Returns ({a: c_a}, {(a, b): c_ab}).
        To be removed with the per-segment solver (_assign_modified_ionization_states).
        """
        return {}, {}

    def _nonCoulomb_charge_coefficients_local(self) -> tuple[dict, dict]:
        """The same coefficients for SegmentBatch (EnergyCalculator provides them; none here)."""
        return {}, {}

    def dipole_temperature_factor(self) -> float:
        """Muñoz 1995-III eq. (12): the side chain-macrodipole term uses a fixed dielectric and is
        scaled by eps(0 C)/eps(T) = exp(+0.004314 (T - 273.15)) (stronger at higher T).
        The GB law (dipole_law="gbx") carries its temperature dependence in eps_w(T) itself: 1."""
        if self.dipole_law in GB_LAWS:
            return 1.0
        return math.exp(0.004314 * (self.T_kelvin - T_REF))

    def _sidechain_dipole_potential(self, idx: int) -> tuple[float, float]:
        """
        Energy (kcal/mol) of a +1 charge on residue idx in the field of the helix macrodipole,
        split into the contributions of the N-terminal and the C-terminal end, before the eps(T)
        factor (dipole_temperature_factor). get_dG_sidechain_macrodipole multiplies these by the
        residue's charge, and the ionisation solver uses them as the helix-state field, so both see
        one electrostatic model. The law is _munoz_dipole_potential (default) or, with
        the GB law ("gbx"), _gb_dipole_potential.
        """
        if self.dipole_law in GB_LAWS:
            return self._gb_dipole_potential(idx)
        return self._munoz_dipole_potential(idx)

    def _assign_sidechain_dipole_potentials(self):
        """
        Helix-state macrodipole field on every charged residue (kcal/mol per +1 charge, including
        the eps(T) factor): the potential the ionisation solver uses for the helix state. It is
        the same law the side chain-macrodipole energy applies (_sidechain_dipole_potential).
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

    def _ionizable_sites(self):
        """
        The titratable groups in the ionisation solver's order: the free N-terminal group, the ionisable side chains
        in sequence order, the free C-terminal group. Returns (sites, nterm_present, cterm_present, succinyl), a site
        being ("Nterm", None), ("SC", idx) or ("Cterm", None).
        """
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
        return sites, nterm_present, cterm_present, succinyl

    def _site_full_charge(self, kind, idx) -> float:
        """Charge of the fully ionised state of a site."""
        if kind == "Nterm":
            return -1.0 if (len(self.seq_list) > 0 and self.seq_list[0] == "Sc") else 1.0
        if kind == "Cterm":
            return -1.0
        return -1.0 if self.seq_list[idx] in self.neg_charge_aa else 1.0

    def _site_base_pka(self, kind, idx) -> float:
        """Intrinsic pKa of a site."""
        if kind == "Nterm":
            return self.nterm_pka
        if kind == "Cterm":
            return self.cterm_pka
        return float(self.seq_pka[idx])

    def _site_pair_distance(self, kind1, idx1, kind2, idx2, helix: bool) -> float:
        """Distance (A) between two ionisable sites in the helix or coil state; 99 for an unmodelled pair."""
        if kind1 != "SC" and kind2 != "SC":
            # terminus-terminus: the same distance in both states (_calculate_r over the whole
            # chain); its energy changes between the states through the charges only
            # (get_dG_terminal_terminal_electrost; point-charge law)
            return self._calculate_r(len(self.seq_list) - 1)
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

    def _assign_modified_ionization_states(self):
        """
        The per-segment solver. To be removed (owner, 2026-10-22): every law solves in SegmentBatch, so nothing in the
        package calls this any more; tests/test_segment_batch.py keeps it as the reference until it is deleted
        (reasoning/nodes/N201.md). Before deleting it, move the references and reasoning in this docstring and its
        comments into SegmentBatch.

        Self-consistent mean-field ionisation in the helix and coil ensembles, and the ionisation
        free energy of the helical segment.

        Each titratable group's pKa is shifted by the electrostatic potential psi it feels as a
        fully charged probe (Lacroix 1998, eqs 8-11): the macrodipole (helix state only) and every
        other ionisable group at its current fractional charge, at helix or coil distances. The
        interaction model is the one the energy terms use: the side chain-macrodipole law
        (_sidechain_dipole_potential), the terminal-macrodipole term with its locality gate, the
        terminal-side chain geometry of _assign_terminal_sidechain_distances and the side
        chain-side chain distances of _assign_sidechain_sidechain_distances.

        psi also carries the derivative of the charge-dependent non-Coulomb terms of the helix state (Table 1
        charged/neutral cells, capping rules, Table IV/V, Petukhov and charged-staple motifs) with respect to the
        group's charged fraction, dG_nonC/dp_a = c_a + sum_b c_ab p_b (_nonCoulomb_charge_coefficients): Lacroix
        1998 takes those terms into account in the pKa, and the mean-field free energy is stationary in p only with
        them.

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
        RT = R_KCAL * self.T_kelvin  # same R as adjust_pKa

        n = len(self.seq_list)
        sites, nterm_present, cterm_present, _ = self._ionizable_sites()
        full_charge = self._site_full_charge
        base_pka = self._site_base_pka

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

        pair_distance = self._site_pair_distance

        nonC_a, nonC_ab = self._nonCoulomb_charge_coefficients()
        nonC_partners = {}
        for (a, b), c in nonC_ab.items():
            nonC_partners.setdefault(a, []).append((b, c))
            nonC_partners.setdefault(b, []).append((a, c))

        def site_potential(kind1, idx1, seq_q, nterm_q, cterm_q, helix):
            """Energy psi (kcal/mol) of the fully charged state of a site in its environment."""
            q1 = full_charge(kind1, idx1)
            psi = 0.0
            if helix and kind1 == "SC" and idx1 in nonC_a:
                # charge-dependent non-Coulomb terms: dG_nonC / dp at the current charges
                psi += nonC_a[idx1] + sum(c * abs(float(seq_q[b])) for b, c in nonC_partners.get(idx1, ()))
            if helix:
                # macrodipole: the same terms as get_dG_terminals_macrodipole and
                # get_dG_sidechain_macrodipole
                if kind1 == "Nterm":
                    psi += self._electrostatic_interaction_energy(
                        qi=self.mu_helix, qj=q1, r=self.terminal_macrodipole_distance_nterm, factor_pi=4.0)
                elif kind1 == "Cterm":
                    psi += self._electrostatic_interaction_energy(
                        qi=-self.mu_helix, qj=q1, r=self.terminal_macrodipole_distance_cterm, factor_pi=4.0)
                else:
                    psi += q1 * self.sidechain_dipole_potential[idx1]
            for kind2, idx2 in sites:
                if kind2 == kind1 and idx2 == idx1:
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

            converged = False
            for sweep in range(1, MAX_ITERATIONS + 1):
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
                    converged = True
                    break
            self.ionization_sweeps[helix] = sweep
            if not converged:
                warnings.warn(
                    f"{'helix' if helix else 'coil'}-state ionisation of segment i={self.i}, j={self.j} of {self.seq} "
                    f"not converged after {MAX_ITERATIONS} sweeps (last change {float(np.max(np.abs(new - old))):.3g})",
                    IonizationConvergenceWarning, stacklevel=2)

            g_ion = 0.0
            psis = np.full(n, np.nan)
            for kind1, idx1 in sites:
                psi = site_potential(kind1, idx1, seq_q, nterm_q, cterm_q, helix)
                if kind1 == "SC":
                    psis[idx1] = psi
                pka = base_pka(kind1, idx1)
                ln_x = (pka - self.pH if is_basic(kind1, idx1) else self.pH - pka) * LN10
                q = abs(float(env_charge(kind1, idx1, seq_q, nterm_q, cterm_q)))
                g_ion += ionization_free_energy(ln_x, psi, q, RT)
            return seq_q, float(nterm_q), float(cterm_q), g_ion, psis

        self.ionization_sweeps = {}  # sweeps used per state (True: helix, False: coil)
        hel_seq, hel_n, hel_c, g_hel, self.site_potential_hel = solve_state(helix=True)
        rc_seq, rc_n, rc_c, g_rc, self.site_potential_rc = solve_state(helix=False)

        self.modified_seq_ionization_hel = hel_seq
        self.modified_nterm_ionization_hel = hel_n
        self.modified_cterm_ionization_hel = hel_c

        self.modified_seq_ionization_rc = rc_seq
        self.modified_nterm_ionization_rc = rc_n
        self.modified_cterm_ionization_rc = rc_c

        self.dG_ionization = g_hel - g_rc
        self.ionization_free_energy_states = {True: g_hel, False: g_rc}

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
        a GB law it is the Generalized Born field (_gb_dipole_potential) and the text above
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

        Each pair takes the point-charge law (_point_pair_energy).

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
            dG = self._point_pair_energy(idx1, idx2)
            if dG is None:  # no modelled change between the states
                continue

            # Store half the energy difference in both triangles of the matrix (give half of the energy to each sidechain)
            energy_matrix[idx1, idx2] = energy_matrix[idx2, idx1] = dG / 2

        return energy_matrix


# Every interacting pair of ionisable groups lies within this many residues: the Lacroix 1998 Table VI distances (side
# chain-side chain, terminal group-side chain, helix and coil) end at i+12 and all of them are below the solver's
# 40 A cutoff; beyond, a pair has 99 A in both states, and neither the energy terms nor the solver count it.
PAIR_RANGE = 12
# Helical segments whose ionisation SegmentBatch solves at once (models.AGADIR): the arrays scale with it.
SEGMENT_BATCH_SIZE = 512


class SegmentBatch:
    """
    The ionisation of every helical segment of a prediction at once (models.AGADIR), or of a single segment (an
    EnergyCalculator built on its own is a batch of one), under both laws (the default and gbx). Same results as the
    per-segment solver of Electrostatics._assign_modified_ionization_states, which is to be removed:

    - the coil state does not depend on the segment (coil distances by sequence separation, the intrinsic charges as
      the start, the point-charge pair law, no macrodipole and no non-Coulomb term), so it is solved once, here, and
      every segment takes that solution. gbx differs from the default only in its side chain-macrodipole field (the
      GB table, read through each segment's sidechain_dipole_potential) and in table values;
    - pair lists, distance tables and the sums cover the pairs within PAIR_RANGE residues, the only ones that
      interact;
    - each state's Gauss-Seidel sweep runs for all segments at once: one array operation per site update, over the
      segments, never over the sites (the 0.005 stopping rule makes the result depend on the iteration path), each
      segment with its own start, potentials and stopping sweep. The pair energies repeat
      _electrostatic_interaction_energy's operations in its order and the transcendental functions are evaluated per
      element, so a segment's charges are the per-segment solver's to the last bit for the same inputs;
    - the non-Coulomb coefficients come from EnergyCalculator._nonCoulomb_charge_coefficients_local, equal to the
      whole-energy differences of _nonCoulomb_charge_coefficients up to rounding (about 1e-14 kcal/mol).

    ref is a segment of this peptide (an EnergyCalculator) whose setup is complete; its ionisation is not used.
    """

    MAX_ITERATIONS = 50
    CONVERGENCE_THRESHOLD = 0.005
    CUTOFF = 40.0

    def __init__(self, ref):
        # Everything here is the peptide's: it does not depend on which segment is helical.
        self.n = len(ref.seq_list)
        self.seq = ref.seq
        # The titratable sites in the per-segment solver's order (N-terminus, side chains, C-terminus). Per site: the
        # charge of its fully ionised state (+1 or -1), its intrinsic pKa, whether it is a base, and its intrinsic
        # charged fraction at this pH, which is where every sweep starts.
        self.sites, self.nterm_present, self.cterm_present, _ = ref._ionizable_sites()
        K = len(self.sites)
        self.q1 = np.array([ref._site_full_charge(kd, ix) for kd, ix in self.sites])
        self.pka = np.array([ref._site_base_pka(kd, ix) for kd, ix in self.sites])
        self.basic = self.q1 > 0
        self.q0 = np.array([ref.nterm_ionization if kd == "Nterm" else ref.cterm_ionization if kd == "Cterm" else
                            float(ref.seq_ionization[ix]) for kd, ix in self.sites])
        # the constants of _electrostatic_interaction_energy (they depend on T and the ionic strength only)
        self.pH, self.T_kelvin = ref.pH, ref.T_kelvin
        self.kappa, self.epsilon_0, self.epsilon_r, self.N_A, self.e = (ref.kappa, ref.epsilon_0, ref.epsilon_r,
                                                                        ref.N_A, ref.e)
        # shared by the segments' setup (PrecomputeParams with batch=...)
        self.seq_pka, self.nterm_pka, self.cterm_pka = ref.seq_pka, ref.nterm_pka, ref.cterm_pka
        self.seq_ionization, self.nterm_ionization, self.cterm_ionization = (ref.seq_ionization, ref.nterm_ionization,
                                                                             ref.cterm_ionization)
        # Only the pairs within PAIR_RANGE ever count, so each segment's setup and pair energies loop over these.
        self.charged_pairs = [p for p in ref.charged_pairs if p[3] - p[2] <= PAIR_RANGE]
        # Coil distances depend on the pair's residues and separation only: one matrix serves every segment.
        self.coil_distances = ref.charged_sidechain_distances_rc
        # A segment's helix distance matrix starts as a copy of this template: NaN, except 99 A for the charged pairs
        # beyond PAIR_RANGE, as the per-segment assignment writes them. The segment's setup fills in the pairs within
        # range (point.py _assign_sidechain_sidechain_distances).
        self.helix_distance_template = np.full((self.n, self.n), np.nan)
        for _, _, i1, i2 in ref.charged_pairs:
            if i2 - i1 > PAIR_RANGE:
                self.helix_distance_template[i1, i2] = self.helix_distance_template[i2, i1] = 99
        self.coil_pair_energy = {}  # (idx1, idx2) -> coil-state pair energy (point.py _point_pair_energy)
        # Site k's partners: the sites whose Coulomb term the per-segment solver can add to k's potential, in its order
        # (by site index). That is every other site except side chains more than PAIR_RANGE apart, which are always
        # at 99 A and dropped by the 40 A cutoff. Pairs with a terminal group are all kept and left to the cutoff,
        # as in the solver.
        self.partners = [[m for m in range(K) if m != k and (
            self.sites[k][0] != "SC" or self.sites[m][0] != "SC" or abs(self.sites[k][1] - self.sites[m][1]) <= PAIR_RANGE)]
            for k in range(K)]
        # The partners' distances are stored flat, one column per (site, partner) slot, site k's slots being
        # offsets[k]:offsets[k + 1]. The arrays below say where each slot's distance is found in a segment's tables
        # (_slot_distances), so a segment's distances take a few array lookups instead of one _site_pair_distance
        # call per pair:
        #   slot_ss:            side chain-side chain slots, read from the side chain distance matrix at
        #                       (slot_ss_i, slot_ss_j);
        #   slot_term[terminus]: terminal group-side chain slots, read from that terminus's distance array at the
        #                       side chain's index;
        #   slot_tt:            the terminus-terminus slots (the same distance in both states and every segment).
        self.offsets = np.cumsum([0] + [len(p) for p in self.partners])
        slots = [(k, m) for k in range(K) for m in self.partners[k]]
        kind = [(self.sites[k][0], self.sites[m][0]) for k, m in slots]
        self.slot_ss = np.array([t for t, (a, b) in enumerate(kind) if a == "SC" and b == "SC"], int)
        self.slot_ss_i = np.array([self.sites[slots[t][0]][1] for t in self.slot_ss], int)
        self.slot_ss_j = np.array([self.sites[slots[t][1]][1] for t in self.slot_ss], int)
        self.slot_term = {}  # terminus -> (slots, side chain indices)
        for term in ("Nterm", "Cterm"):
            ts = [t for t, (a, b) in enumerate(kind) if (a == term and b == "SC") or (a == "SC" and b == term)]
            sc = [self.sites[slots[t][0]][1] if kind[t][0] == "SC" else self.sites[slots[t][1]][1] for t in ts]
            self.slot_term[term] = (np.array(ts, int), np.array(sc, int))
        self.slot_tt = np.array([t for t, (a, b) in enumerate(kind) if a != "SC" and b != "SC"], int)
        self.r_tt = ref._calculate_r(self.n - 1)  # the terminus-terminus distance (_site_pair_distance)
        sc_site = {ix: m for m, (kd, ix) in enumerate(self.sites) if kd == "SC"}
        # The non-Coulomb term of a side chain i couples it to the side chains at i-4, i-3, i+3 and i+4 (Table IV
        # pairs): their site indices, None where that residue is not titratable, in the order the per-segment solver
        # sums them (its nonC_partners list for i holds i-4, i-3, i+3, i+4 in that order).
        self.nonC_partners = [[sc_site.get(ix + d) if kd == "SC" else None for d in (-4, -3, 3, 4)]
                              for kd, ix in self.sites]

        # The coil state, once for the peptide: one sweep with the coil distances and no helix terms.
        r = self._slot_distances([(ref.charged_sidechain_distances_rc, ref.terminal_sidechain_distances_nterm_rc,
                                   ref.terminal_sidechain_distances_cterm_rc)])
        q, psis, g, sweeps = self._sweep(1, r, None, lambda s_i: f"coil-state ionisation of {self.seq}")
        self.coil = self._per_residue(q[0], psis[0]) + (float(g[0]), int(sweeps[0]))

    def _slot_distances(self, tables) -> list:
        """
        The distance (A) of every partner slot, per segment, from each segment's tables (side chain distance
        matrix, N-terminal and C-terminal group distance arrays, in one state): _site_pair_distance's values, with 99
        for an unmodelled pair (NaN in the tables). Returns site k's distances as the (segment, partner) array r[k].
        """
        r_flat = np.full((len(tables), int(self.offsets[-1])), 99.0)
        for s_i, (ss, nterm, cterm) in enumerate(tables):
            row = r_flat[s_i]
            if self.slot_ss.size:
                row[self.slot_ss] = ss[self.slot_ss_i, self.slot_ss_j]
            for term, arr in (("Nterm", nterm), ("Cterm", cterm)):
                ts, sc = self.slot_term[term]
                if ts.size:
                    row[ts] = arr[sc]
            if self.slot_tt.size:
                row[self.slot_tt] = self.r_tt
            row[np.isnan(row)] = 99.0
        return [r_flat[:, self.offsets[k]:self.offsets[k + 1]] for k in range(len(self.sites))]

    def _sweep(self, S, r, helix_terms, label):
        """
        The per-segment solver's solve_state for S segments at once. r: site k's partner distances (_slot_distances);
        helix_terms: None for the coil state, else the helix state's charge-independent terms (const, has_a, ca, cp;
        solve_helix); label(s_i): the name of segment s_i in a non-convergence warning. Returns the converged charges
        and site potentials (segment, site), each segment's ionisation free energy, and its number of sweeps.
        """
        K = len(self.sites)
        q1, pka0, basic, pH = self.q1, self.pka, self.basic, self.pH

        # The charge-independent factors of each pair energy, once per slot. _electrostatic_interaction_energy
        # computes (((e**2 * q1) * q2) / D) * screen, then N_A * . / 4184, with D = 4 pi eps0 eps_r r; the same
        # operations in the same order give the same bits. The screening factor takes math.exp once per distinct
        # distance (a peptide has few), which is the solver's value exactly. inside marks the pairs the 40 A cutoff
        # keeps.
        e2 = self.e ** 2
        c0 = 4.0 * math.pi * self.epsilon_0 * self.epsilon_r
        D = [c0 * (rk * 1e-10) for rk in r]
        screen = []
        for rk in r:
            u, inv = np.unique(rk * 1e-10, return_inverse=True)
            screen.append(np.array([math.exp(-self.kappa * float(x)) for x in u])[inv].reshape(rk.shape))
        inside = [rk < self.CUTOFF for rk in r]
        X = LN10 * R_KCAL * self.T_kelvin  # adjust_pKa's denominator: a potential psi shifts the pKa by psi / X

        def psi_of(k, q):
            """
            Potential (kcal/mol) of site k's fully charged state in every segment, at the charges q (S x K): the
            per-segment solver's site_potential, term by term and in its order. np.where(mask, psi + x, psi) adds x
            only in the segments where the per-segment solver adds it, and leaves the other sums untouched.
            """
            psi = np.zeros(S)
            if helix_terms is not None:  # the helix state only, as in the per-segment solver
                const, has_a, ca, cp = helix_terms
                tot = np.zeros(S)  # the solver's sum() of the Table IV partners starts at 0
                for o, m in enumerate(self.nonC_partners[k]):
                    if m is not None:
                        tot = tot + cp[:, k, o] * np.abs(q[:, m])
                psi = np.where(has_a[:, k], psi + (ca[:, k] + tot), psi)  # non-Coulomb: dG/dp_k = c_k + sum c_kb p_b
                psi = psi + const[:, k]  # macrodipole
            for m_i, m in enumerate(self.partners[k]):  # Coulomb terms with the partners at their current charges
                en = (((e2 * q1[k]) * q[:, m]) / D[k][:, m_i]) * screen[k][:, m_i]
                en = self.N_A * en / 4184
                psi = np.where(inside[k][:, m_i], psi + en, psi)
            return psi

        def ionize(k, pka):
            """Site k's charged fraction at the shifted pKa, per element with the chemistry functions themselves
            (NumPy's vectorised power differs from them in the last bit for some inputs)."""
            f = basic_residue_ionization if basic[k] else acidic_residue_ionization
            return np.array([f(pH, float(x)) for x in pka])

        # Gauss-Seidel sweeps, as in the per-segment solver: from the intrinsic charges, the sites in order, each
        # site's charge updated from its potential at the charges as they stand (including the sites already updated
        # in this sweep). One array operation updates site k in all segments. A segment stops at the end of the first
        # sweep in which every charge moved by less than CONVERGENCE_THRESHOLD; it is then inactive and keeps its
        # charges while the others go on.
        q = np.tile(self.q0, (S, 1))
        active = np.ones(S, bool)
        sweeps = np.zeros(S, int)
        change = np.zeros(S)
        for sweep in range(1, self.MAX_ITERATIONS + 1):
            old = q.copy()
            for k in range(K):
                gamma = 1 if basic[k] else -1
                pka = pka0[k] - (gamma * psi_of(k, q) / X)  # adjust_pKa: a base's pKa falls in a positive potential
                q[:, k] = np.where(active, ionize(k, pka), q[:, k])
            change = np.abs(q - old).max(axis=1) if K else np.zeros(S)
            sweeps[active] = sweep
            active &= ~(change < self.CONVERGENCE_THRESHOLD)
            if not active.any():
                break
        for s_i in np.flatnonzero(active):
            warnings.warn(f"{label(s_i)} not converged after {self.MAX_ITERATIONS} sweeps "
                          f"(last change {float(change[s_i]):.3g})", IonizationConvergenceWarning, stacklevel=3)

        # At the converged charges: each site's potential, and the state's ionisation free energy, the sum over the
        # sites in order of ionization_free_energy (chemistry.py), the cost of each group's ionisation departing from
        # its intrinsic value.
        RT = R_KCAL * self.T_kelvin
        g = np.zeros(S)
        psis = np.zeros((S, K))
        for k in range(K):
            psi = psi_of(k, q)
            psis[:, k] = psi
            ln_x = (pka0[k] - pH if basic[k] else pH - pka0[k]) * LN10
            g = g + np.array([ionization_free_energy(float(ln_x), float(p), abs(float(x)), RT)
                              for p, x in zip(psi, q[:, k])])
        return q, psis, g, sweeps

    def _per_residue(self, q_row, psi_row):
        """One segment's site charges and potentials as the per-segment solver stores them: the charged fraction per
        residue (non-titratable residues keep 0), the terminal groups' charges (0 for a capped end), and the side
        chains' potentials per residue (NaN elsewhere)."""
        seq_q = self.seq_ionization.copy()
        site_psi = np.full(self.n, np.nan)
        n_q = c_q = 0.0
        for k, (kd, ix) in enumerate(self.sites):
            if kd == "Nterm":
                n_q = float(q_row[k])
            elif kd == "Cterm":
                c_q = float(q_row[k])
            else:
                seq_q[ix] = q_row[k]
                site_psi[ix] = psi_row[k]
        return (seq_q, n_q if self.nterm_present else 0.0, c_q if self.cterm_present else 0.0, site_psi)

    def solve_helix(self, calcs) -> None:
        """Solve the helix-state ionisation of the segments calcs (EnergyCalculators of this peptide whose setup is
        complete), and give each its helix and coil ionisation, site potentials and dG_ionization."""
        # Arrays are (segment, site): row s_i is calcs[s_i], column k is self.sites[k].
        S, K = len(calcs), len(self.sites)
        q1 = self.q1

        # Per segment, the parts of each site's helix-state potential that do not depend on the charges:
        #    const        the macrodipole term (a terminal group: its energy with the macrodipole; a side chain: its
        #                 charge times the macrodipole potential at its position);
        #    has_a, ca    the non-Coulomb term's own coefficient c_a, where the segment has one;
        #    cp           its Table IV couplings c_ab to the partners at -4, -3, +3, +4 (0 where absent, which adds
        #                 exactly 0 to the sum).
        const = np.zeros((S, K))
        has_a = np.zeros((S, K), bool)
        ca = np.zeros((S, K))
        cp = np.zeros((S, K, 4))
        for s_i, c in enumerate(calcs):
            nonC_a, nonC_ab = c._nonCoulomb_charge_coefficients_local()
            for k, (kd, ix) in enumerate(self.sites):
                if kd == "Nterm":
                    const[s_i, k] = c._electrostatic_interaction_energy(
                        qi=c.mu_helix, qj=float(q1[k]), r=c.terminal_macrodipole_distance_nterm, factor_pi=4.0)
                elif kd == "Cterm":
                    const[s_i, k] = c._electrostatic_interaction_energy(
                        qi=-c.mu_helix, qj=float(q1[k]), r=c.terminal_macrodipole_distance_cterm, factor_pi=4.0)
                else:
                    const[s_i, k] = float(q1[k]) * c.sidechain_dipole_potential[ix]
                    if ix in nonC_a:
                        has_a[s_i, k] = True
                        ca[s_i, k] = nonC_a[ix]
                        for o, d in enumerate((-4, -3, 3, 4)):
                            cp[s_i, k, o] = nonC_ab.get((ix + d, ix) if d < 0 else (ix, ix + d), 0.0)
        # the helix-state distances of every partner slot, from each segment's own tables
        r = self._slot_distances([(c.sidechain_sidechain_distances_hel, c.terminal_sidechain_distances_nterm,
                                   c.terminal_sidechain_distances_cterm) for c in calcs])
        q, psis, g, sweeps = self._sweep(
            S, r, (const, has_a, ca, cp),
            lambda s_i: f"helix-state ionisation of segment i={calcs[s_i].i}, j={calcs[s_i].j} of {self.seq}")

        # Give each segment the attributes the per-segment solver sets: the helix state from this solve, the coil
        # state from the peptide's (copies, so that no segment's arrays are another's), dG_ionization = helix minus
        # coil.
        rc_seq, rc_n, rc_c, rc_psi, g_rc, rc_sweeps = self.coil
        for s_i, c in enumerate(calcs):
            (c.modified_seq_ionization_hel, c.modified_nterm_ionization_hel, c.modified_cterm_ionization_hel,
             c.site_potential_hel) = self._per_residue(q[s_i], psis[s_i])
            c.modified_seq_ionization_rc = rc_seq.copy()
            c.modified_nterm_ionization_rc = rc_n
            c.modified_cterm_ionization_rc = rc_c
            c.site_potential_rc = rc_psi.copy()
            c.dG_ionization = float(g[s_i]) - g_rc
            c.ionization_free_energy_states = {True: float(g[s_i]), False: g_rc}
            c.ionization_sweeps = {True: int(sweeps[s_i]), False: rc_sweeps}
