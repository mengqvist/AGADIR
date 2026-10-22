"""
Generalized Born electrostatics: dipole_law "gbx" and "gbp". Opt-in and in development; the default law is the
point-charge law (point.py).

Both laws rest on the Generalized Born field of the helix backbone on the charged side chain, tabulated from built
poly-Ala helices (params/sidechain_dipole_gb.tsv, made by gb_pipeline/helix_builds/build_gb_dipole_table.py;
params/README.md), in place of the point-charge side chain-macrodipole law. The field alone was once the law "gb";
it was removed (owner, 2026-10-22) because used alone it counts twice the first-turn and capping electrostatics
that Table 1's measured values already hold (_gb_dipole_potential). _gb_field_potential still gives it, for the
transfer that removes that double count (gb_pipeline/transfer/build_gb_transfer.py).
"gbx": the GB field with every Table 1 value that was measured with the default law in the loop transferred to it (a
potential offset per residue and Table 1 class at that value's reference measurement, params/
sidechain_dipole_gb_transfer.tsv, made by gb_pipeline/transfer/build_gb_transfer.py) and the parameters re-derived
under it (params/gbx_overrides.tsv). See params/README.md "gbx".
"gbp": gbx plus the Generalized Born side chain-side chain pair law (params/sidechain_pair_gb.tsv, made by
gb_pipeline/helix_builds/build_gb_pair_table.py) in place of the point-charge pair term, the exact two-state coupling
of every ionisable group to both laws (each protonation form's own GB energy, so a group's pKa shifts by the
charged-minus-neutral energy), the statistical salt-bridge network for the Table IV acid-base cells, and the Table 1
transfer of all of it (params/sidechain_gbp_transfer.tsv, gb_pipeline/transfer/build_gbp_transfer.py). See
params/README.md "gbp". How every GB table is made, and the structures, MD runs and intermediates behind it:
gb_pipeline/README.md.

The GB laws keep the point-charge law for the terminal groups, for charged residues outside the caps, for table rows
with no sterically allowed structure, and (gbp) for pairs the pair table does not cover.

GBElectrostatics is a mixin of energies.PrecomputeParams and reads the segment set up there.
"""
import math
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import R_KCAL

GB_LAWS = ("gbx", "gbp")
GBX_LAWS = ("gbx", "gbp")  # laws with a Table 1 transfer and parameter overrides
# Tabulated residues, their table form and the sign of its charge:
GB_DIPOLE_FORM ={"K": "K", "R": "R", "H": "H+", "D": "D", "E": "E", "C": "C-", "Y": "Y-"}
GB_DIPOLE_CHARGE = {"K": 1.0, "R": 1.0, "H": 1.0, "D": -1.0, "E": -1.0, "C": -1.0, "Y": -1.0}
GB_NEAR_CAP = 8   # acetyl / amide cap rows cover the positions this close to the capped end
GB_BOTH_CAPS = 16  # helices up to this length have rows with both caps
_GB_TABLE = None


def gb_dipole_table() -> dict:
    """
    params/sidechain_dipole_gb.tsv, loaded once: the kappa grid (1/A); helix rows by (form, ncap, ccap, L, k) as
    (E_in, W[kappa], n_rot), where n_rot is the number of sterically allowed chi1 wells averaged (0: none, the row is
    not used); coil rows by (form, ncap, ccap) as {k: (E_in, W[kappa])} for the 24-residue chains.
    """
    global _GB_TABLE
    if _GB_TABLE is None:
        df = pd.read_csv(files("pyagadir.params").joinpath("sidechain_dipole_gb.tsv"), sep="\t", comment="#")
        wcols = [c for c in df.columns if c.startswith("W_")]
        w = df[wcols].to_numpy(float)
        helix, coil = {}, {}
        for i, r in enumerate(df[["form", "state", "ncap", "ccap", "L", "k", "n_rot", "E_in"]].itertuples(index=False)):
            if r.state == "helix":
                helix[(r.form, r.ncap, r.ccap, int(r.L), int(r.k))] = (float(r.E_in), w[i], int(r.n_rot))
            else:
                coil.setdefault((r.form, r.ncap, r.ccap), {})[int(r.k)] = (float(r.E_in), w[i])
        for form in GB_DIPOLE_FORM.values():
            for L in range(4, 41):
                for k in range(L + 2):
                    if (form, "res", "res", L, k) not in helix:
                        raise ValueError(f"sidechain_dipole_gb.tsv: no row for {form} L={L} k={k}")
        _GB_TABLE = dict(kappa=np.array([float(c[2:]) for c in wcols]), helix=helix, coil=coil,
                         lmax=max(key[3] for key in helix))
    return _GB_TABLE


GBX_CAPPING_BOX_NCAPS = ("S", "T", "D", "N")
_GBX_TABLES = None


def gbx_position_class(aa: str, n_dist: int, c_dist: int, ncap_res: str) -> str:
    """
    Table 1 class of a charged residue for the gbx transfer, following get_dG_Int's column choice: the N-cap; the
    C-cap; C1-C3 (whose charged-residue cells equal Ncen); N1-N4 for Asp/Glu, per N-cap type (acetyl/succinyl "N1",
    residue "N1r"); Ncen otherwise (Lys/Arg/His N1-N4 cells equal Ncen). "N3cb": Glu at N3 behind an S/T/D/N N-cap,
    the capping box, whose bonus was measured in residue-capped peptides.
    """
    if aa == "E" and n_dist == 3 and c_dist > 3 and ncap_res in GBX_CAPPING_BOX_NCAPS:
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


def gbx_tables() -> dict:
    """
    The gbx law's data, loaded once: transfer offsets {(residue, class): dphi} (kcal/mol per +1 charge,
    params/sidechain_dipole_gb_transfer.tsv); from params/gbx_overrides.tsv, parameter-table overrides
    [(table, row, column, value)], first-turn N2 offsets ("first_turn_n2" rows) and end cells
    {(residue, position): kcal/mol} ("end_cell" rows).
    """
    global _GBX_TABLES
    if _GBX_TABLES is None:
        t = pd.read_csv(files("pyagadir.params").joinpath("sidechain_dipole_gb_transfer.tsv"), sep="\t", comment="#")
        o = pd.read_csv(files("pyagadir.params").joinpath("gbx_overrides.tsv"), sep="\t", comment="#")
        _GBX_TABLES = dict(
            transfer={(r.residue, r.cls): float(r.dphi) for r in t.itertuples()},
            overrides=[(r.table, r.row, r.column, float(r.value)) for r in o.itertuples()
                       if r.table.startswith("table_")],
            overrides_meta=[(r.table, r.row, r.column, float(r.value)) for r in o.itertuples()
                            if r.table == "first_turn_n2"],
            end_cells={(r.row, r.column): float(r.value) for r in o.itertuples() if r.table == "end_cell"},
        )
    return _GBX_TABLES


# gbp data files (params/README.md "gbp"). GBP_FILES may point a key at another file (analysis tools); call
# gbp_reset() after changing it.
GBP_FILES = dict(pairs="sidechain_pair_gb.tsv", neutral="sidechain_dipole_gb_neutral.tsv",
                 transfer="sidechain_gbp_transfer.tsv", overrides="gbp_overrides.tsv",
                 desolv="sidechain_dipole_gb_desolv.tsv")
GB_DESOLV_RANGE = 4  # positions from the nearer helix end with their own desolvation row; farther: the interior row
GB_PAIR_CHARGE = {"K": 1.0, "R": 1.0, "H": 1.0, "D": -1.0, "E": -1.0}
GB_PAIR_NEUTRAL = {"K": "K0", "H": "H0", "D": "D0", "E": "E0"}  # Arg: no neutral form (its neutral energies are 0)
GB_PAIR_KMAX = 12
GB_NEUTRAL_FORMS = {"K": ("K0",), "D": ("D0",), "E": ("E0",), "H": ("H0d", "H0e")}
# intrinsic weights of neutral His's tautomers, N-delta1-H (HID) and N-epsilon2-H (HIE): tau:pi ~ 4:1 in water (Reynolds,
# Peat, Freedman & Lyerla 1973, J. Am. Chem. Soc. 95, 328, 13C NMR of L-histidine in basic solution)
GB_HIS_TAUTOMER = {"H0d": 0.2, "H0e": 0.8}
_GBP_TABLES = None


def _gbp_path(key):
    p = GBP_FILES[key]
    return p if "/" in str(p) else files("pyagadir.params").joinpath(p)


def gbp_reset():
    global _GBP_TABLES
    _GBP_TABLES = None
    GBElectrostatics._law_params.pop("gbp", None)


def gbp_tables() -> dict:
    """
    The gbp law's data, loaded once:
    - pairs: {(X, Y, k, state, cls): {(form X, form Y): F[T, kappa]}} with the grid axes T_grid (K) and kappa_grid
      (1/A); the GB free energy of side chains X(i), Y(i + k) over their rotamers (params/sidechain_pair_gb.tsv);
    - neutral: the neutral forms' rows of the GB dipole law, in gb_dipole_table()'s layout (helix, coil);
    - desolv {(form, end, distance): kcal/mol}: the side chain's GB desolvation in the helix minus the coil (its own
      self-energy and the backbone's self-energy change it causes), by distance from the nearer helix end (end "N" or
      "C", 0 = the cap, up to GB_DESOLV_RANGE) or the interior (end "mid", distance -1), params/sidechain_dipole_gb_desolv.tsv;
    - transfer {(residue, class): dphi} (kcal/mol per +1 charge, added to the potential), const {(residue, class):
      kcal/mol} (added at every charge) and neutral offsets {residue: kcal/mol} (class "Neutral", added to the neutral
      form's energy), params/sidechain_gbp_transfer.tsv;
    - overrides, overrides_meta, end_cells as gbx_tables(), from params/gbp_overrides.tsv.
    """
    global _GBP_TABLES
    if _GBP_TABLES is None:
        df = pd.read_csv(_gbp_path("pairs"), sep="\t", comment="#")
        fcols = [c for c in df.columns if c.startswith("F_")]
        Ts = sorted({float(c.split("_")[1]) for c in fcols})
        ks = sorted({float(c.split("_")[2]) for c in fcols})
        grid = df[fcols].to_numpy(float).reshape(len(df), len(Ts), len(ks))
        pairs = {}
        for n, r in enumerate(df[["X", "Y", "k", "state", "cls", "fX", "fY"]].itertuples(index=False)):
            pairs.setdefault((r.X, r.Y, int(r.k), r.state, r.cls), {})[(r.fX, r.fY)] = grid[n]
        nd = pd.read_csv(_gbp_path("neutral"), sep="\t", comment="#")
        wcols = [c for c in nd.columns if c.startswith("W_")]
        w = nd[wcols].to_numpy(float)
        helix, coil = {}, {}
        for i, r in enumerate(nd[["form", "state", "ncap", "ccap", "L", "k", "n_rot", "E_in"]].itertuples(index=False)):
            if r.state == "helix":
                helix[(r.form, r.ncap, r.ccap, int(r.L), int(r.k))] = (float(r.E_in), w[i], int(r.n_rot))
            else:
                coil.setdefault((r.form, r.ncap, r.ccap), {})[int(r.k)] = (float(r.E_in), w[i])
        t = pd.read_csv(_gbp_path("transfer"), sep="\t", comment="#")
        o = pd.read_csv(_gbp_path("overrides"), sep="\t", comment="#")
        ds = pd.read_csv(_gbp_path("desolv"), sep="\t", comment="#")
        desolv = {(r.form, r.end, int(r.dist)): float(r.desolv) for r in ds.itertuples()}
        _GBP_TABLES = dict(
            pairs=pairs, T_grid=np.array(Ts), kappa_grid=np.array(ks),
            neutral=dict(kappa=np.array([float(c[2:]) for c in wcols]), helix=helix, coil=coil,
                         lmax=max(key[3] for key in helix)),
            transfer={(r.residue, r.cls): float(r.dphi) for r in t.itertuples() if r.cls != "Neutral"},
            const={(r.residue, r.cls): float(r.const) for r in t.itertuples() if r.cls != "Neutral"},
            neutral_offset={r.residue: float(r.dphi) for r in t.itertuples() if r.cls == "Neutral"},
            desolv=desolv,
            overrides=[(r.table, r.row, r.column, float(r.value)) for r in o.itertuples()
                       if r.table.startswith("table_")],
            overrides_meta=[(r.table, r.row, r.column, float(r.value)) for r in o.itertuples()
                            if r.table == "first_turn_n2"],
            end_cells={(r.row, r.column): float(r.value) for r in o.itertuples() if r.table == "end_cell"},
        )
    return _GBP_TABLES


def law_tables(dipole_law: str) -> dict:
    """The transfer/override tables of a law with a Table 1 transfer (GBX_LAWS)."""
    return gbp_tables() if dipole_law == "gbp" else gbx_tables()


def gb_pair_grid_value(grid: np.ndarray, T: float, kappa: float) -> float:
    """Bilinear interpolation of a pair free-energy grid F[T, kappa] (held at the edges beyond the grid)."""
    tab = gbp_tables()
    Tg, kg = tab["T_grid"], tab["kappa_grid"]
    T = min(max(T, Tg[0]), Tg[-1])
    kappa = min(max(kappa, kg[0]), kg[-1])
    a = int(np.clip(np.searchsorted(Tg, T) - 1, 0, len(Tg) - 2))
    b = int(np.clip(np.searchsorted(kg, kappa) - 1, 0, len(kg) - 2))
    u = (T - Tg[a]) / (Tg[a + 1] - Tg[a])
    v = (kappa - kg[b]) / (kg[b + 1] - kg[b])
    return float((1 - u) * ((1 - v) * grid[a, b] + v * grid[a, b + 1]) + u * ((1 - v) * grid[a + 1, b] + v * grid[a + 1, b + 1]))


def sb_network_free_energy(cells: dict, RT: float) -> float:
    """
    Free energy of a network of acid-base side-chain hydrogen bonds (the Table IV acid-base cells), counted
    statistically. cells {(i, j): cell (kcal/mol, < 0)}, j - i in (3, 4). Each cell is a two-state free energy,
    cell = -RT ln(1 + K), so K = exp(-cell/RT) - 1. A side chain bridges toward the C-terminus (as i) or toward the
    N-terminus (as j), not both; partners on one side may share it. Z = sum over the allowed sets of bridged pairs of
    prod K, by a transfer matrix over residues whose state records which of the last four residues are bridged to an
    earlier partner. Returns -RT ln Z, which is the cell itself for an isolated pair.
    """
    if not cells:
        return 0.0
    K = {ij: math.exp(-c / RT) - 1.0 for ij, c in cells.items()}
    lo = min(i for i, _ in K)
    hi = max(j for _, j in K)
    states = {(0, 0, 0, 0): 1.0}
    for j in range(lo, hi + 1):
        new = {}
        for (f4, f3, f2, f1), w in states.items():
            opts = [(0, 1.0)]
            k4, k3 = K.get((j - 4, j)), K.get((j - 3, j))
            if k4 is not None and not f4:
                opts.append((1, k4))
            if k3 is not None and not f3:
                opts.append((1, k3))
            if k4 is not None and k3 is not None and not f4 and not f3:
                opts.append((1, k4 * k3))
            for fj, wt in opts:
                key = (f3, f2, f1, fj)
                new[key] = new.get(key, 0.0) + w * wt
        states = new
    return -RT * math.log(sum(states.values()))


class GBElectrostatics:
    """
    The Generalized Born laws: the GB side chain-macrodipole potential, the neutral forms' energies and the pair
    coupling (gbp), and the parts of the other energy terms that change under gbx/gbp (the parameter overrides, the
    Table 1 end cells and first-turn offsets, the acid-base network). Every method here is used only when
    dipole_law is a GB law.
    """

    _law_params = {}  # law -> load_params() with that law's overrides (law_params); gbp_reset() clears "gbp"

    @classmethod
    def law_params(cls, dipole_law: str = "munoz") -> dict:
        """load_params() with the gbx or gbp law's parameter overrides applied to copies of the tables they touch
        (params/gbx_overrides.tsv, params/gbp_overrides.tsv); the other laws use load_params() itself."""
        base = cls.load_params()
        if dipole_law not in GBX_LAWS:
            return base
        if dipole_law not in cls._law_params:
            out = dict(base)
            for table, row, col, value in law_tables(dipole_law)["overrides"]:
                if out[table] is base[table]:
                    out[table] = base[table].copy()
                out[table].loc[row, col] = value
            cls._law_params[dipole_law] = out
        return cls._law_params[dipole_law]

    def _gb_dipole_energy(self, idx: int):
        """
        Helix-minus-coil Generalized Born energy (kcal/mol) of residue idx's charged form with the helix backbone,
        E_helix(L, k) - E_coil(k, L + 1 - k) of _gb_dipole_potential, or None where the table does not apply: a residue
        that is not tabulated, one outside the caps, or a row with no sterically allowed structure (n_rot = 0).
        """
        form = GB_DIPOLE_FORM.get(self.seq_list[idx])
        if form is None or not int(self.ncap_idx) <= idx <= int(self.ccap_idx):
            return None
        e = self._gb_form_energy(gb_dipole_table(), form, idx)
        if e is None:
            return None
        return e[0] - e[1]

    def _gb_field_potential(self, idx: int) -> tuple[float, float]:
        """
        The GB field alone: _gb_dipole_potential without a law's transfer offset or gbp's neutral form, as
        (N-end, C-end) like _sidechain_dipole_potential, and Muñoz's law where the table does not apply. It is the
        potential the Table 1 transfer is measured against (gb_pipeline/transfer/build_gb_transfer.py: dphi =
        phi_default - phi_field); it was the potential of the removed law dipole_law="gb".
        """
        energy = self._gb_dipole_energy(idx)
        if energy is None:
            return self._munoz_dipole_potential_scaled(idx)
        k0, L0 = idx - int(self.ncap_idx), int(self.ccap_idx) - int(self.ncap_idx) - 1
        phi = energy / GB_DIPOLE_CHARGE[self.seq_list[idx]]
        return (phi, 0.0) if k0 <= L0 + 1 - k0 else (0.0, phi)

    def _gb_dipole_potential(self, idx: int) -> tuple[float, float]:
        """
        The GB laws: energy (kcal/mol) of a +1 charge on residue idx from the helix backbone, by Generalized Born
        on built poly-Ala helices (params/sidechain_dipole_gb.tsv, gb_pipeline/helix_builds/build_gb_dipole_table.py), helix minus coil.
        For Lys, Arg, His+, Asp, Glu, Cys- or Tyr- at position k = 0 (N-cap) .. L + 1 (C-cap) of a helix with L
        helical residues:

            E(kappa, T) = E_in + W(kappa) / eps_w(T)                  (the residue's charged form)
            potential   = (E_helix(L, k) - E_coil(k, L + 1 - k)) / z  (z = +1 or -1)

        Caps: an acetyl or succinyl N-cap token uses the acetyl rows, an amide C-cap token the amide rows, any residue
        cap the residue rows. The coil reference spans the same residues with the same caps: the mid-chain coil value
        plus the change near each end (up to eight positions), from the 24-residue coil chains.

        W is interpolated linearly on the table's kappa grid (held at its last value above kappa = 0.8 / A, about
        5.6 M at 0 C). The temperature dependence is eps_w(T), so the eps(T) factor is 1 for this law. The value is
        returned on the nearer end (N on ties); only the sum is used.

        Helices longer than the table take the end-anchored value of the longest tabulated one: a position within
        lmax // 2 of an end takes that position's value, positions farther from both ends its central value.

        Kept on Muñoz's law times its eps(T) factor: charged residues outside the caps (flanks), and table rows with
        no sterically allowed structure (n_rot = 0).

        The law covers the side chain's electrostatics with the backbone in both states, including the local terms
        Table 1's propensities already contain. Its N-cap..C-cap values must be used with table values transferred to
        it (new value = old value + z[potential_munoz - potential_gb] at each value's reference conditions), or the
        first-turn and capping energies are counted twice. dipole_law="gbx" does exactly that: it adds the offset
        dphi(residue, gbx_position_class) of params/sidechain_dipole_gb_transfer.tsv wherever the table applies.

        dipole_law="gbp" couples both protonation forms exactly (reasoning/nodes/N166.md): E = theta E_charged +
        (1 - theta) E_neutral, so the potential is (E_charged - E_neutral) / z (the pKa shift) and _gb_neutral_energy(idx)
        holds E_neutral (the energy term adds it at every charge) for Lys, His, Asp and Glu, with
        params/sidechain_gbp_transfer.tsv's offsets; under gbp every tabulated form's helix energy also carries its
        desolvation (params/sidechain_dipole_gb_desolv.tsv).
        """
        aa = self.seq_list[idx]
        ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)
        energy = self._gb_dipole_energy(idx)
        if energy is None:  # not tabulated, outside the caps, or no allowed structure: Muñoz's law
            return self._munoz_dipole_potential_scaled(idx)
        k0, L0 = idx - ncap_i, ccap_i - ncap_i - 1
        if self.dipole_law == "gbp" and aa in GB_NEUTRAL_FORMS:
            energy -= self._gb_neutral_energy(idx) + gbp_tables()["neutral_offset"].get(aa, 0.0)
        phi = energy / GB_DIPOLE_CHARGE[aa]
        if self.dipole_law in GBX_LAWS:
            cls = gbx_position_class(aa, k0, L0 + 1 - k0, self.seq_list[ncap_i])
            phi += law_tables(self.dipole_law)["transfer"].get((aa, cls), 0.0)
        return (phi, 0.0) if k0 <= L0 + 1 - k0 else (0.0, phi)

    def _gb_form_energy(self, tab: dict, form: str, idx: int):
        """(E_helix, E_coil) of one form on residue idx (N-cap..C-cap) from a GB dipole table (gb_dipole_table() or
        gbp_tables()["neutral"]), or None for a row with no sterically allowed structure (n_rot = 0)."""
        ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)
        nt = "Ac" if self.seq_list[ncap_i] in ("Ac", "Sc") else "res"
        ct = "Am" if self.seq_list[ccap_i] == "Am" else "res"
        L0, k0 = ccap_i - ncap_i - 1, idx - ncap_i
        L, k = L0, k0
        if L > tab["lmax"]:
            half, c = tab["lmax"] // 2, L + 1 - k
            k = min(k, half) if k <= c else tab["lmax"] + 1 - min(c, half)
            L = tab["lmax"]
        c = L + 1 - k
        if nt == "Ac" and ct == "Am" and L <= GB_BOTH_CAPS:
            caps = ("Ac", "Am")
        elif nt == "Ac" and k <= GB_NEAR_CAP:
            caps = ("Ac", "res")
        elif ct == "Am" and c <= GB_NEAR_CAP:
            caps = ("res", "Am")
        else:
            caps = ("res", "res")
        e_in, w, n_rot = tab["helix"][(form, *caps, L, k)]
        if n_rot == 0:
            return None
        kap = self.kappa * 1e-10  # 1/A
        e_h = e_in + np.interp(kap, tab["kappa"], w) / self.epsilon_r
        if self.dipole_law == "gbp":  # the side chain's desolvation (helix minus coil), by distance from the nearer end
            c0 = L0 + 1 - k0
            if min(k0, c0) <= GB_DESOLV_RANGE:
                key = (form, "N", k0) if k0 <= c0 else (form, "C", c0)
            else:
                key = (form, "mid", -1)
            e_h += gbp_tables()["desolv"].get(key, 0.0)
        return e_h, self._gb_coil(form, nt, ct, k0, L0 + 1 - k0, kap, tab)

    def _gb_neutral_energy(self, idx: int) -> float:
        """gbp: the neutral form's GB energy with the backbone, helix minus coil (kcal/mol), of Lys, His, Asp or Glu on
        residue idx (N-cap..C-cap; 0 elsewhere or for a row with no allowed structure). Neutral His is its tautomers'
        mixture, -RT ln sum_t w_t exp(-E_t / RT) in each state (GB_HIS_TAUTOMER)."""
        aa = self.seq_list[idx]
        if aa not in GB_NEUTRAL_FORMS or not int(self.ncap_idx) <= idx <= int(self.ccap_idx):
            return 0.0
        tab = gbp_tables()["neutral"]
        es = [self._gb_form_energy(tab, f, idx) for f in GB_NEUTRAL_FORMS[aa]]
        if any(e is None for e in es):
            return 0.0
        if len(es) == 1:
            return es[0][0] - es[0][1]
        RT = R_KCAL * self.T_kelvin
        wts = [GB_HIS_TAUTOMER[f] for f in GB_NEUTRAL_FORMS[aa]]
        mix = [-RT * math.log(sum(wt * math.exp(-e[s] / RT) for wt, e in zip(wts, es))) for s in (0, 1)]
        return mix[0] - mix[1]

    def _gb_coil(self, form: str, nt: str, ct: str, n_n: int, n_c: int, kap: float, tab: dict = None) -> float:
        """Coil reference of the GB law: the residue's energy in the coil over a range with n_n residues before it and
        n_c after it, with the given caps; the mid-chain value plus each end's change, if within GB_NEAR_CAP."""
        tab = tab or gb_dipole_table()

        def e(caps, k):
            e_in, w = tab["coil"][(form, *caps)][k]
            return e_in + np.interp(kap, tab["kappa"], w) / self.epsilon_r
        res, cap = ("res", "res"), ("Ac", "Am")
        top = max(tab["coil"][(form, *res)])  # the chain's C-cap position (25)
        mid = float(np.mean([e(res, k) for k in range(GB_NEAR_CAP + 1, top - GB_NEAR_CAP)]))
        out = mid
        if n_n <= GB_NEAR_CAP:
            out += e(cap if nt == "Ac" else res, n_n) - mid
        if n_c <= GB_NEAR_CAP:
            out += e(cap if ct == "Am" else res, top - n_c) - mid
        return out

    def _assign_gb_neutral_energies(self):
        """
        gbp: the energy every ionisable residue (N-cap..C-cap) carries at any charge: its neutral form's GB energy
        (helix minus coil) with the neutral transfer offset (Lys, His, Asp, Glu), plus its class's constant transfer
        offset (all tabulated residues). gb_neutral_energy is zero for the other laws.
        """
        n = len(self.seq_list)
        self.gb_neutral_energy = np.zeros(n, dtype=float)
        if self.dipole_law == "gbp":
            tab = gbp_tables()
            ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)
            for idx, aa in enumerate(self.seq_list):
                if aa not in GB_DIPOLE_FORM or not ncap_i <= idx <= ccap_i:
                    continue
                if aa in GB_NEUTRAL_FORMS:
                    self.gb_neutral_energy[idx] = self._gb_neutral_energy(idx) + tab["neutral_offset"].get(aa, 0.0)
                cls = gbx_position_class(aa, idx - ncap_i, ccap_i - idx, self.seq_list[ncap_i])
                self.gb_neutral_energy[idx] += tab["const"].get((aa, cls), 0.0)

    def _gb_pair_coeffs(self, forms: dict, a1: str, a2: str) -> tuple:
        """(U, a1, a2, E_nn) of a gbp pair row at this instance's T and kappa: with q = z theta the pair's energy is
        E = E_nn + q1 a1 + q2 a2 + q1 q2 U, from the four protonation-state free energies
        E(theta1, theta2) = theta1 theta2 E_cc + theta1 (1 - theta2) E_cn + (1 - theta1) theta2 E_nc + (1 - theta1)(1 - theta2) E_nn.
        Arg has no neutral form: its neutral states' energies are 0."""
        T, kap = self.T_kelvin, self.kappa * 1e-10
        cx, cy = GB_DIPOLE_FORM[a1], GB_DIPOLE_FORM[a2]
        nx, ny = GB_PAIR_NEUTRAL.get(a1), GB_PAIR_NEUTRAL.get(a2)

        def F(fx, fy):
            if fx is None or fy is None:
                return 0.0
            return gb_pair_grid_value(forms[(fx, fy)], T, kap)
        e_cc, e_cn, e_nc, e_nn = F(cx, cy), F(cx, ny), F(nx, cy), F(nx, ny)
        z1, z2 = GB_PAIR_CHARGE[a1], GB_PAIR_CHARGE[a2]
        return ((e_cc - e_cn - e_nc + e_nn) / (z1 * z2), (e_cn - e_nn) / z1, (e_nc - e_nn) / z2, e_nn)

    def _assign_gb_pairs(self):
        """
        dipole_law="gbp": the GB pair law for Lys, Arg, His, Asp and Glu pairs 1-12 apart (gbp_tables()["pairs"]),
        stored as gb_pairs[state][(i, j)] = (U, a_i, a_j, E_nn) (_gb_pair_coeffs), state True = helix, False = coil.
          both residues in the segment (N-cap..C-cap): helix row Ncap if i is the N-cap, Ccap if j is the C-cap,
                                                       else int; coil state: the coil row;
          both outside on one side (not straddling):  the coil row in both states (no change);
          otherwise (one in, one out; straddling):     the point-charge law in both states, as the default model.
        Pairs with Cys or Tyr and separations beyond 12 keep the point law.
        """
        self.gb_pairs = {True: {}, False: {}}
        if self.dipole_law != "gbp":
            return
        tab = gbp_tables()["pairs"]
        hel = set(int(i) for i in self.helix_indices)
        nc, cc = int(self.ncap_idx), int(self.ccap_idx)
        for a1, a2, i1, i2 in self.charged_pairs:
            k = i2 - i1
            if a1 not in GB_PAIR_CHARGE or a2 not in GB_PAIR_CHARGE or not 1 <= k <= GB_PAIR_KMAX:
                continue
            both_in = i1 in hel and i2 in hel
            both_out = i1 not in hel and i2 not in hel and not (i1 < nc and i2 > cc)
            if not (both_in or both_out):
                continue
            coil = self._gb_pair_coeffs(tab[(a1, a2, k, "coil", "coil")], a1, a2)
            self.gb_pairs[False][(i1, i2)] = coil
            if both_in:
                cls = "Ncap" if i1 == nc else ("Ccap" if i2 == cc else "int")
                self.gb_pairs[True][(i1, i2)] = self._gb_pair_coeffs(tab[(a1, a2, k, "helix", cls)], a1, a2)
            else:
                self.gb_pairs[True][(i1, i2)] = coil

    def _gb_pair_energy(self, idx1: int, idx2: int):
        """
        gbp: helix-minus-coil GB energy (kcal/mol) of the side chains idx1 < idx2, E = E_nn + q1 a1 + q2 a2 + q1 q2 U
        in each state at its converged charges (_assign_gb_pairs), or None for a pair the GB pair law does not cover.
        get_dG_sidechain_sidechain_electrost gives half of it to each residue.
        """
        c_h = self.gb_pairs[True].get((idx1, idx2))
        if c_h is None:
            return None
        c_r = self.gb_pairs[False][(idx1, idx2)]
        q1h, q2h = self.modified_seq_ionization_hel[idx1], self.modified_seq_ionization_hel[idx2]
        q1r, q2r = self.modified_seq_ionization_rc[idx1], self.modified_seq_ionization_rc[idx2]
        G_hel = c_h[3] + q1h * c_h[1] + q2h * c_h[2] + q1h * q2h * c_h[0]
        G_rc = c_r[3] + q1r * c_r[1] + q2r * c_r[2] + q1r * q2r * c_r[0]
        return G_hel - G_rc

    def _gbx_end_cell(self, aa: str, n_dist: int, c_dist: int) -> float:
        """gbx/gbp: the re-derived first- or last-turn Table 1 cell offset (kcal/mol) of residue aa n_dist from the
        N-cap and c_dist from the C-cap ("end_cell" rows of params/gbx_overrides.tsv, params/gbp_overrides.tsv)."""
        pos = f"C{c_dist}" if c_dist <= 3 else (f"N{n_dist}" if n_dist <= 3 else None)
        return law_tables(self.dipole_law)["end_cells"].get((aa, pos), 0.0)

    def _gbx_first_turn_n2_offset(self, aa: str):
        """gbx/gbp: the re-derived first-turn N2 offset of residue aa ("first_turn_n2" rows of params/gbx_overrides.tsv,
        params/gbp_overrides.tsv), or None where the law keeps energies.FIRST_TURN_N2_OFFSET."""
        for table, row, _col, value in law_tables(self.dipole_law)["overrides_meta"]:
            if table == "first_turn_n2" and row == aa:
                return value
        return None

    def _sb_candidates(self) -> dict:
        """gbp: the acid-base pairs i,i+3 and i,i+4 inside the helix (caps excluded) with a favourable Table IV cell,
        at this temperature (the hydrogen-bond law, EnergyCalculator._hbond_temperature): {(i, j): kcal/mol}."""
        out = {}
        interior = set(self.helix_indices[1:-1])
        for k, table in ((3, self.table_4a_lacroix), (4, self.table_4b_lacroix)):
            for i in interior:
                j = i + k
                if j not in interior:
                    continue
                a, b = self.seq_list[i], self.seq_list[j]
                if {a, b} & {"D", "E"} and {a, b} & {"K", "R", "H"}:
                    v = float(table.loc[a, b]) / 100.0
                    if v < 0:
                        out[(i, j)] = self._hbond_temperature(v)
        return out

    def _sb_network_factor(self) -> float:
        """dipole_law="gbp": the acid-base cells counted statistically (sb_network_free_energy) instead of by the
        one-partner rule. Every candidate pair keeps its cell, scaled by f = G_network / sum(cells), so the acid-base
        cells sum to the network's free energy (an isolated pair keeps its cell)."""
        if getattr(self, "_sb_f", None) is None:
            c = self._sb_candidates()
            tot = sum(c.values())
            self._sb_f = sb_network_free_energy(c, R_KCAL * self.T_kelvin) / tot if tot < 0 else 1.0
        return self._sb_f
