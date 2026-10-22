"""
Generalized Born electrostatics: dipole_law "gbx". Opt-in and in development; the default law is the point-charge
law (point.py).

The law rests on the Generalized Born field of the helix backbone on the charged side chain, tabulated from built
poly-Ala helices (params/sidechain_dipole_gb.tsv, made by gb_pipeline/helix_builds/build_gb_dipole_table.py;
params/README.md), in place of the point-charge side chain-macrodipole law. The field alone was once the law "gb";
it was removed (owner, 2026-10-22) because used alone it counts twice the first-turn and capping electrostatics
that Table 1's measured values already hold (_gb_dipole_potential). _gb_field_potential still gives it, for the
transfer that removes that double count (gb_pipeline/transfer/build_gb_transfer.py).
"gbx": the GB field with every Table 1 value that was measured with the default law in the loop transferred to it (a
potential offset per residue and Table 1 class at that value's reference measurement, params/
sidechain_dipole_gb_transfer.tsv, made by gb_pipeline/transfer/build_gb_transfer.py) and the parameters re-derived
under it (params/gbx_overrides.tsv). See params/README.md "gbx".
"gbp" (gbx plus Generalized Born side chain-side chain pairs, two-state coupling through the neutral forms, GB
desolvation and a statistical acid-base network) was retired on 2026-10-22, when the owner kept one GB law; its
record is in params/README.md "Retired: gbp", and its code at git 5a6e3f8. How the GB tables are made:
gb_pipeline/README.md.

gbx keeps the point-charge law for the terminal groups, for charged residues outside the caps, for table rows with no
sterically allowed structure, and for every side chain-side chain pair.

GBElectrostatics is a mixin of energies.PrecomputeParams and reads the segment set up there.
"""
import math
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import R_KCAL

GB_LAWS = ("gbx",)
GBX_LAWS = ("gbx",)  # laws with a Table 1 transfer and parameter overrides (the GB law)
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


def law_tables(dipole_law: str) -> dict:
    """The transfer and override tables of the GB law (gbx_tables)."""
    return gbx_tables()


class GBElectrostatics:
    """
    The Generalized Born law (gbx): the GB side chain-macrodipole potential and the parts of the other energy terms
    that change under it (the parameter overrides, the Table 1 end cells and first-turn offsets). Every method here
    is used only when dipole_law is the GB law; _gb_field_potential also serves the transfer builder.
    """

    _law_params = {}  # law -> load_params() with that law's overrides (law_params)

    @classmethod
    def law_params(cls, dipole_law: str = "munoz") -> dict:
        """load_params() with the gbx law's parameter overrides applied to copies of the tables they touch
        (params/gbx_overrides.tsv); the other laws use load_params() itself."""
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
        The GB field alone: _gb_dipole_potential without the law's transfer offset, as
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
        """
        aa = self.seq_list[idx]
        ncap_i, ccap_i = int(self.ncap_idx), int(self.ccap_idx)
        energy = self._gb_dipole_energy(idx)
        if energy is None:  # not tabulated, outside the caps, or no allowed structure: Muñoz's law
            return self._munoz_dipole_potential_scaled(idx)
        k0, L0 = idx - ncap_i, ccap_i - ncap_i - 1
        phi = energy / GB_DIPOLE_CHARGE[aa]
        if self.dipole_law in GBX_LAWS:
            cls = gbx_position_class(aa, k0, L0 + 1 - k0, self.seq_list[ncap_i])
            phi += law_tables(self.dipole_law)["transfer"].get((aa, cls), 0.0)
        return (phi, 0.0) if k0 <= L0 + 1 - k0 else (0.0, phi)

    def _gb_form_energy(self, tab: dict, form: str, idx: int):
        """(E_helix, E_coil) of one form on residue idx (N-cap..C-cap) from the GB dipole table (gb_dipole_table()),
        or None for a row with no sterically allowed structure (n_rot = 0)."""
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
        return e_h, self._gb_coil(form, nt, ct, k0, L0 + 1 - k0, kap, tab)


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





    def _gbx_end_cell(self, aa: str, n_dist: int, c_dist: int) -> float:
        """gbx: the re-derived first- or last-turn Table 1 cell offset (kcal/mol) of residue aa n_dist from the
        N-cap and c_dist from the C-cap ("end_cell" rows of params/gbx_overrides.tsv)."""
        pos = f"C{c_dist}" if c_dist <= 3 else (f"N{n_dist}" if n_dist <= 3 else None)
        return law_tables(self.dipole_law)["end_cells"].get((aa, pos), 0.0)

    def _gbx_first_turn_n2_offset(self, aa: str):
        """gbx: the re-derived first-turn N2 offset of residue aa ("first_turn_n2" rows of params/gbx_overrides.tsv),
        or None where the law keeps energies.FIRST_TURN_N2_OFFSET."""
        for table, row, _col, value in law_tables(self.dipole_law)["overrides_meta"]:
            if table == "first_turn_n2" and row == aa:
                return value
        return None


