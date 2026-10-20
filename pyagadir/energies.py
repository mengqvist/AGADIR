import math
from importlib.resources import files

import numpy as np
import pandas as pd

from pyagadir.chemistry import adjust_pKa, acidic_residue_ionization, basic_residue_ionization, calculate_permittivity, debye_screening_kappa, ionization_free_energy
from pyagadir.utils import is_valid_peptide_sequence, is_valid_ncap_ccap, is_valid_conditions
import itertools




class ParamTable:
    """
    Read-only, dict-backed copy of a parameter DataFrame.

    Scalar DataFrame.loc lookups cost ~10 µs each, and a prediction makes tens of
    thousands of them (one EnergyCalculator per helical segment), which made them
    ~70% of the run time. This keeps the same spelling -- table.loc[row, col],
    table.loc[row][col], `row in table.index`, table.columns[k] -- at dict speed.

    It is a snapshot: edits to the source DataFrame are only seen after rebuilding.
    """

    def __init__(self, df: pd.DataFrame):
        self._rows = df.to_dict(orient="index")
        self.index = tuple(df.index)
        self.columns = tuple(df.columns)

    @property
    def loc(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self._rows[row][col]
        return self._rows[key]


# Helix-state distance (A) between a succinyl N-cap's carboxylate and a charged side chain at N1, N2 or N3, used in
# place of Lacroix 1998 Table VI 'N-cap f' (which places a free amine on the backbone: 9.2 A to N2 in the helix vs 9.6 A
# in the coil). Each value is the distance at which this model's own Coulomb law reproduces the helix-minus-coil pair
# energy from continuum electrostatics (APBS, carboxylate on the N-cap residue as the succinyl proxy, helix minus PPII,
# rotamer-averaged, interior dielectric 16, the value at which the same calculation matches the N1/N2 His titrations of
# Cochran et al. 2001, Protein Sci. 10, 463) at 0.1 M and 3 C: acid N2 +0.50, acid N3 +0.35, acid N1 +0.04, base N2
# -0.34 kcal/mol. Pairs not listed keep Table VI. See params/README.md.
SUCCINYL_HELIX_DISTANCE = {("acid", 1): 9.33, ("acid", 2): 3.95, ("acid", 3): 5.27, ("base", 2): 4.80}

# First-turn offset (kcal/mol at 273 K) added to the Table 1 N2 cell of the residues whose first-turn cells were never
# measured in peptides (Lacroix 1998 took the N1-N4 cells from Petukhov's poly-Ala measurements; K, R and H were left
# at their interior value, C, F, Y and W at round numbers). Lys and Arg take the measured peptide value for Leu at N2
# (Petukhov et al. 1998, J. Mol. Biol. 278, 279: +0.25 above the interior), which the peptides with Lys at N2 support
# when each is compared with the other peptides of its own host. H, C, F, Y and W take the N2-minus-interior offsets of
# the single-mutant scan of Tsuboyama et al. 2023 (Nature 620, 434; exposed helix sites, relative to Ala), scaled by
# 0.943 so that Leu's protein offset equals its peptide value. Charge-independent. See params/README.md.
FIRST_TURN_N2_OFFSET = {"K": 0.25, "R": 0.25, "H": -0.059, "C": 0.228, "F": 0.162, "Y": 0.157, "W": 0.141}

# Hydrogen-bond terms share one temperature law (EnergyCalculator._hbond_temperature): Munoz & Serrano 1995-III
# eq. (8) for the backbone bond, dH_ref = -0.898 kcal/mol per bond at 273 K (Lacroix 1998). Side chains and end groups
# whose capping contribution is a hydrogen bond (Munoz & Serrano 1995-III p. 305), and residues that can form
# side-chain hydrogen bonds in the Table IV/V pair terms:
HBOND_DH_REF = -0.898
HBOND_NCAPS = ("S", "T", "N", "D", "Ac", "Sc")
HBOND_CCAPS = ("K", "R", "H", "N", "D")
HBOND_POLAR = ("S", "T", "N", "Q", "D", "E", "K", "R", "H")

# Far end of the helix macrodipole: a charged residue inside the helix feels the nearest end by Munoz & Serrano 1995-II
# eq. 11 and the other end as a half charge (Hol, van Duijnen & Berendsen 1978), screened Coulomb in water. Poisson-
# Boltzmann calculations on built helices (interior dielectric 4; Asp/Glu at N1-N3 and Lys/Arg at C1-C3, helix lengths
# 6-28) give 0.82 times that half-charge law (least squares through the origin over 72 cases), so the far end is scaled
# by this factor. The flank half charges are not. See params/README.md.
FEH_FAR_APBS = 0.82

# Side chain-macrodipole laws. "munoz" (default): the law above. "gb": the Generalized Born field of the helix backbone
# on the charged side chain, tabulated from built poly-Ala helices (params/sidechain_dipole_gb.tsv, made by
# tools/build_gb_dipole_table.py; params/README.md). Tabulated residues, their table form and the sign of its charge:
# "gbx": the GB law with every Table 1 value that was measured with the default law in the loop transferred to it (a
# potential offset per residue and Table 1 class at that value's reference measurement, params/
# sidechain_dipole_gb_transfer.tsv, made by tools/build_gb_transfer.py) and the parameters re-derived under it
# (params/gbx_overrides.tsv). See params/README.md "gbx".
# "gbp": gbx plus the Generalized Born side chain-side chain pair law (params/sidechain_pair_gb.tsv, made by
# tools/build_gb_pair_table.py) in place of the point-charge pair term, the exact two-state coupling of every
# ionisable group to both laws (each protonation form's own GB energy, so a group's pKa shifts by the charged-minus-
# neutral energy), the statistical salt-bridge network for the Table IV acid-base cells, and the Table 1 transfer of
# all of it (params/sidechain_gbp_transfer.tsv, tools/build_gbp_transfer.py). See params/README.md "gbp".
DIPOLE_LAWS = ("munoz", "gb", "gbx", "gbp")
GB_LAWS = ("gb", "gbx", "gbp")
GBX_LAWS = ("gbx", "gbp")  # laws with a Table 1 transfer and parameter overrides
GB_DIPOLE_FORM = {"K": "K", "R": "R", "H": "H+", "D": "D", "E": "E", "C": "C-", "Y": "Y-"}
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
    PrecomputeParams._law_params.pop("gbp", None)


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


class PrecomputeParams:
    """
    Class to load parameters for the AGADIR model and
    for pre-computing distances, and ionization states.
    """

    _params = None # class variable to store the params

    @classmethod
    def load_params(cls):
        """
        Load the parameters for the AGADIR model.
        Only load once, and store in class variable to 
        save time and memory when many instances are created.
        """
        if cls._params is None:

            cls._params = {}

            # get params
            datapath = files("pyagadir.params")

            # load energy contributions for intrinsic propensities, capping, etc.
            cls._params["table_1_lacroix"] = pd.read_csv(
                datapath.joinpath("table_1_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load the hydrophobic staple motif energy contributions
            cls._params["table_2_lacroix"] = pd.read_csv(
                datapath.joinpath("table_2_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load the schellman motif energy contributions
            cls._params["table_3_lacroix"] = pd.read_csv(
                datapath.joinpath("table_3_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load energy contributions for interactions between i and i+3
            cls._params["table_4a_lacroix"] = pd.read_csv(
                datapath.joinpath("table_4a_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load energy contributions for interactions between i and i+4
            cls._params["table_4b_lacroix"] = pd.read_csv(
                datapath.joinpath("table_4b_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load sidechain distances for helices
            cls._params["table_6_helix_lacroix"] = pd.read_csv(
                datapath.joinpath("table_6_helix_lacroix.tsv"),
                index_col="Pos",
                sep="\t",
            ).astype(float)

            # load sidechain distances for coils
            cls._params["table_6_coil_lacroix"] = pd.read_csv(
                datapath.joinpath("table_6_coil_lacroix.tsv"),
                index_col="Pos",
                sep="\t",
            ).astype(float)

            # load N-terminal distances between charged amino acids and the half charge from the helix macrodipole
            cls._params["table_7_ccap_lacroix"] = pd.read_csv(
                datapath.joinpath("table_7_Ccap_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load C-terminal distances between charged amino acids and the half charge from the helix macrodipole
            cls._params["table_7_ncap_lacroix"] = pd.read_csv(
                datapath.joinpath("table_7_Ncap_lacroix.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load empirical sidechain-macrodipole energies from Muñoz & Serrano 1995 II Table 3
            cls._params["table_3_munoz_nterm"] = pd.read_csv(
                datapath.joinpath("table_3_munoz_1995.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            cls._params["table_3_munoz_cterm"] = pd.read_csv(
                datapath.joinpath("table_3_munoz_1995_cterm.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # Coulomb distances for sidechain-macrodipole (differ from Table7 for K/R)
            cls._params["table_7_coulomb_ncap"] = pd.read_csv(
                datapath.joinpath("table_7_coulomb_Ncap.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            cls._params["table_7_coulomb_ccap"] = pd.read_csv(
                datapath.joinpath("table_7_coulomb_Ccap.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

            # load pKa values for for side chain ionization and the N- and C-terminal capping groups
            cls._params["pka_values"] = pd.read_csv(
                datapath.joinpath("pka_values.tsv"),
                index_col="AA",
                sep="\t",
            ).astype(float)

        return cls._params

    @classmethod
    def snapshot_params(cls, dipole_law: str = "munoz") -> dict:
        """
        Fast ParamTable copies of load_params(), for passing as params= to many
        instances. Rebuilt on every call (~10 ms), so take one per prediction:
        edits made to the _params DataFrames between predictions are then honoured.
        """
        return {name: ParamTable(df) for name, df in cls.law_params(dipole_law).items()}

    _law_params = {}

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

    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, debug: bool = False, params: dict = None, dipole_law: str = "munoz"):
        """
        Initialize the PrecomputedParams for a peptide sequence.

        Args:
            seq (str): Peptide sequence.
            i (int): Helix start index, python 0-indexed.
            j (int): Helix length.
            pH (float): Solution pH.
            T (float): Temperature in Celsius.
            ionic_strength (float): Ionic strength of the solution in mol/L.
            ncap (str): N-terminal capping modification (acetylation='Ac', succinylation='Sc').
            ccap (str): C-terminal capping modification (amidation='Am').
            params (dict): Parameter tables from snapshot_params(). If None, load_params() is used.
            dipole_law (str): side chain-macrodipole law, "munoz" (default), "gb", "gbx" or "gbp" (see
                _sidechain_dipole_potential; "gbp" also replaces the side chain-side chain pair law).
        """
        self.debug = debug
        if dipole_law not in DIPOLE_LAWS:
            raise ValueError(f"dipole_law must be one of {DIPOLE_LAWS}, got {dipole_law!r}")
        self.dipole_law = dipole_law

        # load params
        if params is None:
            params = self.law_params(dipole_law) if dipole_law in GBX_LAWS else self.load_params()
        self.table_1_lacroix = params["table_1_lacroix"]
        self.table_2_lacroix = params["table_2_lacroix"]
        self.table_3_lacroix = params["table_3_lacroix"]
        self.table_4a_lacroix = params["table_4a_lacroix"]
        self.table_4b_lacroix = params["table_4b_lacroix"]
        self.table_6_helix_lacroix = params["table_6_helix_lacroix"]
        self.table_6_coil_lacroix = params["table_6_coil_lacroix"]
        self.table_7_ccap_lacroix = params["table_7_ccap_lacroix"]
        self.table_7_ncap_lacroix = params["table_7_ncap_lacroix"]
        self.table_3_munoz_nterm = params["table_3_munoz_nterm"]
        self.table_3_munoz_cterm = params["table_3_munoz_cterm"]
        self.table_7_coulomb_ncap = params["table_7_coulomb_ncap"]
        self.table_7_coulomb_ccap = params["table_7_coulomb_ccap"]
        self.table_pka_values = params["pka_values"]

        is_valid_peptide_sequence(seq)
        is_valid_ncap_ccap(ncap, ccap)
        is_valid_conditions(pH, T, ionic_strength)

        self.seq = seq
        self.seq_list = list(seq)
        if ncap is not None:
            self.seq_list.insert(0, ncap)
        if ccap is not None:
            self.seq_list.append(ccap)
        self.helix = self.seq_list[i:i+j]
        self.i = i
        self.j = j
        self.pH = pH
        self.T_celsius = T
        self.T_kelvin = T + 273.15
        self.ionic_strength = ionic_strength
        self.ncap = ncap
        self.ccap = ccap

        # pre-compute indices for the helix and get some key residues
        self.helix_indices = list(range(i, i+j))

        self.ncap_idx = self.helix_indices[0]
        self.Ncap_AA = self.seq_list[self.ncap_idx]
        self.N1_AA = self.seq_list[self.ncap_idx + 1]
        self.N3_AA = self.seq_list[self.ncap_idx + 3]
        self.N4_AA = self.seq_list[self.ncap_idx + 4]

        self.ccap_idx = self.helix_indices[-1]
        self.cprime_idx = self.ccap_idx + 1
        self.Ccap_AA = self.seq_list[self.ccap_idx]
        if self.cprime_idx < len(self.seq_list):
            self.Cprime_AA = self.seq_list[self.cprime_idx]
        else:
            self.Cprime_AA = None
        self.C3_AA = self.seq_list[self.ccap_idx - 3]

        # pre-compute some flags
        self.has_acetyl = True if ncap == "Ac" else False
        self.has_succinyl = True if ncap == "Sc" else False
        self.has_amide = True if ccap == "Am" else False

        # assign some constants
        self.min_helix_length = 6
        self.mu_helix = 0.5
        self.neg_charge_aa = ["C", "D", "E", "Y"] # residues that get a negative charge when deprotonated
        self.pos_charge_aa = ["K", "R", "H"] # residues that get a positive charge when protonated

        # assign some chemistry constants
        self.kappa = debye_screening_kappa(self.ionic_strength, self.T_kelvin)
        self.epsilon_r = calculate_permittivity(self.T_kelvin) # Relative permittivity of water
        self.epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(Nm^2)
        self.N_A = 6.022e23  # Avogadro's number in mol^-1
        self.e = 1.602e-19  # Elementary charge in Coulombs

        # assign None to all variables
        self.charged_pairs = None
        self.seq_pka = None
        self.nterm_pka = None
        self.cterm_pka = None

        self.seq_ionization = None
        self.nterm_ionization = None
        self.cterm_ionization = None

        self.terminal_macrodipole_distance_nterm = None
        self.terminal_macrodipole_distance_cterm = None

        # modified ionization states    
        self.modified_seq_ionization_hel = None
        self.modified_nterm_ionization_hel = None
        self.modified_cterm_ionization_hel = None
        self.modified_seq_ionization_rc = None
        self.modified_nterm_ionization_rc = None
        self.modified_cterm_ionization_rc = None

        # for printing
        self.category_pad = 15
        self.value_pad = 6

        # find charged pairs
        self._find_charged_pairs()

        # assign pKa values, distances, and ionization states
        self._assign_pka_values()
        self._assign_ionization_states()
        self._assign_terminal_macrodipole_distances()
        self._assign_sidechain_dipole_potentials()
        self._assign_terminal_sidechain_distances()
        self._assign_sidechain_sidechain_distances()
        self._assign_gb_pairs()
        self._assign_modified_ionization_states()

    def _find_charged_pairs(self) -> list[tuple[str, int, int]]:
        """
        Find all pairs of charged residues in a sequence and their global positions.
        """
        charged_amino_acids = self.neg_charge_aa + self.pos_charge_aa

        # Iterate over all pairs of charged residues (pairing only the charged
        # positions; scanning every residue pair was costly, as it runs per segment)
        charged_idx = [idx for idx, aa in enumerate(self.seq_list) if aa in charged_amino_acids]
        self.charged_pairs = [
            (self.seq_list[idx1], self.seq_list[idx2], idx1, idx2)  # Include global positions
            for idx1, idx2 in itertools.combinations(charged_idx, 2)
        ]

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

    @staticmethod
    def _terminal_macrodipole_r(N: int) -> float:
        """Distance from the free terminus to the helix macrodipole pole.

        Uses an empirically fitted polynomial.

        NOTE: Lacroix 1998 states this distance directly as 2.1 A plus 2 A per
        extra residue (r = 2.1 + 2.0*N); the polynomial grows at ~1.7-1.8 A per
        residue instead. Replacing it with the paper's rule is a candidate change
        whose benchmark effect has been measured separately.

        It is NOT replaced here, because doing so alone makes end-to-end
        agreement worse: another term is compensating for this error. See
        reasoning/nodes/N045.md for the measurements and what to fix alongside it.

        This function is only used for terminal-macrodipole distances, not
        for random-coil reference distances in the pKa solver or scsc code.

        Args:
            N: Number of coil residues between the free terminus and the
               helix macrodipole pole (Ncap or Ccap).

        Returns:
            Distance in Ångströms.
        """
        r = 2.0 + 1.80167 * N - 0.045 * N**2 + 0.00333 * N**3
        return max(r, 2.0)

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

    def _make_box(self, title: str):
        """
        Make a box with a title in the middle.
        For making nice looking output when printing.
        """
        box_lines = []
        box_lines.append('+' + '-' * (len(title) + 2) + '+')
        box_lines.append('|' + f'{title.center(len(title) + 2)}' + '|')
        box_lines.append('+' + '-' * (len(title) + 2) + '+')
        return '\n'.join(box_lines)

    def show_inputs(self):
        """
        Print out the inputs for the AGADIR model in a nicely formatted way.
        """
        print(self._make_box("Inputs"))
        print(f'seq = {self.seq}, i = {self.i}, j = {self.j}, pH = {self.pH}, T(celcius) = {self.T_celsius}, ionic_strength = {self.ionic_strength}, ncap = {self.ncap}, ccap = {self.ccap}')
        print("")

    def show_helix(self):
        """
        Print out the helix in a nicely formatted way.
        """
        print(self._make_box("Helix"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"structure:".ljust(self.category_pad)} {"".join(["He".ljust(self.value_pad) if self.i <= idx < self.i + self.j else "STC".ljust(self.value_pad) for idx, aa in enumerate(self.seq_list)])}')
        print("")

    def _assign_pka_values(self):
        """
        Assign pKa values to the sequence and the terminal residues.
        """
        self.seq_pka = np.array([float(self.table_pka_values.loc[aa]["pKa"]) if aa in self.neg_charge_aa + self.pos_charge_aa else np.nan 
                                  for aa in self.seq_list])
        self.nterm_pka = float(self.table_pka_values.loc["Nterm"]["pKa"]) if self.ncap is None else float(self.table_pka_values.loc["Sc"]["pKa"]) if self.ncap == "Sc" else np.nan
        # the alpha-amino pKa depends on the N-terminal residue; a residue-specific row
        # ("Nterm_Y") is used where one has been measured (params/README.md)
        if self.ncap is None and f"Nterm_{self.seq_list[0]}" in self.table_pka_values.index:
            self.nterm_pka = float(self.table_pka_values.loc[f"Nterm_{self.seq_list[0]}"]["pKa"])
        self.cterm_pka = float(self.table_pka_values.loc["Cterm"]["pKa"]) if self.ccap is None else np.nan

    def get_pka_values(self):
        """
        Get pKa values for the sequence and the terminal residues.

        Returns:
            np.ndarray: pKa values for the sequence.
            float: pKa value for the N-terminal capping group.
            float: pKa value for the C-terminal capping group.
        """
        return self.seq_pka, self.nterm_pka, self.cterm_pka
    
    def show_pka_values(self):
        """
        Print out pKa values for the sequence in a nicely formatted way.
        """
        print(self._make_box("pKa values"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"pKa:".ljust(self.category_pad)} {"".join([f"{pka:.2f}".ljust(self.value_pad) for pka in self.seq_pka])}')
        print(f'{"nterm_pka:".ljust(self.category_pad)} {self.nterm_pka:.2f}')
        print(f'{"cterm_pka:".ljust(self.category_pad)} {self.cterm_pka:.2f}')
        print("")

    def _assign_ionization_states(self):
        """
        Compute ionization states for charged residues in the sequence and the terminal residues.

        The ionization states are computed using the pKa values and the pH of the solution.
        """
        self.seq_ionization = np.array([
            acidic_residue_ionization(self.pH, pKa) if aa in self.neg_charge_aa else 
            basic_residue_ionization(self.pH, pKa) if aa in self.pos_charge_aa else 0.0 
            for aa, pKa in zip(self.seq_list, self.seq_pka)
        ])
        
        self.nterm_ionization = (
            basic_residue_ionization(self.pH, self.nterm_pka) if self.ncap is None else
            acidic_residue_ionization(self.pH, self.nterm_pka) if self.ncap == "Sc" else 0.0
        )
        
        self.cterm_ionization = (
            acidic_residue_ionization(self.pH, self.cterm_pka) if self.ccap is None else 0.0
        )
    
    def get_ionization_states(self):
        """
        Get the ionization states for the sequence and the terminal residues.

        Returns:
            np.ndarray: Ionization states for the sequence.
            float: Ionization state for the N-terminal capping group.
            float: Ionization state for the C-terminal capping group.
        """
        return self.seq_ionization, self.nterm_ionization, self.cterm_ionization

    def show_ionization_states(self):
        """
        Print out ionization states for the sequence in a nicely formatted way.
        """
        print(self._make_box("Ionization states"))
        print(f'{"sequence:".ljust(self.category_pad)} {"".join([aa.ljust(self.value_pad) for aa in self.seq_list])}')
        print(f'{"charge:".ljust(self.category_pad)} {"".join([f"{q:.2f}".ljust(self.value_pad) for q in self.seq_ionization])}')
        print(f'{"nterm_charge:".ljust(self.category_pad)} {self.nterm_ionization:.2f}')
        print(f'{"cterm_charge:".ljust(self.category_pad)} {self.cterm_ionization:.2f}')
        print("")

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

    def _gb_dipole_potential(self, idx: int) -> tuple[float, float]:
        """
        dipole_law="gb": energy (kcal/mol) of a +1 charge on residue idx from the helix backbone, by Generalized Born
        on built poly-Ala helices (params/sidechain_dipole_gb.tsv, tools/build_gb_dipole_table.py), helix minus coil.
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
        form = GB_DIPOLE_FORM.get(aa)
        if form is None or not ncap_i <= idx <= ccap_i:
            return self._munoz_dipole_potential_scaled(idx)
        e = self._gb_form_energy(gb_dipole_table(), form, idx)
        if e is None:
            return self._munoz_dipole_potential_scaled(idx)
        energy = e[0] - e[1]
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
        RT = 1.9872e-3 * self.T_kelvin
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

    def _munoz_dipole_potential_scaled(self, idx: int) -> tuple[float, float]:
        """Muñoz's law with its eps(T) factor applied, for the positions the GB law leaves to it."""
        factor = math.exp(0.004314 * (self.T_kelvin - 273.15))
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
        eps_to_0C = self.epsilon_r / calculate_permittivity(273.15)

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
        # gbp: the energy every ionisable residue (N-cap..C-cap) carries at any charge: its neutral form's GB energy
        # (helix minus coil) with the neutral transfer offset (Lys, His, Asp, Glu), plus its class's constant transfer
        # offset (all tabulated residues)
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
                 1995 Table 2) and contributes no helix-coil energy.

        terminal_sidechain_modelled_nterm / _cterm mark the pairs the energy term counts.
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
        Assign the distance between the peptide terminal residues and the helix macrodipole.
        Uses the corrected polynomial distance model (_terminal_macrodipole_r) for
        standard free amine/carboxyl terminals.  Succinylated N-termini (Sc) keep
        the linear _calculate_r model because the succinyl carboxyl extends beyond
        the backbone, making its effective distance longer.
        """
        # N-Terminal Dipole Distance
        if self.ncap == 'Sc':
            # Succinyl carboxyl is farther from helix than a backbone amine
            self.terminal_macrodipole_distance_nterm = self._calculate_r(self.ncap_idx)
        else:
            self.terminal_macrodipole_distance_nterm = self._terminal_macrodipole_r(self.ncap_idx)

        # C-Terminal Dipole Distance
        c_separation = len(self.seq_list) - 1 - self.ccap_idx
        self.terminal_macrodipole_distance_cterm = self._terminal_macrodipole_r(c_separation)

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
           table 6 N' / C' (C'G-cap when the C-cap is Gly) distances; other such pairs are
           not modelled
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
                
            self.sidechain_sidechain_distances_hel[idx1, idx2] = distance_angstrom
            self.sidechain_sidechain_distances_hel[idx2, idx1] = distance_angstrom

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

    def show_all(self):
        """
        Print out all the inputs, helix, pKa values, and ionization states in a nicely formatted way.
        """
        self.show_inputs()
        self.show_helix()
        self.show_pka_values()
        self.show_ionization_states()
        self.show_modified_ionization_states()
        self.show_terminal_macrodipole_distances()
        self.show_sidechain_dipole_potentials()
        self.show_terminal_sidechain_distances()
        self.show_sidechain_sidechain_distances()


class EnergyCalculator(PrecomputeParams):
    """
    Class to calculate the free energy contributions for a peptide sequence.
    """
    def __init__(self, seq: str, i: int, j: int, pH: float, T: float, ionic_strength: float, ncap: str = None, ccap: str = None, params: dict = None, dipole_law: str = "munoz"):
        """
        Initialize the EnergyCalculator for a peptide sequence.

        Args:
            seq (str): Peptide sequence.
            i (int): Helix start index, python 0-indexed.
            j (int): Helix length.
            pH (float): Solution pH.
            T (float): Temperature in Celsius.
            ionic_strength (float): Ionic strength of the solution in mol/L.
            ncap (str): N-terminal capping modification (acetylation='Ac', succinylation='Sc').
            ccap (str): C-terminal capping modification (amidation='Am').
            params (dict): Parameter tables from snapshot_params(). If None, load_params() is used.
            dipole_law (str): side chain-macrodipole law, "munoz" (default) or "gb".
        """
        super().__init__(seq, i, j, pH, T, ionic_strength, ncap, ccap, params=params, dipole_law=dipole_law)
        self._AROMATIC = {"F", "Y", "W"}
        self._ALIPHATIC = {"A", "V", "L", "I", "M"}  # you can expand if you want (e.g. C)
        self.dCp = -0.0015  # kcal/(mol*K)

    def _dCp_hydroph_kcal(self, aa1: str, aa2: str) -> float:
        """
        Hydrophobic heat capacity increment in kcal/(mol*K), helix-formation direction.
        Implements the Muñoz & Serrano linear form (their Eq. 18) with a separate
        aromatic-aromatic scale.
        """
        # Only apply to hydrophobic pairs
        if not ((aa1 in self._ALIPHATIC or aa1 in self._AROMATIC) and (aa2 in self._ALIPHATIC or aa2 in self._AROMATIC)):
            return 0.0

        T = self.T_kelvin
        Tref = 273.15

        if aa1 in self._AROMATIC and aa2 in self._AROMATIC:
            # aromatic-aromatic (smaller magnitude)
            return (-4.0 + 0.025 * (T - Tref)) / 1000.0  # cal -> kcal
        else:
            # aliphatic-aliphatic or aromatic-aliphatic
            return (-8.0 + 0.05 * (T - Tref)) / 1000.0  # cal -> kcal

    def _entropic_cp_correct(self, dG_ref: float, dCp: float) -> float:
        """
        Corrects a purely entropic free energy for temperature using Muñoz 1995 Eq. (10).
        Assumes dH = 0 at all temperatures (hydrophobic interactions).

        Formula: dG(T) = dG_ref * (T/Tref) - T * dCp * ln(T/Tref)
        """
        T = self.T_kelvin
        Tref = 273.15  # 0°C reference temperature

        # Entropic scaling of reference energy
        scaled_ref = dG_ref * (T / Tref)

        # Cp contribution to entropy: -T * dCp * ln(T/Tref)
        cp_term = -T * dCp * np.log(T / Tref)

        return scaled_ref + cp_term

    def get_dG_Int(self) -> np.ndarray:
        """
        Get the intrinsic free energy contributions for a helical segment.
        This accounts for the loss of entropy due to the helix formation.
        The first and last residues are considered to be caps unless they are
        the peptide terminal residues with modifications.
        Temperature: equation (9) of Muñoz & Serrano (1995-III), a purely entropic term,
        dG(t) = dG_ref * t/t_ref - t * dCp * ln(t/t_ref) with t_ref = 273 K.

        Returns:
            np.ndarray: The intrinsic free energy contributions for each amino acid in the helical segment.
        """
        # Initialize energy array
        energy = np.zeros(len(self.seq_list))
        T = self.T_kelvin
        dCp = self.dCp

        # Iterate over the helix and get the intrinsic energy for each residue, 
        # not including residues that are capping for the helical segment
        for idx in self.helix_indices:
            if idx == self.ncap_idx or idx == self.ccap_idx:
                continue

            AA = self.seq_list[idx]

            # Distance from N-cap and C-cap
            n_dist = idx - self.ncap_idx  # 1=N1, 2=N2, ...
            c_dist = self.ccap_idx - idx  # 1=C1, 2=C2, ...

            # Use C-terminal propensities for positions C1, C2, C3 (within 3 of Ccap)
            # Otherwise use N-terminal propensities
            if c_dist <= 3 and f"C{c_dist}" in self.table_1_lacroix.columns:
                col = f"C{c_dist}"
                val = self.table_1_lacroix.loc[AA, col]
                if not np.isnan(val):
                    energy[idx] = val
                else:
                    # Fallback to N-terminal value if C-terminal not available
                    energy[idx] = self.table_1_lacroix.loc[AA, "Ncen"]
            elif n_dist == 1:
                energy[idx] = self.table_1_lacroix.loc[AA, "N1"]
            elif n_dist == 2:
                energy[idx] = self.table_1_lacroix.loc[AA, "N2"]
            elif n_dist == 3:
                energy[idx] = self.table_1_lacroix.loc[AA, "N3"]
            elif n_dist == 4:
                energy[idx] = self.table_1_lacroix.loc[AA, "N4"]
            else:
                energy[idx] = self.table_1_lacroix.loc[AA, "Ncen"]

            if AA in self.pos_charge_aa + ["D", "E"]:
                # Ionization correction for residues that are normally charged at pH 7.
                # K, R, H (pos_charge_aa) have pKa > 6 and are charged at neutral pH.
                # D, E have pKa ~4 and are charged (negative) at neutral pH.
                # Table 1 position-specific values represent the charged form for these.
                # When deionized (extreme pH), interpolate toward the Neutral column.
                # Y and C are excluded: their pKa (~10.1, ~8.3) means they are normally
                # neutral at pH 7, so Table 1 values already represent the neutral form.
                q = abs(self.modified_seq_ionization_hel[idx])
                basic_energy = energy[idx]
                basic_energy_neutral = self.table_1_lacroix.loc[AA, "Neutral"]
                energy[idx] = q * basic_energy + (1 - q) * basic_energy_neutral

            # First-turn N2 offset (FIRST_TURN_N2_OFFSET), after the charged/neutral interpolation so that both
            # charge states pay it; not where the C1-C3 columns apply.
            if n_dist == 2 and c_dist > 3:
                energy[idx] += self._first_turn_n2_offset(AA)

            # gbx/gbp: first- and last-turn cells of charged residues re-derived under the GB law
            # (params/gbx_overrides.tsv, params/gbp_overrides.tsv)
            if self.dipole_law in GBX_LAWS:
                pos = f"C{c_dist}" if c_dist <= 3 else (f"N{n_dist}" if n_dist <= 3 else None)
                energy[idx] += law_tables(self.dipole_law)["end_cells"].get((AA, pos), 0.0)

        # Munoz & Serrano 1995-III eq. (9): dG_Int = -t (dS_ref + dCp ln(t/t_ref)),
        # i.e. dG_ref * t/t_ref - t * dCp * ln(t/t_ref).  The dCp*(t - t_ref) enthalpy term
        # belongs to the H-bond (eq. 8), not here.  t_ref = 273.0 K.
        Tref = 273.0
        scaled_ref = energy * (T / Tref)
        cp_term = -T * dCp * np.log(T / Tref)

        # Apply to all non-zero entries (avoid adding energy to caps/zeros)
        energy = np.where(energy != 0, scaled_ref + cp_term, energy)

        return energy
    
    def _first_turn_n2_offset(self, aa: str) -> float:
        """FIRST_TURN_N2_OFFSET, with the gbx/gbp law's re-derived values (params/gbx_overrides.tsv or
        params/gbp_overrides.tsv, table "first_turn_n2") where it has them."""
        if self.dipole_law in GBX_LAWS:
            for table, row, _col, value in law_tables(self.dipole_law)["overrides_meta"]:
                if table == "first_turn_n2" and row == aa:
                    return value
        return FIRST_TURN_N2_OFFSET.get(aa, 0.0)

    def get_dG_Hbond(self) -> float:
        """
        Get the free energy contribution for hydrogen bonding for a sequence.

        Capping residues, don't count toward hydrogen bonding, which gives 2.
        Additionally, the first 4 helical amino acids are considered to have
        zero net enthalpy since they are nucleating residues.
        This gives a total of 6 residues that don't count toward hydrogen bonding.

        The net H-bond ΔG at 0°C is -0.898 kcal/mol per bond (Lacroix 1998). It is
        treated as enthalpic, with the temperature dependence of equation (8) of
        Muñoz & Serrano (1995-III): ΔG(t) = ΔH_ref + ΔCp (t - t_ref), ΔCp = -1.5
        cal/(mol·K) in the folding direction, t_ref = 273 K (_hbond_temperature, the
        law every hydrogen-bond term uses).

        Returns:
            float: The total free energy contribution for hydrogen bonding in the sequence.
        """
        n_hbonds = max((self.j - 6), 0)

        # Munoz & Serrano 1995-III eq. (8): dG_HBond = dH_ref + dCp (t - t_ref) per bond,
        # with dH_ref = -0.898 (Lacroix 1998), dCp = -0.0015 kcal/(mol K), t_ref = 273.0 K.
        dG_per_bond = self._hbond_temperature(HBOND_DH_REF)
        return dG_per_bond * n_hbonds

    def _hbond_temperature(self, dG_ref):
        """
        The temperature law of every hydrogen-bond term: the backbone bond (get_dG_Hbond), the
        hydrogen-bonding N- and C-caps and end groups (get_dG_Ncap, get_dG_Ccap), the side-chain
        hydrogen bonds of the i,i+3 and i,i+4 terms (Table IV acid-base and polar pairs, the
        Table V motifs) and the Petukhov and charged-staple motifs.

        Muñoz & Serrano (1995-III) eq. (8) treats a hydrogen-bond term as an enthalpy,
        ΔG(t) = ΔH_ref + ΔCp (t - t_ref), with ΔCp = -1.5 cal/(mol·K) per backbone bond in the
        folding direction and t_ref = 273 K; the entropy of fixing the backbone is in
        get_dG_Int. Other hydrogen bonds take the same law as a relative change,
        ΔG_ref × (1 + ΔCp (t - t_ref) / ΔH_ref) with the backbone bond's ΔH_ref = -0.898
        kcal/mol, which for the backbone bond is eq. (8) itself. Muñoz & Serrano apply it to the
        hydrogen-bonding caps and the end groups (p. 305) and neglect it for side-chain pairs.
        The term grows more favourable with temperature, as the backbone term does.

        Args:
            dG_ref: free energy at 273 K (float or array), kcal/mol.
        """
        return dG_ref * (1.0 + self.dCp * (self.T_kelvin - 273.0) / HBOND_DH_REF)

    def get_dG_Ncap(self) -> np.ndarray:
        """
        Get the free energy contribution for N-terminal capping.
        This accounts only for residue capping effects.

        Returns:
            np.ndarray: The free energy contribution.
        """
        energy = np.zeros(len(self.seq_list))

        # Nc-4 	N-cap values when there is a Pro at position N1 and Glu, Asp or Gln at position N3.
        if self.N1_AA == "P" and self.N3_AA in ["E", "D", "Q"]:
            col = "Nc-4"

        # Nc-3 	N-cap values when there is a Glu, Asp or Gln at position N3.
        elif self.N3_AA in ["E", "D", "Q"]:
            col = "Nc-3"

        # Nc-2 	N-cap values when there is a Pro at position N1.
        elif self.N1_AA == "P":
            col = "Nc-2"

        # Nc-1 	Normal N-cap values.
        else:
            col = "Nc-1"
        energy[self.ncap_idx] = self.table_1_lacroix.loc[self.Ncap_AA, col]

        # Lacroix 1998 (G_nonH): the N-capping contribution of Cys is 1 kcal/mol more
        # favourable when it is charged, and that of His 1 kcal/mol more favourable when it
        # is neutral.  Table I holds the neutral forms; weighted by helix-state ionisation.
        if self.Ncap_AA in ("C", "H"):
            q_cap = abs(float(self.modified_seq_ionization_hel[self.ncap_idx]))
            energy[self.ncap_idx] += (-1.0 if self.Ncap_AA == "C" else 1.0) * q_cap

        # Capping box (Harper & Rose 1993): a Ser, Thr, Asp or Asn N-cap and a Glu at N3 form
        # reciprocal side chain-backbone hydrogen bonds. Peptide measurements put it at -0.9
        # kcal/mol beyond the Nc-3 column: Glu vs Ala, Gln and Asp at N3 and Ser vs Ala at the
        # N-cap (Zhou et al. 1994, Proteins 18, 1; Petukhov et al. 1996, Biochemistry 35, 387).
        # Glu beats Gln there by as much as it beats Ala, so the bonus belongs to the charged Glu
        # and is weighted by that Glu's helix-state ionisation.
        if self.Ncap_AA in ("S", "T", "D", "N") and self.N3_AA == "E":
            q_glu = abs(float(self.modified_seq_ionization_hel[self.ncap_idx + 3]))
            energy[self.ncap_idx] += -0.9 * q_glu

        # A Ser, Thr, Asn or Asp N-cap, and the Ac or Sc end group, stabilise the helix by a
        # hydrogen bond (Muñoz & Serrano 1995-III p. 305): their advantage over Ala in the same
        # column, capping box included, takes the hydrogen-bond temperature law.
        if self.Ncap_AA in HBOND_NCAPS:
            ref = self.table_1_lacroix.loc["A", col]
            energy[self.ncap_idx] = ref + self._hbond_temperature(energy[self.ncap_idx] - ref)
        return energy

    def get_dG_Ccap(self) -> np.ndarray:
        """
        Get the free energy contribution for C-terminal capping.

        Returns:
            np.ndarray: The free energy contribution.
        """
        energy = np.zeros(len(self.seq_list))

        # Cc-2 	C-cap values when there is a Pro residue at position C'
        if self.Cprime_AA == "P":
            energy[self.ccap_idx] = self.table_1_lacroix.loc[self.Ccap_AA, "Cc-2"]

        # Cc-1 	Normal C-cap values
        else:
            energy[self.ccap_idx] = self.table_1_lacroix.loc[self.Ccap_AA, "Cc-1"]

        # Lacroix 1998 (G_nonH): uncharged Asp at the C-cap H-bonds the C3 carbonyl as Asn
        # does and takes Asn's C-capping value; Table I holds the charged form.
        if self.Ccap_AA == "D":
            col = "Cc-2" if self.Cprime_AA == "P" else "Cc-1"
            q_cap = abs(float(self.modified_seq_ionization_hel[self.ccap_idx]))
            energy[self.ccap_idx] = (q_cap * self.table_1_lacroix.loc["D", col]
                                     + (1.0 - q_cap) * self.table_1_lacroix.loc["N", col])

        # A Lys, Arg, His, Asn or Asp C-cap hydrogen-bonds a C-terminal carbonyl, and the Am end
        # group's contribution is a hydrogen bond (Muñoz & Serrano 1995-III p. 305): the
        # advantage over Ala (the whole Am value) takes the hydrogen-bond temperature law.
        col = "Cc-2" if self.Cprime_AA == "P" else "Cc-1"
        if self.Ccap_AA in HBOND_CCAPS:
            ref = self.table_1_lacroix.loc["A", col]
            energy[self.ccap_idx] = ref + self._hbond_temperature(energy[self.ccap_idx] - ref)
        elif self.Ccap_AA == "Am":
            energy[self.ccap_idx] = self._hbond_temperature(energy[self.ccap_idx])
        return energy

    def get_dG_staple(self) -> float:
        """
        Get the free energy contribution for the hydrophobic staple motif.
        The hydrophobic interaction is between the N' and N4 residues of the helix.
        The terminology of Richardson & Richardson (1988) is used.
        See https://doi.org/10.1038/nsb0595-380 for more details.

        Returns:
            float: The free energy contribution.
        """
        # Staple motif requires the N' residue before the Ncap, so the first residue of the helix cannot be the first residue of the peptide
        # This should be true regardless of whether there is an N-terminal modification or not
        energy = 0.0
        if self.ncap_idx == 0:
            return energy

        # The hydrophobic staple motif applies whatever the N-cap residue: x 1 in the two capping-box cases below
        # (Lacroix 1998 supplement) and x 0.5 otherwise, including non-polar N-caps (Viguera & Serrano 1999,
        # Protein Sci. 8, 1733, Table 3 note c; params/README.md). The supplement had given non-polar N-caps 0.
        # Table II is indexed by N' (rows) x N4 (columns): "the interactions between
        # different amino acids at positions N' (rows) and N4 (columns) in a hydrophobic
        # staple motif" (Lacroix 1998, supplementary Table II caption), which is also what
        # this method's docstring says.  This lookup previously used Ncap_AA, so the term
        # read the wrong table row for every staple it fired on.
        Nprime_AA = self.seq_list[self.ncap_idx - 1]
        if Nprime_AA in self.table_2_lacroix.index and self.N4_AA in self.table_2_lacroix.columns:
            energy = self.table_2_lacroix.loc[Nprime_AA, self.N4_AA] / 100

            # whenever the N-cap residue is Asn, Asp, Ser, or Thr and the N3 residue is Glu, Asp or Gln, multiply by 1.0
            if self.Ncap_AA in ["N", "D", "S", "T"] and self.N3_AA in ["E", "D", "Q"]:
                energy *= 1.0

            # whenever the N-cap residue is Asp or Asn and the N3 residue is Ser or Thr
            elif self.Ncap_AA in ["N", "D"] and self.N3_AA in ["S", "T"]:
                energy *= 1.0

            # other cases they are multiplied by 0.5
            else:
                energy *= 0.5

        # Apply hydrophobic temperature correction to the N'–N4 interaction
        # N' is the residue BEFORE Ncap
        if energy != 0.0 and self.ncap_idx > 0:
            Nprime = self.seq_list[self.ncap_idx - 1]
            N4 = self.N4_AA

            dCp_h = self._dCp_hydroph_kcal(Nprime, N4)
            if dCp_h != 0.0:
                energy = self._entropic_cp_correct(energy, dCp_h)

        return energy

    def get_dG_schellman(self) -> float:
        """
        Get the free energy contribution for the Schellman motif.
        The Schellman motif is only considered whenever Gly is the C-cap residue,
        where the interaction happens between the C' and C3 residues of the helix.
        The terminology of Richardson & Richardson (1988) is used.

        Returns:
            float: The free energy contribution.
        """
        # The Schellman motif is only considered whenever Gly is the C-cap residue,
        # and there has to be a C' residue after the helix
        energy = 0.0
        if self.Cprime_AA in ["Am", None] or self.Ccap_AA != "G":
            return energy
    
        # get the amino acids governing the Schellman motif and extract the energy
        energy = self.table_3_lacroix.loc[self.C3_AA, self.Cprime_AA] / 100

        return energy

    def get_dG_petukhov_motif(self) -> float:
        """
        Get the free energy contribution for the Petukhov combination motif.

        From Lacroix 1998 (p.175): "free N terminus, capping box motif and an Asp or a Glu
        at position N4. The stabilization is due to a strong interaction between residue N4,
        the N-capping residue and the charged N-terminal group." Contributes -1 kcal/mol.

        Requirements:
        - Free (unblocked) N-terminus
        - Helix starts at the peptide N-terminus (Ncap is position 0)
        - Ncap is a capping box residue (Asp, Asn, Ser, Thr)
        - N3 is Glu, Asp, or Gln (the capping box H-bond partner)
        - N4 is Asp or Glu (charged)

        Returns:
            float: The free energy contribution.
        """
        # Must have a free N-terminus (no Ac/Sc cap)
        if self.ncap is not None:
            return 0.0

        # Helix must start at the peptide N-terminus
        if self.ncap_idx != 0:
            return 0.0

        # Ncap must be a capping box residue
        if self.Ncap_AA not in ["D", "N", "S", "T"]:
            return 0.0

        # N3 must be Glu, Asp, or Gln (capping box partner)
        if self.N3_AA not in ["E", "D", "Q"]:
            return 0.0

        # N4 must be Asp or Glu
        if self.N4_AA not in ["D", "E"]:
            return 0.0

        # Apply the motif energy, scaled by the ionization of N4
        q_N4 = abs(self.modified_seq_ionization_hel[self.ncap_idx + 4])
        energy = -1.0 * q_N4

        # hydrogen bond: the hydrogen-bond temperature law
        energy = self._hbond_temperature(energy)

        return energy

    def get_dG_charged_staple(self) -> float:
        """
        Get the free energy contribution for the charged staple variant.

        From Lacroix 1998 (p.175): When Ser or Thr is at N-cap, the carbonyl of N0 points
        toward N4. If K or R is at N4, it can form a hydrogen bond with that carbonyl.
        Value: -0.3 kcal/mol.

        Requirements:
        - Ncap is Ser or Thr
        - N4 is Lys or Arg
        - There is an N' residue before Ncap (ncap_idx > 0)

        Returns:
            float: The free energy contribution.
        """
        # Need an N' residue before the Ncap
        if self.ncap_idx == 0:
            return 0.0

        # Ncap must be Ser or Thr
        if self.Ncap_AA not in ["S", "T"]:
            return 0.0

        # N4 must be Lys or Arg
        if self.N4_AA not in ["K", "R"]:
            return 0.0

        # Apply energy, scaled by ionization of N4
        q_N4 = abs(self.modified_seq_ionization_hel[self.ncap_idx + 4])
        energy = -0.3 * q_N4

        # hydrogen bond: the hydrogen-bond temperature law
        energy = self._hbond_temperature(energy)

        return energy

    def _acid_base_hbonds(self) -> set:
        """Acid-base (Asp/Glu - Lys/Arg/His) side-chain H-bonds that can coexist in this helix.

        A side chain is in one rotamer at a time: it can reach partners on its N-terminal side
        (i-3, i-4) or on its C-terminal side (i+3, i+4), not both.  Partners on the same side can
        share that rotamer.  Candidates are the i,i+3 and i,i+4 pairs inside the helix (caps
        excluded) with a favourable Table IV value; the strongest are taken first.  The ionic
        part of each pair (the Coulomb term) is not restricted.
        """
        if getattr(self, "_ab_hbonds", None) is not None:
            return self._ab_hbonds
        if self.dipole_law == "gbp":  # statistical network (_sb_network_factor): every candidate carries its cell
            self._ab_hbonds = set(self._sb_candidates())
            return self._ab_hbonds
        cands = []
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
                        cands.append((v, i, j))
        # A side chain points either toward the N-terminus (partners at i-3/i-4) or toward the
        # C-terminus (partners at i+3/i+4), not both.  Partners on the same side can share it.
        direction, chosen = {}, set()
        for v, i, j in sorted(cands):
            if direction.get(i, "up") != "up" or direction.get(j, "down") != "down":
                continue
            direction[i], direction[j] = "up", "down"
            chosen.add((i, j))
        self._ab_hbonds = chosen
        return chosen

    def _sb_candidates(self) -> dict:
        """The acid-base pairs i,i+3 and i,i+4 inside the helix (caps excluded) with a favourable Table IV cell, at
        this temperature (the hydrogen-bond law): {(i, j): kcal/mol}."""
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
            self._sb_f = sb_network_free_energy(c, 1.9872e-3 * self.T_kelvin) / tot if tot < 0 else 1.0
        return self._sb_f

    def get_dG_i3(self) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+3 in the sequence.

        - Table IV corresponds to non-charged interactions.
        - If BOTH residues are ionizable, Table IV applies only to cases where
        at least one is not charged; scale by (1 - p_i * p_j).
        - No abs(q_i*q_j) scaling here; electrostatics handled elsewhere.

        Side chain-side chain interaction between AAi and AAi+3 (Table IVa, non-charged term).
        For pairs where BOTH residues are titratable, suppress the Table-IV contribution
        in the both-charged microstate: multiply by (1 - p_i * p_j), where p = |q|.

        Returns:
            np.ndarray: The free energy contributions for each interaction.
        """
        energy = np.zeros(len(self.seq_list))

        for idx in self.helix_indices[:-3]:
            AAi = self.seq_list[idx]
            AAi3 = self.seq_list[idx + 3]

            # Skip terminal modifications
            if AAi in ["Ac", "Am", "Sc"] or AAi3 in ["Ac", "Am", "Sc"]:
                continue

            # Table IV values are "kcal/mol * 100" -> convert to kcal/mol
            base = self.table_4a_lacroix.loc[AAi, AAi3] / 100.0
            # side chain-side chain pairs only INSIDE the helix (Lacroix 1998, G_SD): a pair
            # involving the N-cap or C-cap residue contributes nothing
            if idx == self.ncap_idx or idx + 3 == self.ccap_idx:
                base = 0.0

            # An Asp/Glu - Lys/Arg/His pair forms a side-chain hydrogen bond whose strength does
            # not depend on salt or on whether the acid is charged (Scholtz et al. 1993; Smith &
            # Scholtz 1998).  Its Table IV value is that hydrogen bond and applies in every
            # ionisation state, where the geometry allows it (_acid_base_hbonds); the ionic part
            # of the pair is the Coulomb term.
            acid_base = bool({AAi, AAi3} & {"D", "E"} and {AAi, AAi3} & {"K", "R", "H"})
            if acid_base and base < 0 and (idx, idx + 3) not in self._acid_base_hbonds():
                base = 0.0

            # Other titratable pairs: Table IV applies to the states that are not both charged.
            if not acid_base and (AAi in (self.pos_charge_aa + self.neg_charge_aa)) and (AAi3 in (self.pos_charge_aa + self.neg_charge_aa)):
                p_i = abs(self.modified_seq_ionization_hel[idx])
                p_j = abs(self.modified_seq_ionization_hel[idx + 3])
                base = base * (1.0 - p_i * p_j)

            # _dCp_hydroph_kcal is non-zero only for hydrophobic pairs; used here as the gate
            dCp_val = self._dCp_hydroph_kcal(AAi, AAi3)
            
            # temperature scaling applies to hydrophobic pairs only
            if dCp_val != 0.0 and base != 0.0:
                # hydrophobic pairs scale entropically (dG_ref * t/t_ref), no dCp_hydroph term:
                # Munoz 1995-III states this term becomes more favourable with temperature
                base = self._entropic_cp_correct(base, 0.0)
            # acid-base and polar pairs are side-chain hydrogen bonds: the hydrogen-bond temperature law
            elif base < 0.0 and (acid_base or (AAi in HBOND_POLAR and AAi3 in HBOND_POLAR)):
                base = self._hbond_temperature(base)
            if acid_base and base < 0.0 and self.dipole_law == "gbp":
                base *= self._sb_network_factor()

            energy[idx] = base

        return energy
    
    def get_dG_i4(self) -> np.ndarray:
        """
        Get the free energy contribution for interaction between each AAi and AAi+4 in the sequence.

        i,i+4 non-charged side chain interactions (Table IVb) + special pH-dependent,
        salt-independent local motifs (Table V), Lacroix supplement.

        Faithful interpretation:
        - Table IV corresponds to non-charged interactions.
        - If BOTH residues are ionizable, Table IV applies only to cases where
            at least one is not charged; scale by (1 - p_i * p_j).
        - Table V terms are ADDED when the relevant residue is charged (weighted by p).
        - No abs(q_i*q_j) scaling here; electrostatics handled elsewhere.

        - Table IVb is a non-charged (or "not both charged") interaction term.
        If BOTH residues are titratable: multiply by (1 - p_i * p_j).

        - Table V: additional interaction energy to ADD when one residue becomes charged
        (pH dependent, NOT affected by ionic strength). Scale by the population of the
        charged form of the relevant residue.
        """
        energy = np.zeros(len(self.seq_list))

        for idx in self.helix_indices[:-4]:
            AAi = self.seq_list[idx]
            AAi4 = self.seq_list[idx + 4]

            # Skip terminal modifications
            if AAi in ["Ac", "Am", "Sc"] or AAi4 in ["Ac", "Am", "Sc"]:
                continue

            # Table IV values are "kcal/mol * 100" -> convert to kcal/mol
            base = self.table_4b_lacroix.loc[AAi, AAi4] / 100.0
            # side chain-side chain pairs only INSIDE the helix (Lacroix 1998, G_SD): a pair
            # involving the N-cap or C-cap residue contributes nothing
            if idx == self.ncap_idx or idx + 4 == self.ccap_idx:
                base = 0.0

            # An Asp/Glu - Lys/Arg/His pair forms a side-chain hydrogen bond whose strength does
            # not depend on salt or on whether the acid is charged (Scholtz et al. 1993; Smith &
            # Scholtz 1998).  Its Table IV value is that hydrogen bond and applies in every
            # ionisation state, where the geometry allows it (_acid_base_hbonds); the ionic part
            # of the pair is the Coulomb term.
            acid_base = bool({AAi, AAi4} & {"D", "E"} and {AAi, AAi4} & {"K", "R", "H"})
            if acid_base and base < 0 and (idx, idx + 4) not in self._acid_base_hbonds():
                base = 0.0

            # Other titratable pairs: suppress Table IV in the both-charged microstate.
            if not acid_base and (AAi in (self.pos_charge_aa + self.neg_charge_aa)) and (AAi4 in (self.pos_charge_aa + self.neg_charge_aa)):
                p_i = abs(self.modified_seq_ionization_hel[idx])
                p_j = abs(self.modified_seq_ionization_hel[idx + 4])
                base = base * (1.0 - p_i * p_j)

            # _dCp_hydroph_kcal is non-zero only for hydrophobic pairs; used here as the gate
            dCp_val = self._dCp_hydroph_kcal(AAi, AAi4)
            
            # temperature scaling applies to hydrophobic pairs only
            if dCp_val != 0.0 and base != 0.0:
                # hydrophobic pairs scale entropically (dG_ref * t/t_ref), no dCp_hydroph term:
                # Munoz 1995-III states this term becomes more favourable with temperature
                base = self._entropic_cp_correct(base, 0.0)
            # acid-base and polar pairs are side-chain hydrogen bonds: the hydrogen-bond temperature law
            elif base < 0.0 and (acid_base or (AAi in HBOND_POLAR and AAi4 in HBOND_POLAR)):
                base = self._hbond_temperature(base)
            if acid_base and base < 0.0 and self.dipole_law == "gbp":
                base *= self._sb_network_factor()

            extra = 0.0  # Table V hydrogen-bond motifs
            aro_his = 0.0  # Table V aromatic-His+ motif (a ring-cation contact, not a hydrogen bond)

            # --- Table V add-ons (orientation matters: position i -> position i+4) ---
            # Table V values are G_helix for charged side-chain interactions.
            # The same interaction exists (weaker) in the coil state. We subtract
            # the coil contribution using 1/d Coulomb scaling with Table 6 distances:
            #   ΔG = G_hel × (1 − d_hel / d_coil)
            # Pairs with one uncharged residue use HelixRest/RcoilRest distances.
            # Validated on Huyghues-Despointes 1993 Asp-scan: Q-D⁻ gives -0.227
            # (optimal -0.225, vs paper raw -0.5 which over-stabilises by +8.9%).
            d_hel_4 = float(self.table_6_helix_lacroix.loc['HelixRest', 'i+4'])
            d_coil_4 = float(self.table_6_coil_lacroix.loc['RcoilRest', 'i+4'])
            coil_corr_4 = 1.0 - d_hel_4 / d_coil_4  # ≈ 0.455

            # FYW (i) with His+ (i+4): -0.4 kcal/mol when His is at C1 or C-cap; otherwise divide by 3.
            # No coil subtraction for this motif: the aromatic ring and the imidazolium are in contact only when
            # both side chains lie on one face of the helix, so the interaction is short-range and scarcely
            # formed in the coil. The 1/d coil share above assumes a Coulomb interaction that persists in the
            # coil, which does not describe a contact. With the full value the C-peptide family (Shoemaker et
            # al. 1987, Fairman et al. 1989, Mitchinson & Baldwin 1986), where Fairman et al. report that His-12+
            # stabilises the helix through Phe-8, is predicted 0.9-1.7 helix points (RMSE) closer.
            if AAi in ["F", "Y", "W"] and AAi4 == "H":
                p_his = abs(self.modified_seq_ionization_hel[idx + 4])  # population of His+
                # His is "C1" if it is the residue just before C-cap; "C-cap" if it is C-cap itself
                his_is_C1_or_Ccap = (idx + 4 == self.ccap_idx) or (idx + 4 == self.ccap_idx - 1)
                val = -0.4 if his_is_C1_or_Ccap else (-0.4 / 3.0)
                aro_his += p_his * val

            # Gln (i) with Asp- (i+4): -0.5 kcal/mol (paper Table V)
            if AAi == "Q" and AAi4 == "D":
                p_asp = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Asp-
                extra += p_asp * (-0.5) * coil_corr_4

            # Glu- (i) with Asn (i+4): -0.5 kcal/mol
            if AAi == "E" and AAi4 == "N":
                p_glu = abs(self.modified_seq_ionization_hel[idx])  # population of Glu-
                extra += p_glu * (-0.5) * coil_corr_4

            # Gln (i) with Glu- (i+4): -0.1 kcal/mol
            if AAi == "Q" and AAi4 == "E":
                p_glu = abs(self.modified_seq_ionization_hel[idx + 4])  # population of Glu-
                extra += p_glu * (-0.1) * coil_corr_4

            # the Gln-Asp-, Glu--Asn and Gln-Glu- motifs are side-chain hydrogen bonds: the hydrogen-bond
            # temperature law; the aromatic-His+ contact is temperature-independent
            energy[idx] = base + self._hbond_temperature(extra) + aro_his

        return energy

    def get_dG_terminals_macrodipole(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate interaction energies between N- and C-terminal backbone charges and the helix macrodipole.
        The energy is added to the residue carrying the macrodipole charge.

        Returns:
            tuple[np.ndarray, np.ndarray]: Interaction energies for N-terminal and C-terminal residues.
        """
        N_term = np.zeros(len(self.seq_list))
        C_term = np.zeros(len(self.seq_list))

        # Calculate the interaction energy between the N-terminal and the helix macrodipole.
        # Distance cutoff: when the terminal is >= 6 coil residues from the helix
        # start (ncap_idx >= 6), the interaction is zero.
        if self.ncap_idx < 6:
            N_term[self.ncap_idx] = self._electrostatic_interaction_energy(
                qi=self.mu_helix,
                qj=self.modified_nterm_ionization_hel,
                r=self.terminal_macrodipole_distance_nterm,
                factor_pi=4.0
            )

        # Calculate the interaction energy between the C-terminal and the helix macrodipole
        c_separation = len(self.seq_list) - 1 - self.ccap_idx
        if c_separation < 6:
            C_term[self.ccap_idx] = self._electrostatic_interaction_energy(
                qi=-self.mu_helix,
                qj=self.modified_cterm_ionization_hel,
                r=self.terminal_macrodipole_distance_cterm,
                factor_pi=4.0
            )

        return N_term, C_term

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
        
    def get_dG_terminals_sidechain_electrost(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate electrostatic interaction energies between terminal backbone charges
        and charged sidechains in the helical segment.

        Uses ionization-adjusted sidechain charges (from seq_ionization) so that
        residues with pKa far from pH contribute proportionally (e.g. Y at pH 4
        is uncharged → zero interaction).  Terminal charges use pH-dependent
        ionization from the pKa solver (modified_nterm/cterm_ionization_hel),
        consistent with the terminal-macrodipole function.

        Only helix-interior charged sidechains participate; flanking coil residues
        are excluded because their helix/coil geometry is identical → ΔG = 0.

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

        # Locality gate (AGADIR/Lacroix-style): only include terminal-sidechain terms
        # when the terminal is in-helix or is the immediate neighbor (N' / C').
        nterm_local = (self.ncap_idx <= 1)
        # Same window at the C-terminus: the free carboxylate at the C-cap or at C'
        # (Lacroix 1998 supplementary Table VI gives distances for both positions).
        cterm_local = (self.ccap_idx >= (n - 2))

        # If neither terminal can contribute, bail early
        if not (nterm_present and nterm_local) and not (cterm_present and cterm_local):
            return energy_N, energy_C

        # Only charged side chains inside the helix (cap positions excluded) at the distances of
        # _assign_terminal_sidechain_distances (Lacroix 1998 supplementary Table VI); a pair that
        # geometry does not model contributes nothing.
        interior_indices = self.helix_indices[1:-1] if len(self.helix_indices) > 2 else []
        for idx in interior_indices:
            if self.seq_list[idx] not in self.neg_charge_aa + self.pos_charge_aa:
                continue
            q_sc = float(self.seq_ionization[idx])
            if q_sc == 0.0:
                continue

            if nterm_present and nterm_local and self.terminal_sidechain_modelled_nterm[idx]:
                q_nterm = float(self.modified_nterm_ionization_hel)  # pH-dependent NH3+ charge
                d_hel = float(self.terminal_sidechain_distances_nterm[idx])
                d_rc = float(self.terminal_sidechain_distances_nterm_rc[idx])
                G_hel = self._electrostatic_interaction_energy(qi=q_nterm, qj=q_sc, r=d_hel, factor_pi=4.0) if d_hel < 40.0 else 0.0
                G_rc = self._electrostatic_interaction_energy(qi=q_nterm, qj=q_sc, r=d_rc, factor_pi=4.0) if d_rc < 40.0 else 0.0
                energy_N[idx] = G_hel - G_rc

            if cterm_present and cterm_local and self.terminal_sidechain_modelled_cterm[idx]:
                q_cterm = float(self.modified_cterm_ionization_hel)
                d_hel = float(self.terminal_sidechain_distances_cterm[idx])
                d_rc = float(self.terminal_sidechain_distances_cterm_rc[idx])
                G_hel = self._electrostatic_interaction_energy(qi=q_cterm, qj=q_sc, r=d_hel, factor_pi=4.0) if d_hel < 40.0 else 0.0
                G_rc = self._electrostatic_interaction_energy(qi=q_cterm, qj=q_sc, r=d_rc, factor_pi=4.0) if d_rc < 40.0 else 0.0
                energy_C[idx] = G_hel - G_rc

        return energy_N, energy_C

    def get_dG_ionization(self) -> float:
        """
        Ionisation free energy of the helical segment relative to the coil (kcal/mol): the part of
        the mean-field electrostatic free energy that the q-weighted energy terms leave out (see
        _assign_modified_ionization_states). Not scaled by temperature beyond RT.
        """
        return float(self.dG_ionization)

    def get_dG_terminal_terminal_electrost(self) -> float:
        """
        Calculate the electrostatic interaction between the free N-terminal
        backbone charge (NH3+) and the free C-terminal backbone charge (COO-).

        No such interaction is modelled, so this is always 0.0.

        Returns:
            float: The terminal-terminal electrostatic free energy in kcal/mol.
        """
        # No terminal-to-terminal interaction is computed.  On free/free poly-alanine,
        # which has no charged side chains to confuse the comparison, this function
        # returned -0.12952 where the calibration requires 0.
        #
        # This term and the C-terminal/side-chain term were a compensating pair: this one
        # was spurious, that one was disabled, and on doubly-free peptides the first stood
        # in for the second (-0.1487 against a required -0.1532 at pH 4).  Removing either
        # alone made the fit worse.  Both are corrected together.
        return 0.0

    def get_dG_sidechain_sidechain_electrost(self) -> np.ndarray:
        """
        Calculate the electrostatic free energy contribution for charged residue sidechains
        inside and outside the helical segment, using Lacroix et al. (1998) equations.
        Half of the energy is added to each of the charged sidechain residues.

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

            # gbp: the GB pair law, E = E_nn + q1 a1 + q2 a2 + q1 q2 U in each state (_assign_gb_pairs)
            c_h = self.gb_pairs[True].get((idx1, idx2))
            if c_h is not None:
                c_r = self.gb_pairs[False][(idx1, idx2)]
                q1h, q2h = self.modified_seq_ionization_hel[idx1], self.modified_seq_ionization_hel[idx2]
                q1r, q2r = self.modified_seq_ionization_rc[idx1], self.modified_seq_ionization_rc[idx2]
                G_hel = c_h[3] + q1h * c_h[1] + q2h * c_h[2] + q1h * q2h * c_h[0]
                G_rc = c_r[3] + q1r * c_r[1] + q2r * c_r[2] + q1r * q2r * c_r[0]
                energy_matrix[idx1, idx2] = energy_matrix[idx2, idx1] = (G_hel - G_rc) / 2
                continue

            # Get the distances between the charged sidechains
            helix_dist = self.sidechain_sidechain_distances_hel[idx1, idx2]
            coil_dist = self.charged_sidechain_distances_rc[idx2, idx1]

            # A pair the helical-state model does not consider is assigned 99 A: of the
            # residues outside the helix only the caps and N'/C' interact with helical
            # residues (Lacroix 1998), pairs straddling the helix are not modelled, and
            # pairs >= 13 apart are out of range.  Such a pair has no modelled change between
            # the two states, so it contributes nothing.  Subtracting its coil-state energy
            # alone would charge the helix for an interaction it was never allowed to keep.
            if helix_dist >= 99:
                continue
                
            # Get the ionization states of the charged sidechains
            q1_hel = self.modified_seq_ionization_hel[idx1]
            q2_hel = self.modified_seq_ionization_hel[idx2]
            q1_rc = self.modified_seq_ionization_rc[idx1]
            q2_rc = self.modified_seq_ionization_rc[idx2]

            # Calculate electrostatic interaction energies with adjusted ionization states, Lacroix Eq 6.
            # factor_pi=4.0: calibrated from YGGS AA reference (scsc-only segments).
            G_hel = self._electrostatic_interaction_energy(qi=q1_hel, qj=q2_hel, r=helix_dist, factor_pi=4.0)
            G_rc = self._electrostatic_interaction_energy(qi=q1_rc, qj=q2_rc, r=coil_dist, factor_pi=4.0)

            # Store half the energy difference in both triangles of the matrix (give half of the energy to each sidechain)
            energy_diff = (G_hel - G_rc) / 2
            energy_matrix[idx1, idx2] = energy_diff  # Upper triangle
            energy_matrix[idx2, idx1] = energy_diff  # Lower triangle

        return energy_matrix
    