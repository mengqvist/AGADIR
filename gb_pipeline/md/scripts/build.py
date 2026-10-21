"""Build a solvated, fully helical peptide for OpenMM (N149).

build(seq, ff) -> (topology, positions, forcefield). Termini are free (NH3+, COO-),
as in Escobedo's L3XQ8 segments. Helix: phi -60, psi -40 (their restraint targets).
"""
import io
import math
import random

import numpy as np
import PeptideBuilder
from PeptideBuilder import Geometry
from Bio.PDB import PDBIO
from openmm import CustomTorsionForce, Vec3, unit
from openmm.app import ForceField, Modeller, PDBFile, Topology

FORCEFIELDS = {
    # CHARMM36m (par_all36m_prot.prm, toppar_c36_jul24) with CHARMM TIP3P
    "c36m": ("charmm36_2024.xml", "charmm36_2024/water.xml"),
}


# Side-chain start rotamers, by preset name: PeptideBuilder Geometry attributes (degrees).
# W t90: chi1 180, chi2 +90 (indole flipped relative to t-105); N149 W test.
ROTAMER_PRESETS = {("W", "t90"): {"N_CA_CB_CG_diangle": 180.0, "CA_CB_CG_CD1_diangle": 90.0}}


def helix_pdb(seq, rotamers=None):
    """rotamers: {0-based position: preset name}; other residues keep PeptideBuilder defaults."""
    geos = []
    for i, aa in enumerate(seq):
        g = Geometry.geometry(aa)
        g.phi, g.psi_im1 = -60.0, -40.0
        for attr, val in ROTAMER_PRESETS.get((aa, (rotamers or {}).get(i)), {}).items():
            setattr(g, attr, val)
        geos.append(g)
    s = PeptideBuilder.make_structure_from_geos(geos)
    PeptideBuilder.add_terminal_OXT(s)
    buf = io.StringIO()
    out = PDBIO()
    out.set_structure(s)
    out.save(buf)
    buf.seek(0)
    return PDBFile(buf)


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


AMIDES = {"GLN": ("OE1", "CD", "NE2", "HE21", "HE22"), "ASN": ("OD1", "CG", "ND2", "HD21", "HD22")}


def fix_amide_names(topology, positions):
    """Place the two amide H of Gln/Asn in the amide plane by the PDB convention: HE21 (HD21) trans
    to OE1 (OD1), HE22 (HD22) cis. Modeller.addHydrogens names them arbitrarily (and sometimes
    off-plane), which swaps the observable (Escobedo's Q4 HE21) and CHARMM's 0.32/0.30 charges."""
    for r in topology.residues():
        if r.name not in AMIDES:
            continue
        o, c, n, h1, h2 = AMIDES[r.name]
        i = {a.name: a.index for a in r.atoms()}
        P = {k: np.array(positions[i[k]].value_in_unit(unit.nanometer)) for k in (o, c, n)}
        u = P[n] - P[c]
        u /= np.linalg.norm(u)
        v = (P[o] - P[c]) - np.dot(P[o] - P[c], u) * u
        v /= np.linalg.norm(v)
        for name, side in ((h1, -1.0), (h2, 1.0)):  # 120 deg C-N-H, 0.101 nm
            positions[i[name]] = unit.Quantity(Vec3(*(P[n] + 0.101 * (0.5 * u + side * 0.866 * v))),
                                               unit.nanometer)


def build(seq, ff="c36m", padding_nm=1.2, ionic_M=0.05, rotamers=None):
    pdb = helix_pdb(seq, rotamers)
    forcefield = ForceField(*FORCEFIELDS[ff])
    mod = Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(forcefield, pH=7.0)
    fix_amide_names(mod.topology, mod.positions)
    mod.addSolvent(forcefield, padding=padding_nm * unit.nanometer, boxShape="dodecahedron",
                   ionicStrength=ionic_M * unit.molar, positiveIon="Na+", negativeIon="Cl-")
    return mod.topology, mod.positions, forcefield


def load(pdb_path, ff="c36m"):
    """Reload a system written by build() (start.pdb) for a checkpoint restart."""
    pdb = PDBFile(pdb_path)
    return pdb.topology, pdb.positions, ForceField(*FORCEFIELDS[ff])


# ---------------------------------------------------------------------------------------------
# Stage 1b (N153): Lacroix 4g/4h/4f 16-mers with chosen termini, coil or helix starts, one fixed
# rhombic dodecahedron for every system, Cl- counter-ions only (no added salt).

S1B_PEPTIDES = {"4g": "YGGSAKAAAAAARAAA", "4h": "YGGSKAAAAAAARAAA", "4f": "YGGSAAKAAAAARAAA"}
# variant -> (peptide, N-terminus, Lys form). The C-terminus is always the amide (CHARMM CT2).
# N-terminus: NH3 = free NH3+ (NTER), NH2 = free neutral amine (NNEU), ACE = acetyl (ACE patch).
S1B_VARIANTS = {
    "4g": ("4g", "NH3", "LYS"), "4h": ("4h", "NH3", "LYS"), "4f": ("4f", "NH3", "LYS"),  # Step 1
    "Ac4g": ("4g", "ACE", "LYS"), "Ac4h": ("4h", "ACE", "LYS"), "Ac4f": ("4f", "ACE", "LYS"),  # Step 2
    "4gneu": ("4g", "NH2", "LSN"),  # Step 2, the pH-12 analogue (Tyr and Arg kept in their neutral-pH forms)
}
COIL_BASINS = ((-75.0, 145.0), (-120.0, 130.0))  # PPII and beta (phi, psi), drawn with equal weight
BASIN_SD = 15.0  # degrees, Gaussian spread around each basin centre
HELIX_RES = (5, 15)  # 1-based residues restrained to -60/-40 in the helix state; Ser4 is the N-cap
S1B_BOX_NM = 6.5  # image distance of the dodecahedron, the same for every system (N153 Step 0)
START_MAX_NM = 5.0  # largest backbone + CB span accepted for a start: >= 1.5 nm from its image at 6.5 nm
CLASH_NM = 0.30  # backbone + CB heavy atoms of residues >= 3 apart must be at least this far apart


def _draw(rng, n):
    b = rng.integers(len(COIL_BASINS), size=n)
    return [(COIL_BASINS[k][0] + rng.normal(0, BASIN_SD), COIL_BASINS[k][1] + rng.normal(0, BASIN_SD)) for k in b]


def _peptide_pdb(seq, nterm, angles):
    """Heavy-atom PDB with separate ACE (if nterm == 'ACE') and NH2 residues. angles: (phi, psi) per
    residue of the extended chain (caps included)."""
    ext = ("G" if nterm == "ACE" else "") + seq + "G"
    geos = []
    for i, aa in enumerate(ext):
        g = Geometry.geometry(aa)
        g.phi = angles[i][0]
        g.psi_im1 = angles[i - 1][1] if i else angles[0][1]
        geos.append(g)
    s = PeptideBuilder.make_structure_from_geos(geos)
    buf = io.StringIO()
    out = PDBIO()
    out.set_structure(s)
    out.save(buf)
    keep, first, last = [], 1 if nterm == "ACE" else None, len(ext)
    for line in buf.getvalue().splitlines():
        if not line.startswith("ATOM"):
            continue
        rn, an = int(line[22:26]), line[12:16].strip()
        if rn == first:  # Gly -> acetyl: CA becomes the methyl carbon, N is dropped
            if an not in ("CA", "C", "O"):
                continue
            line = line[:12] + (" CH3" if an == "CA" else line[12:16]) + line[16:17] + "ACE" + line[20:]
        elif rn == last:  # Gly -> amide: keep N only
            if an != "N":
                continue
            line = line[:17] + "NH2" + line[20:]
        keep.append(line)
    return PDBFile(io.StringIO("\n".join(keep) + "\nEND\n"))


def _clash_span(pdb):
    """Backbone + CB heavy atoms: (smallest distance between residues at least 3 apart, largest
    distance), in nm. Both are the same for 4g, 4h and 4f, which differ only beyond CB."""
    names = {"N", "CA", "C", "O", "CB", "CH3"}
    atoms = [(a.residue.index, np.array(pdb.positions[a.index].value_in_unit(unit.nanometer)))
             for a in pdb.topology.atoms() if a.name in names]
    r = np.array([a[0] for a in atoms])
    x = np.array([a[1] for a in atoms])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return d[np.abs(r[:, None] - r[None]) >= 3].min(), d.max()


def s1b_start(variant, state, seed, max_draws=1000):
    """Heavy-atom start for one replica: the first draw that is clash-free and spans <= START_MAX_NM.
    The draw depends on (state, seed, N-terminus) only, not on the peptide: 4g, 4h and 4f replica r
    start from the same backbone. Returns (pdb, angles, draws)."""
    pep, nterm, _ = S1B_VARIANTS[variant]
    seq = S1B_PEPTIDES[pep]
    rng = np.random.default_rng(seed)
    off = 1 if nterm == "ACE" else 0
    for draw in range(1, max_draws + 1):
        ang = _draw(rng, len(seq) + 2)  # one spare entry before and after the peptide
        if state == "helix":
            for i in range(HELIX_RES[0], HELIX_RES[1] + 1):
                ang[i] = (-60.0, -40.0)
        ext = ang[1 - off:len(seq) + 2]  # drop the leading spare when there is no ACE
        pdb = _peptide_pdb(seq, nterm, ext)
        clash, span = _clash_span(pdb)
        if clash >= CLASH_NM and span <= START_MAX_NM:
            return pdb, ext, draw
    raise RuntimeError(f"no clash-free start for {variant} {state} seed {seed}")


def _merge_caps(top, pos, nterm):
    """Move ACE atoms into residue 1 (CHARMM ACE patch names CAY HY1-3 CY OY) and the NH2 atoms into
    the last residue (CT2 names NT HT1 HT2). HT1 (+0.32) is the H cis to O, HT2 (+0.30) trans, from
    the CT2 internal coordinates in top_all36_prot.rtf (CA-C-NT-HT1 = 180). Atom order is kept."""
    ace = {"CH3": "CAY", "H1": "HY1", "H2": "HY2", "H3": "HY3", "C": "CY", "O": "OY"}
    res = list(top.residues())
    P = np.array(pos.value_in_unit(unit.nanometer))
    ct = {a.name: a for a in res[-2].atoms()}
    nh2 = {a.name: a for a in res[-1].atoms()}
    hn = [a for a in res[-1].atoms() if a.element.symbol == "H"]
    # Modeller puts these two H at arbitrary (often off-plane) positions: place them in the amide plane,
    # 120 deg C-N-H, 0.101 nm (as fix_amide_names), HT1 on the O side.
    u = P[nh2["N"].index] - P[ct["C"].index]
    u /= np.linalg.norm(u)
    v = P[ct["O"].index] - P[ct["C"].index]
    v -= np.dot(v, u) * u
    v /= np.linalg.norm(v)
    for h, side in zip(hn, (1.0, -1.0)):
        P[h.index] = P[nh2["N"].index] + 0.101 * (0.5 * u + side * 0.866 * v)
    ct2 = {"N": "NT", hn[0].name: "HT1", hn[1].name: "HT2"}
    new = Topology()
    new.setPeriodicBoxVectors(top.getPeriodicBoxVectors())
    chain = new.addChain()
    amap = {}
    groups = [(res[0], None)] if nterm != "ACE" else [(res[1], res[0])]
    body = res[2:-2] if nterm == "ACE" else res[1:-2]
    for r, cap in groups:
        nr = new.addResidue(r.name, chain, r.id)
        for a in (list(cap.atoms()) if cap else []):
            amap[a] = new.addAtom(ace[a.name], a.element, nr)
        for a in r.atoms():
            amap[a] = new.addAtom(a.name, a.element, nr)
    for r in body:
        nr = new.addResidue(r.name, chain, r.id)
        for a in r.atoms():
            amap[a] = new.addAtom(a.name, a.element, nr)
    nr = new.addResidue(res[-2].name, chain, res[-2].id)
    for a in res[-2].atoms():
        amap[a] = new.addAtom(a.name, a.element, nr)
    for a in res[-1].atoms():
        amap[a] = new.addAtom(ct2[a.name], a.element, nr)
    for b in top.bonds():
        new.addBond(amap[b[0]], amap[b[1]])
    order = sorted(amap, key=lambda a: amap[a].index)
    return new, unit.Quantity([Vec3(*P[a.index]) for a in order], unit.nanometer)


def s1b_build(variant, state, seed, box_nm=S1B_BOX_NM, ff="c36m"):
    """Solvated Stage 1b system. Returns (topology, positions, forcefield, info)."""
    pep, nterm, lys = S1B_VARIANTS[variant]
    seq = S1B_PEPTIDES[pep]
    pdb, angles, draws = s1b_start(variant, state, seed)
    random.seed(seed)  # Modeller places some H and picks the waters replaced by ions with Python's random
    forcefield = ForceField(*FORCEFIELDS[ff])
    mod = Modeller(pdb.topology, pdb.positions)
    variants = [None] * mod.topology.getNumResidues()
    if lys == "LSN":
        for r in mod.topology.residues():
            if r.name == "LYS":
                variants[r.index] = "LYN"
    mod.addHydrogens(pH=7.0, variants=variants)  # no force field here: it cannot type the separate caps
    if nterm == "NH2":  # neutral amine (NNEU): drop one of the three N-terminal H, keep two
        first = next(r for r in mod.topology.residues() if r.name != "ACE")
        mod.delete([next(a for a in first.atoms() if a.name == "H3")])
    top, pos = _merge_caps(mod.topology, mod.positions, nterm)
    if nterm != "ACE":  # free amine: CHARMM names HT1..HT3 (equal charges, so the order is immaterial)
        r0 = next(top.residues())
        n0 = next(a for a in r0.atoms() if a.name == "N")
        hs = [b[0] if b[1] is n0 else b[1] for b in top.bonds() if n0 in b and "H" in (b[0].element.symbol, b[1].element.symbol)]
        for k, a in enumerate(sorted(hs, key=lambda a: a.index)):
            a.name = f"HT{k + 1}"
    mod = Modeller(top, pos)
    d = box_nm
    vecs = (Vec3(d, 0, 0), Vec3(0, d, 0), Vec3(d / 2, d / 2, d * np.sqrt(2) / 2)) * unit.nanometer
    mod.addSolvent(forcefield, boxVectors=vecs, neutralize=True, ionicStrength=0 * unit.molar,
                   positiveIon="Na+", negativeIon="Cl-")
    info = {"variant": variant, "peptide": pep, "sequence": seq, "nterm": nterm, "cterm": "CT2", "lys": lys,
            "state": state, "seed": seed, "draws": draws, "box_nm": d,
            "phi_psi": [f"{p:.1f}/{q:.1f}" for p, q in angles]}
    return mod.topology, mod.positions, forcefield, info


def add_helix_restraints(system, top, k=5.0, residues=HELIX_RES):
    """phi/psi of the given 1-based peptide residues to -60/-40, harmonic, k in kJ mol-1 rad-2 (Stage 0 form)."""
    tors = CustomTorsionForce("0.5*k_hx*d^2; d = dtheta - 2*pi*floor((dtheta + pi)/(2*pi)); dtheta = theta - theta0")
    tors.addGlobalParameter("k_hx", k)
    tors.addGlobalParameter("pi", math.pi)
    tors.addPerTorsionParameter("theta0")
    res = [r for r in top.residues() if r.chain.index == 0]
    at = {(r.index, a.name): a.index for r in res for a in r.atoms()}
    for i in range(residues[0] - 1, residues[1]):  # 0-based peptide residue index (caps are merged)
        tors.addTorsion(at[i - 1, "C"], at[i, "N"], at[i, "CA"], at[i, "C"], [math.radians(-60)])
        tors.addTorsion(at[i, "N"], at[i, "CA"], at[i, "C"], at[i + 1, "N"], [math.radians(-40)])
    system.addForce(tors)
    return tors.getNumTorsions()


def matched_templates(forcefield, top):
    """Template matched to each peptide residue, patches included (uses OpenMM's matcher, as createSystem
    does; the public getMatchingTemplates ignores patches). Returns [(resname, template name)]."""
    data = ForceField._SystemData(top)
    tmpl = forcefield._matchAllResiduesToTemplates(data, top, {}, False)
    return [(r.name, tmpl[r].name) for r in top.residues() if r.chain.index == 0]
