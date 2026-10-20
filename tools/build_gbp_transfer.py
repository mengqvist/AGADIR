"""Transfer the Table 1 values measured with the default electrostatics in the loop to dipole_law="gbp":
params/sidechain_gbp_transfer.tsv.

gbp changes three things the Table 1 values were measured with: the side chain-macrodipole law (GB, as gbx), the side
chain-side chain pair law (GB pairs in place of point charges) and the coupling of each ionisable group to both (each
protonation form's own GB energy). At every reference measurement of tools/build_gb_transfer.REFS (the source's host
with the guest in the class, its temperature, ionic strength and pH), the offsets make the gbp model reproduce the
default model's guest-minus-host difference of the segment free energy, weighted over the segments that place the
guest in the class by their statistical weights in the default model (as tools/build_gb_transfer.py):
    ddG = [dG(gbp) - dG(default)](guest) - [dG(gbp) - dG(default)](Ala host)  ->  0
The guest's interactions with charged host residues (its pair electrostatics and its i,i+3 / i,i+4 Table IV terms with
them) are left out of both sides (--with-partners keeps them), so the offsets carry only the guest's own
electrostatics and the pair law governs its partners in the Table 1 reference hosts as everywhere else.
- dphi (kcal/mol per +1 charge, added to the GB potential, as gbx's offsets): a change moves ddG by z theta (theta:
  the guest's charged fraction in the helix), so dphi -= ddG / (z theta);
- where gbp makes the guest neutral at its reference although the default model has it charged (theta_gbp < 0.25 and
  theta_default > theta_gbp + 0.25, at any iteration; His at the N-cap at pH 5 needed it before the desolvation
  term), no per-charge offset can reproduce the value (the guest's free energy is then bounded by its neutral
  state's). There dphi matches the guest's
  site potential psi (its charged minus neutral state's energy, at the converged charges) to the default model's,
  dphi += (psi_default - psi_gbp) / z, so its ionisation at its reference measurement is the default model's, and a
  constant (kcal/mol, added at every charge) matches the free energy: const -= ddG;
- neutral offset (class "Neutral", Lys, His, Asp, Glu; added to the neutral form's energy, so it acts on the neutral
  state only): Table 1's single Neutral value, an interior one, at the Ncen host (Chakrabartty et al. 1994) at a pH
  where the guest is neutral (Asp/Glu pH 2, His pH 9, Lys pH 12.5): offset -= ddG / (1 - theta).
All are solved together by iteration, starting from the current transfer file's values (all 0 for a first build),
until every |ddG| < 0.0005 kcal/mol and |dpsi| < 0.001.
Run: python tools/build_gbp_transfer.py [--out pyagadir/params/sidechain_gbp_transfer.tsv]   (the gbp data files in
energies.GBP_FILES, apart from the transfer itself, must be the ones the offsets are for)
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gb_transfer as BT  # noqa: E402
import pyagadir.energies as E  # noqa: E402
from pyagadir.energies import EnergyCalculator, gbx_position_class  # noqa: E402
from pyagadir.models import AGADIR  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "pyagadir/params/sidechain_gbp_transfer.tsv"
NEUTRAL_PH = {"D": 2.0, "E": 2.0, "H": 9.0, "K": 12.5}


def segments(cls, aa, ref, pH):
    seq = ref["seq"].replace("X", aa)
    idx = ref["pos"] - 1 + (1 if ref["ncap"] else 0)
    m = BT._Recorder(method="1s", T=ref["T"], M=ref["I"], pH=pH)
    m.weights = {}
    with contextlib.redirect_stdout(io.StringIO()):
        m.predict(seq, ncap=ref["ncap"], ccap=ref["ccap"])
    sl = ([ref["ncap"]] if ref["ncap"] else []) + list(seq) + ([ref["ccap"]] if ref["ccap"] else [])
    segs = [(i, j, K) for (i, j), K in m.weights.items()
            if i <= idx <= i + j - 1 and gbx_position_class(aa, idx - i, i + j - 1 - idx, sl[i]) == cls]
    if not segs:
        raise RuntimeError(f"no segment places {aa} in class {cls}")
    return seq, idx, segs


CHARGED = set("KRHDE")
EXCLUDE_PARTNERS = True  # --with-partners turns it off


def guest_partner_terms(ec, idx):
    """The guest's interactions with charged host residues in one segment: its pair electrostatics (both halves of
    get_dG_sidechain_sidechain_electrost) and its i,i+3 / i,i+4 Table IV terms with charged partners (kcal/mol)."""
    if idx is None:
        return 0.0
    seq = ec.seq_list
    m = ec.get_dG_sidechain_sidechain_electrost()
    out = float(np.sum(m[idx, :]) + np.sum(m[:, idx]))
    i3, i4 = ec.get_dG_i3(), ec.get_dG_i4()
    for k, e in ((3, i3), (4, i4)):
        for a, b in ((idx - k, idx), (idx, idx + k)):
            if 0 <= a and b < len(seq) and (seq[b] if a == idx else seq[a]) in CHARGED:
                out += float(e[a])
    return out


def seg_dG(seq, pH, segs, ref, law, idx=None):
    """K-weighted means over the segments: free energy, and the guest's helix-state charged fraction and site
    potential (the energy of its charged minus its neutral state at the converged charges, kcal/mol). With
    EXCLUDE_PARTNERS the guest's pair and acid-base terms with charged host residues are left out of the free energy
    (both laws alike), so the offsets carry only the guest's own electrostatics; the pair law then governs its partners
    in the Table 1 reference host as everywhere else."""
    a = AGADIR(method="1s", T=ref["T"], M=ref["I"], pH=pH, dipole_law=law)
    with contextlib.redirect_stdout(io.StringIO()):
        a.predict(seq, ncap=ref["ncap"], ccap=ref["ccap"])
    a.debug = False
    params = EnergyCalculator.snapshot_params(law)
    num = den = th = ps = 0.0
    for i, j, K in segs:
        a.energy_calculator = EnergyCalculator(a.result.seq, i, j, pH, ref["T"], ref["I"], ncap=ref["ncap"],
                                               ccap=ref["ccap"], params=params, dipole_law=law)
        num += K * (float(a._calc_dG_Hel(i=i, j=j))
                    - (guest_partner_terms(a.energy_calculator, idx) if EXCLUDE_PARTNERS else 0.0))
        if idx is not None:
            th += K * abs(float(a.energy_calculator.modified_seq_ionization_hel[idx]))
            ps += K * float(a.energy_calculator.site_potential_hel[idx])
        den += K
    return num / den, th / den, ps / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--with-partners", action="store_true", help="match the guest's partner terms too (leaks the pair "
                    "laws' differences at the references into the offsets)")
    a = ap.parse_args()
    global EXCLUDE_PARTNERS
    EXCLUDE_PARTNERS = not a.with_partners
    tab = E.gbp_tables()
    work = []  # (kind, cls, aa, ref, pH, seq, host, idx, segs, default ddG part, default psi, default theta)
    for cls, ref in BT.REFS.items():
        for aa in ref["res"]:
            pH = ref.get("pH_res", {}).get(aa, ref["pH"])
            seq, idx, segs = segments(cls, aa, ref, pH)
            host = ref["seq"].replace("X", "A")
            g = seg_dG(seq, pH, segs, ref, "munoz", idx)
            d0 = g[0] - seg_dG(host, pH, segs, ref, "munoz")[0]
            work.append(("charged", cls, aa, ref, pH, seq, host, idx, segs, d0, g[2], g[1]))
    for aa, pH in NEUTRAL_PH.items():
        ref = BT.REFS["Ncen"]
        seq, idx, segs = segments("Ncen", aa, ref, pH)
        host = ref["seq"].replace("X", "A")
        d0 = seg_dG(seq, pH, segs, ref, "munoz", idx)[0] - seg_dG(host, pH, segs, ref, "munoz")[0]
        work.append(("neutral", "Neutral", aa, ref, pH, seq, host, idx, segs, d0, None, 0.0))
    mode = {}
    for it in range(a.max_iter):
        worst, wpsi, res = 0.0, 0.0, []
        for kind, cls, aa, ref, pH, seq, host, idx, segs, d0, p0, t0 in work:
            g, th, ps = seg_dG(seq, pH, segs, ref, "gbp", idx)
            h = seg_dG(host, pH, segs, ref, "gbp")[0]
            dd = (g - h) - d0
            # switched (once, permanently) when gbp has the guest neutral where the default has it charged, at the
            # first evaluation or once the per-charge iteration has driven it there
            if mode.get((cls, aa)) != "const":
                mode[(cls, aa)] = "const" if kind == "charged" and th < 0.25 and t0 > th + 0.25 else "charge"
            dp = (p0 - ps) if mode[(cls, aa)] == "const" else 0.0
            res.append((kind, cls, aa, dd, th, dp))
            worst, wpsi = max(worst, abs(dd)), max(wpsi, abs(dp))
        for kind, cls, aa, dd, th, dp in res:
            if kind == "neutral":
                tab["neutral_offset"][aa] = tab["neutral_offset"].get(aa, 0.0) - dd / max(1.0 - th, 0.5)
            elif mode[(cls, aa)] == "const":
                tab["transfer"][(aa, cls)] = tab["transfer"].get((aa, cls), 0.0) + dp / BT.Z[aa]
                tab["const"][(aa, cls)] = tab["const"].get((aa, cls), 0.0) - dd
            else:
                tab["transfer"][(aa, cls)] = tab["transfer"].get((aa, cls), 0.0) - dd / (BT.Z[aa] * max(th, 0.2))
        print(f"iteration {it}: max |ddG| {worst:.4f} kcal/mol, max |dpsi| {wpsi:.4f}", flush=True)
        if worst < 0.0005 and wpsi < 0.001:
            break
    else:
        for kind, cls, aa, dd, th, dp in res:
            if abs(dd) > 0.0005 or abs(dp) > 0.001:
                print(f"  unconverged {aa} {cls}: ddG {dd:+.4f}, dpsi {dp:+.4f}, theta {th:.3f}", flush=True)
    for (cls, aa), m in mode.items():
        if m == "const":
            print(f"  {aa} {cls}: neutral under gbp at its reference; potential matched + constant", flush=True)
    lines = ["# dipole_law=\"gbp\" transfer (tools/build_gbp_transfer.py): dphi (kcal/mol per +1 charge) added to the GB "
             "potential and const (kcal/mol) added at every charge, per Table 1 class; class Neutral: dphi = the offset "
             f"(kcal/mol) of the neutral form's energy. gbp files: pairs {Path(str(E.GBP_FILES['pairs'])).name}, "
             f"neutral {Path(str(E.GBP_FILES['neutral'])).name}, desolv {Path(str(E.GBP_FILES['desolv'])).name}, "
             f"overrides {Path(str(E.GBP_FILES['overrides'])).name}; "
             f"last max |ddG| {worst:.4f}, |dpsi| {wpsi:.4f}",
             "residue\tcls\tdphi\tconst\treference\tT_C\tI_M\tpH"]
    for kind, cls, aa, ref, pH, *_ in work:
        if kind == "neutral":
            v, c = tab["neutral_offset"][aa], 0.0
        else:
            v, c = tab["transfer"][(aa, cls)], tab["const"][(aa, cls)]
        lines.append(f"{aa}\t{cls}\t{v:.4f}\t{c:.4f}\t{ref['src']}\t{ref['T']:g}\t{ref['I']:.5f}\t{pH:g}")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
