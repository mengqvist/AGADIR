"""Transfer the Table 1 values that were measured with the default side chain-macrodipole law in the loop to the
Generalized Born law: params/sidechain_dipole_gb_transfer.tsv, the potential offsets of dipole_law="gbx".

A Table 1 value for a charged residue in a given class of positions is, in effect, its measurement minus the default
law's electrostatics at that measurement's conditions. The GB law (params/sidechain_dipole_gb.tsv) contains all of
the side chain's electrostatics with the backbone, so used as is it counts that part twice. For every such value,
at its reference measurement (the source's host peptide with the residue in the class, at the source's temperature
and ionic strength):

    dphi(X, class) = phi_default - phi_gb      (kcal/mol per +1 charge, EnergyCalculator.sidechain_dipole_potential)

and dipole_law="gbx" uses phi_gb + dphi(X, class) wherever the GB table applies, so energy and ionisation equal the
default model at each reference while the GB law governs elsewhere (other positions in the class, salts, lengths,
temperatures). The table-equivalent change of the charged cell is z * dphi (z = -1 Asp/Glu/Cys/Tyr, +1 Lys/Arg/His).

The guest sits in many helical segments of its host: dphi is the mean over the segments that place it in the class,
weighted by each segment's statistical weight in the default model (the reference helicity is preserved to first
order). Classes follow energies.gbx_position_class (get_dG_Int's column choice; N1-N4 per N-cap type; the capping
box). References:
- N-cap (D E K R H C Y): Doig & Baldwin 1995 Table 1, NH2-XAKAAAAKAAAAKAAGY-CONH2, 0 C, 1 M NaCl (Cys, Tyr at pH 9.55,
  where they are ionised; the others at pH 5);
- C-cap (D E K R H): Doig & Baldwin 1995 Table 3, Ac-YGAAKAAAAKAAAAKAX-CO2H, 0 C, 1 M, pH 7;
- Ncen, also C1-C3 and Lys/Arg/His N1-N4 (D E K R H): Chakrabartty et al. 1994, Ac-YGGKAAAAKAXAAKAAAAK-NH2, 0 C, 1 M;
- N1-N4 (D E): Petukhov et al. 1998's host Ac-AAAAAAAARAAARGGY-NH2 with the guest at Nk, 5 C, 5 mM sodium phosphate
  pH 7; residue N-caps ("N1r".."N4r"): the same host with an Ala N-cap and a free N-terminus;
- N3 behind an S/T/D/N N-cap (Glu, the capping box): Zhou et al. 1994, Ac-YMSEDELKAAEAAFKRHGPT-NH2, 4 C, pH 6,
  0.01 M.
Run: python tools/build_gb_transfer.py [--out pyagadir/params/sidechain_dipole_gb_transfer.tsv]
"""
import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from pyagadir.energies import EnergyCalculator, gbx_position_class
from pyagadir.models import AGADIR
from pyagadir.validation import titrated_ionic_strength

OUT = Path(__file__).resolve().parents[1] / "pyagadir/params/sidechain_dipole_gb_transfer.tsv"
Z = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 1, "C": -1, "Y": -1}
I_PET = titrated_ionic_strength(0.005, 7.0, {"sodium_phosphate_M": 0.005, "prepared_pH": 7.0})

REFS = {
    "Ncap": dict(src="Doig & Baldwin 1995 Table 1", seq="XAKAAAAKAAAAKAAGY", ncap=None, ccap="Am", pos=1, T=0.0,
                 I=titrated_ionic_strength(1.0, 5.0), pH=5.0, res="DEKRHCY", pH_res={"C": 9.55, "Y": 9.55}),
    "Ccap": dict(src="Doig & Baldwin 1995 Table 3", seq="YGAAKAAAAKAAAAKAX", ncap="Ac", ccap=None, pos=17, T=0.0,
                 I=titrated_ionic_strength(1.0, 7.0), pH=7.0, res="DEKRH"),
    "Ncen": dict(src="Chakrabartty et al. 1994", seq="YGGKAAAAKAXAAKAAAAK", ncap="Ac", ccap="Am", pos=11, T=0.0,
                 I=titrated_ionic_strength(1.0, 7.0), pH=7.0, res="DEKRH"),
    "N3cb": dict(src="Zhou et al. 1994", seq="YMSEDXLKAAEAAFKRHGPT", ncap="Ac", ccap="Am", pos=6, T=4.0,
                 I=titrated_ionic_strength(0.01, 6.0), pH=6.0, res="E"),
}
for k in (1, 2, 3, 4):
    s = list("AAAAAAAARAAARGGY")
    s[k - 1] = "X"
    REFS[f"N{k}"] = dict(src="Petukhov et al. 1998 host", seq="".join(s), ncap="Ac", ccap="Am", pos=k, T=5.0,
                         I=I_PET, pH=7.0, res="DE")
    REFS[f"N{k}r"] = dict(src="Petukhov et al. 1998 host, Ala N-cap", seq="A" + "".join(s), ncap=None, ccap="Am",
                          pos=k + 1, T=5.0, I=I_PET, pH=7.0, res="DE")


class _Recorder(AGADIR):
    """AGADIR that records each segment's statistical weight."""

    def _calc_K(self, dG_Hel):
        K = super()._calc_K(dG_Hel)
        self.weights[(self.energy_calculator.i, self.energy_calculator.j)] = K
        return K


def transfer(cls, aa, ref):
    seq = ref["seq"].replace("X", aa)
    pH = ref.get("pH_res", {}).get(aa, ref["pH"])
    off = 1 if ref["ncap"] else 0
    idx = ref["pos"] - 1 + off
    m = _Recorder(method="1s", T=ref["T"], M=ref["I"], pH=pH)
    m.weights = {}
    with contextlib.redirect_stdout(io.StringIO()):
        m.predict(seq, ncap=ref["ncap"], ccap=ref["ccap"])
    sl = ([ref["ncap"]] if ref["ncap"] else []) + list(seq) + ([ref["ccap"]] if ref["ccap"] else [])
    params = EnergyCalculator.snapshot_params()
    num = den = 0.0
    for (i, j), K in m.weights.items():
        if not i <= idx <= i + j - 1 or gbx_position_class(aa, idx - i, i + j - 1 - idx, sl[i]) != cls:
            continue
        phi = {}
        for law in ("munoz", "gb"):
            ec = EnergyCalculator(seq, i, j, pH, ref["T"], ref["I"], ncap=ref["ncap"], ccap=ref["ccap"],
                                  params=params, dipole_law=law)
            phi[law] = float(ec.sidechain_dipole_potential[idx])
        num += K * (phi["munoz"] - phi["gb"])
        den += K
    if den == 0.0:
        raise RuntimeError(f"no segment places {aa} in class {cls}")
    return num / den, pH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    lines = ["# dipole_law=\"gbx\" transfer offsets (tools/build_gb_transfer.py): dphi = phi_default - phi_gb (kcal/mol per "
             "+1 charge) at each Table 1 value's reference; table-equivalent cell change = z * dphi",
             "residue\tcls\tdphi\tcell_change\treference\tT_C\tI_M\tpH"]
    for cls, ref in REFS.items():
        for aa in ref["res"]:
            d, pH = transfer(cls, aa, ref)
            lines.append(f"{aa}\t{cls}\t{d:.4f}\t{Z[aa] * d:+.4f}\t{ref['src']}\t{ref['T']:g}\t{ref['I']:.5f}\t{pH:g}")
            print(lines[-1], flush=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
