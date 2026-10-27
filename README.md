# pyAGADIR: α-helix prediction for peptides

pyAGADIR is a Python implementation of AGADIR, the statistical-mechanical model of α-helix formation in monomeric peptides by Muñoz & Serrano (1994, 1995, 1997) and Lacroix et al. (1998). From a peptide sequence and the solution conditions (temperature, pH, ionic strength), it predicts the helicity of each residue and of the whole peptide.

The energy parameters come from the supplementary tables of Lacroix et al. (1998). Where a value departs from the published tables, [`pyagadir/params/README.md`](pyagadir/params/README.md) gives the change and the measurements behind it.


## Installation

pyAGADIR needs Python 3.10 or later. Its only dependencies are NumPy and pandas. Install it from GitHub:

```bash
pip install git+https://github.com/mengqvist/AGADIR.git
# or, in a uv project:
uv add git+https://github.com/mengqvist/AGADIR.git
```

Do not run `pip install pyagadir`: the package of that name on PyPI is the original 2023 version this project was forked from (see [License](#license)), not this one.

### Development

```bash
git clone https://github.com/mengqvist/AGADIR.git
cd AGADIR
uv sync          # editable install plus pytest, matplotlib and the analysis tools
uv run pytest    # about 2 minutes
```

Without uv, `pip install -e . pytest matplotlib` in a virtual environment does the same for the package and tests.


## Predicting helicity

```python
from pyagadir import AGADIR

model = AGADIR(T=4.0, M=0.15, pH=7.0)  # °C, ionic strength in mol/L, pH (these are the defaults)
result = model.predict("ILKSLEEFLKVTLRSTRQT", ncap="Ac", ccap="Am")

print(result.get_percent_helix())                # mean helicity of the peptide, %
print(result.get_helical_propensity().round(1))  # helicity of each position, %
```

```text
Predicting helical propensity for sequence: ILKSLEEFLKVTLRSTRQT, method: 1s, T(C): 4.0, M: 0.15, pH: 7.0, ncap: Ac, ccap: Am
28.55
[ 0.   5.3  5.6  6.   6.3 57.  57.2 57.4 57.6 57.  52.4 47.8 45.  41.8
 21.6 10.1  7.4  4.6  1.4  0.8  0. ]
```

- **Units.** Both results are percentages (0-100). The per-position array includes the Ac and Am positions, so it has two more entries than the sequence. `get_percent_helix()` averages over the amino acids only. `predict()` prints the conditions it ran at, as shown above.
- **Input.** Sequences use the 20 standard one-letter codes, in either case. Together with its caps, a peptide must span at least 6 positions. The temperature must be at least 0 °C, the pH between 0 and 14, and the ionic strength 0 or more.
- **Terminal modifications.** `ncap="Ac"` (acetyl) or `"Sc"` (succinyl), and `ccap="Am"` (amide). Each one occupies a position in the sequence and can act as a capping residue. Acetyl and amide remove the terminal charge; succinyl removes the amine's positive charge but adds a negative one. Use them to match capped synthetic peptides. For a helix cut out of a protein, use Ac and Am: the helix in the protein has no free terminal charges.
- **Method.** `method="1s"` (the default) is AGADIR's one-sequence approximation, which allows at most one helical segment per molecule. A residue's helicity is the summed statistical weight of the segments in which it is helical, divided by the partition function (the weights of all segments, with the coil counting 1). `method="r"` divides by each residue's own sum instead and ignores the competition between segments, so it always gives equal or higher values. It is kept for comparison; the validation below uses `"1s"`.

### Helix-dipole electrostatics

The helix macrodipole is a +1/2 charge at the N-terminal end of the helix and a -1/2 charge at the C-terminal end (Hol et al. 1978). Every charge feels it by the law of every other charge pair in the model, screened Coulomb in water: charged side chains inside the helix feel both ends at their tabulated distances (Lacroix et al. 1998, Table VII), flanking side chains and the free terminal groups the nearby end. Table 1's values for charged residues were measured with Muñoz & Serrano's (1995) near-end law in the loop; each is carried over to this rule at its reference measurement (`params/macrodipole_transfer.tsv`). The derivation and its effect are in [`pyagadir/params/README.md`](pyagadir/params/README.md), "Helix macrodipole".

### Charged side-chain pairs

An acid-base pair three or four residues apart interacts through a hydrogen bond (not screened by salt) and through the
Coulomb attraction of the two charges (screened). Each
side chain is in one state at a time: a set of contacts with partners three or four residues away, or its N-terminal
capping motif, or nothing. A contact carries both the bond and the close-range attraction. Which contacts one side chain
can hold together follows the helix geometry: a Lys can pair with two acids only on its C-terminal side, and no side
chain reaches partners on opposite faces at once (Wolny et al. 2017; Iqbalsyah & Doig 2005). The helix sums over all
these states rather than picking one network. Each isolated pair keeps the free energy measured for it (Smith & Scholtz
1998; Iqbalsyah & Doig 2005; Huyghues-Despointes et al. 1993), with a separate value for the bond when one partner is
neutral; charge-dense block repeats such as EEEEKKKK are predictions, not fitted. The derivation, the pair strengths beside the published values, and the tests are
in [`pyagadir/params/README.md`](pyagadir/params/README.md), "Charged side-chain pairs".

### Energy terms of one helical segment

`EnergyCalculator` gives the free-energy terms of a single helical segment, the quantities AGADIR sums over all segments. Energies are in kcal/mol, and negative values favour the helix. A segment is given by `i`, the index of its N-cap, and `j`, its length including both caps. Indices are 0-based and count the Ac/Sc and Am positions. For `Ac-ILKSLEEFLKVTLRSTRQT-Am` with `i=1, j=10`:

```text
index     0   1     2   3   4   5   6   7   8   9   10    11  ...  20
residue   Ac  I     L   K   S   L   E   E   F   L   K     V   ...  Am
role      N'  Ncap  N1  N2  N3  N4  C4  C3  C2  C1  Ccap  C'
```

```python
from pyagadir.energies import EnergyCalculator

calc = EnergyCalculator("ILKSLEEFLKVTLRSTRQT", i=1, j=10, pH=7.0, T=4.0, ionic_strength=0.15,
                        ncap="Ac", ccap="Am")
calc.get_dG_Int()    # intrinsic helix propensity of each position
calc.get_dG_Hbond()  # main chain-main chain hydrogen bonds
```

| Method | Term | Returns |
| --- | --- | --- |
| `get_dG_Int()` | Intrinsic helix propensities | array, per position |
| `get_dG_Hbond()` | Main chain-main chain hydrogen bonds | float |
| `get_dG_Ncap()`, `get_dG_Ccap()` | N- and C-capping | array |
| `get_dG_staple()` | Hydrophobic staple (N'-N4) | float |
| `get_dG_schellman()` | Schellman motif (C-terminal Gly) | float |
| `get_dG_petukhov_motif()` | Free N-terminus, capping box and Asp/Glu at N4 | float |
| `get_dG_charged_staple()` | Lys/Arg at N4 behind a Ser/Thr N-cap | float |
| `get_dG_i3()`, `get_dG_i4()` | Side chain-side chain, i,i+3 and i,i+4 | array |
| `get_dG_sidechain_sidechain_electrost()` | Charged side-chain pairs | matrix |
| `get_dG_terminals_sidechain_electrost()` | Terminal charges with side chains | (N, C) arrays |
| `get_dG_terminals_macrodipole()` | Terminal charges with the helix dipole | (N, C) arrays |
| `get_dG_sidechain_macrodipole()` | Side chains with the helix dipole | (N, C) arrays |


## Protein structures

`pyagadir.pdb_elements` splits one chain of a PDB file into helices, strands, loops and other elements and scores each one. Secondary structure comes from a built-in DSSP, so no external programs are needed. Helices get AGADIR's helicity as isolated peptides, with two flanking residues on each side and Ac/Am ends. Strands and loops get empirical features from studies of designed mini-proteins (Rocklin et al. 2017; Kim et al. 2022; Tsuboyama et al. 2023). These are in their sources' units (log-odds, kcal/mol, Å²), not folding free energies. The top of [`pyagadir/element_scores.py`](pyagadir/element_scores.py) describes each score.

```bash
python -m pyagadir.pdb_elements protein.pdb --chain A --json scores.json
```

`--T`, `--pH` and `--M` set the AGADIR conditions (defaults 25 °C, pH 7, 0.15 M), and `--help` lists the other options. From Python:

```python
from pyagadir.pdb_elements import score_structure

report = score_structure("protein.pdb")
for element in report["elements"]:
    print(element.kind, element.first_residue, element.last_residue, element.scores)
print(report["summary"])
```

A clone of the repository has two example structures in `tests/data/structures/`.


## Validation

The validation data are in `data/`, so these commands run from a clone set up as under [Development](#development). Each takes about 2 minutes.

```bash
uv run python -m pyagadir.validation             # writes the comparison plots to figures/
uv run python -m pyagadir.scoring --metric rmse  # one score per panel; also mse, mae, pearsonr
```

`score_all()` and the `score_*` functions in `pyagadir/scoring.py` take the same `metric=` argument and can be called from an optimiser. The reference data were digitised from the papers with WebPlotDigitizer.

Lacroix et al. (1998), Fig. 3b: pH dependence of helicity.

<img src="figures/lacroix_figure_3b.png" width="400" alt="Lacroix et al. 1998, Figure 3b">

Lacroix et al. (1998), Fig. 4a-h: pH titrations.

<img src="figures/lacroix_figure_4.png" width="800" alt="Lacroix et al. 1998, Figure 4">

Huyghues-Despointes et al. (1993), Fig. 1: a single Asp at each position, at pH 2 and pH 7.

<img src="figures/huygues_despointes_figure_1a.png" width="800" alt="Huyghues-Despointes et al. 1993, Figure 1">

Muñoz & Serrano (1997), Fig. 4A-C: repeats of AAQAA, AAKAA and AEAAKA.

<img src="figures/munoz_1997_figure_4.png" width="400" alt="Muñoz & Serrano 1997, Figure 4">

Muñoz & Serrano (1995, paper III), Fig. 3: temperature dependence. Panel 3D is replaced by the designed peptides of Scholtz et al. (1991) and Marqusee & Baldwin (1987).

<img src="figures/munoz_1995_figure_3.png" width="800" alt="Muñoz & Serrano 1995, Figure 3">


## References

**AGADIR**

- Muñoz, V. & Serrano, L. (1994). Elucidating the folding problem of helical peptides using empirical parameters. *Nat. Struct. Biol.* 1, 399-409. https://doi.org/10.1038/nsb0694-399
- Muñoz, V. & Serrano, L. (1995). Elucidating the folding problem of helical peptides using empirical parameters. II. Helix macrodipole effects and rational modification of the helical content of natural peptides. *J. Mol. Biol.* 245, 275-296. https://doi.org/10.1006/jmbi.1994.0023
- Muñoz, V. & Serrano, L. (1995). Elucidating the folding problem of helical peptides using empirical parameters. III. Temperature and pH dependence. *J. Mol. Biol.* 245, 297-308. https://doi.org/10.1006/jmbi.1994.0024
- Muñoz, V. & Serrano, L. (1997). Development of the multiple sequence approximation within the AGADIR model of α-helix formation: comparison with Zimm-Bragg and Lifson-Roig formalisms. *Biopolymers* 41, 495-509. https://doi.org/10.1002/(SICI)1097-0282(19970415)41:5%3C495::AID-BIP2%3E3.0.CO;2-H
- Lacroix, E., Viguera, A. R. & Serrano, L. (1998). Elucidating the folding problem of α-helices: local motifs, long-range electrostatics, ionic-strength dependence and prediction of NMR parameters. *J. Mol. Biol.* 284, 173-191. https://doi.org/10.1006/jmbi.1998.2145

**Validation data and electrostatics**

- Baker, E. G. et al. (2015). Local and macroscopic electrostatic interactions in single α-helices. *Nat. Chem. Biol.* 11, 221-228. https://doi.org/10.1038/nchembio.1739
- Huyghues-Despointes, B. M. P., Scholtz, J. M. & Baldwin, R. L. (1993). Effect of a single aspartate on helix stability at different positions in a neutral alanine-based peptide. *Protein Sci.* 2, 1604-1611. https://doi.org/10.1002/pro.5560021006
- Huyghues-Despointes, B. M. P., Scholtz, J. M. & Baldwin, R. L. (1993). Helical peptides with three pairs of Asp-Arg and Glu-Arg residues in different orientations and spacings. *Protein Sci.* 2, 80-85.
- Iqbalsyah, T. M. & Doig, A. J. (2005). Anticooperativity in a Glu-Lys-Glu salt bridge triplet in an isolated α-helical peptide. *Biochemistry* 44, 10449-10456. https://doi.org/10.1021/bi0508690
- Marqusee, S. & Baldwin, R. L. (1987). Helix stabilization by Glu-...Lys+ salt bridges in short peptides of de novo design. *Proc. Natl. Acad. Sci. USA* 84, 8898-8902. https://doi.org/10.1073/pnas.84.24.8898
- Scholtz, J. M., Qian, H., York, E. J., Stewart, J. M. & Baldwin, R. L. (1991). Parameters of helix-coil transition theory for alanine-based peptides of varying chain lengths in water. *Biopolymers* 31, 1463-1470. https://doi.org/10.1002/bip.360311304
- Smith, J. S. & Scholtz, J. M. (1998). Energetics of polar side-chain interactions in helical peptides: salt effects on ion pairs and hydrogen bonds. *Biochemistry* 37, 33-40.
- Wolny, M. et al. (2017). Characterization of long and stable de novo single alpha-helix domains provides novel insight into their stability. *Sci. Rep.* 7, 44341. https://doi.org/10.1038/srep44341

**Structure scores**

- Rocklin, G. J. et al. (2017). Global analysis of protein folding using massively parallel design, synthesis, and testing. *Science* 357, 168-175. https://doi.org/10.1126/science.aan0693
- Kim, T.-E. et al. (2022). Dissecting the stability determinants of a challenging de novo protein fold using massively parallel design and experimentation. *Proc. Natl. Acad. Sci. USA* 119, e2122676119. https://doi.org/10.1073/pnas.2122676119
- Tsuboyama, K. et al. (2023). Mega-scale experimental analysis of protein folding stability in biology and design. *Nature* 620, 434-444. https://doi.org/10.1038/s41586-023-06328-6


## License

MIT; see [LICENSE](LICENSE). pyAGADIR began as a fork of Alex Reis's 2023 implementation, [reisalex/AGADIR](https://github.com/reisalex/AGADIR).
