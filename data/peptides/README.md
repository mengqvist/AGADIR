# Extracted peptide datasets

Flat, per-paper tables of measured peptide helicity, pulled from the primary literature
under `doc/`. Each `<key>.tsv` has a matching entry in `manifest.json` carrying the
citation, the **verbatim** conditions quote, the caps convention, and the paper's own
ellipticity-to-helicity conversion.

## These are NOT the scored benchmark

`data/validation/` holds the four datasets the investigation scores against, and
`question.md`'s success conditions are defined over those. Nothing here is scored yet.
Bringing a dataset in is a deliberate act — adding several at once would make condition 1
("no dataset regresses") unmeasurable overnight, because every correction would regress
something somewhere.

## Format

One tidy row per (peptide, quantity). Columns:

| column | meaning |
| --- | --- |
| `peptide_id` | the paper's own label |
| `sequence` | bare sequence, caps stripped into their own columns |
| `ncap` / `ccap` | `Ac`, `Sc`, `Am`, or empty for free — **read from the paper's notation, never inferred** |
| `n_res` | length of `sequence` |
| `pH`, `T_C`, `ionic_strength_M` | per row, so each row stands alone. `ionic_strength_M` is the stated salt and buffer; it leaves out the HCl or NaOH that set the pH unless the manifest entry has `ionic_strength_includes_titrant` (see below) |
| `quantity` | what `value` is: `minus_theta222`, `helix_percent`, … |
| `value`, `error`, `unit` | as printed, with the paper's own error where given |
| `source_table` | which table in the paper |

## Two rules these files follow

**Store what the paper reports, not what we derive.** Where a paper gives both
`-[theta]222` and `% helix`, both are stored as separate rows and the conversion is
recorded in the manifest rather than applied. Three different theta_H conventions already
appear in this corpus — Chen 1974's `39,500(1-2.57/n)`, Scholtz 1991's
`40,000(1-2.5/n)`, and a flat `-33,750` — and `munoz1997_figure4A` is wrong *because* a
helicity was re-derived under a mismatched one (`reasoning/nodes/N050.md`@2).

**Every extractor is a script under `reasoning/evidence/scripts/` with a parse-loss
guard.** The first Petukhov run silently dropped five of thirty peptides to a control
character in the PDF text; it produced a smaller table rather than an error. Each extractor
now counts the rows it can see against the rows it emits and raises on a mismatch.

## Published and common-frame helicity

Each row stores what the paper printed. Papers turn [θ]222 into % helix with different baselines (Chen et al. 1974,
Scholtz et al. 1991, Luo & Baldwin 1997, or a laboratory's own pair), and on the same signal these differ by up to 11
helix points (Merutka et al. 1991's fitted −40,500 helix baseline against Scholtz's −34,100 at n = 17). The evaluation
(`reasoning/evidence/scripts/extracted_direct.py`) therefore reports two helicities for every row it reads:

- **published** (`measured_paper`): the printed % or fraction; for a row with only [θ]222, that θ converted with the
  paper's own stated pair (recorded in `manifest.json`).
- **common frame** (`measured_common`): the same measurement on one conversion, Scholtz et al. (1991, *Biopolymers* 31,
  1463–1470) eqs 3–4 with temperature: [θ]H = −40,000(1 − 2.5/n) + 100T, [θ]C = 640 − 45T (n = residues, T in °C),
  defined once in `reasoning/evidence/scripts/helix_frames.py`. It comes from the row's own θ, else from the same
  paper's θ at the same conditions, else from inverting the paper's printed conversion; with none of these it equals the
  published value. `basis_method` says which.

Model development and evaluation use the common frame. The published value is kept on every row and never overwritten,
so either basis can be selected (`HELIX_BASIS=common|paper`). No aromatic-band correction is applied. A Tyr, Phe or Trp at
a chain end can move CD helicity by about ±15 points: in our comparison of Song et al. (2008)'s CD with their 13C′ NMR on
the same peptides, C-terminal GY peptides read 13–16 points low by CD, and a free-amine YGG peptide reads 13 points high.
The size of the shift does not carry over from one host to another.

**Independent support.** The common frame's temperature terms match a global fit to published fully helical peptides and
coiled coils (their Table S3) plus an (AAKAA)n-GY length series: ∂[θ]H∞/∂T = +100 ± 30 and ∂[θ]C/∂T = −45 ± 5 deg cm² dmol⁻¹ °C⁻¹, with
[θ]H∞ = −41,000 ± 1,400 (Zavrtanik, Lah & Hadži 2024, *J. Phys. Chem. B* 128, 2652–2663, Table 1). The steeper +250T
helix slope of Luo & Baldwin (1997, *Biochemistry* 36, 8413–8421) is about twice what stable helices show (Song et al.
2008, *Biopolymers* 89, 747–760; Zavrtanik et al. 2024). **Not supported:** the coil intercept +640. The same fit gives
+2,100 ± 300 (Luo & Baldwin: +2,220) and attributes the lower value to treating short peptides as pure coil. Adopting it
would raise low-helicity rows by up to about 3.5 points; the frame keeps +640 for now. The common frame is a shared
convention, not an absolute scale.

**Units.** `minus_theta222` is per residue throughout. Where a paper reports ellipticity per peptide unit
(N_res + 1 for blocked peptides; Zavrtanik et al. 2024) or per backbone amide (Andersen et al. 1996), the extractor
rescales it to per residue, the one place these files store a derived value. The factor is in the row's `flags` and
in the manifest.

## Figure data

Some datasets are read from figures rather than tables (the C-peptide / S-peptide
analogue papers, Shoemaker 1987, Mitchinson & Baldwin 1986 and Fairman 1989, and the
Cochran 2001 pH titrations). Those rows carry `digitised` in `flags`; the manifest entry
says how the axes were calibrated, how the figure values were checked against any printed
values in the same paper, and the reading uncertainty (about ±200 deg cm2 dmol-1 in
`-[theta]222`). The abscissa is stored as read (pH to 0.05, temperature to 0.5 °C), with
NaCl snapped to the figure's nominal concentrations. Values printed in the same papers
are stored as ordinary rows alongside.

The C-peptide papers state a fourth helicity convention (0 % = +3,000, 100 % = -27,000
deg cm2 dmol-1 at 3 °C); as with the others it is recorded in the manifest, not applied.

## Per-residue NMR data (`nmr/`)

`nmr/` holds per-residue NMR measurements: one row per (peptide, residue), with columns `position` (the
residue number in that peptide's own chain) and `residue` added after `n_res`. `quantity` names what is stored
(for example `CaH_shift_ppm`). These are not helicities. They live in a subdirectory because the scripts that read
`data/peptides/*.tsv` treat every row as one measurement condition and need a pH, and the NMR papers often do
not state one. The manifest key is the path relative to this directory (`nmr/<key>`).

## Measurements the model cannot score (`unscorable/`)

`unscorable/` keeps measurements that the scoring scripts must not read: peptides with a residue the model cannot
represent (ornithine, written `O` in `sequence`, flag `nonstandard_residue`; norleucine and the C-peptide's homoserine
lactone end, written `X` with the residue named in the flag), and figures whose temperature or buffer
the paper does not state (`T_C` left empty). The format is the main one. The scripts glob `data/peptides/*.tsv`
only, so nothing here is scored. The manifest key is the path relative to this directory (`unscorable/<key>`).

## Non-helicity rows in the main files

A few main files carry rows that are not helicities (for example `Tm_C`, a printed melting midpoint, whose
`T_C` is the midpoint itself). The scoring scripts read only `helix_percent`, `helix_fraction`,
`minus_theta222` and the printed free energies below, and skip the rest.

Other estimates a paper prints beside its 222-nm helicity keep their own names and are not scored: spectral fits
(`helix_percent_cd_shape`, a polylysine reference fit, Bruch et al. 1991; `helix_percent_cdsstr` and
`helix_percent_bestsel`, deconvolutions, Süveges et al. 2009 and Varga et al. 2025), a figure that shows a stored
[theta]222 series again under the authors' conversion (`helix_fraction_luo_baldwin`, Sivaramakrishnan et al. 2008
Fig. 1A, the same melt as its SI Fig. S2), helix contents from amide hydrogen-exchange kinetics
(`helix_percent_nh_exchange`, Padmanabhan & Baldwin 1994) and series normalised to one
of their own points (`theta222_ratio_<reference>`, e.g. `theta222_ratio_6C`, `theta222_ratio_0M_NaCl`; a consumer
needs an absolute [theta]222 at the reference to use them).

## Printed free energies (`dG_helix_kcal_mol`)

Park et al. 1993 print two-state free energies instead of ellipticities: `dG_helix_kcal_mol` is their
ΔG = −RT ln([θ]obs − [θ]coil)/([θ]helix − [θ]obs) (helix −35,600, coil +300 at 0 °C), so f = 1/(1 + exp(ΔG/RT)).
Their Fig. 4B is printed relative to pH 2 (`ddG_vs_pH2_kcal_mol`); a consumer adds the same peptide's Table I pH-2 ΔG.
Both are stored as printed, and `reasoning/evidence/scripts/extracted_direct.py` converts them.

## Salt flags

Rows name their salt in `flags` (`KCl 0.5 M`, `CaCl2 0.5 M`). `ionic_strength_M` is the ionic strength (3c for
CaCl2), and rows with a divalent salt also carry `divalent_salt`, so a test of divalent against monovalent salt at equal
ionic strength can select them (Park et al. 1993, Fig. 2A). `same_samples_as=<source>` marks a point that shows a sample
already stored from another table or figure. Scorers exclude it, as they exclude `reprint=` rows. `extrapolated`
(in `unscorable/`) marks a value the authors extrapolated to conditions they did not measure.

## Ionic strength and the titrant

Papers state their salt and buffer, rarely the HCl or NaOH that brought each sample to its pH. That titrant raises the
ionic strength at the pH extremes: at pH 2 the free HCl alone adds 10 mM. `ionic_strength_M` records the stated salt
and buffer. A consumer adds the free acid or base with its counter-ion, 10^-pH + 10^(pH-14) M, with
`pyagadir.validation.titrated_ionic_strength(ionic_strength_M, pH)`. The three datasets whose coded value already
includes the titrant (Viguera et al. 1995; Richardson et al. 1999 and 2004) carry `ionic_strength_includes_titrant`
in `manifest.json`. The term is a lower bound: re-speciation of the buffer itself is not included, except for the
Lacroix 1998 panels in `data/validation/`, whose buffer is computed in full.

