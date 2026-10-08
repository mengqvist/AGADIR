# Extracted peptide datasets

Flat, per-paper tables of measured peptide helicity, pulled from the primary literature
under `doc/`. Each `<key>.tsv` has a matching entry in `manifest.json` carrying the
citation, the **verbatim** conditions quote, the caps convention, and the paper's own
ellipticity-to-helicity conversion.

## These are NOT the scored benchmark

`pyagadir/data/validation/` holds the four datasets the investigation scores against, and
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
| `pH`, `T_C`, `ionic_strength_M` | per row, so each row stands alone |
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
