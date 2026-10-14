# Data Directory Structure

This directory contains the measured literature data that pyagadir is validated and benchmarked against. The model does not read it when making predictions.

## Structure
- `validation/`: Data digitised from figures in the AGADIR papers, used by `pyagadir.validation` (plots) and `pyagadir.scoring` (scores)
- `peptides/`: Per-paper tables of measured peptide helicity from the primary literature, with citations and conditions in `manifest.json` (see `peptides/README.md`)

Related directories:
- `pyagadir/params/`: Parameter tables used by the model, shipped with the package
- `figures/`: Generated plots comparing pyagadir results with the validation data

## File Formats
- Validation data: .json
- Peptide datasets: .tsv, with a manifest.json
- Parameter tables: .tsv
- Figures: .png

Note: The figures can be regenerated with `python -m pyagadir.validation`.
