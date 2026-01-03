# Regression Datasets

This directory stores reference capture sequences used to validate SLAM and splat
stability across releases.

Expected layout:

- `data/maps/regression/sequence_<name>/frames/`
- `data/maps/regression/sequence_<name>/manifest.json`

Each dataset should include:
- RGB frames and metadata JSONs (timestamped)
- Optional depth `.npy` files
- A short README describing the scenario and expected behavior

Update `index.json` whenever adding or removing datasets.
