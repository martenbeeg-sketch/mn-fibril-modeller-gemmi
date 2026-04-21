# mn-fibril-modeller-gemmi

Experimental copy of `mn-fibril-modeller` for a Gemmi-first structure backend migration.

This project is intentionally separate from the stable app so we can rework parsing, propagation, and merge/export internals without destabilizing the working Biopython version.

## Current Setup

- separate project folder
- separate Python package: `mn_fibril_modeller_gemmi`
- conda environment file included (`environment.yml`)
- dedicated launcher: `app-gemmi`
- Gemmi included from the start as the intended primary structure library

## Installation On A New Computer

1. Clone the repository.
```bash
git clone <YOUR_REPO_URL>
cd mn-fibril-modeller-gemmi
```

2. Create the conda environment.
```bash
conda env create -f environment.yml
```

3. Activate the environment.
```bash
conda activate mn-fibril-modeller
```

4. (Optional) Reinstall editable mode after pulling updates.
```bash
pip install -e .
```

## Run The App

```bash
conda activate mn-fibril-modeller
app-gemmi
```

Use a custom port if needed:

```bash
app-gemmi --port 8502
```

Headless mode is also supported:

```bash
app-gemmi --headless
```

## Optional: Rosetta Optimization Requirements

Rosetta optimization uses Docker and expects the image:

```text
ovo-proteinmpnn-fastrelax
```

If Docker or the image is unavailable, propagation/inspection/export still work, but Rosetta optimization will be disabled in the UI.

## Development

Run tests:

```bash
conda activate mn-fibril-modeller
pytest
```
