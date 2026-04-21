# mn-fibril-modeller-gemmi

Experimental copy of `mn-fibril-modeller` for a Gemmi-first structure backend migration.

This project is intentionally separate from the stable app so we can rework parsing, propagation, and merge/export internals without destabilizing the working Biopython version.

## Current Setup

- separate project folder
- separate Python package: `mn_fibril_modeller_gemmi`
- shared conda env: `mn-fibril-modeller`
- dedicated launcher: `app-gemmi`
- Gemmi included from the start as the intended primary structure library

## Shared Environment

This copy is meant to reuse the existing conda environment instead of creating a new one.

```bash
cd /home/user/programs/ovo/mn-fibril-modeller-gemmi
conda activate mn-fibril-modeller
pip install -e .
```

## Run The Experimental App

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

## Purpose

The goal of this copy is to move structure operations toward a Gemmi-first implementation while preserving the current UI and workflow as a familiar baseline.

The stable app remains in:

```text
/home/user/programs/ovo/mn-fibril-modeller
```

The experimental Gemmi app lives in:

```text
/home/user/programs/ovo/mn-fibril-modeller-gemmi
```
