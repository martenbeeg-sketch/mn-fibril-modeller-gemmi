---
name: mn-fibril-modeller-gemmi
summary: Experimental Gemmi-first fork of the MN Fibril Modeller app for structure parsing, propagation, visualization, and Docker-backed Rosetta optimization.
---

# MN Fibril Modeller Gemmi

This repository is an experimental branch of the MN Fibril Modeller project that reworks structure handling around the Gemmi library while preserving the existing Streamlit-based UI and fibril modelling workflow.

## Project purpose

- Build a Gemmi-first implementation for protein fibril structure parsing and manipulation.
- Preserve the stable app's workflow while allowing rewrites of parsing, propagation, and merge/export internals.
- Support both PDB and mmCIF structure formats.
- Provide an interactive Streamlit application for selecting chains, assigning protofibrils, propagating models, and previewing merged output.
- Allow optional Docker-backed Rosetta FastRelax optimization for refined structure output.

## Key components

- `mn_fibril_modeller_gemmi/app.py`
  - Streamlit application entrypoint.
  - Loads uploaded structures, renders Mol* visualizations, manages chain selection, protofibril assignment, growth configuration, propagation, merged export, and optimization flows.

- `mn_fibril_modeller_gemmi/cli.py`
  - CLI entrypoint for the `app-gemmi` console script.
  - Launches Streamlit on a free port and supports headless mode.

- `mn_fibril_modeller_gemmi/core/pdb_io.py`
  - Structure format detection and parsing utilities.
  - Supports both Biopython and Gemmi parsing backends. 
  - Prioritize Gemmi for all new structural manipulations; Biopython is legacy/fallback.
  - Implements chain extraction, centroids, ordering, and serialization with PDB/mmCIF output.

- `mn_fibril_modeller_gemmi/core/propagation.py`
  - Core fibril propagation logic.
  - Builds propagated models by copying and transforming chains.
  - Creates merged protofibril visualizations and handles PDB/mmCIF chain-ID limits.

- `mn_fibril_modeller_gemmi/core/rosetta.py`
  - Docker-backed Rosetta FastRelax integration. Docker image is called ovo-proteinmpnn-fastrelax.
  - Provides optimization helper functions and environment checks.

- `mn_fibril_modeller_gemmi/viewer/molstar_custom_component/`
  - Custom Streamlit component wrappers for Mol* visualization.
  - Defines `ChainVisualization`, `StructureVisualization`, and component rendering utilities.

## Usage

- Install editable package in the shared `mn-fibril-modeller` Conda environment.
- Run with `app-gemmi`.
- Use `--port` to choose a port and `--headless` to avoid opening a browser.

## Important details

- The project is intentionally separate from the stable `mn-fibril-modeller` app.
- Gemmi is the intended primary structure backend for parsing and serialization.
- The app preserves UI-driven fibril modelling while enabling backend migration and refactors.
- Tests cover propagation and merged visualization behavior.
