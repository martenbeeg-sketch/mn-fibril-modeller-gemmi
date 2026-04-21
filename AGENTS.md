# AI Agent Reference for `mn-fibril-modeller-gemmi`

## Project Identity & Core Tech Stack

`mn-fibril-modeller-gemmi` is an experimental, Gemmi-first fork of the MN fibril modeller app. It preserves the original Streamlit UI workflow while moving structural parsing, propagation, and output serialization toward a Gemmi-centric backend.

Primary libraries:
- `streamlit` for the interactive web UI and stateful session handling
- `gemmi` as the primary structural biology backend for PDB/mmCIF parsing and coordinate transformations
- `biopython` as a secondary compatibility layer and fallback for legacy structure operations
- `numpy` and `pandas` for numeric processing, chain metadata, and dataframe rendering
- `docker` via a Docker-backed Rosetta integration for structure optimization

## Architectural Map

### Root-level entrypoints
- `mn_fibril_modeller_gemmi/app.py`
  - Streamlit application entrypoint
  - Contains UI panels, file upload handling, session state hooks, visualization callbacks, and orchestration of propagation/export/optimization workflows
- `mn_fibril_modeller_gemmi/cli.py`
  - CLI launcher for `app-gemmi`
  - Picks a free port and runs Streamlit with the Gemmi app module

### Core backend
- `mn_fibril_modeller_gemmi/core/pdb_io.py`
  - Primary structure parsing and serialization utilities
  - Defines `detect_structure_format`, `parse_structure_gemmi`, and `serialize_structure_gemmi`
  - Provides chain inspection helpers, centroid calculations, and orderings used by the UI and propagation logic
- `mn_fibril_modeller_gemmi/core/propagation.py`
  - Core fibril propagation and merged export logic
  - Implements transform extraction, chain cloning, and merged protofibril building
  - Must be cross-referenced before any proposed 3D transformation or propagation change
- `mn_fibril_modeller_gemmi/core/rosetta.py`
  - Docker-backed Rosetta FastRelax integration
  - Exposes `run_docker_rosetta_optimization` and Docker availability helpers

### Viewer layer
- `mn_fibril_modeller_gemmi/viewer/molstar_custom_component/`
  - Custom Mol* frontend component wrappers used by Streamlit
  - Renders chain-level structure previews and selection-driven visualizations

### Experimental Rosetta container helper
- `mn_fibril_modeller_gemmi/rosetta/container_pyrosetta_fastrelax.py`
  - Container-side script invoked by Docker
  - Runs Rosetta FastRelax on an input PDB and writes scores/output

## Data Flow

1. User uploads a PDB/mmCIF file in `app.py`
2. `app.py` uses `core.pdb_io.parse_structure_gemmi` to parse the structure and `chain_rows_from_pdb` to derive chain metadata
3. The UI populates chain selection state and viewer components from those parsed chains
4. Propagation is executed through `core.propagation.build_propagated_model`
   - input: raw structure text, selected kept chains, protofibril configs, structure format
   - output: propagated PDB preview and propagation metadata
5. Merged export uses `core.propagation.build_merged_protofibril_visualization_result`
   - input: propagated PDB, protofibril configs, protofibril membership, residue gap
   - output: merged protofibril structure preview and backend metadata
6. Optimization uses `core.rosetta.run_docker_rosetta_optimization`
   - input: final structure text, optimization mode, Docker image
   - output: optimized structure text and Rosetta score data

## State & Data Handling

### Streamlit session state

`app.py` stores application state in `st.session_state` for:
- uploaded structure text and detected format
- selected chain IDs for keep/transform operations
- protofibril count and per-protofibril chain assignments
- propagation previews: `propagated_pdb_preview`, `propagation_result_preview`
- merged export previews: `merged_protofibril_pdb_preview`, `merged_protofibril_backend`
- optimization outputs: `optimized_structure_preview`, `optimized_structure_scores`, `optimized_structure_mode`
- final output selection: `final_structure_preview`, `final_structure_source`

### Data passing between modules

- UI functions in `app.py` pass plain strings and typed dictionaries into core helpers
- `core/pdb_io.py` exposes low-level schema-free parsing and serialization functions for raw structure text
- `core/propagation.py` consumes structured text, chain selection state, and protofibril configuration dictionaries
- `core/rosetta.py` accepts a structure string and outputs a PDB string plus score metadata

## Strategic Priorities

- Treat `gemmi` as the primary backend for structural format handling and coordinate operations.
- Keep `biopython` as a secondary compatibility/fallback implementation only when necessary.
- Preserve the current Streamlit-driven UI workflow while refactoring backend internals.
- Respect the Docker-backed Rosetta integration as a separate optimization path rather than a first-class propagation engine.

## AI Interaction Rules (Budget Saving)

- Favor small, incremental code changes over full-file rewrites.
- Use Python type hints consistently in all new code and refactors.
- Avoid conversational fluff; respond with direct, implementation-focused guidance.
- When suggesting 3D transformation or propagation changes, explicitly cross-reference `mn_fibril_modeller_gemmi/core/propagation.py` first.
- Prefer fixes that preserve current UI state management and flow in `app.py` instead of replacing large sections.
- If a proposed change touches state handling, verify the existing `st.session_state` keys before editing.
