from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from mn_fibril_modeller_gemmi.core.pdb_io import chain_rows_from_pdb, detect_structure_format, parse_structure_gemmi
from mn_fibril_modeller_gemmi.core.propagation import (
    build_merged_protofibril_visualization_pdb,
    build_merged_protofibril_visualization_result,
    build_propagated_model,
)
from mn_fibril_modeller_gemmi.core.rosetta import (
    is_docker_available,
    run_docker_rosetta_optimization,
)
from mn_fibril_modeller_gemmi.viewer.molstar_custom_component import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)


st.set_page_config(page_title="MN Fibril Modeller Gemmi", layout="wide")

CHAIN_COLORS = [
    "#6C5CE7",
    "#00B894",
    "#E17055",
    "#0984E3",
    "#E84393",
    "#FDCB6E",
    "#00CEC9",
    "#2D3436",
    "#55EFC4",
    "#A29BFE",
]

PROTOFIBRIL_COLORS = [
    "#E84393",
    "#0984E3",
    "#00B894",
    "#F39C12",
    "#6C5CE7",
    "#D35400",
]

TOP_CHAIN_COLOR = "#00C853"
BOTTOM_CHAIN_COLOR = "#D50000"
TRANSFORM_CHAIN_COLOR = "#FFD54F"
VIEWER_COMFORT_CHAIN_LIMIT = 62
PDB_CHAIN_ID_LIMIT = 62
MERGED_RESIDUE_GAP = 20
INSPECT_MAX_ATOMS = 3000
INSPECT_MAX_PAIR_CHECKS = 500000
INSPECT_MAX_REPORTED_CONTACTS = 100
SHOW_PROPAGATION_DEBUG_UI = False


def _uploaded_text(uploaded_file) -> str:
    return uploaded_file.getvalue().decode("utf-8")


def _stable_signature(payload) -> str:
    return json.dumps(payload, sort_keys=True)


def _request_prepare_merged_export():
    st.session_state["prepare_merged_protofibril_export_requested"] = True


def _reset_rosetta_advanced_defaults():
    st.session_state["rosetta_coordinate_constraint_weight"] = 1.0
    st.session_state["rosetta_max_iter"] = 200
    st.session_state["rosetta_constrain_to_start_coords"] = True


def _get_final_structure_text() -> str | None:
    return st.session_state.get("final_structure_preview") or st.session_state.get("propagated_pdb_preview")


def _structure_signature(structure_text: str | None) -> str | None:
    if not structure_text:
        return None
    return f"{len(structure_text)}:{hash(structure_text)}"


def _format_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: {exc!r}"


def _as_optional_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_row(scores: dict, preferred_keys: list[str], state: str | None = None) -> dict:
    row = {key: scores.get(key) for key in preferred_keys if key in scores}
    if state is not None:
        row = {"state": state, **row}
    return row


def _all_score_row(scores: dict, state: str | None = None) -> dict:
    ordered = dict(sorted(scores.items(), key=lambda item: item[0]))
    if state is not None:
        return {"state": state, **ordered}
    return ordered


def _run_propagation_with_feedback(pdb_text: str, build_settings: dict):
    progress = st.progress(0, text="Preparing propagation job...")
    status = st.empty()
    debug_mode = bool(st.session_state.get("propagation_debug_mode", False))
    debug_events: list[dict] = []
    st.session_state["propagation_debug_events"] = []

    def _progress_update(
        *,
        completed_progress_steps: int,
        total_progress_steps: int,
        protofibril_index: int,
        iteration: int,
        addition_unit: int,
        direction: str,
    ):
        if total_progress_steps <= 0:
            ratio = 0.0
        else:
            ratio = completed_progress_steps / total_progress_steps
        progress_value = min(95, 10 + int(ratio * 80))
        direction_text = {
            "Add to top": f"adding {addition_unit} chain(s) to the top",
            "Add to bottom": f"adding {addition_unit} chain(s) to the bottom",
            "Add to both ends": f"adding {addition_unit} chain(s) to both ends",
        }.get(direction, direction)
        status.caption(
            f"Propagating protofibril {protofibril_index}, iteration {iteration}: {direction_text}."
        )
        progress.progress(
            progress_value,
            text=f"Protofibril {protofibril_index}, iteration {iteration}",
        )

    try:
        status.caption("Validating selected chains and propagation settings.")
        progress.progress(10, text="Preparing propagation...")
        propagation_result = build_propagated_model(
            pdb_text=pdb_text,
            keep_chain_ids=st.session_state.get("selected_kept_chains", []),
            protofibril_configs=build_settings["protofibril_configs"],
            structure_format=st.session_state.get("current_structure_format"),
            progress_callback=_progress_update,
            debug_mode=debug_mode,
            debug_sink=debug_events,
        )
        st.session_state["propagation_debug_events"] = debug_events

        status.caption("Storing propagated chain membership and output structure.")
        progress.progress(97, text="Finalizing propagated structure...")
        st.session_state["propagated_pdb_preview"] = propagation_result["pdb"]
        st.session_state["propagation_result_preview"] = propagation_result
        st.session_state["optimized_structure_preview"] = None
        st.session_state["optimized_structure_scores"] = None
        st.session_state["optimized_structure_scores_before"] = None
        st.session_state["optimized_structure_scores_after"] = None
        st.session_state["optimized_structure_mode"] = None
        st.session_state["optimized_inspect_summary"] = None
        st.session_state["inspect_rosetta_scores"] = None
        st.session_state["final_structure_preview"] = propagation_result["pdb"]
        st.session_state["final_structure_source"] = "propagated"
        st.session_state["inspect_summary"] = None
        st.session_state["inspect_signature"] = None
        st.session_state["merged_protofibril_pdb_preview"] = None

        status.caption("Propagation finished.")
        progress.progress(100, text="Propagation finished.")
        return propagation_result
    except Exception:
        st.session_state["propagation_debug_events"] = debug_events
        raise
    finally:
        progress.empty()
        status.empty()


def _run_merged_export_with_feedback(propagated_pdb: str):
    progress = st.progress(0, text="Preparing merged export...")
    status = st.empty()

    def _progress_update(
        *,
        processed_source_chains: int,
        total_source_chains: int,
        protofibril_index: int,
        source_chain_id: str,
    ):
        if total_source_chains <= 0:
            ratio = 0.0
        else:
            ratio = processed_source_chains / total_source_chains
        progress_value = min(95, 40 + int(ratio * 55))
        current_count = min(processed_source_chains + 1, total_source_chains)
        status.caption(
            f"Merging protofibril {protofibril_index}: processing source chain {source_chain_id} "
            f"({current_count}/{total_source_chains})."
        )
        progress.progress(
            progress_value,
            text=f"Protofibril {protofibril_index}: {current_count}/{total_source_chains} source chains",
        )

    try:
        status.caption("Step 1/4: collecting propagated protofibril membership.")
        progress.progress(15, text="Collecting merged export inputs...")

        membership = (st.session_state.get("propagation_result_preview") or {}).get("protofibril_chain_membership", [])
        configs = st.session_state.get("protofibril_configs_preview", [])

        status.caption("Step 2/4: ordering propagated chains within each protofibril.")
        progress.progress(40, text="Ordering propagated chains...")

        status.caption("Step 3/4: merging each protofibril into one output chain.")
        progress.progress(70, text="Collapsing protofibrils into merged chains...")
        merge_result = build_merged_protofibril_visualization_result(
            pdb_text=propagated_pdb,
            protofibril_configs=configs,
            protofibril_chain_membership=membership,
            residue_gap=MERGED_RESIDUE_GAP,
            progress_callback=_progress_update,
        )
        st.session_state["merged_protofibril_pdb_preview"] = merge_result["structure_text"]
        st.session_state["merged_protofibril_backend"] = merge_result["backend"]
        st.session_state["merged_protofibril_backend_fallback_reason"] = merge_result["fallback_reason"]

        status.caption("Step 4/4: finalizing merged export.")
        progress.progress(100, text="Merged export finished.")
    finally:
        progress.empty()
        status.empty()


def _run_optimization_with_feedback(
    propagated_structure_text: str,
    optimization_mode: str,
    coordinate_constraint_weight: float = 1.0,
    max_iter: int = 200,
    constrain_to_start_coords: bool = True,
):
    progress = st.progress(0, text="Preparing Rosetta optimization...")
    status = st.empty()
    try:
        status.caption("Preparing merged protofibril optimization input.")
        progress.progress(10, text="Building merged optimization input...")
        membership = (st.session_state.get("propagation_result_preview") or {}).get("protofibril_chain_membership", [])
        configs = st.session_state.get("protofibril_configs_preview", [])
        merged_result = build_merged_protofibril_visualization_result(
            pdb_text=propagated_structure_text,
            protofibril_configs=configs,
            protofibril_chain_membership=membership,
            residue_gap=MERGED_RESIDUE_GAP,
        )
        optimization_input_structure = merged_result["structure_text"]
        st.session_state["optimization_input_structure_preview"] = optimization_input_structure

        status.caption("Preparing Docker-backed Rosetta FastRelax run.")
        progress.progress(35, text="Preparing Rosetta input...")
        status.caption(f"Running Rosetta optimization on merged structure in Docker: {optimization_mode}.")
        progress.progress(65, text="Running Rosetta...")
        stage_labels = {
            "starting_container": "Starting container",
            "init_pyrosetta": "Initializing PyRosetta",
            "load_pose": "Loading pose",
            "configure_relax": "Configuring FastRelax",
            "run_fastrelax": "Running FastRelax",
            "run_fastrelax_backbone_only": "Running FastRelax (backbone only)",
            "run_backbone_minimization_only": "Running backbone-only minimization (no repacking)",
            "run_fastrelax_sidechains_only": "Running FastRelax (side chains only)",
            "run_fastrelax_coupled": "Running FastRelax (coupled backbone + side chains)",
            "fastrelax_done": "FastRelax finished",
            "write_output_pdb": "Writing optimized structure",
            "write_scores": "Writing Rosetta scores",
            "done": "Rosetta run finished",
        }

        def _rosetta_status_update(
            *,
            elapsed_seconds: int,
            stage: str = "running",
            current_iteration: int | None = None,
            max_iter: int | None = None,
        ):
            stage_text = stage_labels.get(stage, stage.replace("_", " "))
            iteration_text = ""
            if current_iteration is not None and max_iter is not None and max_iter > 0:
                iteration_text = f" | iter {current_iteration}/{max_iter}"
            progress.progress(
                65,
                text=f"Running Rosetta... {stage_text}{iteration_text} ({elapsed_seconds}s)",
            )
            status.caption(
                f"Running Rosetta optimization on merged structure in Docker: {optimization_mode}. "
                f"Stage: {stage_text}{iteration_text}. {elapsed_seconds}s elapsed."
            )

        result = run_docker_rosetta_optimization(
            structure_text=optimization_input_structure,
            optimization_mode=optimization_mode,
            structure_format=detect_structure_format(optimization_input_structure),
            status_callback=_rosetta_status_update,
            coordinate_constraint_weight=coordinate_constraint_weight,
            max_iter=max_iter,
            constrain_to_start_coords=constrain_to_start_coords,
        )
        status.caption("Storing optimized structure and Rosetta scores.")
        progress.progress(90, text="Finalizing optimized structure...")
        st.session_state["optimized_structure_preview"] = result["structure_text"]
        st.session_state["optimized_structure_scores"] = result["scores"]
        st.session_state["optimized_structure_scores_before"] = result.get("scores_before")
        st.session_state["optimized_structure_scores_after"] = result.get("scores_after")
        st.session_state["optimized_structure_mode"] = optimization_mode
        st.session_state["final_structure_preview"] = result["structure_text"]
        st.session_state["final_structure_source"] = "optimized"
        status.caption("Checking post-optimization clashes.")
        progress.progress(96, text="Running full clash inspection after optimization...")
        st.session_state["optimized_inspect_summary"] = _compute_inspect_summary(
            result["structure_text"],
            max_atoms=None,
            max_pair_checks=None,
        )
        st.session_state["merged_protofibril_pdb_preview"] = None
        st.session_state["merged_protofibril_backend"] = None
        st.session_state["merged_protofibril_backend_fallback_reason"] = None
        status.caption("Rosetta optimization finished.")
        progress.progress(100, text="Optimization finished.")
        return result
    finally:
        progress.empty()
        status.empty()


def _sidebar_summary():
    st.sidebar.title("MN Fibril Modeller Gemmi")
    st.sidebar.caption("Experimental Gemmi-first fibril building workspace")
    st.sidebar.markdown(
        "\n".join(
            [
                "- load a fibril PDB",
                "- inspect chain composition",
                "- choose source chains for transform extraction",
                "- prepare for propagation, QC, and cleanup",
            ]
        )
    )


def _render_intro():
    st.title("MN Fibril Modeller Gemmi")
    st.write(
        "Experimental copy of the fibril modeller for Gemmi-first backend work. Load a fibril structure file, choose the chains that define the unit you want to model, and prepare a propagation workflow for building longer assemblies."
    )


def _render_upload_panel():
    st.subheader("Input Structure")
    return st.file_uploader("Upload a fibril PDB or mmCIF", type=["pdb", "cif", "mmcif", "mcif"])


def _render_chain_table(pdb_text: str):
    st.subheader("Chain Overview")
    rows = chain_rows_from_pdb(pdb_text)
    if not rows:
        st.warning("No polymer chains were detected in the uploaded structure.")
        return rows
    st.dataframe(pd.DataFrame(rows), width="stretch")
    return rows


def _initialize_chain_state(chain_ids: list[str]):
    current_kept_chains = st.session_state.get("selected_kept_chains", [])
    if not current_kept_chains or any(chain_id not in chain_ids for chain_id in current_kept_chains):
        st.session_state["selected_kept_chains"] = chain_ids[:]
    current_transform_chains = st.session_state.get("selected_transform_chains", [])
    if not current_transform_chains or any(chain_id not in chain_ids for chain_id in current_transform_chains):
        default_pair = st.session_state["selected_kept_chains"][:2]
        st.session_state["selected_transform_chains"] = default_pair
    current_proto_count = int(st.session_state.get("protofibril_count", 1))
    if current_proto_count < 1:
        current_proto_count = 1
    st.session_state["protofibril_count"] = current_proto_count
    for proto_index in range(1, current_proto_count + 1):
        key = f"protofibril_{proto_index}_chains"
        current_proto_chains = st.session_state.get(key, [])
        if any(chain_id not in chain_ids for chain_id in current_proto_chains):
            st.session_state[key] = []


def _chain_color_map(chain_ids: list[str]) -> dict[str, str]:
    return {chain_id: CHAIN_COLORS[index % len(CHAIN_COLORS)] for index, chain_id in enumerate(chain_ids)}


def _extract_clicked_chain_ids(component_value) -> list[str]:
    if not isinstance(component_value, str):
        return []
    try:
        import json

        parsed = json.loads(component_value)
    except Exception:
        return []
    selections = parsed.get("sequenceSelections", [])
    chain_ids = []
    for selection in selections:
        chain_id = selection.get("chainId")
        if isinstance(chain_id, str) and chain_id and chain_id not in chain_ids:
            chain_ids.append(chain_id)
    return chain_ids


def _render_structure_viewer(pdb_text: str, chain_rows):
    st.subheader("Structure Preview")
    chain_ids = [row["chain_id"] for row in chain_rows]
    color_map = _chain_color_map(chain_ids)
    selected_from_viewer = st.session_state.get("selected_kept_chains", chain_ids[:])
    selected_set = set(selected_from_viewer)

    chain_visualizations = []
    for chain_id in chain_ids:
        is_selected = chain_id in selected_set
        chain_visualizations.append(
            ChainVisualization(
                chain_id=chain_id,
                color="uniform",
                color_params={"value": (color_map[chain_id] if is_selected else "#C7CEDB").replace("#", "0x")},
                representation_type="cartoon+ball-and-stick" if is_selected else "cartoon",
                label=f"Chain {chain_id}",
            )
        )

    component_value = molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=pdb_text,
                representation_type="cartoon",
                chains=chain_visualizations,
            )
        ],
        key=f"fibril_viewer_{'_'.join(chain_ids)}_{'_'.join(selected_from_viewer)}",
        height=720,
        show_controls=True,
        selection_mode=True,
        force_reload=False,
    )

    clicked_chain_ids = _extract_clicked_chain_ids(component_value)
    if clicked_chain_ids and clicked_chain_ids != st.session_state.get("selected_kept_chains", []):
        st.session_state["selected_kept_chains"] = clicked_chain_ids
        next_pair = clicked_chain_ids[:2]
        if len(next_pair) == 2:
            st.session_state["selected_transform_chains"] = next_pair
        elif len(clicked_chain_ids) == 1:
            st.session_state["selected_transform_chains"] = clicked_chain_ids
        else:
            st.session_state["selected_transform_chains"] = []
        st.rerun()

    st.caption(
        "Click chains in the viewer to choose which chains to keep. Use Ctrl/Cmd-click in Mol* to select multiple chains."
    )


def _render_protofibril_assignment_viewer(pdb_text: str, chain_rows, protofibrils):
    st.markdown("**Protofibril Assignment Preview**")
    chain_ids = [row["chain_id"] for row in chain_rows]
    chain_visualizations = []

    assigned_chain_to_color = {}
    for proto_index, protofibril in enumerate(protofibrils):
        color = PROTOFIBRIL_COLORS[proto_index % len(PROTOFIBRIL_COLORS)]
        for chain_id in protofibril["chains"]:
            assigned_chain_to_color[chain_id] = color

    for chain_id in chain_ids:
        color = assigned_chain_to_color.get(chain_id, "#C7CEDB")
        chain_visualizations.append(
            ChainVisualization(
                chain_id=chain_id,
                color="uniform",
                color_params={"value": color.replace("#", "0x")},
                representation_type="cartoon+ball-and-stick" if chain_id in assigned_chain_to_color else "cartoon",
                label=f"Chain {chain_id}",
            )
        )

    assignment_key = "protofibril_assignment_preview"
    molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=pdb_text,
                representation_type="cartoon",
                chains=chain_visualizations,
            )
        ],
        key=assignment_key,
        height=720,
        show_controls=True,
        force_reload=False,
    )
    if protofibrils:
        legend = []
        for proto_index, protofibril in enumerate(protofibrils):
            color = PROTOFIBRIL_COLORS[proto_index % len(PROTOFIBRIL_COLORS)]
            chain_text = ", ".join(protofibril["chains"]) if protofibril["chains"] else "none"
            legend.append(f"`{protofibril['name']}`: {chain_text} ({color})")
        st.caption(" | ".join(legend))


def _render_growth_selection_viewer(pdb_text: str, chain_rows, configs):
    st.markdown("**Growth Selection Preview**")
    configured_chain_ids = []
    for config in configs:
        for chain_id in config["chains"]:
            if chain_id not in configured_chain_ids:
                configured_chain_ids.append(chain_id)
    chain_ids = configured_chain_ids
    if not chain_ids:
        st.info("Assign chains to at least one protofibril to see the growth preview.")
        return
    base_chain_to_color = {}
    for config in configs:
        base_color = PROTOFIBRIL_COLORS[(config["protofibril_index"] - 1) % len(PROTOFIBRIL_COLORS)]
        for chain_id in config["chains"]:
            base_chain_to_color[chain_id] = base_color

    highlighted_top = set()
    highlighted_bottom = set()
    highlighted_reference = set()
    highlighted_propagated = set()
    for config in configs:
        direction = config.get("propagation_direction", "Add to both ends")
        use_top = direction in {"Add to top", "Add to both ends"}
        use_bottom = direction in {"Add to bottom", "Add to both ends"}
        proto_chain_order = list(config.get("chains", []))
        addition_unit = int(config.get("addition_unit", 1))
        if use_top and config["top_chain"]:
            highlighted_top.add(config["top_chain"])
        if use_bottom and config["bottom_chain"]:
            highlighted_bottom.add(config["bottom_chain"])
        # Reference chains should be only the "paired chain" combobox values (pair source, index 0).
        top_pair = config.get("top_reference_pair", [])
        bottom_pair = config.get("bottom_reference_pair", [])
        if use_top and len(top_pair) == 2:
            highlighted_reference.add(top_pair[0])
        if use_bottom and len(bottom_pair) == 2:
            highlighted_reference.add(bottom_pair[0])

        # Propagation-view highlighting follows configured addition-unit size.
        # This indicates the chain subset participating in each propagation step.
        if proto_chain_order:
            capped_unit = max(1, min(addition_unit, len(proto_chain_order)))
            if direction == "Add to top":
                highlighted_propagated.update(proto_chain_order[-capped_unit:])
            elif direction == "Add to bottom":
                highlighted_propagated.update(proto_chain_order[:capped_unit])
            else:
                highlighted_propagated.update(proto_chain_order[:capped_unit])
                highlighted_propagated.update(proto_chain_order[-capped_unit:])

    top_bottom_visualizations = []
    reference_visualizations = []

    for chain_id in chain_ids:
        base_color = base_chain_to_color.get(chain_id, "#D0D5DD")

        top_bottom_color = "#C7CEDB"
        top_bottom_representation = "cartoon"
        if chain_id in highlighted_reference:
            top_bottom_color = TRANSFORM_CHAIN_COLOR
            top_bottom_representation = "cartoon+ball-and-stick"
        if chain_id in highlighted_bottom:
            top_bottom_color = BOTTOM_CHAIN_COLOR
            top_bottom_representation = "cartoon+ball-and-stick"
        if chain_id in highlighted_top:
            top_bottom_color = TOP_CHAIN_COLOR
            top_bottom_representation = "cartoon+ball-and-stick"
        top_bottom_visualizations.append(
            ChainVisualization(
                chain_id=chain_id,
                color="uniform",
                color_params={"value": top_bottom_color.replace("#", "0x")},
                representation_type=top_bottom_representation,
                label=f"Chain {chain_id}",
            )
        )

        reference_color = "#C7CEDB"
        reference_representation = "cartoon"
        if chain_id in highlighted_propagated:
            reference_color = TRANSFORM_CHAIN_COLOR
            reference_representation = "cartoon+ball-and-stick"
        reference_visualizations.append(
            ChainVisualization(
                chain_id=chain_id,
                color="uniform",
                color_params={"value": reference_color.replace("#", "0x")},
                representation_type=reference_representation,
                label=f"Chain {chain_id}",
            )
        )

    viewer_height = 820
    st.markdown(
        (
            f"Legend: "
            f"<span style='color:{TOP_CHAIN_COLOR}; font-weight:600;'>top chain</span> | "
            f"<span style='color:{BOTTOM_CHAIN_COLOR}; font-weight:600;'>bottom chain</span> | "
            f"<span style='color:{TRANSFORM_CHAIN_COLOR}; font-weight:600;'>reference chain (left) / propagated chain (right)</span>"
        ),
        unsafe_allow_html=True,
    )
    top_bottom_key = "growth_top_bottom_preview"
    reference_key = "growth_reference_preview"
    left, right = st.columns([1, 1], gap="medium")

    with left:
        modes = {config.get("propagation_direction", "Add to both ends") for config in configs}
        active_mode = next(iter(modes)) if len(modes) == 1 else "Mixed"
        if active_mode == "Add to top":
            st.caption(
                f"Top chains: {', '.join(sorted(highlighted_top)) or 'none'} | "
                f"Reference chains: {', '.join(sorted(highlighted_reference)) or 'none'}"
            )
        elif active_mode == "Add to bottom":
            st.caption(
                f"Bottom chains: {', '.join(sorted(highlighted_bottom)) or 'none'} | "
                f"Reference chains: {', '.join(sorted(highlighted_reference)) or 'none'}"
            )
        else:
            st.caption(
                f"Top/Bottom chains: top = {', '.join(sorted(highlighted_top)) or 'none'}, "
                f"bottom = {', '.join(sorted(highlighted_bottom)) or 'none'} | "
                f"Reference chains: {', '.join(sorted(highlighted_reference)) or 'none'}"
            )
        molstar_custom_component(
            structures=[
                StructureVisualization(
                    pdb=pdb_text,
                    representation_type="cartoon",
                    chains=top_bottom_visualizations,
                )
            ],
            key=top_bottom_key,
            height=viewer_height,
            width="100%",
            show_controls=True,
            force_reload=False,
        )

    with right:
        st.caption(f"Propagated chains: {', '.join(sorted(highlighted_propagated)) or 'none'}")
        molstar_custom_component(
            structures=[
                StructureVisualization(
                    pdb=pdb_text,
                    representation_type="cartoon",
                    chains=reference_visualizations,
                )
            ],
            key=reference_key,
            height=viewer_height,
            width="100%",
            show_controls=True,
            force_reload=False,
        )


def _render_step_header(step_number: int, title: str, description: str):
    st.markdown(f"### Step {step_number}. {title}")
    st.caption(description)


def _render_step_1_input(uploaded_file):
    _render_step_header(1, "Load Structure", "Upload the experimental fibril assembly you want to inspect and extend.")
    structure_format = st.session_state.get("current_structure_format", "pdb")
    st.caption(f"Detected input format: `{structure_format}`")
    return uploaded_file


def _render_step_2_chain_selection(chain_rows):
    _render_step_header(
        2,
        "Select Chains To Keep",
        "Choose which chains from the uploaded structure should be kept for modelling. Everything else will be ignored.",
    )
    chain_ids = [row["chain_id"] for row in chain_rows]
    _initialize_chain_state(chain_ids)

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("**Chain Overview**")
        st.dataframe(pd.DataFrame(chain_rows), width="stretch")
        selected_kept_chains = st.multiselect(
            "Chains to keep",
            options=chain_ids,
            default=st.session_state.get("selected_kept_chains", chain_ids[:]),
            help="Select the chains you want to keep available for manual protofibril definition.",
        )
        st.session_state["selected_kept_chains"] = selected_kept_chains

        button_cols = st.columns(2)
        if button_cols[0].button("Use all chains", width="stretch"):
            st.session_state["selected_kept_chains"] = chain_ids
            st.rerun()
        if button_cols[1].button("Clear selection", width="stretch"):
            st.session_state["selected_kept_chains"] = []
            st.session_state["selected_transform_chains"] = []
            st.rerun()

        selection = st.session_state.get("selected_kept_chains", [])
        if selection:
            st.success(f"Chains kept for modelling: {', '.join(selection)}")
        else:
            st.warning("Select at least one chain to continue.")

    with right:
        return st.session_state.get("selected_kept_chains", [])


def _render_step_3_define_propagation(pdb_text: str, chain_rows):
    _render_step_header(
        3,
        "Define Protofibrils",
        "Manually state how many protofibrils you want to model and assign the kept chains to each protofibril.",
    )
    chain_ids = st.session_state.get("selected_kept_chains", [])
    if not chain_ids:
        st.info("Finish step 2 first by selecting the chains you want to keep.")
        return None

    protofibril_count = st.number_input(
        "How many protofibrils should be modelled?",
        min_value=1,
        max_value=max(1, len(chain_ids)),
        value=int(st.session_state.get("protofibril_count", 1)),
        step=1,
    )
    st.session_state["protofibril_count"] = int(protofibril_count)

    protofibrils = []
    already_reserved_chains = set()
    for proto_index in range(1, int(protofibril_count) + 1):
        st.markdown(f"**Protofibril {proto_index}**")
        key = f"protofibril_{proto_index}_chains"
        existing_selection = [chain_id for chain_id in st.session_state.get(key, []) if chain_id in chain_ids]
        available_options = [chain_id for chain_id in chain_ids if chain_id not in already_reserved_chains or chain_id in existing_selection]
        selected_proto_chains = st.multiselect(
            f"Chains in protofibril {proto_index}",
            options=available_options,
            default=existing_selection,
            key=f"{key}_selector",
            help="Manually assign the kept chains that belong to this protofibril. Chains already used in earlier protofibrils are removed from later selectors.",
        )
        st.session_state[key] = selected_proto_chains
        protofibrils.append(
            {
                "name": f"Protofibril {proto_index}",
                "chains": selected_proto_chains,
            }
        )
        already_reserved_chains.update(selected_proto_chains)

    assigned_chains = set()
    duplicate_chains = set()
    for protofibril in protofibrils:
        for chain_id in protofibril["chains"]:
            if chain_id in assigned_chains:
                duplicate_chains.add(chain_id)
            assigned_chains.add(chain_id)

    unassigned_chains = [chain_id for chain_id in chain_ids if chain_id not in assigned_chains]
    assigned_lengths = [len(protofibril["chains"]) for protofibril in protofibrils if protofibril["chains"]]
    mismatched_proto_lengths = len(set(assigned_lengths)) > 1
    if duplicate_chains:
        st.warning(f"These chains are assigned to more than one protofibril: {', '.join(sorted(duplicate_chains))}")
    if unassigned_chains:
        st.info(f"Unassigned kept chains: {', '.join(unassigned_chains)}")
    if mismatched_proto_lengths:
        st.warning(
            "All protofibrils must currently contain the same number of chains. "
            f"Current assigned sizes: {', '.join(str(length) for length in assigned_lengths)}."
        )
    if not duplicate_chains and not unassigned_chains and any(proto["chains"] for proto in protofibrils):
        st.success("All kept chains are assigned exactly once.")
    with st.expander("Show protofibril assignment viewer", expanded=False):
        _render_protofibril_assignment_viewer(pdb_text, chain_rows, protofibrils)

    return {
        "kept_chains": chain_ids,
        "protofibrils": protofibrils,
    }


def _render_step_4_build(chain_rows):
    _render_step_header(
        4,
        "Configure Iterative Growth",
        "Set one shared growth schedule for all protofibrils, then define the stack ends and reference pair for each protofibril.",
    )
    kept_chains = st.session_state.get("selected_kept_chains", [])
    protofibril_count = int(st.session_state.get("protofibril_count", 0))
    if not kept_chains or protofibril_count < 1:
        st.info("Complete the previous steps first.")
        return None

    assigned_proto_chain_lists = [
        st.session_state.get(f"protofibril_{proto_index}_chains", [])
        for proto_index in range(1, protofibril_count + 1)
        if st.session_state.get(f"protofibril_{proto_index}_chains", [])
    ]
    if assigned_proto_chain_lists and len({len(chains) for chains in assigned_proto_chain_lists}) > 1:
        st.error(
            "All assigned protofibrils must have the same chain count before propagation can be configured."
        )
        return {"protofibril_configs": []}
    max_shared_addition_unit = min((len(chains) for chains in assigned_proto_chain_lists), default=1)
    # Streamlit keeps widget state across reruns; clamp stale values when the
    # allowed max shrinks (e.g., after changing protofibril assignments).
    current_addition_unit = int(st.session_state.get("global_addition_unit", 1))
    if current_addition_unit > max_shared_addition_unit:
        st.session_state["global_addition_unit"] = max_shared_addition_unit
    elif current_addition_unit < 1:
        st.session_state["global_addition_unit"] = 1

    st.markdown("**Shared Growth Settings**")
    global_propagation_direction = st.radio(
        "Growth direction for all protofibrils",
        options=["Add to top", "Add to bottom", "Add to both ends"],
        key="global_growth_direction",
        help="The same direction will be used for every protofibril in this modelling run.",
    )
    global_addition_unit = st.number_input(
        "Addition unit size for all protofibrils",
        min_value=1,
        max_value=max_shared_addition_unit,
        value=int(st.session_state.get("global_addition_unit", 1)),
        step=1,
        key="global_addition_unit",
        help="This is the number of chains added per propagation step, shared across all protofibrils.",
    )
    global_units_to_add = st.number_input(
        "Number of addition units for all protofibrils",
        min_value=1,
        max_value=100,
        value=int(st.session_state.get("global_units_to_add", 3)),
        step=1,
        key="global_units_to_add",
        help="This is the number of iterative propagation cycles, shared across all protofibrils.",
    )
    st.caption(
        f"Addition unit size is capped by the smallest assigned protofibril. "
        f"Current maximum: {max_shared_addition_unit} chain(s)."
    )
    current_structure_format = st.session_state.get("current_structure_format", "pdb")

    active_protofibril_count = sum(
        1
        for proto_index in range(1, protofibril_count + 1)
        if st.session_state.get(f"protofibril_{proto_index}_chains", [])
    )
    per_protofibril_growth_per_iteration = (
        global_addition_unit if global_propagation_direction != "Add to both ends" else global_addition_unit * 2
    )
    projected_total_added_chains = active_protofibril_count * int(global_units_to_add) * int(per_protofibril_growth_per_iteration)
    projected_total_chain_count = len(kept_chains) + projected_total_added_chains

    st.caption(
        f"Projected total chain count: {projected_total_chain_count} "
        f"({len(kept_chains)} kept + {projected_total_added_chains} added across {active_protofibril_count} protofibril(s))."
    )
    if projected_total_chain_count > VIEWER_COMFORT_CHAIN_LIMIT:
        st.warning(
            f"Projected total chains exceed {VIEWER_COMFORT_CHAIN_LIMIT}, which is where the current viewer often becomes hard to read. "
            "You can still proceed and download the propagated structure."
        )
    if current_structure_format == "pdb" and projected_total_chain_count > PDB_CHAIN_ID_LIMIT:
        st.info(
            f"Projected total chains exceed {PDB_CHAIN_ID_LIMIT}. Generated outputs are written as mmCIF "
            "so longer chain identifiers can be used."
        )
    elif current_structure_format == "mmcif" and projected_total_chain_count > PDB_CHAIN_ID_LIMIT:
        st.info(
            "Generated outputs are written as mmCIF, so propagation can continue past the legacy single-character PDB chain-ID limit. "
            "Large structures may still be slower to render."
        )

    def _sync_selectbox_key_to_options(key: str, options: list[str]) -> None:
        if not options:
            return
        current_value = st.session_state.get(key)
        if current_value not in options:
            st.session_state[key] = options[0]

    configs = []
    for proto_index in range(1, protofibril_count + 1):
        proto_key = f"protofibril_{proto_index}_chains"
        proto_chains = st.session_state.get(proto_key, [])
        st.markdown(f"**Protofibril {proto_index} Growth Settings**")
        if not proto_chains:
            st.info("Assign chains to this protofibril in step 3 first.")
            continue

        if len(proto_chains) < 2:
            st.warning("This protofibril needs at least two chains to define a propagation pair.")
            continue

        two_chain_mode = len(proto_chains) == 2
        top_chain = proto_chains[-1]
        bottom_chain = proto_chains[0]
        top_reference_pair = []
        bottom_reference_pair = []

        if global_propagation_direction in {"Add to bottom", "Add to both ends"}:
            bottom_chain_key = f"proto_{proto_index}_bottom_chain"
            _sync_selectbox_key_to_options(bottom_chain_key, proto_chains)
            bottom_chain = st.selectbox(
                f"Bottom chain for protofibril {proto_index}",
                options=proto_chains,
                key=bottom_chain_key,
            )
            bottom_partner_options = [chain_id for chain_id in proto_chains if chain_id != bottom_chain]
            if not bottom_partner_options:
                st.warning("Select at least two chains to define the bottom propagation pair.")
                continue
            if two_chain_mode:
                bottom_partner = bottom_partner_options[0]
            else:
                bottom_partner_key = f"proto_{proto_index}_bottom_partner"
                _sync_selectbox_key_to_options(bottom_partner_key, bottom_partner_options)
                bottom_partner = st.selectbox(
                    f"Chain paired with bottom chain for protofibril {proto_index}",
                    options=bottom_partner_options,
                    key=bottom_partner_key,
                    help="This chain and the bottom chain define the stacking step used to extend the bottom end.",
                )
            bottom_reference_pair = [bottom_partner, bottom_chain]
            st.caption(f"Bottom propagation pair: {bottom_partner} -> {bottom_chain}")

        if global_propagation_direction in {"Add to top", "Add to both ends"}:
            if global_propagation_direction == "Add to both ends":
                top_chain_options = [chain_id for chain_id in proto_chains if chain_id != bottom_chain]
            else:
                top_chain_options = list(proto_chains)
            if not top_chain_options:
                st.warning("No valid top-chain options remain after applying bottom-chain constraints.")
                continue
            top_chain_key = f"proto_{proto_index}_top_chain"
            _sync_selectbox_key_to_options(top_chain_key, top_chain_options)
            top_chain = st.selectbox(
                f"Top chain for protofibril {proto_index}",
                options=top_chain_options,
                key=top_chain_key,
            )
            if global_propagation_direction == "Add to both ends":
                if two_chain_mode:
                    # Special 2-chain case: the opposite chain is the required partner.
                    top_partner_options = [chain_id for chain_id in proto_chains if chain_id != top_chain]
                else:
                    top_partner_options = [
                        chain_id for chain_id in proto_chains if chain_id not in {top_chain, bottom_chain}
                    ]
            else:
                top_partner_options = [chain_id for chain_id in proto_chains if chain_id != top_chain]
            if not top_partner_options:
                st.warning("Select at least two distinct chains to define the top propagation pair.")
                continue
            if two_chain_mode:
                top_partner = top_partner_options[0]
            else:
                top_partner_key = f"proto_{proto_index}_top_partner"
                _sync_selectbox_key_to_options(top_partner_key, top_partner_options)
                top_partner = st.selectbox(
                    f"Chain paired with top chain for protofibril {proto_index}",
                    options=top_partner_options,
                    key=top_partner_key,
                    help="This chain and the top chain define the stacking step used to extend the top end.",
                )
            top_reference_pair = [top_partner, top_chain]
            st.caption(f"Top propagation pair: {top_partner} -> {top_chain}")

        if global_propagation_direction == "Add to top":
            # Keep a stable bottom anchor for membership ordering even when not growing bottom.
            bottom_chain = next((chain_id for chain_id in proto_chains if chain_id != top_chain), proto_chains[0])
        elif global_propagation_direction == "Add to bottom":
            # Keep a stable top anchor for membership ordering even when not growing top.
            top_chain = next((chain_id for chain_id in reversed(proto_chains) if chain_id != bottom_chain), proto_chains[-1])

        if two_chain_mode:
            # Special case: with exactly two kept chains, enforce reciprocal reference pairs.
            top_reference_pair = [bottom_chain, top_chain]
            bottom_reference_pair = [top_chain, bottom_chain]
            st.caption(
                "Two-chain mode: reciprocal propagation pairs are auto-assigned "
                f"({bottom_chain} -> {top_chain} and {top_chain} -> {bottom_chain})."
            )

        if global_propagation_direction == "Add to top":
            st.caption("Current run uses the top propagation pair only.")
        elif global_propagation_direction == "Add to bottom":
            st.caption("Current run uses the bottom propagation pair only.")
        else:
            st.caption("Current run uses both top and bottom propagation pairs.")

        per_iteration_growth = global_addition_unit if global_propagation_direction != "Add to both ends" else global_addition_unit * 2
        projected_growth = int(global_units_to_add) * int(per_iteration_growth)
        st.caption(f"Protofibril {proto_index} projected added chains: {projected_growth}")

        configs.append(
            {
                "protofibril_index": proto_index,
                "chains": proto_chains,
                "top_chain": top_chain,
                "bottom_chain": bottom_chain,
                "top_reference_pair": top_reference_pair,
                "bottom_reference_pair": bottom_reference_pair,
                "addition_unit": int(global_addition_unit),
                "propagation_direction": global_propagation_direction,
                "units_to_add": int(global_units_to_add),
            }
        )

    if configs:
        st.info("Propagation builds the rigid fibril geometry only. Inspection and optional optimization happen afterward.")
        with st.expander("Show growth selection viewers", expanded=False):
            _render_growth_selection_viewer(st.session_state.get("current_pdb_text", ""), chain_rows, configs)
    return {
        "protofibril_configs": configs,
        "global_units_to_add": int(global_units_to_add),
        "global_addition_unit": int(global_addition_unit),
        "global_propagation_direction": global_propagation_direction,
    }


def _render_step_5_review(configs: list[dict] | None = None):
    _render_step_header(
        5,
        "Iteration Plan",
        "Review the exact rigid propagation sequence that will be executed from the current settings above.",
    )
    configs = configs if configs is not None else st.session_state.get("protofibril_configs_preview", [])
    if not configs:
        st.info("Complete the growth configuration step first.")
        return

    lines = []
    for config in configs:
        lines.append(
            f"Protofibril {config['protofibril_index']}: start with chains {', '.join(config['chains'])} "
            f"(bottom = {config['bottom_chain']}, top = {config['top_chain']})."
        )
        if config["top_reference_pair"]:
            lines.append(
                f"Protofibril {config['protofibril_index']}: top propagation pair = "
                f"{config['top_reference_pair'][0]} -> {config['top_reference_pair'][1]}."
            )
        if config["bottom_reference_pair"]:
            lines.append(
                f"Protofibril {config['protofibril_index']}: bottom propagation pair = "
                f"{config['bottom_reference_pair'][0]} -> {config['bottom_reference_pair'][1]}."
            )
        per_iteration_added = config["addition_unit"]
        if config["propagation_direction"] == "Add to top":
            direction_detail = f"add {per_iteration_added} chain(s) to the top end"
        elif config["propagation_direction"] == "Add to bottom":
            direction_detail = f"add {per_iteration_added} chain(s) to the bottom end"
        else:
            direction_detail = (
                f"add {per_iteration_added} chain(s) to the top end and "
                f"{per_iteration_added} chain(s) to the bottom end"
            )
        for iteration in range(1, config["units_to_add"] + 1):
            lines.append(
                f"Protofibril {config['protofibril_index']}, iteration {iteration}: "
                f"{direction_detail}."
            )
        total_added = config["units_to_add"] * (
            config["addition_unit"] if config["propagation_direction"] != "Add to both ends" else config["addition_unit"] * 2
        )
        lines.append(
            f"Protofibril {config['protofibril_index']}: total projected added chains = {total_added}."
        )

    st.markdown("\n".join([f"- {line}" for line in lines]))
    st.info("This plan updates immediately when you change anything in the steps above.")


def _render_propagation_debug_panel():
    debug_events = st.session_state.get("propagation_debug_events") or []
    if not debug_events:
        return
    with st.expander("Propagation debug log", expanded=False):
        st.caption(
            f"Captured {len(debug_events)} debug events from the most recent propagation attempt."
        )
        st.dataframe(pd.DataFrame(debug_events), width="stretch")
        st.code(json.dumps(debug_events, indent=2, default=str), language="json")


def _render_propagated_model_preview(propagated_pdb: str):
    st.markdown("**Propagated Model Preview**")
    propagated_rows = chain_rows_from_pdb(propagated_pdb)
    st.caption(f"Propagated chain count: {len(propagated_rows)}")
    st.dataframe(pd.DataFrame(propagated_rows), width="stretch")
    chain_ids = [row["chain_id"] for row in propagated_rows]
    propagation_result = st.session_state.get("propagation_result_preview") or {}
    membership_rows = propagation_result.get("protofibril_chain_membership", []) or []
    seed_chain_ids = {
        str(row.get("chain_id"))
        for row in membership_rows
        if str(row.get("origin", "")).lower() == "seed"
    }
    if seed_chain_ids:
        st.caption(
            "Original chains are highlighted in blue: "
            f"{', '.join(sorted(seed_chain_ids))}"
        )
        chain_visualizations = []
        for chain_id in chain_ids:
            is_seed = chain_id in seed_chain_ids
            chain_visualizations.append(
                ChainVisualization(
                    chain_id=chain_id,
                    color="uniform",
                    color_params={"value": ("0x1E88E5" if is_seed else "0xC7CEDB")},
                    representation_type=("cartoon+ball-and-stick" if is_seed else "cartoon"),
                    label=f"Chain {chain_id}{' (original)' if is_seed else ' (propagated)'}",
                )
            )
        structures = [
            StructureVisualization(
                pdb=propagated_pdb,
                representation_type="cartoon",
                chains=chain_visualizations,
            )
        ]
    else:
        structures = [
            StructureVisualization(
                pdb=propagated_pdb,
                # Render the full propagated structure directly from the
                # serialized mmCIF/PDB text. This avoids chain-id field mapping
                # mismatches (auth_asym_id vs label_asym_id) that can hide
                # generated PF* chains in chain-filtered views.
                color="chain-id",
                representation_type="cartoon",
            )
        ]
    molstar_custom_component(
        structures=structures,
        key=f"propagated_preview_{'_'.join(chain_ids)}",
        height=720,
        show_controls=True,
        force_reload=False,
    )


def _render_propagated_membership_table():
    propagation_result = st.session_state.get("propagation_result_preview") or {}
    membership_rows = propagation_result.get("protofibril_chain_membership", [])
    if not membership_rows:
        return
    st.markdown("**Propagated Protofibril Membership**")
    st.caption("This table shows the final ordered chain assignment after propagation. The merged export uses this mapping.")
    st.dataframe(pd.DataFrame(membership_rows), width="stretch")


def _compute_inspect_summary(
    structure_text: str,
    progress_callback=None,
    max_atoms: int | None = INSPECT_MAX_ATOMS,
    max_pair_checks: int | None = INSPECT_MAX_PAIR_CHECKS,
) -> dict:
    structure = parse_structure_gemmi(structure_text, detect_structure_format(structure_text))
    model = structure[0]
    atoms = []
    backbone_names = {"N", "CA", "C", "O"}
    for chain in model:
        for residue in chain:
            if residue.het_flag != "A":
                continue
            residue_label = f"{chain.name}:{residue.seqid.num}:{residue.name}"
            for atom in residue:
                if atom.element.name == "H":
                    continue
                atoms.append(
                    {
                        "chain_id": chain.name,
                        "residue_label": residue_label,
                        "atom_name": atom.name.strip(),
                        "is_backbone": atom.name.strip() in backbone_names,
                        "coord": np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
                    }
                )

    sampled = False
    original_atom_count = len(atoms)
    if max_atoms is not None and len(atoms) > max_atoms:
        sampled = True
        step = max(1, len(atoms) // max_atoms)
        atoms = atoms[::step][:max_atoms]

    severe_backbone = 0
    sidechain_clashes = 0
    min_distance = None
    checked_pairs = 0
    backbone_contacts = []
    sidechain_contacts = []
    backbone_residue_set = set()
    sidechain_residue_set = set()
    total_outer = len(atoms)
    for left_index in range(len(atoms)):
        if progress_callback is not None and total_outer > 0 and (left_index % 25 == 0 or left_index == total_outer - 1):
            progress_callback(current=left_index + 1, total=total_outer)
        left_atom = atoms[left_index]
        for right_index in range(left_index + 1, len(atoms)):
            if max_pair_checks is not None and checked_pairs >= max_pair_checks:
                break
            right_atom = atoms[right_index]
            if left_atom["chain_id"] == right_atom["chain_id"]:
                continue
            checked_pairs += 1
            distance = float(np.linalg.norm(left_atom["coord"] - right_atom["coord"]))
            if min_distance is None or distance < min_distance:
                min_distance = distance
            if left_atom["is_backbone"] and right_atom["is_backbone"] and distance < 1.2:
                severe_backbone += 1
                backbone_residue_set.update([left_atom["residue_label"], right_atom["residue_label"]])
                if len(backbone_contacts) < INSPECT_MAX_REPORTED_CONTACTS:
                    backbone_contacts.append(
                        {
                            "residue_a": left_atom["residue_label"],
                            "atom_a": left_atom["atom_name"],
                            "residue_b": right_atom["residue_label"],
                            "atom_b": right_atom["atom_name"],
                            "distance": round(distance, 3),
                        }
                    )
            elif distance < 2.2:
                sidechain_clashes += 1
                sidechain_residue_set.update([left_atom["residue_label"], right_atom["residue_label"]])
                if len(sidechain_contacts) < INSPECT_MAX_REPORTED_CONTACTS:
                    sidechain_contacts.append(
                        {
                            "residue_a": left_atom["residue_label"],
                            "atom_a": left_atom["atom_name"],
                            "residue_b": right_atom["residue_label"],
                            "atom_b": right_atom["atom_name"],
                            "distance": round(distance, 3),
                        }
                    )
        if max_pair_checks is not None and checked_pairs >= max_pair_checks:
            break

    if severe_backbone > 0:
        recommendation = "Optimize backbone + side chains"
        rationale = "Severe inter-chain backbone overlap was detected."
    elif sidechain_clashes > 0:
        recommendation = "Optimize side chains only"
        rationale = "The backbone looks usable, but inter-chain steric clashes suggest side-chain cleanup."
    else:
        recommendation = "No optimization needed"
        rationale = "No concerning inter-chain clashes were detected with the current heuristic checks."

    return {
        "chain_count": len(chain_rows_from_pdb(structure_text)),
        "heavy_atom_count": len(atoms),
        "heavy_atom_count_original": original_atom_count,
        "sampled": sampled,
        "checked_pairs": checked_pairs,
        "min_interchain_distance": None if min_distance is None else round(min_distance, 2),
        "severe_backbone_overlaps": severe_backbone,
        "sidechain_clashes": sidechain_clashes,
        "backbone_overlap_residues": sorted(backbone_residue_set),
        "sidechain_clash_residues": sorted(sidechain_residue_set),
        "backbone_overlap_contacts": backbone_contacts,
        "sidechain_clash_contacts": sidechain_contacts,
        "recommendation": recommendation,
        "rationale": rationale,
    }


def _render_step_7_inspect():
    _render_step_header(
        7,
        "Inspect",
        "Evaluate the propagated geometry before deciding whether optimization is needed.",
    )
    structure_text = st.session_state.get("propagated_pdb_preview")
    if not structure_text:
        st.info("Build a propagated structure first.")
        return

    inspect_mode = "Full (exhaustive)"
    st.caption("Inspection depth: full (exhaustive). All heavy atoms and all inter-chain pairs are checked.")

    current_signature = _stable_signature(
        {
            "structure": _structure_signature(structure_text),
            "inspect_mode": inspect_mode,
        }
    )
    run_inspect_clicked = st.button(
        "Run inspection",
        key="run_inspection",
        type="primary",
        width="stretch",
    )
    if st.session_state.get("inspect_signature") != current_signature:
        if st.session_state.get("inspect_summary"):
            st.warning("Current inspection results are outdated for this structure. Click `Run inspection` to refresh.")
        else:
            st.info("No inspection results yet. Click `Run inspection` to evaluate clashes.")

    if run_inspect_clicked:
        inspect_progress = st.progress(0, text="Preparing inspection...")
        inspect_status = st.empty()

        def _inspect_progress_update(*, current: int, total: int):
            pct = max(1, min(99, int((current / max(total, 1)) * 100)))
            inspect_status.caption(f"Checking inter-chain contacts: {current}/{total} atoms scanned.")
            inspect_progress.progress(pct, text="Inspecting clashes...")

        try:
            max_atoms = None
            max_pair_checks = None
            st.session_state["inspect_summary"] = _compute_inspect_summary(
                structure_text,
                progress_callback=_inspect_progress_update,
                max_atoms=max_atoms,
                max_pair_checks=max_pair_checks,
            )
            inspect_progress.progress(90, text="Preparing Rosetta pre-optimization scoring...")
            st.session_state["inspect_rosetta_scores"] = None
            if is_docker_available():
                inspect_status.caption("Building merged Rosetta scoring input.")
                membership = (st.session_state.get("propagation_result_preview") or {}).get("protofibril_chain_membership", [])
                configs = st.session_state.get("protofibril_configs_preview", [])
                merged_result = build_merged_protofibril_visualization_result(
                    pdb_text=structure_text,
                    protofibril_configs=configs,
                    protofibril_chain_membership=membership,
                    residue_gap=MERGED_RESIDUE_GAP,
                )
                inspect_status.caption("Running Rosetta score-only pass (pre-optimization baseline).")
                score_result = run_docker_rosetta_optimization(
                    structure_text=merged_result["structure_text"],
                    optimization_mode="Score only",
                    structure_format=detect_structure_format(merged_result["structure_text"]),
                )
                st.session_state["inspect_rosetta_scores"] = (
                    score_result.get("scores_before")
                    or score_result.get("scores")
                    or {}
                )
            else:
                inspect_status.caption("Docker not available. Skipping Rosetta pre-optimization score.")
            st.session_state["inspect_signature"] = current_signature
        except Exception as exc:
            st.session_state["inspect_summary"] = {
                "chain_count": len(chain_rows_from_pdb(structure_text)),
                "heavy_atom_count": 0,
                "heavy_atom_count_original": 0,
                "sampled": False,
                "checked_pairs": 0,
                "min_interchain_distance": "n/a",
                "severe_backbone_overlaps": 0,
                "sidechain_clashes": 0,
                "recommendation": "No optimization needed",
                "rationale": "Inspection failed; optimization/export is still available.",
            }
            st.session_state["inspect_rosetta_scores"] = None
            st.session_state["inspect_signature"] = current_signature
            st.warning(f"Inspection failed, but you can still continue: {_format_exception_message(exc)}")
        finally:
            inspect_progress.empty()
            inspect_status.empty()

    summary = st.session_state.get("inspect_summary") or {}
    if not summary:
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Chains", summary.get("chain_count", 0))
    metric_cols[1].metric("Heavy Atoms", summary.get("heavy_atom_count", 0))
    metric_cols[2].metric("Min Inter-chain Distance", summary.get("min_interchain_distance", "n/a"))
    metric_cols[3].metric("Side-chain Clashes", summary.get("sidechain_clashes", 0))
    if summary.get("sampled"):
        st.caption(
            f"Inspection used a sampled atom set ({summary.get('heavy_atom_count', 0)} of "
            f"{summary.get('heavy_atom_count_original', 0)} heavy atoms) and capped pair checks "
            f"at {INSPECT_MAX_PAIR_CHECKS}."
        )
    else:
        st.caption("Inspection ran in full mode: all heavy atoms and all inter-chain pairs were checked.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "severe_backbone_overlaps": summary.get("severe_backbone_overlaps", 0),
                    "sidechain_clashes": summary.get("sidechain_clashes", 0),
                    "checked_interchain_pairs": summary.get("checked_pairs", 0),
                    "recommended_action": summary.get("recommendation", "n/a"),
                }
            ]
        ),
        width="stretch",
    )
    if summary.get("recommendation") == "No optimization needed":
        st.success(summary.get("rationale", "No optimization needed."))
    elif summary.get("recommendation") == "Optimize side chains only":
        st.warning(summary.get("rationale", "Side-chain cleanup is recommended."))
    else:
        st.error(summary.get("rationale", "Backbone and side-chain optimization is recommended."))

    st.markdown("**Residues Involved In Detected Contacts**")
    residue_col1, residue_col2 = st.columns(2)
    with residue_col1:
        st.caption("Backbone-overlap residues")
        backbone_residues = summary.get("backbone_overlap_residues", [])
        if backbone_residues:
            st.code("\n".join(backbone_residues[:80]), language="text")
        else:
            st.caption("No backbone-overlap residues detected.")
    with residue_col2:
        st.caption("Side-chain clash residues")
        sidechain_residues = summary.get("sidechain_clash_residues", [])
        if sidechain_residues:
            st.code("\n".join(sidechain_residues[:80]), language="text")
        else:
            st.caption("No side-chain clash residues detected.")

    backbone_contacts = summary.get("backbone_overlap_contacts", [])
    sidechain_contacts = summary.get("sidechain_clash_contacts", [])
    if backbone_contacts:
        st.markdown("**Backbone Overlap Contacts (Top Entries)**")
        st.dataframe(pd.DataFrame(backbone_contacts), width="stretch")
    if sidechain_contacts:
        st.markdown("**Side-Chain Clash Contacts (Top Entries)**")
        st.dataframe(pd.DataFrame(sidechain_contacts), width="stretch")

    inspect_scores = st.session_state.get("inspect_rosetta_scores") or {}
    if inspect_scores:
        preferred_keys = ["total_score", "fa_atr", "fa_rep", "coordinate_constraint", "fa_dun"]
        present_scores = {key: inspect_scores.get(key) for key in preferred_keys if key in inspect_scores}
        if present_scores:
            st.markdown("**Rosetta Score Of Propagated Structure (Step 7 Baseline)**")
            st.dataframe(pd.DataFrame([present_scores]), width="stretch")
        with st.expander("Show all Rosetta score terms (Step 7 baseline)", expanded=False):
            st.dataframe(pd.DataFrame([_all_score_row(inspect_scores)]), width="stretch")
    elif run_inspect_clicked and not is_docker_available():
        st.info("Rosetta baseline score skipped because Docker is not available.")


def _render_step_8_optimize():
    _render_step_header(
        8,
        "Optimize",
        "Decide whether to keep the propagated structure as-is or prepare for a later optimization pass.",
    )
    propagated_text = st.session_state.get("propagated_pdb_preview")
    if not propagated_text:
        st.info("Build and inspect a propagated structure first.")
        return

    summary = st.session_state.get("inspect_summary") or {}
    default_mode = "No optimization" if summary.get("recommendation") == "No optimization needed" else "Backbone only"

    optimization_options = [
        "No optimization",
        "Backbone only",
    ]
    selected_mode = st.radio(
        "Optimization mode",
        options=optimization_options,
        index=optimization_options.index(default_mode),
        key="optimization_mode",
        help="This setting controls what the final structure should become once the optimization backend is connected.",
    )
    with st.expander("Advanced Rosetta settings", expanded=False):
        st.caption(
            "Defaults are pre-filled and recommended for fibril cleanup. "
            "You can change them when you need tighter or looser movement."
        )
        coordinate_constraint_weight = st.number_input(
            "Coordinate constraint weight",
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            value=1.0,
            key="rosetta_coordinate_constraint_weight",
            help=(
                "Default: 1.0. Recommended range: 0.5-2.0 for restrained cleanup. "
                "0.0 disables coordinate-constraint energy even if constraints are enabled. "
                "Higher values keep atoms closer to the starting coordinates."
            ),
        )
        max_iter = st.number_input(
            "FastRelax max iterations",
            min_value=1,
            step=10,
            value=200,
            key="rosetta_max_iter",
            help="Default: 200. Higher values allow longer minimization per run.",
        )
        constrain_to_start_coords = st.checkbox(
            "Constrain relax to start coordinates",
            value=True,
            key="rosetta_constrain_to_start_coords",
            help="Default: enabled. Keeps optimization close to the propagated geometry.",
        )
        constraint_behavior_text = {
            "No optimization": (
                "No Rosetta run is performed in this mode, so coordinate constraints are not applied."
            ),
            "Backbone only": (
                "Backbone-only mode: only backbone torsions can move. "
                "With constraints ON, backbone stays close to start coordinates; "
                "with constraints OFF, backbone can move freely."
            ),
        }
        current_behavior = constraint_behavior_text.get(selected_mode, "")
        if current_behavior:
            st.info(current_behavior)
        st.caption(
            "Coordinate-constraint weight guidance: 0.0-10.0 allowed; "
            "0.5-2.0 recommended for most fibril cleanup runs; "
            ">2.0 is very restrictive."
        )
        if coordinate_constraint_weight == 0.0:
            st.warning(
                "Coordinate constraint weight is 0.0: coordinate-constraint energy is effectively disabled."
            )
        elif coordinate_constraint_weight > 2.0:
            st.warning(
                "Coordinate constraint weight is high (>2.0): optimization may become over-restrained."
            )
        st.caption(
            f"Current setting: constraints are {'ON' if constrain_to_start_coords else 'OFF'}."
        )
        if not constrain_to_start_coords:
            st.warning(
                "Constraints are OFF: allowed degrees of freedom for the selected mode are unconstrained."
            )
        st.button(
            "Reset advanced settings to defaults",
            key="reset_rosetta_advanced_defaults",
            on_click=_reset_rosetta_advanced_defaults,
        )

    if selected_mode == "No optimization":
        st.success("The propagated structure will be kept as the final structure.")
        st.session_state["optimized_structure_preview"] = None
        st.session_state["optimized_structure_scores"] = None
        st.session_state["optimized_structure_scores_before"] = None
        st.session_state["optimized_structure_scores_after"] = None
        st.session_state["optimized_structure_mode"] = None
        st.session_state["optimized_inspect_summary"] = None
        st.session_state["final_structure_preview"] = propagated_text
        st.session_state["final_structure_source"] = "propagated"
    else:
        docker_available = is_docker_available()
        if docker_available:
            st.caption("Docker is available. Rosetta FastRelax will run with image `ovo-proteinmpnn-fastrelax`.")
        else:
            st.error("Docker is not available on PATH in this environment, so Rosetta optimization cannot be started.")

        run_clicked = st.button(
            "Run Rosetta optimization",
            key="run_rosetta_optimization",
            type="primary",
            width="stretch",
            disabled=not docker_available,
        )
        if run_clicked:
            try:
                _run_optimization_with_feedback(
                    propagated_text,
                    selected_mode,
                    coordinate_constraint_weight=float(st.session_state.get("rosetta_coordinate_constraint_weight", 1.0)),
                    max_iter=int(st.session_state.get("rosetta_max_iter", 200)),
                    constrain_to_start_coords=bool(st.session_state.get("rosetta_constrain_to_start_coords", True)),
                )
                st.success(f"Rosetta optimization finished with mode: {selected_mode}.")
            except Exception as exc:
                st.session_state["optimized_structure_preview"] = None
                st.session_state["optimized_structure_scores"] = None
                st.session_state["optimized_structure_scores_before"] = None
                st.session_state["optimized_structure_scores_after"] = None
                st.session_state["optimized_structure_mode"] = None
                st.session_state["optimized_inspect_summary"] = None
                st.session_state["final_structure_preview"] = propagated_text
                st.session_state["final_structure_source"] = "propagated"
                st.error(f"Rosetta optimization failed: {_format_exception_message(exc)}")

        if st.session_state.get("optimized_structure_preview"):
            st.success(
                f"Optimized structure available. Final structure source: {st.session_state.get('optimized_structure_mode', selected_mode)}."
            )
            st.caption("Optimization was run on a merged protofibril representation to keep Rosetta input PDB-safe.")
            before = st.session_state.get("inspect_summary") or {}
            after = st.session_state.get("optimized_inspect_summary") or {}
            if before or after:
                st.markdown("**Clash Check: Before vs After Optimization (full check)**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "state": "before",
                                "min_interchain_distance": _as_optional_float(before.get("min_interchain_distance")),
                                "severe_backbone_overlaps": int(before.get("severe_backbone_overlaps", 0))
                                if before.get("severe_backbone_overlaps") is not None
                                else None,
                                "sidechain_clashes": int(before.get("sidechain_clashes", 0))
                                if before.get("sidechain_clashes") is not None
                                else None,
                            },
                            {
                                "state": "after",
                                "min_interchain_distance": _as_optional_float(after.get("min_interchain_distance")),
                                "severe_backbone_overlaps": int(after.get("severe_backbone_overlaps", 0))
                                if after.get("severe_backbone_overlaps") is not None
                                else None,
                                "sidechain_clashes": int(after.get("sidechain_clashes", 0))
                                if after.get("sidechain_clashes") is not None
                                else None,
                            },
                        ]
                    ),
                    width="stretch",
                )
            scores_before = st.session_state.get("optimized_structure_scores_before") or {}
            scores_after = st.session_state.get("optimized_structure_scores_after") or {}
            scores = st.session_state.get("optimized_structure_scores") or {}
            if scores_before or scores_after or scores:
                preferred_keys = [
                    "total_score",
                    "fa_atr",
                    "fa_rep",
                    "coordinate_constraint",
                    "fa_dun",
                ]
                if not scores_before and scores:
                    scores_before = {}
                    scores_after = scores
                score_rows = []
                if scores_before:
                    score_rows.append(
                        {
                            "state": "before",
                            **{key: scores_before.get(key) for key in preferred_keys},
                        }
                    )
                if scores_after:
                    score_rows.append(
                        {
                            "state": "after",
                            **{key: scores_after.get(key) for key in preferred_keys},
                        }
                    )
                if score_rows:
                    st.markdown("**Rosetta Score: Before vs After Optimization**")
                    st.dataframe(pd.DataFrame(score_rows), width="stretch")
                    with st.expander("Show all Rosetta score terms (before vs after)", expanded=False):
                        full_rows = []
                        if scores_before:
                            full_rows.append(_all_score_row(scores_before, state="before"))
                        if scores_after:
                            full_rows.append(_all_score_row(scores_after, state="after"))
                        elif scores and not scores_before:
                            full_rows.append(_all_score_row(scores, state="after"))
                        if full_rows:
                            st.dataframe(pd.DataFrame(full_rows), width="stretch")
        else:
            st.info(
                "If you run Rosetta optimization, the optimized structure will become the final structure automatically. "
                "Until then, the propagated structure remains the final structure."
            )
            st.session_state["final_structure_preview"] = propagated_text
            st.session_state["final_structure_source"] = f"propagated_pending_{selected_mode.lower().replace(' ', '_')}"


def _render_merged_export_feedback():
    propagation_result = st.session_state.get("propagation_result_preview") or {}
    membership_rows = propagation_result.get("protofibril_chain_membership", [])
    if not membership_rows:
        return

    st.markdown("**Merged Export Plan**")
    st.caption(
        f"Each propagated protofibril will be collapsed into one output chain. "
        f"When one source chain is appended after another, residue numbering is restarted with an offset gap of "
        f"{MERGED_RESIDUE_GAP} so the original chain breaks remain visible in the numbering."
    )

    grouped_rows: dict[int, list[dict]] = {}
    for row in membership_rows:
        grouped_rows.setdefault(int(row["protofibril_index"]), []).append(row)

    summary_rows = []
    for proto_index, rows in sorted(grouped_rows.items()):
        ordered_rows = sorted(rows, key=lambda row: int(row["position_in_protofibril"]))
        summary_rows.append(
            {
                "protofibril_index": proto_index,
                "merged_output_chain": f"one merged chain for protofibril {proto_index}",
                "source_chain_order": " -> ".join(row["chain_id"] for row in ordered_rows),
                "source_chain_count": len(ordered_rows),
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), width="stretch")
    st.caption(
        "The merged visualization export is a convenience view built from the current final structure. "
        "Its residue numbering keeps chain-break offsets so the stacked source chains stay easy to inspect."
    )


def _render_step_9_export(uploaded_name: str, pdb_text: str):
    _render_step_header(
        9,
        "Export",
        "Download the current structure, the final propagated/optimized structure, and the merged visualization structure.",
    )
    _render_export_placeholder(uploaded_name, pdb_text)


def _download_metadata(uploaded_name: str, default_stem: str, structure_text: str):
    structure_format = detect_structure_format(structure_text)
    suffix = ".cif" if structure_format == "mmcif" else ".pdb"
    mime = "chemical/x-cif" if structure_format == "mmcif" else "chemical/x-pdb"
    stem = (uploaded_name or default_stem).rsplit(".", 1)[0]
    return stem, suffix, mime


def _optimization_mode_slug(mode: str | None) -> str:
    mapping = {
        "Backbone only": "bb",
    }
    return mapping.get(mode or "", "prop")


def _build_session_bundle_zip(uploaded_name: str, original_text: str) -> tuple[bytes, str]:
    original_stem, original_suffix, _ = _download_metadata(uploaded_name, "input_structure", original_text)
    propagated_text = st.session_state.get("propagated_pdb_preview")
    optimized_text = st.session_state.get("optimized_structure_preview")
    final_text = _get_final_structure_text()
    protofibril_configs = st.session_state.get("protofibril_configs_preview", [])

    propagation_result = st.session_state.get("propagation_result_preview") or {}
    propagation_result_meta = {
        "structure_format": propagation_result.get("structure_format"),
        "chain_count": len(propagation_result.get("chain_rows", []) or []),
        "protofibril_chain_membership_count": len(propagation_result.get("protofibril_chain_membership", []) or []),
        "protofibril_chain_membership": propagation_result.get("protofibril_chain_membership", []),
    }

    settings_payload = {
        "app": "mn-fibril-modeller-gemmi",
        "selected_kept_chains": st.session_state.get("selected_kept_chains", []),
        "protofibril_count": st.session_state.get("protofibril_count"),
        "protofibril_configs_preview": st.session_state.get("protofibril_configs_preview", []),
        "current_build_signature": st.session_state.get("current_build_signature"),
        "built_build_signature": st.session_state.get("built_build_signature"),
        "optimization_mode": st.session_state.get("optimization_mode"),
        "optimized_structure_mode": st.session_state.get("optimized_structure_mode"),
        "final_structure_source": st.session_state.get("final_structure_source"),
        "rosetta_coordinate_constraint_weight": st.session_state.get("rosetta_coordinate_constraint_weight", 1.0),
        "rosetta_max_iter": st.session_state.get("rosetta_max_iter", 200),
        "rosetta_constrain_to_start_coords": st.session_state.get("rosetta_constrain_to_start_coords", True),
        "inspect_summary": st.session_state.get("inspect_summary"),
        "optimized_inspect_summary": st.session_state.get("optimized_inspect_summary"),
        "inspect_rosetta_scores": st.session_state.get("inspect_rosetta_scores"),
        "optimized_structure_scores_before": st.session_state.get("optimized_structure_scores_before"),
        "optimized_structure_scores_after": st.session_state.get("optimized_structure_scores_after"),
        "propagation_result_meta": propagation_result_meta,
        "merged_protofibril_backend": st.session_state.get("merged_protofibril_backend"),
        "merged_protofibril_backend_fallback_reason": st.session_state.get("merged_protofibril_backend_fallback_reason"),
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("settings.json", json.dumps(settings_payload, indent=2, sort_keys=True, default=str))
        zf.writestr(f"structures/original{original_suffix}", original_text)

        if propagated_text:
            _, propagated_suffix, _ = _download_metadata(uploaded_name, "propagated_structure", propagated_text)
            zf.writestr(f"structures/propagated{propagated_suffix}", propagated_text)
            try:
                merged_text = build_merged_protofibril_visualization_pdb(
                    pdb_text=propagated_text,
                    protofibril_configs=protofibril_configs,
                    protofibril_chain_membership=propagation_result.get("protofibril_chain_membership", []),
                    residue_gap=MERGED_RESIDUE_GAP,
                )
                zf.writestr("structures/propagated_merged_single_chain.pdb", merged_text)
            except Exception as exc:
                zf.writestr("structures/propagated_merged_single_chain_error.txt", _format_exception_message(exc))

        if optimized_text:
            _, optimized_suffix, _ = _download_metadata(uploaded_name, "optimized_structure", optimized_text)
            optimized_mode = _optimization_mode_slug(st.session_state.get("optimized_structure_mode"))
            zf.writestr(f"structures/minimized_{optimized_mode}{optimized_suffix}", optimized_text)

        if final_text and final_text not in {propagated_text, optimized_text}:
            _, final_suffix, _ = _download_metadata(uploaded_name, "final_structure", final_text)
            final_mode = _optimization_mode_slug(st.session_state.get("optimized_structure_mode"))
            zf.writestr(f"structures/final_{final_mode}{final_suffix}", final_text)

    zip_buffer.seek(0)
    return zip_buffer.getvalue(), f"{original_stem}_session_bundle.zip"


def _render_export_placeholder(uploaded_name: str, pdb_text: str):
    original_stem, original_suffix, original_mime = _download_metadata(uploaded_name, "input_structure", pdb_text)
    st.download_button(
        "Download original structure",
        data=pdb_text,
        file_name=f"{original_stem}{original_suffix}",
        mime=original_mime,
    )
    propagated_pdb = st.session_state.get("propagated_pdb_preview")
    final_structure = _get_final_structure_text()
    if propagated_pdb:
        propagated_name, propagated_suffix, propagated_mime = _download_metadata(uploaded_name, "propagated_structure", propagated_pdb)
        st.download_button(
            "Download propagated structure",
            data=propagated_pdb,
            file_name=f"{propagated_name}_propagated{propagated_suffix}",
            mime=propagated_mime,
        )
    optimized_structure = st.session_state.get("optimized_structure_preview")
    if optimized_structure:
        minimized_name, minimized_suffix, minimized_mime = _download_metadata(uploaded_name, "optimized_structure", optimized_structure)
        mode_slug = _optimization_mode_slug(st.session_state.get("optimized_structure_mode"))
        st.download_button(
            f"Download minimized structure ({mode_slug})",
            data=optimized_structure,
            file_name=f"{minimized_name}_minimized_{mode_slug}{minimized_suffix}",
            mime=minimized_mime,
        )
    elif final_structure and final_structure != propagated_pdb:
        final_name, final_suffix, final_mime = _download_metadata(uploaded_name, "final_structure", final_structure)
        st.download_button(
            "Download final structure",
            data=final_structure,
            file_name=f"{final_name}_final{final_suffix}",
            mime=final_mime,
        )
    bundle_zip, bundle_name = _build_session_bundle_zip(uploaded_name, pdb_text)
    st.download_button(
        "Download full session bundle (zip)",
        data=bundle_zip,
        file_name=bundle_name,
        mime="application/zip",
    )
    if final_structure:
        _render_merged_export_feedback()


def main():
    _sidebar_summary()
    _render_intro()

    uploaded_file = _render_upload_panel()
    if uploaded_file is None:
        st.info("Upload a PDB to begin.")
        return

    pdb_text = _uploaded_text(uploaded_file)
    st.session_state["current_pdb_text"] = pdb_text
    uploaded_suffix = uploaded_file.name.rsplit(".", 1)[-1] if "." in uploaded_file.name else None
    st.session_state["current_structure_format"] = detect_structure_format(pdb_text, uploaded_suffix)
    chain_rows = chain_rows_from_pdb(pdb_text)

    _render_step_1_input(uploaded_file)
    if not chain_rows:
        st.warning("No polymer chains were detected in the uploaded structure.")
        return

    _render_step_2_chain_selection(chain_rows)
    _render_structure_viewer(pdb_text, chain_rows)
    _render_step_3_define_propagation(pdb_text, chain_rows)
    build_settings = _render_step_4_build(chain_rows)
    if build_settings:
        st.session_state["protofibril_configs_preview"] = build_settings["protofibril_configs"]
    _render_step_5_review(st.session_state.get("protofibril_configs_preview", []))
    if build_settings:
        _render_step_header(
            6,
            "Propagate",
            "Build the propagated fibril preview from the current iteration plan.",
        )
        if SHOW_PROPAGATION_DEBUG_UI:
            st.checkbox(
                "Enable propagation debug mode",
                key="propagation_debug_mode",
                value=False,
                help=(
                    "Stores transform/addition/serialization diagnostics for this run. "
                    "Useful when propagation fails with non-obvious errors."
                ),
            )
        else:
            st.session_state["propagation_debug_mode"] = False
        current_build_signature = _stable_signature(
            {
                "keep_chain_ids": st.session_state.get("selected_kept_chains", []),
                "protofibril_configs": build_settings["protofibril_configs"],
                "uploaded_name": uploaded_file.name if uploaded_file else None,
            }
        )
        st.session_state["current_build_signature"] = current_build_signature

        build_col, status_col = st.columns([0.25, 0.75])
        build_clicked = build_col.button(
            "Build propagation preview",
            type="primary",
            width="stretch",
            disabled=not bool(build_settings["protofibril_configs"]),
        )

        if not build_settings["protofibril_configs"]:
            st.session_state["propagated_pdb_preview"] = None
            st.session_state["propagation_result_preview"] = None
            st.session_state["optimized_structure_preview"] = None
            st.session_state["optimized_structure_scores"] = None
            st.session_state["optimized_structure_scores_before"] = None
            st.session_state["optimized_structure_scores_after"] = None
            st.session_state["optimized_structure_mode"] = None
            st.session_state["optimized_inspect_summary"] = None
            st.session_state["inspect_rosetta_scores"] = None
            st.session_state["final_structure_preview"] = None
            st.session_state["final_structure_source"] = None
            st.session_state["inspect_summary"] = None
            st.session_state["inspect_signature"] = None
            st.session_state["merged_protofibril_pdb_preview"] = None
            st.session_state["merged_protofibril_backend"] = None
            st.session_state["merged_protofibril_backend_fallback_reason"] = None
            st.session_state["built_build_signature"] = None
        elif build_clicked:
            try:
                propagation_result = _run_propagation_with_feedback(pdb_text, build_settings)
                st.session_state["merged_protofibril_pdb_preview"] = None
                st.session_state["merged_protofibril_backend"] = None
                st.session_state["merged_protofibril_backend_fallback_reason"] = None
                st.session_state["built_build_signature"] = current_build_signature
                status_col.success("Propagation preview rebuilt with the current settings.")
            except Exception as exc:
                st.session_state["propagated_pdb_preview"] = None
                st.session_state["propagation_result_preview"] = None
                st.session_state["optimized_structure_preview"] = None
                st.session_state["optimized_structure_scores"] = None
                st.session_state["optimized_structure_scores_before"] = None
                st.session_state["optimized_structure_scores_after"] = None
                st.session_state["optimized_structure_mode"] = None
                st.session_state["optimized_inspect_summary"] = None
                st.session_state["inspect_rosetta_scores"] = None
                st.session_state["final_structure_preview"] = None
                st.session_state["final_structure_source"] = None
                st.session_state["inspect_summary"] = None
                st.session_state["inspect_signature"] = None
                st.session_state["merged_protofibril_pdb_preview"] = None
                st.session_state["merged_protofibril_backend"] = None
                st.session_state["merged_protofibril_backend_fallback_reason"] = None
                st.session_state["built_build_signature"] = None
                st.error(f"Propagation preview failed: {_format_exception_message(exc)}")
        elif st.session_state.get("built_build_signature") != current_build_signature:
            status_col.info("Settings changed. Click `Build propagation preview` to update the propagated model.")
        elif st.session_state.get("propagated_pdb_preview"):
            status_col.caption("Preview is up to date with the current settings.")
        if SHOW_PROPAGATION_DEBUG_UI and (
            st.session_state.get("propagation_debug_mode") or st.session_state.get("propagation_debug_events")
        ):
            _render_propagation_debug_panel()
    if st.session_state.get("propagated_pdb_preview"):
        st.success("Propagation build finished. Download buttons are available below.")
        st.info("This preview is the rigid propagated structure. Inspect it first, then decide whether optimization is needed.")
        _render_propagated_model_preview(st.session_state["propagated_pdb_preview"])
        _render_propagated_membership_table()
        _render_step_7_inspect()
        _render_step_8_optimize()
        _render_step_9_export(uploaded_file.name, pdb_text)

    with st.expander("Raw PDB Header", expanded=False):
        st.code(pdb_text[:4000], language="text")
        if len(pdb_text) > 4000:
            st.caption("Preview truncated to the first 4000 characters.")

    if not st.session_state.get("propagated_pdb_preview"):
        _render_step_9_export(uploaded_file.name, pdb_text)


if __name__ == "__main__":
    main()
