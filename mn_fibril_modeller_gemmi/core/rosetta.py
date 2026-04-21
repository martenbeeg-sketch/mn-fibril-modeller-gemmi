from __future__ import annotations

import json
import queue
import re
import shutil
import string
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from mn_fibril_modeller_gemmi.core.pdb_io import (
    detect_structure_format,
    parse_structure_gemmi,
    serialize_structure_gemmi,
)


DEFAULT_DOCKER_ROSETTA_IMAGE = "ovo-proteinmpnn-fastrelax"
PDB_CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def is_docker_available() -> bool:
    return shutil.which("docker") is not None


def get_default_docker_rosetta_image() -> str:
    return DEFAULT_DOCKER_ROSETTA_IMAGE


def _build_pdb_safe_chain_map(structure: "gemmi.Structure") -> dict[str, str]:
    model = structure[0]
    source_chain_ids = [chain.name for chain in model]
    used_targets: set[str] = set()
    mapping: dict[str, str] = {}

    # Keep already-safe one-letter IDs when possible.
    for chain_id in source_chain_ids:
        if len(chain_id) == 1 and chain_id in PDB_CHAIN_IDS and chain_id not in used_targets:
            mapping[chain_id] = chain_id
            used_targets.add(chain_id)

    available_ids = [chain_id for chain_id in PDB_CHAIN_IDS if chain_id not in used_targets]
    available_index = 0
    for chain_id in source_chain_ids:
        if chain_id in mapping:
            continue
        if available_index >= len(available_ids):
            raise RuntimeError(
                "Rosetta optimization requires PDB-safe one-letter chain IDs, but this structure exceeds "
                f"the available {len(PDB_CHAIN_IDS)} PDB chain identifiers. Try optimizing a reduced/merged view first."
            )
        mapping[chain_id] = available_ids[available_index]
        available_index += 1
    return mapping


def _rename_chains_in_gemmi_structure(structure: "gemmi.Structure", chain_map: dict[str, str]) -> "gemmi.Structure":
    renamed = structure.clone()
    model = renamed[0]
    for chain in model:
        chain.name = chain_map.get(chain.name, chain.name)
    return renamed


def run_docker_rosetta_optimization(
    structure_text: str,
    optimization_mode: str,
    structure_format: str | None = None,
    status_callback=None,
    coordinate_constraint_weight: float = 1.0,
    max_iter: int = 200,
    constrain_to_start_coords: bool = True,
) -> dict:
    if not is_docker_available():
        raise RuntimeError("Docker is not available on PATH.")

    if optimization_mode not in {
        "Score only",
        "Backbone only",
        "Side chains only",
        "Backbone + side chains (coupled)",
    }:
        raise ValueError(f"Unsupported optimization mode: {optimization_mode}")

    docker_image = get_default_docker_rosetta_image()
    input_format = detect_structure_format(structure_text, structure_format)
    input_structure = parse_structure_gemmi(structure_text, input_format)
    chain_map = _build_pdb_safe_chain_map(input_structure)
    reverse_chain_map = {value: key for key, value in chain_map.items()}
    rosetta_input_structure = _rename_chains_in_gemmi_structure(input_structure, chain_map)
    input_pdb = serialize_structure_gemmi(rosetta_input_structure, "pdb")

    package_root = Path(__file__).resolve().parents[2]
    script_path = "/app/mn_fibril_modeller_gemmi/rosetta/container_pyrosetta_fastrelax.py"
    mode_to_container = {
        "Score only": "score_only",
        "Backbone only": "backbone_only",
        "Side chains only": "sidechains_only",
        "Backbone + side chains (coupled)": "coupled_backbone_sidechains",
    }
    container_mode = mode_to_container[optimization_mode]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_pdb_path = tmp_path / "input_structure.pdb"
        output_pdb_path = tmp_path / "optimized_structure.pdb"
        output_json_path = tmp_path / "scores.json"
        input_pdb_path.write_text(input_pdb, encoding="utf-8")

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{package_root}:/app:ro",
            "-v",
            f"{tmp_dir}:/work",
            "-w",
            "/work",
            docker_image,
            "python3",
            script_path,
            "/work/input_structure.pdb",
            "/work/optimized_structure.pdb",
            "--scores-json",
            "/work/scores.json",
            "--mode",
            container_mode,
            "--coord-cst-weight",
            str(float(coordinate_constraint_weight)),
            "--max-iter",
            str(int(max_iter)),
        ]
        if constrain_to_start_coords:
            cmd.append("--constrain-to-start")
        else:
            cmd.append("--no-constrain-to-start")
        started_at = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log_queue: queue.Queue[str] = queue.Queue()
        live_logs: list[str] = []
        current_stage = "starting_container"

        def _log_reader():
            if process.stdout is None:
                return
            for line in process.stdout:
                stripped = line.strip()
                log_queue.put(stripped)

        reader_thread = threading.Thread(target=_log_reader, daemon=True)
        reader_thread.start()
        current_iteration: int | None = None

        while process.poll() is None:
            while True:
                try:
                    log_line = log_queue.get_nowait()
                except queue.Empty:
                    break
                live_logs.append(log_line)
                if log_line.startswith("MN_STAGE "):
                    current_stage = log_line.split(" ", 1)[1].strip()
                lower_line = log_line.lower()
                iter_match = re.search(r"\biter(?:ation)?\s*[:=]?\s*(\d+)\b", lower_line)
                if iter_match:
                    try:
                        current_iteration = int(iter_match.group(1))
                    except ValueError:
                        current_iteration = current_iteration
            if status_callback is not None:
                elapsed = int(time.time() - started_at)
                status_callback(
                    elapsed_seconds=elapsed,
                    stage=current_stage,
                    current_iteration=current_iteration,
                    max_iter=int(max_iter),
                )
            time.sleep(1.0)

        while True:
            try:
                log_line = log_queue.get_nowait()
            except queue.Empty:
                break
            live_logs.append(log_line)
            if log_line.startswith("MN_STAGE "):
                current_stage = log_line.split(" ", 1)[1].strip()
            lower_line = log_line.lower()
            iter_match = re.search(r"\biter(?:ation)?\s*[:=]?\s*(\d+)\b", lower_line)
            if iter_match:
                try:
                    current_iteration = int(iter_match.group(1))
                except ValueError:
                    current_iteration = current_iteration

        stdout = "\n".join(live_logs).strip()
        stderr = ""
        if process.returncode != 0:
            message = stderr.strip() or stdout.strip() or f"Docker run failed with exit code {process.returncode}"
            raise RuntimeError(message)

        optimized_pdb = output_pdb_path.read_text(encoding="utf-8")
        optimized_structure = parse_structure_gemmi(optimized_pdb, "pdb")
        restored_structure = _rename_chains_in_gemmi_structure(optimized_structure, reverse_chain_map)
        optimized_structure_text = serialize_structure_gemmi(
            restored_structure,
            "mmcif" if input_format == "mmcif" or any(len(chain_id) > 1 for chain_id in chain_map) else "pdb",
        )
        scores = {}
        scores_before = {}
        scores_after = {}
        if output_json_path.exists():
            parsed_scores = json.loads(output_json_path.read_text(encoding="utf-8"))
            if isinstance(parsed_scores, dict) and "before" in parsed_scores and "after" in parsed_scores:
                scores_before = parsed_scores.get("before") or {}
                scores_after = parsed_scores.get("after") or {}
                scores = scores_after
            else:
                scores = parsed_scores if isinstance(parsed_scores, dict) else {}
                scores_after = scores
        return {
            "structure_text": optimized_structure_text,
            "structure_format": detect_structure_format(optimized_structure_text),
            "scores": scores,
            "scores_before": scores_before,
            "scores_after": scores_after,
            "docker_image": docker_image,
            "optimization_mode": optimization_mode,
        }
