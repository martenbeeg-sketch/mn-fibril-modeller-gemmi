from __future__ import annotations

import io
from typing import Dict, List

import gemmi
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser


MMCIF_SS_PREFIXES = (
    "_struct_conf.",
    "_struct_conf_type.",
    "_struct_sheet.",
    "_struct_sheet_order.",
    "_struct_sheet_range.",
    "_pdbx_struct_sheet_hbond.",
)


def detect_structure_format(structure_text: str, format_hint: str | None = None) -> str:
    if format_hint:
        normalized = format_hint.lower().lstrip(".")
        if normalized in {"cif", "mmcif", "mcif"}:
            return "mmcif"
        return "pdb"

    stripped = structure_text.lstrip()
    if stripped.startswith("data_") or "_atom_site." in structure_text:
        return "mmcif"
    return "pdb"


def parse_structure(structure_text: str, format_hint: str | None = None):
    structure_format = detect_structure_format(structure_text, format_hint)
    if structure_format == "mmcif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("input_structure", io.StringIO(structure_text))


def parse_pdb(pdb_text: str):
    return parse_structure(pdb_text, "pdb")


def parse_structure_gemmi(structure_text: str, format_hint: str | None = None) -> gemmi.Structure:
    structure_format = detect_structure_format(structure_text, format_hint)
    if structure_format == "mmcif":
        document = gemmi.cif.read_string(structure_text)
        structure = gemmi.make_structure_from_block(document.sole_block())
        _canonicalize_chain_names_from_auth_asym_id(structure, document.sole_block())
        return structure
    return gemmi.read_pdb_string(structure_text)


def _canonicalize_chain_names_from_auth_asym_id(structure: gemmi.Structure, block: gemmi.cif.Block) -> None:
    """Rename Gemmi chain names to auth_asym_id when mapping is unambiguous.

    Gemmi may expose mmCIF chains via label_asym_id (e.g. Axp), while the app
    viewer and selection logic use auth_asym_id (e.g. A). This canonicalization
    keeps app-level chain IDs consistent for mmCIF input/output.
    """
    try:
        label_col = block.find_values("_atom_site.label_asym_id")
        auth_col = block.find_values("_atom_site.auth_asym_id")
    except Exception:
        return
    if not label_col or not auth_col:
        return

    mapping: dict[str, set[str]] = {}
    for label, auth in zip(label_col, auth_col):
        label_s = str(label).strip()
        auth_s = str(auth).strip()
        if not label_s or not auth_s or label_s in {".", "?"} or auth_s in {".", "?"}:
            continue
        mapping.setdefault(label_s, set()).add(auth_s)

    # Keep only 1:1 label->auth pairs and avoid collisions on target names.
    one_to_one = {label: next(iter(auths)) for label, auths in mapping.items() if len(auths) == 1}
    target_counts: dict[str, int] = {}
    for target in one_to_one.values():
        target_counts[target] = target_counts.get(target, 0) + 1

    safe_mapping = {
        label: target
        for label, target in one_to_one.items()
        if target_counts.get(target, 0) == 1
    }
    if not safe_mapping:
        return

    for model in structure:
        for chain in model:
            replacement = safe_mapping.get(chain.name)
            if replacement:
                chain.name = replacement


def _ensure_auth_atom_id_in_mmcif(structure_text: str) -> str:
    if "_atom_site.auth_atom_id" in structure_text and "_atom_site.auth_comp_id" in structure_text:
        return structure_text

    lines = structure_text.splitlines()
    output_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip() == "loop_":
            header_index = index + 1
            headers: list[str] = []
            while header_index < len(lines) and lines[header_index].lstrip().startswith("_"):
                headers.append(lines[header_index].strip())
                header_index += 1
            if headers and headers[0].startswith("_atom_site."):
                if "_atom_site.label_atom_id" not in headers:
                    output_lines.append(line)
                    output_lines.extend(lines[index + 1 : header_index])
                    index = header_index
                    continue
                new_headers = list(headers)
                insertion_plan: list[tuple[int, str, str]] = []
                if "_atom_site.auth_atom_id" not in headers:
                    label_atom_idx = headers.index("_atom_site.label_atom_id")
                    insertion_plan.append((label_atom_idx + 1, "_atom_site.auth_atom_id", "_atom_site.label_atom_id"))
                if "_atom_site.auth_comp_id" not in headers and "_atom_site.label_comp_id" in headers:
                    label_comp_idx = headers.index("_atom_site.label_comp_id")
                    insertion_plan.append((label_comp_idx + 1, "_atom_site.auth_comp_id", "_atom_site.label_comp_id"))
                insertion_plan.sort(key=lambda item: item[0])
                applied_insertions = 0
                adjusted_insertions: list[tuple[int, str]] = []
                for insert_at, header_name, source_header in insertion_plan:
                    adjusted_at = insert_at + applied_insertions
                    new_headers = new_headers[:adjusted_at] + [header_name] + new_headers[adjusted_at:]
                    adjusted_insertions.append((adjusted_at, source_header))
                    applied_insertions += 1
                output_lines.append(line)
                output_lines.extend(new_headers)
                data_index = header_index
                label_seq_idx = headers.index("_atom_site.label_seq_id") if "_atom_site.label_seq_id" in headers else None
                auth_seq_idx = headers.index("_atom_site.auth_seq_id") if "_atom_site.auth_seq_id" in headers else None
                while data_index < len(lines):
                    next_stripped = lines[data_index].strip()
                    if next_stripped == "loop_" or lines[data_index].lstrip().startswith("_") or next_stripped.startswith("data_"):
                        break
                    tokens = lines[data_index].split()
                    if len(tokens) == len(headers):
                        original_tokens = list(tokens)
                        if label_seq_idx is not None and auth_seq_idx is not None:
                            if original_tokens[label_seq_idx] in {".", "?"} and original_tokens[auth_seq_idx] not in {".", "?"}:
                                original_tokens[label_seq_idx] = original_tokens[auth_seq_idx]
                                tokens[label_seq_idx] = original_tokens[auth_seq_idx]
                        for adjusted_at, source_header in adjusted_insertions:
                            source_idx = headers.index(source_header)
                            tokens = tokens[:adjusted_at] + [original_tokens[source_idx]] + tokens[adjusted_at:]
                        output_lines.append(" ".join(tokens))
                    else:
                        output_lines.append(lines[data_index])
                    data_index += 1
                index = data_index
                continue
        output_lines.append(line)
        index += 1
    return "\n".join(output_lines) + ("\n" if structure_text.endswith("\n") else "")


def normalize_mmcif_for_biopandas(structure_text: str) -> str:
    return _ensure_auth_atom_id_in_mmcif(structure_text)


def _strip_secondary_structure_annotations(structure_text: str, output_format: str) -> str:
    if output_format == "pdb":
        kept_lines = []
        for line in structure_text.splitlines():
            record = line[:6].strip()
            if record in {"HELIX", "SHEET", "TURN"}:
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines) + ("\n" if structure_text.endswith("\n") else "")

    lines = structure_text.splitlines()
    kept_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == "loop_":
            header_index = index + 1
            headers: list[str] = []
            while header_index < len(lines) and lines[header_index].lstrip().startswith("_"):
                headers.append(lines[header_index].strip())
                header_index += 1
            if headers and any(header.startswith(MMCIF_SS_PREFIXES) for header in headers):
                data_index = header_index
                while data_index < len(lines):
                    next_stripped = lines[data_index].strip()
                    if next_stripped == "loop_" or lines[data_index].lstrip().startswith("_") or next_stripped.startswith("data_"):
                        break
                    data_index += 1
                index = data_index
                continue
        if stripped.startswith(MMCIF_SS_PREFIXES):
            index += 1
            continue
        kept_lines.append(line)
        index += 1
    return "\n".join(kept_lines) + ("\n" if structure_text.endswith("\n") else "")


def serialize_structure_gemmi(structure: gemmi.Structure, output_format: str = "mmcif") -> str:
    if output_format == "mmcif":
        structure_for_write = structure.clone()
        for model in structure_for_write:
            for chain in model:
                chain_name = str(chain.name).strip()
                if not chain_name:
                    chain_name = "X"
                chain.name = chain_name
                for residue in chain:
                    residue.subchain = chain_name
                    entity_id = str(residue.entity_id).strip() if residue.entity_id is not None else ""
                    if not entity_id or entity_id in {".", "?"}:
                        residue.entity_id = "1"
        structure_for_write.remove_empty_chains()
        # Keep Gemmi's entity/asym mapping coherent for generated chains.
        structure_for_write.setup_entities()
        structure_for_write.add_entity_types()
        structure_for_write.add_entity_ids()
        document = structure_for_write.make_mmcif_document()
        return _strip_secondary_structure_annotations(document.as_string(), output_format)
    return _strip_secondary_structure_annotations(structure.make_pdb_string(), output_format)


def polymer_residues(chain) -> List:
    return [residue for residue in chain if residue.id[0] == " "]


def residue_id_string(residue) -> str:
    seq_num = residue.id[1]
    insertion_code = residue.id[2].strip()
    return f"{seq_num}{insertion_code}" if insertion_code else str(seq_num)


def _gemmi_polymer_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    # Match the current Biopython-based app behavior: keep standard polymer-style
    # residues by het flag, even when Gemmi's stricter polymer perception would
    # drop minimal/incomplete test residues.
    return [residue for residue in chain if residue.het_flag == "A"]


def _gemmi_residue_id_string(residue: gemmi.Residue) -> str:
    insertion_code = residue.seqid.icode.strip()
    seq_num = residue.seqid.num
    return f"{seq_num}{insertion_code}" if insertion_code else str(seq_num)


def chain_rows_from_pdb(pdb_text: str, format_hint: str | None = None) -> List[Dict[str, object]]:
    structure = parse_structure_gemmi(pdb_text, format_hint)
    rows: List[Dict[str, object]] = []
    if len(structure) == 0:
        return rows

    model = structure[0]
    for chain in model:
        residues = _gemmi_polymer_residues(chain)
        if not residues:
            continue
        rows.append(
            {
                "chain_id": chain.name,
                "residue_count": len(residues),
                "start_residue": _gemmi_residue_id_string(residues[0]),
                "end_residue": _gemmi_residue_id_string(residues[-1]),
            }
        )
    return rows


def chain_lengths_from_pdb(pdb_text: str, format_hint: str | None = None) -> Dict[str, int]:
    return {row["chain_id"]: row["residue_count"] for row in chain_rows_from_pdb(pdb_text, format_hint)}


def chain_centroids_from_pdb(pdb_text: str, format_hint: str | None = None) -> Dict[str, tuple[float, float, float]]:
    structure = parse_structure_gemmi(pdb_text, format_hint)
    centroids: Dict[str, tuple[float, float, float]] = {}
    if len(structure) == 0:
        return centroids

    model = structure[0]
    for chain in model:
        residues = _gemmi_polymer_residues(chain)
        if not residues:
            continue

        coords: list[tuple[float, float, float]] = []
        for residue in residues:
            ca_atom = next((atom for atom in residue if atom.name.strip() == "CA"), None)
            if ca_atom is not None:
                coords.append((float(ca_atom.pos.x), float(ca_atom.pos.y), float(ca_atom.pos.z)))
            else:
                coords.extend(
                    (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                    for atom in residue
                )
        if not coords:
            continue

        coord_array = np.array(coords, dtype=float)
        centroid = coord_array.mean(axis=0)
        centroids[chain.name] = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    return centroids


def ordered_chain_ids_from_pdb(
    pdb_text: str, chain_ids: List[str] | None = None, axis: str = "z", format_hint: str | None = None
) -> List[str]:
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    axis_index = axis_to_index.get(axis, 2)
    centroids = chain_centroids_from_pdb(pdb_text, format_hint)
    selected_ids = chain_ids or list(centroids)
    selected_ids = [chain_id for chain_id in selected_ids if chain_id in centroids]
    return sorted(selected_ids, key=lambda chain_id: centroids[chain_id][axis_index])


def _infer_fibril_axis(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 2:
        axis = coords[-1] - coords[0]
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        axis = axis / norm
    else:
        pairwise = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        nearest_indices = np.argmin(pairwise, axis=1)
        orientation_tensor = np.zeros((3, 3), dtype=float)
        for index, neighbor_index in enumerate(nearest_indices):
            vector = coords[neighbor_index] - coords[index]
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            unit_vector = vector / norm
            orientation_tensor += np.outer(unit_vector, unit_vector)
        eigenvalues, eigenvectors = np.linalg.eigh(orientation_tensor)
        axis = eigenvectors[:, np.argmax(eigenvalues)]

    dominant_component = int(np.argmax(np.abs(axis)))
    if axis[dominant_component] < 0:
        axis = -axis
    return axis


def principal_axis_ordered_chain_ids_from_pdb(
    pdb_text: str, chain_ids: List[str] | None = None, format_hint: str | None = None
) -> List[str]:
    centroids = chain_centroids_from_pdb(pdb_text, format_hint)
    selected_ids = chain_ids or list(centroids)
    selected_ids = [chain_id for chain_id in selected_ids if chain_id in centroids]
    if len(selected_ids) <= 2:
        return selected_ids

    coords = np.array([centroids[chain_id] for chain_id in selected_ids], dtype=float)
    centered = coords - coords.mean(axis=0)
    principal_axis = _infer_fibril_axis(centered)
    projections = centered @ principal_axis
    ordered_indices = np.argsort(projections)
    return [selected_ids[index] for index in ordered_indices]


def suggest_protofibril_groups_from_pdb(
    pdb_text: str, chain_ids: List[str] | None = None, format_hint: str | None = None
) -> List[List[str]]:
    centroids = chain_centroids_from_pdb(pdb_text, format_hint)
    selected_ids = chain_ids or list(centroids)
    selected_ids = [chain_id for chain_id in selected_ids if chain_id in centroids]
    if len(selected_ids) <= 1:
        return [selected_ids] if selected_ids else []
    if len(selected_ids) == 2:
        return [selected_ids]

    coords = np.array([centroids[chain_id] for chain_id in selected_ids], dtype=float)
    centered = coords - coords.mean(axis=0)
    principal_axis = _infer_fibril_axis(centered)
    projections = centered @ principal_axis
    projected_coords = np.outer(projections, principal_axis)
    orthogonal_coords = centered - projected_coords

    pairwise = np.linalg.norm(
        orthogonal_coords[:, np.newaxis, :] - orthogonal_coords[np.newaxis, :, :],
        axis=2,
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest = np.min(pairwise, axis=1)
    finite_nearest = nearest[np.isfinite(nearest)]
    if finite_nearest.size == 0:
        return [principal_axis_ordered_chain_ids_from_pdb(pdb_text, selected_ids, format_hint)]

    threshold = float(np.median(finite_nearest) * 1.5)
    parent = list(range(len(selected_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(selected_ids)):
        for right in range(left + 1, len(selected_ids)):
            if pairwise[left, right] <= threshold:
                union(left, right)

    groups: Dict[int, List[str]] = {}
    for index, chain_id in enumerate(selected_ids):
        groups.setdefault(find(index), []).append(chain_id)

    ordered_groups = [
        principal_axis_ordered_chain_ids_from_pdb(pdb_text, group_chain_ids, format_hint)
        for group_chain_ids in groups.values()
    ]
    return sorted(ordered_groups, key=lambda group_chain_ids: (-len(group_chain_ids), group_chain_ids))
