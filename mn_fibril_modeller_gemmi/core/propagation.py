from __future__ import annotations

import copy
import io
import math
import string
import tempfile

import gemmi
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFIO, PDBIO, Superimposer
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from biopandas.mmcif import PandasMmcif
from biopandas.pdb import PandasPdb

from mn_fibril_modeller_gemmi.core.pdb_io import (
    detect_structure_format,
    normalize_mmcif_for_biopandas,
    parse_structure,
    parse_structure_gemmi,
    polymer_residues,
    residue_id_string,
    serialize_structure_gemmi,
)


PDB_CHAIN_IDS = string.ascii_uppercase + string.ascii_lowercase + string.digits


def _debug_log(debug_sink: list[dict] | None, event: str, **payload):
    if debug_sink is None:
        return
    record = {"event": event}
    record.update(payload)
    debug_sink.append(record)


def filter_pdb_to_chains(
    pdb_text: str,
    keep_chain_ids: list[str],
    input_format: str | None = None,
    output_format: str = "pdb",
) -> str:
    keep_chain_ids_set = set(keep_chain_ids)
    structure = parse_structure_gemmi(pdb_text, input_format).clone()
    for model in structure:
        chain_names_to_remove = [chain.name for chain in model if chain.name not in keep_chain_ids_set]
        for chain_name in chain_names_to_remove:
            model.remove_chain(chain_name)
    return serialize_structure_gemmi(structure, output_format)


def _matching_atoms(source_chain, target_chain):
    source_residues = {residue_id_string(residue): residue for residue in polymer_residues(source_chain)}
    target_residues = {residue_id_string(residue): residue for residue in polymer_residues(target_chain)}
    common_ids = [residue_id for residue_id in source_residues if residue_id in target_residues]

    source_atoms = []
    target_atoms = []
    backbone_atom_names = ("N", "CA", "C", "O")
    for residue_id in common_ids:
        source_residue = source_residues[residue_id]
        target_residue = target_residues[residue_id]
        for atom_name in backbone_atom_names:
            if atom_name in source_residue and atom_name in target_residue:
                source_atoms.append(source_residue[atom_name])
                target_atoms.append(target_residue[atom_name])

    if len(source_atoms) < 3:
        source_atoms = []
        target_atoms = []
        for residue_id in common_ids:
            source_residue = source_residues[residue_id]
            target_residue = target_residues[residue_id]
            if "CA" in source_residue and "CA" in target_residue:
                source_atoms.append(source_residue["CA"])
                target_atoms.append(target_residue["CA"])

    if len(source_atoms) < 3:
        raise ValueError("Not enough matching atoms were found to compute a stable chain transform.")

    return source_atoms, target_atoms


def _compute_chain_transform(model, source_chain_id: str, target_chain_id: str):
    source_chain = model[source_chain_id]
    target_chain = model[target_chain_id]
    source_atoms, target_atoms = _matching_atoms(source_chain, target_chain)
    superimposer = Superimposer()
    superimposer.set_atoms(target_atoms, source_atoms)
    rotation, translation = superimposer.rotran
    return rotation, translation, float(superimposer.rms)


def _extract_helical_metadata_from_mmcif_text(structure_text: str) -> dict[str, object] | None:
    try:
        document = gemmi.cif.read_string(structure_text)
        block = document.sole_block()
    except Exception:
        return None

    def _first_value(tag: str) -> str | None:
        try:
            values = block.find_values(tag)
        except Exception:
            return None
        if not values:
            return None
        value = str(values[0]).strip()
        if not value or value in {".", "?"}:
            return None
        return value

    angle_text = _first_value("_em_helical_entity.angular_rotation_per_subunit")
    rise_text = _first_value("_em_helical_entity.axial_rise_per_subunit")
    symmetry_text = _first_value("_em_helical_entity.axial_symmetry")
    if angle_text is None or rise_text is None:
        return None

    try:
        angle_deg = float(angle_text)
        rise_angstrom = float(rise_text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(angle_deg) or not math.isfinite(rise_angstrom):
        return None
    if abs(rise_angstrom) < 1e-8:
        return None

    return {
        "angle_deg_per_subunit": angle_deg,
        "rise_angstrom_per_subunit": rise_angstrom,
        "axial_symmetry": symmetry_text,
    }


def _gemmi_chain_centroid(chain: gemmi.Chain) -> np.ndarray:
    coords: list[np.ndarray] = []
    for residue in chain:
        ca_atom = next((atom for atom in residue if atom.name.strip() == "CA"), None)
        if ca_atom is not None:
            coords.append(np.array([float(ca_atom.pos.x), float(ca_atom.pos.y), float(ca_atom.pos.z)], dtype=float))
        else:
            for atom in residue:
                coords.append(np.array([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)], dtype=float))
    if not coords:
        raise ValueError(f"Chain {chain.name} has no atoms for centroid estimation.")
    stacked = np.vstack(coords)
    return stacked.mean(axis=0)


def _estimate_protofibril_axis(
    model: gemmi.Model,
    proto_chain_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    centroids: dict[str, np.ndarray] = {}
    ordered_centroids: list[np.ndarray] = []
    for chain_id in proto_chain_ids:
        chain = _find_gemmi_chain(model, chain_id)
        centroid = _gemmi_chain_centroid(chain)
        centroids[chain_id] = centroid
        ordered_centroids.append(centroid)
    if len(ordered_centroids) < 2:
        raise ValueError("At least two chains are required to estimate a protofibril axis.")

    centroid_stack = np.vstack(ordered_centroids)
    axis_point = centroid_stack.mean(axis=0)
    centered = centroid_stack - axis_point
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("Failed to estimate protofibril axis from chain centroids.")
    axis_unit = vh[0].astype(float)
    axis_norm = float(np.linalg.norm(axis_unit))
    if axis_norm < 1e-10:
        raise ValueError("Estimated protofibril axis has near-zero length.")
    axis_unit = axis_unit / axis_norm

    first_centroid = ordered_centroids[0]
    last_centroid = ordered_centroids[-1]
    if float(np.dot(last_centroid - first_centroid, axis_unit)) < 0.0:
        axis_unit = -axis_unit

    return axis_point, axis_unit, centroids


def _rotation_matrix_from_axis_angle(axis_unit: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    x, y, z = axis_unit.astype(float)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def _helical_pair_step_count(
    source_chain_id: str,
    target_chain_id: str,
    *,
    centroids: dict[str, np.ndarray],
    axis_unit: np.ndarray,
    rise_per_subunit: float,
) -> int:
    source_centroid = centroids[source_chain_id]
    target_centroid = centroids[target_chain_id]
    delta = target_centroid - source_centroid
    axial_distance = float(np.dot(delta, axis_unit))
    estimated_steps = int(round(abs(axial_distance) / abs(rise_per_subunit)))
    return max(1, estimated_steps)


def _build_helical_screw_transform(
    *,
    axis_point: np.ndarray,
    axis_unit: np.ndarray,
    angle_deg: float,
    rise_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_column = _rotation_matrix_from_axis_angle(axis_unit, angle_deg)
    rotation_row = rotation_column.T
    translation = axis_point - rotation_column @ axis_point + rise_angstrom * axis_unit
    return rotation_row, translation


def _next_available_chain_id(used_chain_ids: set[str], output_format: str = "pdb") -> str:
    if output_format == "mmcif":
        index = 1
        while True:
            candidate = f"PF{index}"
            if candidate not in used_chain_ids:
                return candidate
            index += 1
    for chain_id in PDB_CHAIN_IDS:
        if chain_id not in used_chain_ids:
            return chain_id
    raise ValueError("No free PDB chain identifiers are available.")


def _clone_chain_with_transform(source_chain, new_chain_id: str, rotation: np.ndarray, translation: np.ndarray):
    cloned_chain = copy.deepcopy(source_chain)
    cloned_chain.id = new_chain_id
    for atom in cloned_chain.get_atoms():
        atom.transform(rotation, translation)
    return cloned_chain


def _gemmi_transform(rotation: np.ndarray, translation: np.ndarray) -> gemmi.Transform:
    transform = gemmi.Transform()
    # Bio.PDB atom.transform applies coordinates as coord @ rotation + translation.
    # Gemmi Transform applies matrix * coord + vector, so we need the transpose
    # to preserve the same rigid-body transform convention.
    transform.mat.fromlist(rotation.T.tolist())
    transform.vec.fromlist(translation.tolist())
    return transform


def _find_gemmi_chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain:
    chain = model.find_chain(chain_id)
    if chain is None:
        raise KeyError(f"Chain {chain_id} was not found in the Gemmi model.")
    return chain


def _clone_chain_with_transform_gemmi(
    source_chain: gemmi.Chain,
    new_chain_id: str,
    rotation: np.ndarray,
    translation: np.ndarray,
    source_entity_id: str | None = None,
) -> gemmi.Chain:
    cloned_chain = source_chain.clone()
    cloned_chain.name = new_chain_id
    # Gemmi chain cloning preserves coordinates but may keep/clear metadata in a
    # way that creates invalid mmCIF asym/entity mappings for generated chains.
    # Reassign per-residue subchain and entity id explicitly.
    source_residues = [res for res in source_chain]
    cloned_residues = [res for res in cloned_chain]
    default_entity_id = str(source_entity_id).strip() if source_entity_id is not None else ""
    if not default_entity_id or default_entity_id in {".", "?"}:
        default_entity_id = "1"
    for src_residue, cloned_residue in zip(source_residues, cloned_residues):
        cloned_residue.subchain = new_chain_id
        src_entity_id = str(src_residue.entity_id).strip() if src_residue.entity_id is not None else ""
        if source_entity_id:
            cloned_residue.entity_id = str(source_entity_id)
        elif src_entity_id and src_entity_id not in {".", "?"}:
            cloned_residue.entity_id = src_entity_id
        else:
            cloned_residue.entity_id = default_entity_id
    transform = _gemmi_transform(rotation, translation)
    for residue in cloned_chain:
        for atom in residue:
            transformed = transform.apply(atom.pos)
            atom.pos = [transformed.x, transformed.y, transformed.z]
    return cloned_chain


def _validate_chain_coordinates(
    chain: gemmi.Chain,
    *,
    proto_index: int,
    direction: str,
    source_chain_id: str,
):
    max_abs_coord = 0.0
    for residue in chain:
        for atom in residue:
            x = float(atom.pos.x)
            y = float(atom.pos.y)
            z = float(atom.pos.z)
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                raise ValueError(
                    f"Propagation produced non-finite coordinates while adding to {direction} "
                    f"(protofibril {proto_index}, source chain {source_chain_id}). "
                    "Check selected top/bottom reference pairs."
                )
            max_abs_coord = max(max_abs_coord, abs(x), abs(y), abs(z))
    if max_abs_coord > 1_000_000.0:
        raise ValueError(
            f"Propagation transform is unstable (|coord|={max_abs_coord:.2f}) while adding to {direction} "
            f"(protofibril {proto_index}, source chain {source_chain_id}). "
            "This usually means the selected reference pair points in the wrong direction."
        )


def _chain_ca_coords(chain: gemmi.Chain) -> np.ndarray:
    coords = []
    for residue in chain:
        for atom in residue:
            if atom.name.strip() == "CA":
                coords.append([float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)])
                break
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _detect_overlapping_chain(
    model: gemmi.Model,
    candidate_chain: gemmi.Chain,
    *,
    ignore_chain_ids: set[str],
    min_ca_atoms: int = 8,
    median_distance_threshold: float = 0.35,
) -> str | None:
    candidate_ca = _chain_ca_coords(candidate_chain)
    if candidate_ca.shape[0] < min_ca_atoms:
        return None

    for chain in model:
        if chain.name in ignore_chain_ids:
            continue
        target_ca = _chain_ca_coords(chain)
        if target_ca.shape[0] < min_ca_atoms:
            continue
        deltas = candidate_ca[:, None, :] - target_ca[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        nearest = np.min(distances, axis=1)
        if float(np.median(nearest)) < median_distance_threshold:
            return chain.name
    return None


def _chain_entity_id_map(structure: gemmi.Structure, model: gemmi.Model) -> dict[str, str]:
    subchain_to_entity: dict[str, str] = {}
    for entity in structure.entities:
        entity_name = str(entity.name)
        for subchain_id in getattr(entity, "subchains", []):
            subchain_to_entity[str(subchain_id)] = entity_name

    chain_to_entity: dict[str, str] = {}
    for chain in model:
        spans = list(chain.subchains())
        if not spans:
            continue
        first_span = spans[0]
        subchain_id = first_span.subchain_id() if hasattr(first_span, "subchain_id") else str(first_span)
        mapped_entity = subchain_to_entity.get(str(subchain_id))
        if mapped_entity:
            chain_to_entity[chain.name] = mapped_entity
    return chain_to_entity


def _mmcif_chain_sanity_summary(model: gemmi.Model) -> dict:
    invalid_chain_names = []
    invalid_residue_fields = []
    chain_count = 0
    residue_count = 0
    atom_count = 0

    for chain in model:
        chain_count += 1
        chain_name = str(chain.name).strip()
        if not chain_name:
            invalid_chain_names.append("<empty>")
        for residue in chain:
            residue_count += 1
            atom_count += len(residue)
            subchain = str(residue.subchain).strip()
            entity_id = str(residue.entity_id).strip() if residue.entity_id is not None else ""
            if not subchain or subchain in {".", "?"}:
                invalid_residue_fields.append(
                    f"{chain_name}:{residue.seqid.num} missing subchain"
                )
            if not entity_id or entity_id in {".", "?"}:
                invalid_residue_fields.append(
                    f"{chain_name}:{residue.seqid.num} missing entity_id"
                )
            if len(invalid_residue_fields) >= 8:
                break
        if len(invalid_residue_fields) >= 8:
            break

    return {
        "chain_count": chain_count,
        "residue_count": residue_count,
        "atom_count": atom_count,
        "invalid_chain_names": invalid_chain_names[:8],
        "invalid_residue_examples": invalid_residue_fields[:8],
    }


def _serialize_structure(structure, output_format: str = "pdb") -> str:
    buffer = io.StringIO()
    intermediate_format = "mmcif" if output_format == "mmcif" else "pdb"
    io_writer = MMCIFIO() if intermediate_format == "mmcif" else PDBIO()
    io_writer.set_structure(structure)
    io_writer.save(buffer)
    gemmi_structure = parse_structure_gemmi(buffer.getvalue(), intermediate_format)
    return serialize_structure_gemmi(gemmi_structure, output_format)


def _build_merged_protofibril_visualization_pdb_biopython(
    pdb_text: str,
    protofibril_configs: list[dict] | None = None,
    protofibril_chain_membership: list[dict] | None = None,
    residue_gap: int = 20,
    structure_format: str | None = None,
    progress_callback=None,
) -> str:
    output_format = detect_structure_format(pdb_text, structure_format)
    structure = parse_structure(pdb_text, output_format)
    model = next(structure.get_models())

    merged_structure = Structure("merged_protofibrils")
    merged_model = Model(0)
    merged_structure.add(merged_model)

    membership_by_protofibril: dict[int, list[dict]] = {}
    if protofibril_chain_membership:
        for row in protofibril_chain_membership:
            membership_by_protofibril.setdefault(int(row["protofibril_index"]), []).append(row)
        proto_entries = sorted(membership_by_protofibril.items())
    else:
        proto_entries = [
            (proto_index, [{"chain_id": chain_id, "position_in_protofibril": position} for position, chain_id in enumerate(config.get("chains", []), start=1)])
            for proto_index, config in enumerate(protofibril_configs or [], start=1)
        ]

    total_source_chains = sum(
        len(
            [
                row["chain_id"]
                for row in sorted(membership_rows, key=lambda row: int(row.get("position_in_protofibril", 0)))
            ]
        )
        for _, membership_rows in proto_entries
    )
    processed_source_chains = 0

    for proto_index, membership_rows in proto_entries:
        proto_chain_ids = [
            row["chain_id"]
            for row in sorted(membership_rows, key=lambda row: int(row.get("position_in_protofibril", 0)))
        ]
        if not proto_chain_ids:
            continue
        merged_chain_id = f"PF{proto_index}" if output_format == "mmcif" else PDB_CHAIN_IDS[proto_index - 1]
        merged_chain = Chain(merged_chain_id)
        residue_offset = 0

        for source_chain_id in proto_chain_ids:
            if progress_callback is not None:
                progress_callback(
                    processed_source_chains=processed_source_chains,
                    total_source_chains=total_source_chains,
                    protofibril_index=proto_index,
                    source_chain_id=source_chain_id,
                )
            source_chain = model[source_chain_id]
            source_residues = polymer_residues(source_chain)
            if not source_residues:
                processed_source_chains += 1
                continue
            min_seq = min(residue.id[1] for residue in source_residues)
            max_seq = max(residue.id[1] for residue in source_residues)

            for residue in source_residues:
                cloned_residue = copy.deepcopy(residue)
                cloned_residue.detach_parent()
                hetflag, seq_id, insertion_code = cloned_residue.id
                normalized_seq_id = seq_id - min_seq + 1 + residue_offset
                cloned_residue.id = (hetflag, normalized_seq_id, insertion_code)
                merged_chain.add(cloned_residue)

            residue_offset += (max_seq - min_seq + 1) + residue_gap
            processed_source_chains += 1

        merged_model.add(merged_chain)

    return _serialize_structure(merged_structure, output_format)


def _build_proto_entries(
    protofibril_configs: list[dict] | None = None,
    protofibril_chain_membership: list[dict] | None = None,
):
    membership_by_protofibril: dict[int, list[dict]] = {}
    if protofibril_chain_membership:
        for row in protofibril_chain_membership:
            membership_by_protofibril.setdefault(int(row["protofibril_index"]), []).append(row)
        return sorted(membership_by_protofibril.items())
    return [
        (
            proto_index,
            [{"chain_id": chain_id, "position_in_protofibril": position} for position, chain_id in enumerate(config.get("chains", []), start=1)],
        )
        for proto_index, config in enumerate(protofibril_configs or [], start=1)
    ]


def _read_structure_with_biopandas(
    structure_text: str,
    structure_format: str,
    source_chain_ids: list[str],
) -> PandasPdb:
    source_chain_ids_set = set(source_chain_ids)
    structure_text_for_biopandas = serialize_structure_gemmi(
        parse_structure_gemmi(structure_text, structure_format),
        "mmcif",
    )
    structure_text_for_biopandas = normalize_mmcif_for_biopandas(structure_text_for_biopandas)
    with tempfile.NamedTemporaryFile("w+", suffix=".cif") as handle:
        handle.write(structure_text_for_biopandas)
        handle.flush()
        pandas_mmcif = PandasMmcif().read_mmcif(handle.name)
    pandas_pdb = pandas_mmcif.convert_to_pandas_pdb()
    atom_table = pandas_mmcif.df.get("ATOM")
    residue_name_column = None
    for candidate in ("auth_comp_id", "label_comp_id"):
        if (
            atom_table is not None
            and candidate in atom_table.columns
            and not atom_table[candidate].isna().all()
        ):
            residue_name_column = candidate
            break
    for record_name in ("ATOM", "HETATM"):
        record_df = pandas_pdb.df.get(record_name)
        if record_df is None or record_df.empty:
            continue
        filtered_df = record_df[record_df["chain_id"].isin(source_chain_ids_set)].copy().reset_index(drop=True)
        if atom_table is not None and residue_name_column is not None:
            atom_record_df = atom_table[atom_table["group_PDB"] == record_name].reset_index(drop=True)
            atom_record_df = atom_record_df[atom_record_df["auth_asym_id"].isin(source_chain_ids_set)].reset_index(drop=True)
            if len(atom_record_df) == len(filtered_df):
                filtered_df["residue_name"] = atom_record_df[residue_name_column].astype(str).to_numpy()
        pandas_pdb.df[record_name] = filtered_df
    return pandas_pdb


def _build_merged_protofibril_visualization_pdb_biopandas(
    pdb_text: str,
    protofibril_configs: list[dict] | None = None,
    protofibril_chain_membership: list[dict] | None = None,
    residue_gap: int = 20,
    structure_format: str | None = None,
    progress_callback=None,
) -> str:
    proto_entries = _build_proto_entries(
        protofibril_configs=protofibril_configs,
        protofibril_chain_membership=protofibril_chain_membership,
    )
    source_chain_ids = list(
        dict.fromkeys(
            row["chain_id"]
            for _, membership_rows in proto_entries
            for row in sorted(membership_rows, key=lambda row: int(row.get("position_in_protofibril", 0)))
        )
    )
    source = _read_structure_with_biopandas(
        pdb_text,
        detect_structure_format(pdb_text, structure_format),
        source_chain_ids,
    )
    total_source_chains = sum(
        len(
            [
                row["chain_id"]
                for row in sorted(membership_rows, key=lambda row: int(row.get("position_in_protofibril", 0)))
            ]
        )
        for _, membership_rows in proto_entries
    )
    processed_source_chains = 0

    atom_counter = 1
    line_counter = 0
    merged_atom_frames = {"ATOM": [], "HETATM": []}
    for proto_index, membership_rows in proto_entries:
        proto_chain_ids = [
            row["chain_id"]
            for row in sorted(membership_rows, key=lambda row: int(row.get("position_in_protofibril", 0)))
        ]
        if not proto_chain_ids:
            continue
        merged_chain_id = PDB_CHAIN_IDS[proto_index - 1]
        residue_offset = 0

        for source_chain_id in proto_chain_ids:
            if progress_callback is not None:
                progress_callback(
                    processed_source_chains=processed_source_chains,
                    total_source_chains=total_source_chains,
                    protofibril_index=proto_index,
                    source_chain_id=source_chain_id,
                )
            chain_frames = []
            for record_name in ("ATOM", "HETATM"):
                record_df = source.df.get(record_name)
                if record_df is None or record_df.empty:
                    continue
                chain_df = record_df[record_df["chain_id"] == source_chain_id].copy()
                if not chain_df.empty:
                    chain_df["record_name"] = record_name
                    chain_frames.append(chain_df)

            if not chain_frames:
                processed_source_chains += 1
                continue

            chain_df = pd.concat(chain_frames, ignore_index=True).sort_values("line_idx").reset_index(drop=True)
            residue_keys = list(dict.fromkeys(zip(chain_df["residue_number"], chain_df["insertion"].fillna(""))))
            min_seq = min(int(key[0]) for key in residue_keys)
            max_seq = max(int(key[0]) for key in residue_keys)
            residue_number_map = {
                key: (int(key[0]) - min_seq + 1 + residue_offset)
                for key in residue_keys
            }

            chain_df["residue_number"] = [
                residue_number_map[(int(res_num), ins if isinstance(ins, str) else "")]
                for res_num, ins in zip(chain_df["residue_number"], chain_df["insertion"].fillna(""))
            ]
            chain_df["chain_id"] = merged_chain_id
            chain_df["atom_number"] = range(atom_counter, atom_counter + len(chain_df))
            chain_df["line_idx"] = range(line_counter, line_counter + len(chain_df))
            atom_counter += len(chain_df)
            line_counter += len(chain_df)

            for record_name in ("ATOM", "HETATM"):
                record_df = chain_df[chain_df["record_name"] == record_name].copy()
                record_df["record_name"] = record_name
                if not record_df.empty:
                    merged_atom_frames[record_name].append(record_df)

            residue_offset += (max_seq - min_seq + 1) + residue_gap
            processed_source_chains += 1

    output_ppdb = PandasPdb()
    output_ppdb.df["ATOM"] = _sanitize_pandas_pdb_frame(
        pd.concat(merged_atom_frames["ATOM"], ignore_index=True)
        if merged_atom_frames["ATOM"] else pd.DataFrame(columns=source.df["ATOM"].columns)
    )
    output_ppdb.df["HETATM"] = _sanitize_pandas_pdb_frame(
        pd.concat(merged_atom_frames["HETATM"], ignore_index=True)
        if merged_atom_frames["HETATM"] else pd.DataFrame(columns=source.df["HETATM"].columns)
    )
    output_ppdb.df["OTHERS"] = pd.DataFrame(columns=source.df["OTHERS"].columns if "OTHERS" in source.df else [])
    return output_ppdb.to_pdb_stream(records=("ATOM", "HETATM")).getvalue()


def build_merged_protofibril_visualization_pdb(
    pdb_text: str,
    protofibril_configs: list[dict] | None = None,
    protofibril_chain_membership: list[dict] | None = None,
    residue_gap: int = 20,
    structure_format: str | None = None,
    progress_callback=None,
) -> str:
    return build_merged_protofibril_visualization_result(
        pdb_text=pdb_text,
        protofibril_configs=protofibril_configs,
        protofibril_chain_membership=protofibril_chain_membership,
        residue_gap=residue_gap,
        structure_format=structure_format,
        progress_callback=progress_callback,
    )["structure_text"]


def build_merged_protofibril_visualization_result(
    pdb_text: str,
    protofibril_configs: list[dict] | None = None,
    protofibril_chain_membership: list[dict] | None = None,
    residue_gap: int = 20,
    structure_format: str | None = None,
    progress_callback=None,
) -> dict:
    structure_text = _build_merged_protofibril_visualization_pdb_biopandas(
        pdb_text=pdb_text,
        protofibril_configs=protofibril_configs,
        protofibril_chain_membership=protofibril_chain_membership,
        residue_gap=residue_gap,
        structure_format=structure_format,
        progress_callback=progress_callback,
    )
    return {"structure_text": structure_text, "backend": "biopandas", "fallback_reason": None}


def _format_merge_backend_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: {exc!r}"


def _sanitize_pandas_pdb_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in sanitized.columns:
        if pd.api.types.is_object_dtype(sanitized[column]) or column in {
            "record_name",
            "blank_1",
            "blank_2",
            "blank_3",
            "blank_4",
            "alt_loc",
            "insertion",
            "segment_id",
            "element_symbol",
            "chain_id",
            "atom_name",
            "residue_name",
        }:
            sanitized[column] = sanitized[column].fillna("").astype(str)
    return sanitized


def build_propagated_model(
    pdb_text: str,
    keep_chain_ids: list[str],
    protofibril_configs: list[dict],
    structure_format: str | None = None,
    progress_callback=None,
    debug_mode: bool = False,
    debug_sink: list[dict] | None = None,
) -> dict:
    if not debug_mode:
        debug_sink = None

    input_format = detect_structure_format(pdb_text, structure_format)
    output_format = "mmcif"
    _debug_log(
        debug_sink,
        "propagation_start",
        input_format=input_format,
        requested_keep_chain_count=len(keep_chain_ids),
        protofibril_count=len(protofibril_configs),
        output_format=output_format,
    )
    assigned_chain_counts = {len(config["chains"]) for config in protofibril_configs if config.get("chains")}
    if len(assigned_chain_counts) > 1:
        raise ValueError(
            "All protofibrils must currently have the same chain count for propagation."
        )
    for config in protofibril_configs:
        if int(config["addition_unit"]) > len(config["chains"]):
            raise ValueError(
                f"Protofibril {config['protofibril_index']} has {len(config['chains'])} chain(s), "
                f"so the addition unit cannot be {config['addition_unit']}."
            )
    modeled_chain_ids = [
        chain_id
        for chain_id in keep_chain_ids
        if any(chain_id in config.get("chains", []) for config in protofibril_configs)
    ]
    if not modeled_chain_ids:
        modeled_chain_ids = keep_chain_ids
    _debug_log(
        debug_sink,
        "modeled_chains",
        modeled_chain_ids=modeled_chain_ids,
        modeled_chain_count=len(modeled_chain_ids),
    )
    kept_structure_text = filter_pdb_to_chains(
        pdb_text,
        modeled_chain_ids,
        input_format=input_format,
        output_format=output_format,
    )
    # Keep transform estimation on a PDB-converted alignment view of the same
    # filtered chains, which is more robust than MMCIFParser against column
    # normalization edge-cases in generated CIF text.
    kept_alignment_pdb = filter_pdb_to_chains(
        pdb_text,
        modeled_chain_ids,
        input_format=input_format,
        output_format="pdb",
    )
    biopython_structure = parse_structure(kept_alignment_pdb, "pdb")
    biopython_model = next(biopython_structure.get_models())
    gemmi_structure = parse_structure_gemmi(kept_structure_text, output_format)
    gemmi_model = gemmi_structure[0]
    chain_entity_map = _chain_entity_id_map(gemmi_structure, gemmi_model)
    used_chain_ids = {chain.name for chain in gemmi_model}
    _debug_log(
        debug_sink,
        "post_filter_structure",
        kept_chain_count=len(used_chain_ids),
        kept_chain_ids=sorted(list(used_chain_ids)),
    )

    propagation_metadata = []
    transforms = {}
    uses_helical_metadata = any(
        str(config.get("transform_mode", "chain_fit")) == "helical_metadata"
        for config in protofibril_configs
    )
    helical_metadata: dict[str, object] | None = None
    if uses_helical_metadata:
        if input_format != "mmcif":
            raise ValueError(
                "Helical metadata transform mode requires mmCIF input with _em_helical_entity fields."
            )
        helical_metadata = _extract_helical_metadata_from_mmcif_text(pdb_text)
        if helical_metadata is None:
            raise ValueError(
                "mmCIF helical metadata is missing or invalid. "
                "Expected _em_helical_entity.angular_rotation_per_subunit and "
                "_em_helical_entity.axial_rise_per_subunit."
            )
        _debug_log(
            debug_sink,
            "helical_metadata_detected",
            angle_deg_per_subunit=float(helical_metadata["angle_deg_per_subunit"]),
            rise_angstrom_per_subunit=float(helical_metadata["rise_angstrom_per_subunit"]),
            axial_symmetry=helical_metadata.get("axial_symmetry"),
        )

    for config in protofibril_configs:
        proto_index = config["protofibril_index"]
        top_pair = config.get("top_reference_pair", [])
        bottom_pair = config.get("bottom_reference_pair", [])
        top_transform = None
        bottom_transform = None
        top_rms = None
        bottom_rms = None
        transform_mode = str(config.get("transform_mode", "chain_fit"))
        helical_parameters: dict[str, object] | None = None

        if transform_mode == "helical_metadata":
            assert helical_metadata is not None
            angle_per_subunit = float(config.get("helical_twist_deg_per_subunit", helical_metadata["angle_deg_per_subunit"]))
            rise_per_subunit = float(config.get("helical_rise_angstrom_per_subunit", helical_metadata["rise_angstrom_per_subunit"]))
            if not math.isfinite(angle_per_subunit) or not math.isfinite(rise_per_subunit):
                raise ValueError("Helical metadata transform values must be finite numbers.")
            if abs(rise_per_subunit) < 1e-8:
                raise ValueError("Helical rise per subunit must be non-zero.")
            axis_point, axis_unit, centroids = _estimate_protofibril_axis(gemmi_model, list(config.get("chains", [])))

            top_step_units = 0
            bottom_step_units = 0
            if len(top_pair) == 2:
                top_step_units = _helical_pair_step_count(
                    top_pair[0],
                    top_pair[1],
                    centroids=centroids,
                    axis_unit=axis_unit,
                    rise_per_subunit=rise_per_subunit,
                )
                top_transform = _build_helical_screw_transform(
                    axis_point=axis_point,
                    axis_unit=axis_unit,
                    angle_deg=angle_per_subunit * top_step_units,
                    rise_angstrom=rise_per_subunit * top_step_units,
                )
            if len(bottom_pair) == 2:
                bottom_step_units = _helical_pair_step_count(
                    bottom_pair[0],
                    bottom_pair[1],
                    centroids=centroids,
                    axis_unit=axis_unit,
                    rise_per_subunit=rise_per_subunit,
                )
                bottom_transform = _build_helical_screw_transform(
                    axis_point=axis_point,
                    axis_unit=axis_unit,
                    angle_deg=-angle_per_subunit * bottom_step_units,
                    rise_angstrom=-rise_per_subunit * bottom_step_units,
                )
            _debug_log(
                debug_sink,
                "transform_helical",
                protofibril_index=proto_index,
                angle_deg_per_subunit=angle_per_subunit,
                rise_angstrom_per_subunit=rise_per_subunit,
                top_step_units=top_step_units,
                bottom_step_units=bottom_step_units,
                axis_point=[float(axis_point[0]), float(axis_point[1]), float(axis_point[2])],
                axis_unit=[float(axis_unit[0]), float(axis_unit[1]), float(axis_unit[2])],
            )
            helical_parameters = {
                "axial_symmetry": helical_metadata.get("axial_symmetry"),
                "angle_deg_per_subunit": angle_per_subunit,
                "rise_angstrom_per_subunit": rise_per_subunit,
                "top_step_units": top_step_units,
                "bottom_step_units": bottom_step_units,
                "top_angle_deg": angle_per_subunit * top_step_units,
                "top_rise_angstrom": rise_per_subunit * top_step_units,
                "bottom_angle_deg": -angle_per_subunit * bottom_step_units,
                "bottom_rise_angstrom": -rise_per_subunit * bottom_step_units,
                "axis_point": [float(axis_point[0]), float(axis_point[1]), float(axis_point[2])],
                "axis_unit": [float(axis_unit[0]), float(axis_unit[1]), float(axis_unit[2])],
            }
        else:
            if len(top_pair) == 2:
                top_rotation, top_translation, top_rms = _compute_chain_transform(biopython_model, top_pair[0], top_pair[1])
                top_transform = (top_rotation, top_translation)
                _debug_log(
                    debug_sink,
                    "transform_top",
                    protofibril_index=proto_index,
                    source=top_pair[0],
                    target=top_pair[1],
                    rms=float(top_rms),
                    translation_norm=float(np.linalg.norm(top_translation)),
                )
            if len(bottom_pair) == 2:
                bottom_rotation, bottom_translation, bottom_rms = _compute_chain_transform(biopython_model, bottom_pair[0], bottom_pair[1])
                bottom_transform = (bottom_rotation, bottom_translation)
                _debug_log(
                    debug_sink,
                    "transform_bottom",
                    protofibril_index=proto_index,
                    source=bottom_pair[0],
                    target=bottom_pair[1],
                    rms=float(bottom_rms),
                    translation_norm=float(np.linalg.norm(bottom_translation)),
                )

        transforms[proto_index] = {
            "top": top_transform,
            "bottom": bottom_transform,
        }
        propagation_metadata.append(
            {
                "protofibril_index": proto_index,
                "transform_mode": transform_mode,
                "top_reference_pair": top_pair,
                "bottom_reference_pair": bottom_pair,
                "top_rms": top_rms,
                "bottom_rms": bottom_rms,
                "helical_parameters": helical_parameters,
            }
        )

    current_edges = {
        config["protofibril_index"]: {
            "top_chain_id": config["top_chain"],
            "bottom_chain_id": config["bottom_chain"],
        }
        for config in protofibril_configs
    }
    protofibril_chain_order = {
        config["protofibril_index"]: list(config["chains"])
        for config in protofibril_configs
    }

    addition_log = []

    def _config_work_units(config: dict) -> int:
        addition_unit = int(config.get("addition_unit", 1))
        units_to_add = int(config.get("units_to_add", 0))
        direction = str(config.get("propagation_direction", "Add to both ends"))
        direction_multiplier = 2 if direction == "Add to both ends" else 1
        return units_to_add * addition_unit * direction_multiplier

    total_progress_steps = sum(_config_work_units(config) for config in protofibril_configs)
    completed_progress_steps = 0
    max_iterations = max((config["units_to_add"] for config in protofibril_configs), default=0)
    for iteration in range(1, max_iterations + 1):
        for config in protofibril_configs:
            if iteration > config["units_to_add"]:
                continue
            proto_index = config["protofibril_index"]
            addition_unit = config["addition_unit"]
            direction = config["propagation_direction"]
            top_transform = transforms[proto_index]["top"]
            bottom_transform = transforms[proto_index]["bottom"]

            if direction in {"Add to top", "Add to both ends"}:
                if top_transform is None:
                    raise ValueError(f"Protofibril {proto_index} is missing a valid top propagation pair.")
                rotation, translation = top_transform
                for _ in range(addition_unit):
                    source_chain = _find_gemmi_chain(gemmi_model, current_edges[proto_index]["top_chain_id"])
                    source_chain_id = source_chain.name
                    new_chain_id = _next_available_chain_id(used_chain_ids, output_format)
                    _debug_log(
                        debug_sink,
                        "addition_top_start",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    cloned_chain = _clone_chain_with_transform_gemmi(
                        source_chain,
                        new_chain_id,
                        rotation,
                        translation,
                        source_entity_id=chain_entity_map.get(source_chain_id),
                    )
                    _debug_log(
                        debug_sink,
                        "addition_top_cloned",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    _debug_log(
                        debug_sink,
                        "addition_top_pre_validate",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    try:
                        _validate_chain_coordinates(
                            cloned_chain,
                            proto_index=proto_index,
                            direction="top",
                            source_chain_id=source_chain_id,
                        )
                        _debug_log(
                            debug_sink,
                            "addition_top_post_validate",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                    except Exception as exc:
                        _debug_log(
                            debug_sink,
                            "addition_top_validate_failed",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        raise RuntimeError(
                            f"Top addition validation failed at iteration {iteration} "
                            f"(protofibril {proto_index}, source {source_chain_id}, new {new_chain_id}): {exc}"
                        ) from exc
                    try:
                        _debug_log(
                            debug_sink,
                            "addition_top_pre_overlap_check",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                        overlapping_chain_id = _detect_overlapping_chain(
                            gemmi_model,
                            cloned_chain,
                            ignore_chain_ids={source_chain_id},
                        )
                        _debug_log(
                            debug_sink,
                            "addition_top_post_overlap_check",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            overlapping_chain_id=overlapping_chain_id,
                        )
                    except Exception as exc:
                        _debug_log(
                            debug_sink,
                            "addition_top_overlap_check_failed",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        raise RuntimeError(
                            f"Top overlap check failed at iteration {iteration} "
                            f"(protofibril {proto_index}, source {source_chain_id}, new {new_chain_id}): {exc}"
                        ) from exc
                    if overlapping_chain_id is not None:
                        _debug_log(
                            debug_sink,
                            "addition_top_overlap",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            overlapping_chain_id=overlapping_chain_id,
                        )
                        raise ValueError(
                            f"Top growth for protofibril {proto_index} would place a new chain on top of existing "
                            f"chain {overlapping_chain_id}. Adjust top chain/reference selection."
                        )
                    try:
                        _debug_log(
                            debug_sink,
                            "addition_top_pre_add_chain",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                        gemmi_model.add_chain(cloned_chain)
                        _debug_log(
                            debug_sink,
                            "addition_top_post_add_chain",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                    except Exception as exc:
                        _debug_log(
                            debug_sink,
                            "addition_top_add_chain_failed",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        raise RuntimeError(
                            f"Top add-chain failed at iteration {iteration} "
                            f"(protofibril {proto_index}, source {source_chain_id}, new {new_chain_id}): {exc}"
                        ) from exc
                    if source_chain_id in chain_entity_map:
                        chain_entity_map[new_chain_id] = chain_entity_map[source_chain_id]
                    used_chain_ids.add(new_chain_id)
                    current_edges[proto_index]["top_chain_id"] = new_chain_id
                    protofibril_chain_order[proto_index].append(new_chain_id)
                    addition_log.append(
                        {
                            "protofibril_index": proto_index,
                            "iteration": iteration,
                            "direction": "top",
                            "source_chain_id": source_chain_id,
                            "new_chain_id": new_chain_id,
                        }
                    )
                    _debug_log(
                        debug_sink,
                        "addition_top",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    completed_progress_steps += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_progress_steps=completed_progress_steps,
                            total_progress_steps=total_progress_steps,
                            protofibril_index=proto_index,
                            iteration=iteration,
                            addition_unit=addition_unit,
                            direction=direction,
                        )

            if direction in {"Add to bottom", "Add to both ends"}:
                if bottom_transform is None:
                    raise ValueError(f"Protofibril {proto_index} is missing a valid bottom propagation pair.")
                rotation, translation = bottom_transform
                for _ in range(addition_unit):
                    source_chain = _find_gemmi_chain(gemmi_model, current_edges[proto_index]["bottom_chain_id"])
                    source_chain_id = source_chain.name
                    new_chain_id = _next_available_chain_id(used_chain_ids, output_format)
                    _debug_log(
                        debug_sink,
                        "addition_bottom_start",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    cloned_chain = _clone_chain_with_transform_gemmi(
                        source_chain,
                        new_chain_id,
                        rotation,
                        translation,
                        source_entity_id=chain_entity_map.get(source_chain_id),
                    )
                    _debug_log(
                        debug_sink,
                        "addition_bottom_cloned",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    _debug_log(
                        debug_sink,
                        "addition_bottom_pre_validate",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    try:
                        _validate_chain_coordinates(
                            cloned_chain,
                            proto_index=proto_index,
                            direction="bottom",
                            source_chain_id=source_chain_id,
                        )
                        _debug_log(
                            debug_sink,
                            "addition_bottom_post_validate",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                    except Exception as exc:
                        _debug_log(
                            debug_sink,
                            "addition_bottom_validate_failed",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        raise RuntimeError(
                            f"Bottom addition validation failed at iteration {iteration} "
                            f"(protofibril {proto_index}, source {source_chain_id}, new {new_chain_id}): {exc}"
                        ) from exc
                    try:
                        _debug_log(
                            debug_sink,
                            "addition_bottom_pre_overlap_check",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                        overlapping_chain_id = _detect_overlapping_chain(
                            gemmi_model,
                            cloned_chain,
                            ignore_chain_ids={source_chain_id},
                        )
                        _debug_log(
                            debug_sink,
                            "addition_bottom_post_overlap_check",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            overlapping_chain_id=overlapping_chain_id,
                        )
                    except Exception as exc:
                        _debug_log(
                            debug_sink,
                            "addition_bottom_overlap_check_failed",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        raise RuntimeError(
                            f"Bottom overlap check failed at iteration {iteration} "
                            f"(protofibril {proto_index}, source {source_chain_id}, new {new_chain_id}): {exc}"
                        ) from exc
                    if overlapping_chain_id is not None:
                        _debug_log(
                            debug_sink,
                            "addition_bottom_overlap",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            overlapping_chain_id=overlapping_chain_id,
                        )
                        raise ValueError(
                            f"Bottom growth for protofibril {proto_index} would place a new chain on top of existing "
                            f"chain {overlapping_chain_id}. Adjust bottom chain/reference selection."
                        )
                    try:
                        _debug_log(
                            debug_sink,
                            "addition_bottom_pre_add_chain",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                        gemmi_model.add_chain(cloned_chain)
                        _debug_log(
                            debug_sink,
                            "addition_bottom_post_add_chain",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                        )
                    except Exception as exc:
                        _debug_log(
                            debug_sink,
                            "addition_bottom_add_chain_failed",
                            protofibril_index=proto_index,
                            iteration=iteration,
                            source_chain_id=source_chain_id,
                            new_chain_id=new_chain_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        raise RuntimeError(
                            f"Bottom add-chain failed at iteration {iteration} "
                            f"(protofibril {proto_index}, source {source_chain_id}, new {new_chain_id}): {exc}"
                        ) from exc
                    if source_chain_id in chain_entity_map:
                        chain_entity_map[new_chain_id] = chain_entity_map[source_chain_id]
                    used_chain_ids.add(new_chain_id)
                    current_edges[proto_index]["bottom_chain_id"] = new_chain_id
                    protofibril_chain_order[proto_index].insert(0, new_chain_id)
                    addition_log.append(
                        {
                            "protofibril_index": proto_index,
                            "iteration": iteration,
                            "direction": "bottom",
                            "source_chain_id": source_chain_id,
                            "new_chain_id": new_chain_id,
                        }
                    )
                    _debug_log(
                        debug_sink,
                        "addition_bottom",
                        protofibril_index=proto_index,
                        iteration=iteration,
                        source_chain_id=source_chain_id,
                        new_chain_id=new_chain_id,
                    )
                    completed_progress_steps += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_progress_steps=completed_progress_steps,
                            total_progress_steps=total_progress_steps,
                            protofibril_index=proto_index,
                            iteration=iteration,
                            addition_unit=addition_unit,
                            direction=direction,
                        )

    addition_index_by_chain_id = {entry["new_chain_id"]: entry for entry in addition_log}
    protofibril_chain_membership = []
    for config in protofibril_configs:
        proto_index = config["protofibril_index"]
        seed_chain_ids = set(config["chains"])
        for position, chain_id in enumerate(protofibril_chain_order[proto_index], start=1):
            if chain_id in seed_chain_ids:
                origin = "seed"
                iteration = 0
                added_to_end = "seed"
                source_chain_id = None
            else:
                log_entry = addition_index_by_chain_id[chain_id]
                origin = "propagated"
                iteration = log_entry["iteration"]
                added_to_end = log_entry["direction"]
                source_chain_id = log_entry["source_chain_id"]
            protofibril_chain_membership.append(
                {
                    "chain_id": chain_id,
                    "protofibril_index": proto_index,
                    "position_in_protofibril": position,
                    "origin": origin,
                    "iteration": iteration,
                    "added_to_end": added_to_end,
                    "source_chain_id": source_chain_id,
                }
            )

    pre_serialize_atom_count = 0
    pre_serialize_residue_count = 0
    pre_serialize_chain_count = 0
    for chain in gemmi_model:
        pre_serialize_chain_count += 1
        for residue in chain:
            pre_serialize_residue_count += 1
            pre_serialize_atom_count += len(residue)
    _debug_log(
        debug_sink,
        "pre_serialize",
        chain_count=pre_serialize_chain_count,
        residue_count=pre_serialize_residue_count,
        atom_count=pre_serialize_atom_count,
        output_format=output_format,
    )

    try:
        serialized_structure = serialize_structure_gemmi(gemmi_structure, output_format)
    except Exception as exc:
        sanity = _mmcif_chain_sanity_summary(gemmi_model)
        _debug_log(
            debug_sink,
            "serialize_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            chain_count=pre_serialize_chain_count,
            residue_count=pre_serialize_residue_count,
            atom_count=pre_serialize_atom_count,
            mmcif_sanity=sanity,
        )
        raise RuntimeError(
            f"Failed to serialize propagated {output_format} structure: {type(exc).__name__}: {exc}. "
            f"Sanity: {sanity}"
        ) from exc

    return {
        "pdb": serialized_structure,
        "kept_pdb": kept_structure_text,
        "structure_format": output_format,
        "propagation_metadata": propagation_metadata,
        "addition_log": addition_log,
        "protofibril_chain_membership": protofibril_chain_membership,
    }
