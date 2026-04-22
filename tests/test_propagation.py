import pytest

from mn_fibril_modeller_gemmi.core.pdb_io import chain_centroids_from_pdb, parse_structure_gemmi, serialize_structure_gemmi
from mn_fibril_modeller_gemmi.core.propagation import (
    _extract_helical_metadata_from_mmcif_text,
    build_merged_protofibril_visualization_pdb,
    build_merged_protofibril_visualization_result,
    build_propagated_model,
)
from mn_fibril_modeller_gemmi.core.pdb_io import chain_rows_from_pdb


LINEAR_STACK_PDB = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  GLY A   1       0.000   0.000   1.000  1.00 20.00           C
ATOM      3  C   GLY A   1       0.000   0.000   2.000  1.00 20.00           C
ATOM      4  O   GLY A   1       0.000   0.000   3.000  1.00 20.00           O
ATOM      5  N   GLY B   1       0.000   0.000   5.000  1.00 20.00           N
ATOM      6  CA  GLY B   1       0.000   0.000   6.000  1.00 20.00           C
ATOM      7  C   GLY B   1       0.000   0.000   7.000  1.00 20.00           C
ATOM      8  O   GLY B   1       0.000   0.000   8.000  1.00 20.00           O
TER
END
"""


def _inject_em_helical_metadata(mmcif_text: str, *, angle_deg: float, rise_angstrom: float, symmetry: str = "C1") -> str:
    lines = mmcif_text.splitlines()
    if not lines:
        return mmcif_text
    header = lines[0]
    tail = lines[1:]
    metadata_lines = [
        f"_em_helical_entity.axial_symmetry {symmetry}",
        f"_em_helical_entity.angular_rotation_per_subunit {angle_deg}",
        f"_em_helical_entity.axial_rise_per_subunit {rise_angstrom}",
    ]
    merged = [header, *metadata_lines, *tail]
    return "\n".join(merged) + ("\n" if mmcif_text.endswith("\n") else "")


def test_build_propagated_model_adds_chain_to_top():
    result = build_propagated_model(
        pdb_text=LINEAR_STACK_PDB,
        keep_chain_ids=["A", "B"],
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
                "top_chain": "B",
                "bottom_chain": "A",
                "top_reference_pair": ["A", "B"],
                "bottom_reference_pair": [],
                "addition_unit": 1,
                "propagation_direction": "Add to top",
                "units_to_add": 1,
                "relax_mode": "No relax",
            }
        ],
    )
    centroids = chain_centroids_from_pdb(result["pdb"])
    assert set(centroids) == {"A", "B", "PF1"}
    assert centroids["A"][2] < centroids["B"][2] < centroids["PF1"][2]


def test_build_propagated_model_adds_chain_to_both_ends():
    result = build_propagated_model(
        pdb_text=LINEAR_STACK_PDB,
        keep_chain_ids=["A", "B"],
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
                "top_chain": "B",
                "bottom_chain": "A",
                "top_reference_pair": ["A", "B"],
                "bottom_reference_pair": ["A", "B"],
                "addition_unit": 1,
                "propagation_direction": "Add to both ends",
                "units_to_add": 1,
                "relax_mode": "No relax",
            }
        ],
    )
    centroids = chain_centroids_from_pdb(result["pdb"])
    assert len(centroids) == 4
    z_values = sorted(centroid[2] for centroid in centroids.values())
    assert z_values[0] < 1.5
    assert z_values[-1] > 10.0
    membership = result["protofibril_chain_membership"]
    assert [row["chain_id"] for row in membership] == ["PF2", "A", "B", "PF1"]
    assert [row["position_in_protofibril"] for row in membership] == [1, 2, 3, 4]
    assert membership[0]["added_to_end"] == "bottom"
    assert membership[-1]["added_to_end"] == "top"


def test_build_merged_protofibril_visualization_pdb_merges_to_single_chain_per_protofibril():
    merged_pdb = build_merged_protofibril_visualization_pdb(
        pdb_text=LINEAR_STACK_PDB,
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
            }
        ],
        residue_gap=10,
    )
    rows = chain_rows_from_pdb(merged_pdb)
    assert rows == [{"chain_id": "A", "residue_count": 2, "start_residue": "1", "end_residue": "12"}]


def test_build_merged_protofibril_visualization_pdb_uses_propagated_membership_order():
    propagation_result = build_propagated_model(
        pdb_text=LINEAR_STACK_PDB,
        keep_chain_ids=["A", "B"],
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
                "top_chain": "B",
                "bottom_chain": "A",
                "top_reference_pair": ["A", "B"],
                "bottom_reference_pair": ["A", "B"],
                "addition_unit": 1,
                "propagation_direction": "Add to both ends",
                "units_to_add": 1,
                "relax_mode": "No relax",
            }
        ],
    )
    merged_pdb = build_merged_protofibril_visualization_pdb(
        pdb_text=propagation_result["pdb"],
        protofibril_chain_membership=propagation_result["protofibril_chain_membership"],
    )
    rows = chain_rows_from_pdb(merged_pdb)
    assert rows == [{"chain_id": "A", "residue_count": 4, "start_residue": "1", "end_residue": "64"}]


def test_build_merged_protofibril_visualization_result_prefers_biopandas_for_simple_case():
    result = build_merged_protofibril_visualization_result(
        pdb_text=LINEAR_STACK_PDB,
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
            }
        ],
    )
    assert result["backend"] == "biopandas"
    rows = chain_rows_from_pdb(result["structure_text"])
    assert rows == [{"chain_id": "A", "residue_count": 2, "start_residue": "1", "end_residue": "22"}]
    assert result["structure_text"].splitlines()[0][17:20] == "GLY"


def test_build_propagated_model_switches_to_mmcif_when_chain_count_exceeds_pdb_limit():
    result = build_propagated_model(
        pdb_text=LINEAR_STACK_PDB,
        keep_chain_ids=["A", "B"],
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
                "top_chain": "B",
                "bottom_chain": "A",
                "top_reference_pair": ["A", "B"],
                "bottom_reference_pair": [],
                "addition_unit": 1,
                "propagation_direction": "Add to top",
                "units_to_add": 70,
                "relax_mode": "No relax",
            }
        ],
        structure_format="pdb",
    )
    assert result["structure_format"] == "mmcif"


def test_extract_helical_metadata_from_mmcif_text():
    mmcif_text = """\
data_test
_em_helical_entity.axial_symmetry C1
_em_helical_entity.angular_rotation_per_subunit -1.19
_em_helical_entity.axial_rise_per_subunit 4.78
"""
    metadata = _extract_helical_metadata_from_mmcif_text(mmcif_text)
    assert metadata is not None
    assert metadata["axial_symmetry"] == "C1"
    assert metadata["angle_deg_per_subunit"] == -1.19
    assert metadata["rise_angstrom_per_subunit"] == 4.78


def test_build_propagated_model_helical_mode_requires_mmcif():
    with pytest.raises(ValueError, match="requires mmCIF input"):
        build_propagated_model(
            pdb_text=LINEAR_STACK_PDB,
            keep_chain_ids=["A", "B"],
            protofibril_configs=[
                {
                    "protofibril_index": 1,
                    "chains": ["A", "B"],
                    "top_chain": "B",
                    "bottom_chain": "A",
                    "top_reference_pair": ["A", "B"],
                    "bottom_reference_pair": [],
                    "addition_unit": 1,
                    "propagation_direction": "Add to top",
                    "units_to_add": 1,
                    "transform_mode": "helical_metadata",
                }
            ],
            structure_format="pdb",
        )


def test_build_propagated_model_helical_mode_uses_mmcif_helical_metadata():
    base_structure = parse_structure_gemmi(LINEAR_STACK_PDB, "pdb")
    mmcif_text = serialize_structure_gemmi(base_structure, "mmcif")
    mmcif_with_helical = _inject_em_helical_metadata(mmcif_text, angle_deg=0.0, rise_angstrom=5.0)

    result = build_propagated_model(
        pdb_text=mmcif_with_helical,
        keep_chain_ids=["A", "B"],
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
                "top_chain": "B",
                "bottom_chain": "A",
                "top_reference_pair": ["A", "B"],
                "bottom_reference_pair": [],
                "addition_unit": 1,
                "propagation_direction": "Add to top",
                "units_to_add": 1,
                "transform_mode": "helical_metadata",
            }
        ],
        structure_format="mmcif",
    )
    centroids = chain_centroids_from_pdb(result["pdb"], "mmcif")
    assert set(centroids) == {"A", "B", "PF1"}
    assert centroids["A"][2] < centroids["B"][2] < centroids["PF1"][2]


def test_build_propagated_model_helical_mode_uses_manual_override_values():
    base_structure = parse_structure_gemmi(LINEAR_STACK_PDB, "pdb")
    mmcif_text = serialize_structure_gemmi(base_structure, "mmcif")
    mmcif_with_helical = _inject_em_helical_metadata(mmcif_text, angle_deg=0.0, rise_angstrom=5.0)

    result = build_propagated_model(
        pdb_text=mmcif_with_helical,
        keep_chain_ids=["A", "B"],
        protofibril_configs=[
            {
                "protofibril_index": 1,
                "chains": ["A", "B"],
                "top_chain": "B",
                "bottom_chain": "A",
                "top_reference_pair": ["A", "B"],
                "bottom_reference_pair": [],
                "addition_unit": 1,
                "propagation_direction": "Add to top",
                "units_to_add": 1,
                "transform_mode": "helical_metadata",
                "helical_twist_deg_per_subunit": 0.0,
                "helical_rise_angstrom_per_subunit": 10.0,
            }
        ],
        structure_format="mmcif",
    )
    metadata = result["propagation_metadata"][0]["helical_parameters"]
    assert metadata is not None
    assert metadata["angle_deg_per_subunit"] == 0.0
    assert metadata["rise_angstrom_per_subunit"] == 10.0
    assert metadata["top_rise_angstrom"] == 10.0
