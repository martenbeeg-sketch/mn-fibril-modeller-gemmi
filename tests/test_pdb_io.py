import io

from Bio.PDB import MMCIFIO

from mn_fibril_modeller_gemmi.core.pdb_io import (
    chain_centroids_from_pdb,
    chain_lengths_from_pdb,
    chain_rows_from_pdb,
    detect_structure_format,
    ordered_chain_ids_from_pdb,
    parse_structure,
    principal_axis_ordered_chain_ids_from_pdb,
    suggest_protofibril_groups_from_pdb,
)


EXAMPLE_PDB = """\
ATOM      1  N   GLY A   1      11.104  13.207   9.302  1.00 20.00           N
ATOM      2  CA  GLY A   1      12.053  12.114   9.451  1.00 20.00           C
ATOM      3  C   GLY A   1      11.392  10.762   9.196  1.00 20.00           C
ATOM      4  N   ALA B   2      14.104  15.207  11.302  1.00 20.00           N
ATOM      5  CA  ALA B   2      15.053  14.114  11.451  1.00 20.00           C
ATOM      6  C   ALA B   2      14.392  12.762  11.196  1.00 20.00           C
TER
END
"""


def test_chain_rows_from_pdb_reports_polymer_chains():
    rows = chain_rows_from_pdb(EXAMPLE_PDB)
    assert rows == [
        {"chain_id": "A", "residue_count": 1, "start_residue": "1", "end_residue": "1"},
        {"chain_id": "B", "residue_count": 1, "start_residue": "2", "end_residue": "2"},
    ]


def test_chain_lengths_from_pdb_returns_simple_mapping():
    assert chain_lengths_from_pdb(EXAMPLE_PDB) == {"A": 1, "B": 1}


def test_chain_rows_from_mmcif_reports_polymer_chains():
    structure = parse_structure(EXAMPLE_PDB, "pdb")
    handle = io.StringIO()
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(handle)
    mmcif_text = handle.getvalue()
    assert detect_structure_format(mmcif_text) == "mmcif"
    rows = chain_rows_from_pdb(mmcif_text)
    assert rows == [
        {"chain_id": "A", "residue_count": 1, "start_residue": "1", "end_residue": "1"},
        {"chain_id": "B", "residue_count": 1, "start_residue": "2", "end_residue": "2"},
    ]


def test_chain_centroids_from_pdb_returns_chain_positions():
    centroids = chain_centroids_from_pdb(EXAMPLE_PDB)
    assert set(centroids) == {"A", "B"}
    assert centroids["A"][2] < centroids["B"][2]


def test_ordered_chain_ids_from_pdb_sorts_by_axis():
    assert ordered_chain_ids_from_pdb(EXAMPLE_PDB, ["A", "B"], axis="z") == ["A", "B"]


THREE_CHAIN_PDB = """\
ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLY B   1       1.000   0.000   5.000  1.00 20.00           C
ATOM      3  CA  GLY C   1       2.000   0.000  10.000  1.00 20.00           C
TER
END
"""


PROTOFIBRIL_GROUP_PDB = """\
ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  GLY B   1       0.100   0.100   5.000  1.00 20.00           C
ATOM      3  CA  GLY C   1       0.200   0.100  10.000  1.00 20.00           C
ATOM      4  CA  GLY D   1      10.000   0.000   0.000  1.00 20.00           C
ATOM      5  CA  GLY E   1      10.100   0.100   5.000  1.00 20.00           C
ATOM      6  CA  GLY F   1      10.200   0.100  10.000  1.00 20.00           C
TER
END
"""


def test_principal_axis_ordered_chain_ids_from_pdb_sorts_along_fibril_axis():
    assert principal_axis_ordered_chain_ids_from_pdb(THREE_CHAIN_PDB, ["C", "A", "B"]) == ["A", "B", "C"]


def test_suggest_protofibril_groups_from_pdb_detects_two_parallel_stacks():
    groups = suggest_protofibril_groups_from_pdb(PROTOFIBRIL_GROUP_PDB, ["A", "B", "C", "D", "E", "F"])
    assert groups == [["A", "B", "C"], ["D", "E", "F"]]
