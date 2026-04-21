import dataclasses
import json
from typing import Literal


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore[arg-type]
        return super().default(o)


@dataclasses.dataclass
class ChainVisualization:
    chain_id: str
    color: Literal[
        "uniform",
        "chain-id",
        "hydrophobicity",
        "plddt",
        "molecule-type",
        "secondary-structure",
        "residue-name",
        "residue-charge",
    ] = "uniform"
    color_params: dict | None = None
    representation_type: Literal[
        "cartoon", "cartoon+ball-and-stick", "molecular-surface", "gaussian-surface", "ball-and-stick"
    ] = "cartoon"
    residues: list[int] | None = None
    label: str | None = None


@dataclasses.dataclass
class StructureVisualization:
    pdb: str
    color: Literal[
        "uniform",
        "chain-id",
        "hydrophobicity",
        "plddt",
        "molecule-type",
        "secondary-structure",
        "residue-name",
        "residue-charge",
    ] = "uniform"
    color_params: dict | None = None
    representation_type: (
        Literal["cartoon", "cartoon+ball-and-stick", "molecular-surface", "gaussian-surface", "ball-and-stick"] | None
    ) = "cartoon"
    highlighted_selections: list[str] | None = None
    chains: list[ChainVisualization] | None = None
    contigs: None = None

    def __post_init__(self):
        if self.chains:
            if isinstance(self.chains, ChainVisualization):
                self.chains = [self.chains]
            elif not isinstance(self.chains, list):
                raise ValueError(
                    f"Invalid type for chains, expected ChainVisualization or list of ChainVisualization, got: {type(self.chains).__name__}"
                )

    def to_dict(self):
        return dataclasses.asdict(self)
