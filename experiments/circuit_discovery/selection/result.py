"""Common return type for every selection method."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class SelectionResult:
    method: str
    components: List[str]                  # the selected circuit
    n_components: int
    params: Dict                           # the kwargs the method ran with
    diagnostics: Dict = field(default_factory=dict)
    # diagnostics is method-specific provenance:
    #   component_coverage: 'curve' (list of (rank, name, signed, cum, frac))
    #   path_coverage:      'n_paths_taken', 'total_path_mass', 'curve'
    #   top_k_components:   'curve'

    def to_json(self) -> Dict:
        return {
            "circuit":      self.components,    # the key verify_circuit reads
            "n_components": self.n_components,
            "method":       self.method,
            "params":       self.params,
            "diagnostics":  self.diagnostics,
        }