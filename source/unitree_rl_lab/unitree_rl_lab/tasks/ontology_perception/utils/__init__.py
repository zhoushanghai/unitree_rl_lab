"""
Purpose: Utility namespace for ontology perception tasks.
Main contents: voxel-map generation and scripted action helpers for data collection and debugging.
"""

from .scripted_explorer import ScriptedExplorer  # noqa: F401
from .voxel_map import compute_explored_mask, compute_local_voxel_gt  # noqa: F401

