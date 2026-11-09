"""Shim: SparseDriveV2 imports things from navsim.navsim_v1.*; map them to
the canonical navsim.* tree."""
import sys as _sys
import importlib as _importlib

_aliases = [
    "common",
    "common.dataclasses",
    "common.dataloader",
    "common.enums",
    "evaluate",
    "evaluate.pdm_score",
    "planning",
    "planning.metric_caching",
    "planning.metric_caching.metric_cache",
    "planning.simulation",
    "planning.simulation.planner",
    "planning.simulation.planner.pdm_planner",
    "planning.simulation.planner.pdm_planner.simulation",
    "planning.simulation.planner.pdm_planner.simulation.pdm_simulator",
    "planning.simulation.planner.pdm_planner.scoring",
    "planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
    "planning.simulation.planner.pdm_planner.utils",
    "planning.simulation.planner.pdm_planner.utils.pdm_array_representation",
    "planning.simulation.planner.pdm_planner.utils.pdm_enums",
]
for _name in _aliases:
    _src = _importlib.import_module(f"navsim.{_name}")
    _sys.modules[f"navsim.navsim_v1.{_name}"] = _src
