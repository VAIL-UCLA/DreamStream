from typing import Any, List, Dict, Optional, Union, Tuple, Sequence
import os
import multiprocessing as mp
import concurrent.futures as cf
from omegaconf import OmegaConf
from hydra.utils import instantiate
import lzma
import pickle

import numpy as np
import numpy.typing as npt
import torch

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.navsim_v1.common.dataclasses import PDMResults, Trajectory
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.navsim_v1.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.navsim_v1.planning.metric_caching.metric_cache import MetricCache

from .pdm_score_v1 import pdm_score


# Resolve the scoring-parameters YAML by file-relative path so it works whether
# the trainer is launched from the repo root or a Ray worker on a remote node
# with a different CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PDM_CFG_PATH = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "..", "planning", "script", "config",
    "pdm_scoring", "default_scoring_parameters_v1.yaml",
))

# Default backend = the original module-local ProcessPoolExecutor (single node).
# Set SDV2_USE_RAY=1 to dispatch training-time scoring to Ray actors.
_USE_RAY = os.environ.get("SDV2_USE_RAY", "0") == "1"


def _init_pool():
    global SIMULATOR, SCORER
    pdm_cfg = OmegaConf.load(PDM_CFG_PATH)
    SIMULATOR = instantiate(pdm_cfg.simulator)
    SCORER    = instantiate(pdm_cfg.scorer)
    SCORER.train_mode = True


def _pdm_worker(args):
    cache, traj_np = args
    with lzma.open(cache, "rb") as f:
        metric_cache = pickle.load(f)

    results = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=traj_np,                # (G, T, C)
        future_sampling=SIMULATOR.proposal_sampling,
        simulator=SIMULATOR,                    # 全局对象，见 initializer
        scorer=SCORER,
    )
    return results


# Single-node ProcessPoolExecutor backend (only built when not using Ray).
_pdm_pool = (
    None if _USE_RAY else
    cf.ProcessPoolExecutor(
        max_workers=16,
        mp_context=mp.get_context("spawn"),
        initializer=_init_pool,
    )
)


# Ray-backed pool — lazy because importing this module should not require a
# live Ray cluster (handy for offline eval / debugging).
_ray_pool = None

if _USE_RAY:
    import ray
    from ray.util.actor_pool import ActorPool

    @ray.remote
    class _PDMActor:
        """One PDM scoring worker. Same logic as `_pdm_worker`, but the
        simulator/scorer live on `self` instead of module globals so each Ray
        actor can hold its own state across calls."""

        def __init__(self):
            pdm_cfg = OmegaConf.load(PDM_CFG_PATH)
            self.simulator = instantiate(pdm_cfg.simulator)
            self.scorer = instantiate(pdm_cfg.scorer)
            self.scorer.train_mode = True

        def score(self, cache_path, traj_np):
            with lzma.open(cache_path, "rb") as f:
                metric_cache = pickle.load(f)
            return pdm_score(
                metric_cache=metric_cache,
                model_trajectory=traj_np,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            )

    def _build_ray_pool():
        if not ray.is_initialized():
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
        # Each DDP rank builds its own pool of scoring actors.
        num_actors = int(os.environ.get("SDV2_RAY_NUM_ACTORS", "16"))
        cpus_per_actor = float(os.environ.get("SDV2_RAY_CPUS_PER_ACTOR", "4"))
        actors = [
            _PDMActor.options(num_cpus=cpus_per_actor).remote()
            for _ in range(num_actors)
        ]
        return ActorPool(actors)


def get_pdm_score_para(trajectory, metric_cache_path):
    global _ray_pool
    B, G = trajectory.shape[:2]
    traj_np = trajectory.detach().cpu().numpy()

    # Split the G dimension across actors when requested. SDV2_G_SPLIT=K means
    # each sample's G candidate trajectories are scored in K parallel chunks,
    # giving B * K total jobs per rank. K=1 (default) preserves the original
    # one-job-per-sample behavior.
    g_split = max(1, int(os.environ.get("SDV2_G_SPLIT", "1")))
    chunk_size = (G + g_split - 1) // g_split
    # (sample_index, chunk_index) for each submitted job, in submission order
    sample_chunk_idx = []
    inputs = []
    for b in range(B):
        for k in range(g_split):
            start = k * chunk_size
            end = min(start + chunk_size, G)
            if start >= end:
                break
            sample_chunk_idx.append((b, k))
            inputs.append((metric_cache_path[b], traj_np[b, start:end]))

    if _USE_RAY:
        if _ray_pool is None:
            _ray_pool = _build_ray_pool()
        chunk_results = list(_ray_pool.map(
            lambda actor, args: actor.score.remote(*args),
            inputs,
        ))
    else:
        futures = [_pdm_pool.submit(_pdm_worker, inp) for inp in inputs]
        chunk_results = [f.result() for f in futures]

    if g_split == 1:
        return chunk_results

    # Reassemble per-sample dicts by concatenating chunks along the G axis.
    by_sample = [[] for _ in range(B)]
    for (b, _), res in zip(sample_chunk_idx, chunk_results):
        by_sample[b].append(res)
    sub_scores = []
    for chunks in by_sample:
        merged = {
            key: np.concatenate([c[key] for c in chunks], axis=0)
            for key in chunks[0]
        }
        sub_scores.append(merged)
    return sub_scores