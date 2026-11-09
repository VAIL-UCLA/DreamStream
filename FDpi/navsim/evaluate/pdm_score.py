import numpy as np
import numpy.typing as npt

from typing import List

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
from navsim.planning.metric_caching.metric_cache import MetricCache

from navsim.common.dataclasses import Trajectory

def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )

    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    # TODO: Refactor & add / modify existing metrics.
    pred_idx = 1

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]
    driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]

    score = scores[pred_idx]

    return PDMResults(
        no_at_fault_collisions,
        drivable_area_compliance,
        ego_progress,
        time_to_collision_within_bound,
        comfort,
        driving_direction_compliance,
        score,
    )

def ade_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    proposals: npt.NDArray[np.float64],
) -> PDMResults:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    initial_ego_state = metric_cache.ego_state

    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )

    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    trajectory_ade = np.mean(np.linalg.norm((simulated_states[0] - simulated_states[1])[:, :2],axis=1))
    
    proposal_states = []
    for proposal in proposals:
        proposal_traj = Trajectory(proposal)
        proposal_pred_traj = transform_trajectory(proposal_traj, initial_ego_state)
        proposal_pred_states = get_trajectory_as_array(proposal_pred_traj, future_sampling, initial_ego_state.time_point)
        proposal_states.append(proposal_pred_states)
    proposal_states = np.stack(proposal_states, axis=0)
    
    proposal_simulated_states = simulator.simulate_proposals(proposal_states, initial_ego_state)
    proposal_ades = np.mean(np.linalg.norm((simulated_states[0] - proposal_simulated_states)[:, :, :2], axis=2), axis=1)
    
    ade_results = {
        "trajectory_ade": trajectory_ade,
        # "proposal_ades": proposal_ades,
        "min_proposal_ade": proposal_ades.min(),
    }
    
    # scores = scorer.score_proposals(
    #     simulated_states,
    #     metric_cache.observation,
    #     metric_cache.centerline,
    #     metric_cache.route_lane_ids,
    #     metric_cache.drivable_area_map,
    # )

    # # TODO: Refactor & add / modify existing metrics.
    # pred_idx = 1

    # no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx]
    # drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx]

    # ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx]
    # time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx]
    # comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx]
    # driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx]

    # score = scores[pred_idx]

    return ade_results

# ---------------------------------------------------------------------------
# Batched scoring entrypoint used by DiffusionDriveV2.
# Accepts a single Trajectory (or a list / numpy array of trajectories) and
# returns either a single PDMResults or (list[PDMResults], simulated_pred).
# ---------------------------------------------------------------------------
from typing import Sequence as _Sequence  # avoid clobbering existing imports


def _pdm_score_single(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> PDMResults:
    """Single-trajectory PDM scoring (matches DiffusionDriveV2's helper)."""
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    pdm_states = get_trajectory_as_array(
        pdm_trajectory, future_sampling, initial_ego_state.time_point
    )
    pred_states = get_trajectory_as_array(
        pred_trajectory, future_sampling, initial_ego_state.time_point
    )
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states[None, ...]], axis=0
    )

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    pred_idx = 1
    return PDMResults(
        scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred_idx],
        scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.TTC, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred_idx],
        scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred_idx],
        scores[pred_idx],
    )


def pdm_score_para(
    metric_cache: MetricCache,
    model_trajectory,                  # Trajectory | Sequence[Trajectory] | np.ndarray
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
):
    """Batched PDM scoring.

    * Single Trajectory  -> returns PDMResults
    * list / ndarray of trajectories -> returns (List[PDMResults], simulated_pred)
    """
    if isinstance(model_trajectory, Trajectory):
        return _pdm_score_single(
            metric_cache, model_trajectory, future_sampling, simulator, scorer
        )

    if isinstance(model_trajectory, np.ndarray):
        traj_list = [Trajectory(model_trajectory[i]) for i in range(model_trajectory.shape[0])]
    else:
        traj_list = list(model_trajectory)

    n = len(traj_list)
    if n == 0:
        raise ValueError("Empty trajectory batch.")

    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory, future_sampling, initial_ego_state.time_point
    )

    pred_states_batch = []
    for traj in traj_list:
        pred_world = transform_trajectory(traj, initial_ego_state)
        pred_states_batch.append(
            get_trajectory_as_array(pred_world, future_sampling, initial_ego_state.time_point)
        )
    pred_states_batch = np.stack(pred_states_batch, axis=0)
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states_batch], axis=0
    )

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)
    scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    results = []
    for i in range(1, n + 1):
        results.append(
            PDMResults(
                scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, i],
                scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, i],
                scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, i],
                scorer._weighted_metrics[WeightedMetricIndex.TTC, i],
                scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, i],
                scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, i],
                scores[i],
            )
        )
    return results, simulated_states[1:]
