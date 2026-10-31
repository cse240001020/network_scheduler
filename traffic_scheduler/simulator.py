"""Synthetic traffic and queue simulator used by the baseline and AI policies.

This is a flow-level discrete-event model. It is intentionally small enough to
run on a laptop and transparent enough to audit; it is not a packet emulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
from typing import Iterable, Sequence


FEATURE_NAMES = (
    "flow_size_bytes",
    "path_base_capacity_mb_s",
    "path_index",
    "queued_flow_count",
    "queued_bytes",
    "buffer_occupancy_fraction",
    "estimated_queue_wait_ms",
    "estimated_capacity_mb_s",
    "recent_mean_latency_ms",
    "recent_capacity_trend_mb_s",
    "capacity_lag1_multiplier",
    "capacity_lag2_multiplier",
    "capacity_lag3_multiplier",
    "ms_since_capacity_observation",
    "completed_flows_observed",
    "arrival_rate_flows_s",
    "time_in_scenario_s",
)

CAPACITY_MULTIPLIERS = (0.35, 0.65, 1.0, 1.4)


@dataclass(frozen=True)
class Flow:
    flow_id: int
    arrival_ms: float
    size_bytes: int


@dataclass
class PathState:
    base_capacity_mb_s: float
    available_at_ms: float = 0.0
    pending: deque[tuple[float, int, float, float]] = field(default_factory=deque)
    queued_bytes: int = 0
    completed_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    observed_capacity_mb_s: float = 0.0
    previous_observed_capacity_mb_s: float = 0.0
    observed_capacity_history: deque[float] = field(default_factory=deque)
    last_observation_ms: float | None = None
    completed_count: int = 0

    def __post_init__(self) -> None:
        self.observed_capacity_mb_s = self.base_capacity_mb_s
        self.previous_observed_capacity_mb_s = self.base_capacity_mb_s
        self.observed_capacity_history = deque(
            [self.base_capacity_mb_s] * 3, maxlen=3
        )


@dataclass(frozen=True)
class FlowResult:
    flow_id: int
    arrival_ms: float
    size_bytes: int
    policy: str
    path_id: int
    dropped: bool
    queue_delay_ms: float
    service_ms: float
    latency_ms: float


@dataclass(frozen=True)
class SimulationSummary:
    policy: str
    flows_offered: int
    flows_admitted: int
    flows_dropped: int
    drop_rate_percent: float
    mean_latency_ms: float
    p95_latency_ms: float
    throughput_mb_s: float
    path_bytes: tuple[int, ...]


@dataclass
class Scenario:
    flows: list[Flow]
    capacities_mb_s: tuple[float, ...]
    arrival_rate_flows_s: float
    buffer_limit_bytes: int
    seed: int
    state_bin_ms: float = 250.0
    multipliers: list[list[float]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        flow_count: int,
        arrival_rate_flows_s: float,
        median_flow_bytes: int,
        capacities_mb_s: Sequence[float],
        buffer_limit_mb: int,
        seed: int,
    ) -> "Scenario":
        if flow_count <= 0:
            raise ValueError("flow_count must be positive")
        if arrival_rate_flows_s <= 0 or median_flow_bytes <= 0:
            raise ValueError("arrival rate and median flow size must be positive")
        if not capacities_mb_s or any(capacity <= 0 for capacity in capacities_mb_s):
            raise ValueError("each path must have a positive capacity")

        rng = random.Random(seed)
        arrival_ms = 0.0
        flows: list[Flow] = []
        sigma = 0.9
        for flow_id in range(flow_count):
            arrival_ms += rng.expovariate(arrival_rate_flows_s / 1000.0)
            size = int(rng.lognormvariate(math.log(median_flow_bytes), sigma))
            size = max(512, min(size, 4_000_000))
            flows.append(Flow(flow_id, arrival_ms, size))

        scenario = cls(
            flows=flows,
            capacities_mb_s=tuple(float(value) for value in capacities_mb_s),
            arrival_rate_flows_s=float(arrival_rate_flows_s),
            buffer_limit_bytes=int(buffer_limit_mb * 1_000_000),
            seed=seed,
        )
        scenario._generate_capacity_states()
        return scenario

    def _generate_capacity_states(self) -> None:
        max_time_ms = self.flows[-1].arrival_ms if self.flows else 0.0
        # Leave room for the final queued flows to complete.
        max_time_ms += 10_000.0
        bin_count = max(1, int(max_time_ms / self.state_bin_ms) + 2)
        self.multipliers = []
        for path_index in range(len(self.capacities_mb_s)):
            rng = random.Random(self.seed * 1009 + path_index * 9176 + 31)
            state = 2  # start at normal capacity
            states: list[float] = []
            for _ in range(bin_count):
                if rng.random() < 0.08:
                    state = rng.randrange(len(CAPACITY_MULTIPLIERS))
                states.append(CAPACITY_MULTIPLIERS[state])
            self.multipliers.append(states)

    def multiplier_at(self, path_index: int, time_ms: float) -> float:
        bin_index = min(int(time_ms / self.state_bin_ms), len(self.multipliers[path_index]) - 1)
        return self.multipliers[path_index][bin_index]


class SchedulerSimulator:
    """Run a routing policy against one fixed arrival and capacity scenario."""

    def __init__(self, scenario: Scenario, buffer_limit_bytes: int | None = None):
        self.scenario = scenario
        self.buffer_limit_bytes = buffer_limit_bytes or scenario.buffer_limit_bytes
        self.paths = [PathState(capacity) for capacity in scenario.capacities_mb_s]
        self.round_robin_cursor = 0
        self.path_bytes = [0 for _ in self.paths]

    def _retire_completed(self, path: PathState, now_ms: float) -> None:
        while path.pending and path.pending[0][0] <= now_ms:
            finish_ms, size_bytes, service_ms, arrival_ms = path.pending.popleft()
            path.queued_bytes -= size_bytes
            latency = finish_ms - arrival_ms
            path.completed_latencies.append(latency)
            path.completed_count += 1
            if service_ms > 0.0:
                sample_capacity = (size_bytes / service_ms) / 1000.0
                path.previous_observed_capacity_mb_s = path.observed_capacity_mb_s
                path.observed_capacity_mb_s = 0.75 * path.observed_capacity_mb_s + 0.25 * sample_capacity
                path.observed_capacity_history.append(sample_capacity)
                path.last_observation_ms = finish_ms

    def _features(self, flow: Flow, path_index: int) -> list[float]:
        path = self.paths[path_index]
        estimated_capacity = max(path.observed_capacity_mb_s, path.base_capacity_mb_s * 0.05)
        queue_wait_ms = (path.queued_bytes / 1000.0) / estimated_capacity
        recent_latency = (
            sum(path.completed_latencies) / len(path.completed_latencies)
            if path.completed_latencies
            else 0.0
        )
        trend = path.observed_capacity_mb_s - path.previous_observed_capacity_mb_s
        capacity_lags = [value / path.base_capacity_mb_s for value in reversed(path.observed_capacity_history)]
        observation_age_ms = (
            max(0.0, flow.arrival_ms - path.last_observation_ms)
            if path.last_observation_ms is not None
            else 10_000.0
        )
        return [
            float(flow.size_bytes),
            path.base_capacity_mb_s,
            float(path_index),
            float(len(path.pending)),
            float(path.queued_bytes),
            path.queued_bytes / self.buffer_limit_bytes,
            queue_wait_ms,
            estimated_capacity,
            recent_latency,
            trend,
            *capacity_lags,
            observation_age_ms,
            float(path.completed_count),
            self.scenario.arrival_rate_flows_s,
            flow.arrival_ms / 1000.0,
        ]

    def _select_path(
        self,
        policy: str,
        flow: Flow,
        feature_rows: list[list[float]],
        model: object | None,
        chooser_rng: random.Random,
    ) -> int:
        available = [
            index for index, path in enumerate(self.paths)
            if path.queued_bytes + flow.size_bytes <= self.buffer_limit_bytes
        ]
        candidates = available or list(range(len(self.paths)))
        if policy == "round_robin":
            selected = self.round_robin_cursor
            self.round_robin_cursor = (self.round_robin_cursor + 1) % len(self.paths)
            return selected
        if policy == "random":
            return chooser_rng.randrange(len(self.paths))
        if policy == "shortest_queue":
            return min(candidates, key=lambda index: len(self.paths[index].pending))

        if policy == "least_delay":
            estimates = [row[6] + row[0] / (row[7] * 1000.0) for row in feature_rows]
            return min(candidates, key=estimates.__getitem__)

        if policy == "persistence_delay":
            # A no-learning forecast: assume the most recently observed
            # capacity state persists until this flow is served.
            lag_index = FEATURE_NAMES.index("capacity_lag1_multiplier")
            estimates = [
                row[6] + row[0] / (row[1] * row[lag_index] * 1000.0)
                for row in feature_rows
            ]
            return min(candidates, key=estimates.__getitem__)

        if policy == "oracle":
            # The oracle sees the simulator's current hidden capacity state and
            # exact queue finish time. It is a reference policy, not deployable.
            def actual_completion_delay(index: int) -> float:
                path = self.paths[index]
                start_ms = max(flow.arrival_ms, path.available_at_ms)
                capacity = path.base_capacity_mb_s * self.scenario.multiplier_at(index, start_ms)
                service_ms = flow.size_bytes / (capacity * 1000.0)
                return start_ms - flow.arrival_ms + service_ms

            return min(candidates, key=actual_completion_delay)

        if policy in ("ai_capacity_random_forest", "ai_capacity_hist_gradient_boosting"):
            if model is None:
                raise ValueError(f"policy {policy!r} requires a trained capacity model")
            predicted_multipliers = model.predict_multiplier(feature_rows)  # type: ignore[attr-defined]
            estimates = [
                row[6] + row[0] / (row[1] * float(predicted_multipliers[index]) * 1000.0)
                for index, row in enumerate(feature_rows)
            ]
            return min(candidates, key=estimates.__getitem__)

        if policy in ("ai_random_forest", "ai_hist_gradient_boosting"):
            if model is None:
                raise ValueError(f"policy {policy!r} requires a trained model")
            predictions = model.predict(feature_rows)  # type: ignore[attr-defined]
            return min(candidates, key=lambda index: float(predictions[index]))

        raise ValueError(f"unknown policy: {policy}")

    def run(
        self,
        policy: str,
        *,
        model: object | None = None,
        collect_training_rows: bool = False,
        collect_capacity_rows: bool = False,
    ) -> tuple[list[FlowResult], list[dict[str, float]], list[dict[str, float]]]:
        results: list[FlowResult] = []
        training_rows: list[dict[str, float]] = []
        capacity_rows: list[dict[str, float]] = []
        chooser_rng = random.Random(self.scenario.seed + 811)
        path_bytes = [0 for _ in self.paths]

        for flow in self.scenario.flows:
            for path in self.paths:
                self._retire_completed(path, flow.arrival_ms)

            feature_rows = [self._features(flow, index) for index in range(len(self.paths))]
            if collect_capacity_rows:
                for path_index, (path, feature_row) in enumerate(zip(self.paths, feature_rows)):
                    if path.queued_bytes + flow.size_bytes > self.buffer_limit_bytes:
                        continue
                    estimated_start_ms = flow.arrival_ms + feature_row[6]
                    row = {name: value for name, value in zip(FEATURE_NAMES, feature_row)}
                    future_multiplier = self.scenario.multiplier_at(path_index, estimated_start_ms)
                    row["target_capacity_state"] = float(CAPACITY_MULTIPLIERS.index(future_multiplier))
                    capacity_rows.append(row)
            selected = self._select_path(policy, flow, feature_rows, model, chooser_rng)
            path = self.paths[selected]
            if path.queued_bytes + flow.size_bytes > self.buffer_limit_bytes:
                results.append(
                    FlowResult(flow.flow_id, flow.arrival_ms, flow.size_bytes, policy, selected,
                               True, 0.0, 0.0, 0.0)
                )
                continue

            start_ms = max(flow.arrival_ms, path.available_at_ms)
            queue_delay_ms = start_ms - flow.arrival_ms
            actual_capacity = (
                path.base_capacity_mb_s
                * self.scenario.multiplier_at(selected, start_ms)
            )
            service_ms = flow.size_bytes / (actual_capacity * 1000.0)
            finish_ms = start_ms + service_ms
            latency_ms = finish_ms - flow.arrival_ms
            path.available_at_ms = finish_ms
            path.queued_bytes += flow.size_bytes
            path.pending.append((finish_ms, flow.size_bytes, service_ms, flow.arrival_ms))
            path_bytes[selected] += flow.size_bytes
            results.append(
                FlowResult(flow.flow_id, flow.arrival_ms, flow.size_bytes, policy, selected,
                           False, queue_delay_ms, service_ms, latency_ms)
            )

            if collect_training_rows:
                row = {name: value for name, value in zip(FEATURE_NAMES, feature_rows[selected])}
                row["target_latency_ms"] = latency_ms
                training_rows.append(row)

        self.path_bytes = path_bytes
        return results, training_rows, capacity_rows


def summarize(policy: str, results: Iterable[FlowResult], path_bytes: Sequence[int]) -> SimulationSummary:
    rows = list(results)
    admitted = [row for row in rows if not row.dropped]
    dropped = len(rows) - len(admitted)
    latencies = sorted(row.latency_ms for row in admitted)
    p95 = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_bytes = sum(row.size_bytes for row in admitted)
    if admitted:
        duration_s = (max(row.arrival_ms + row.latency_ms for row in admitted)
                      - min(row.arrival_ms for row in rows)) / 1000.0
    else:
        duration_s = 0.0
    throughput = total_bytes / duration_s / 1_000_000.0 if duration_s > 0.0 else 0.0
    return SimulationSummary(
        policy=policy,
        flows_offered=len(rows),
        flows_admitted=len(admitted),
        flows_dropped=dropped,
        drop_rate_percent=(100.0 * dropped / len(rows)) if rows else 0.0,
        mean_latency_ms=mean_latency,
        p95_latency_ms=p95,
        throughput_mb_s=throughput,
        path_bytes=tuple(path_bytes),
    )
