"""Training, repeated evaluation, and paired statistical comparisons."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .model import (
    feature_importances,
    evaluate_capacity_models,
    evaluate_persistence_validation,
    load_model,
    save_model,
    train_capacity_models,
    train_latency_models,
)
from .simulator import FEATURE_NAMES, Scenario, SchedulerSimulator, summarize


TRAIN_CAPACITY_SETS = (
    (55.0, 35.0, 20.0),
    (60.0, 30.0, 15.0),
    (70.0, 25.0, 15.0),
    (45.0, 35.0, 20.0),
    (80.0, 30.0, 15.0),
    (50.0, 40.0, 10.0),
)
TRAIN_ARRIVAL_RATES = (300.0, 450.0, 600.0, 750.0, 900.0)
TRAIN_MEDIAN_SIZES = (32_000, 64_000, 96_000, 128_000)
POLICY_MODELS = {
    "ai_random_forest": "random_forest",
    "ai_hist_gradient_boosting": "hist_gradient_boosting",
}
CAPACITY_POLICY_MODELS = {
    "ai_capacity_random_forest": "random_forest",
    "ai_capacity_hist_gradient_boosting": "hist_gradient_boosting",
}
AI_POLICIES = (*POLICY_MODELS, *CAPACITY_POLICY_MODELS)
POLICIES = (
    "round_robin",
    "shortest_queue",
    "least_delay",
    "persistence_delay",
    "oracle",
    *POLICY_MODELS,
    *CAPACITY_POLICY_MODELS,
)
POLICY_COLORS = {
    "round_robin": "#8993a4",
    "shortest_queue": "#4f86c6",
    "least_delay": "#31a37a",
    "persistence_delay": "#00a6a6",
    "oracle": "#222222",
    "ai_random_forest": "#ee9b36",
    "ai_hist_gradient_boosting": "#d35f5f",
    "ai_capacity_random_forest": "#845ec2",
    "ai_capacity_hist_gradient_boosting": "#008f7a",
}


def _training_scenarios(count: int, flows: int, seed: int) -> list[tuple[str, Scenario]]:
    rng = random.Random(seed + 70_001)
    scenarios: list[tuple[str, Scenario]] = []
    for index in range(count):
        scenario_seed = seed + 10_000 + index * 37
        capacities = TRAIN_CAPACITY_SETS[index % len(TRAIN_CAPACITY_SETS)]
        rate = TRAIN_ARRIVAL_RATES[(index * 3 + index // 2) % len(TRAIN_ARRIVAL_RATES)]
        median_size = TRAIN_MEDIAN_SIZES[(index * 5 + 1) % len(TRAIN_MEDIAN_SIZES)]
        rate *= rng.uniform(0.9, 1.1)
        median_size = int(median_size * rng.uniform(0.9, 1.1))
        scenario = Scenario.create(
            flow_count=flows,
            arrival_rate_flows_s=rate,
            median_flow_bytes=median_size,
            capacities_mb_s=capacities,
            buffer_limit_mb=64,
            seed=scenario_seed,
        )
        scenarios.append((f"train_{index:02d}", scenario))
    return scenarios


def _evaluation_scenarios(
    flows: int, repetitions: int, seed: int
) -> list[tuple[str, str, Scenario]]:
    # These regimes are not copies of the training grid. Repetitions vary only
    # the random traffic and congestion-state seeds within each fixed regime.
    specs = (
        ("light", 260.0, 48_000, (60.0, 35.0, 20.0)),
        ("balanced", 570.0, 72_000, (55.0, 35.0, 20.0)),
        ("short_bursts", 820.0, 48_000, (65.0, 28.0, 14.0)),
        ("large_flows", 410.0, 116_000, (48.0, 38.0, 18.0)),
        ("overloaded", 900.0, 128_000, (48.0, 32.0, 18.0)),
    )
    scenarios: list[tuple[str, str, Scenario]] = []
    for regime_index, (regime, rate, median_size, capacities) in enumerate(specs):
        for repetition in range(repetitions):
            scenario_id = f"{regime}_rep_{repetition:02d}"
            scenario_seed = seed + 90_000 + regime_index * 10_000 + repetition * 103
            scenario = Scenario.create(
                flow_count=flows,
                arrival_rate_flows_s=rate,
                median_flow_bytes=median_size,
                capacities_mb_s=capacities,
                buffer_limit_mb=64,
                seed=scenario_seed,
            )
            scenarios.append((scenario_id, regime, scenario))
    return scenarios


def _write_dict_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_dict_rows(path, rows, ["scenario_id", *FEATURE_NAMES, "target_latency_ms"])


def _stratified_bootstrap_ci(
    groups: dict[str, list[float]], seed: int, samples: int = 1200
) -> tuple[float, float]:
    groups = {name: values for name, values in groups.items() if values}
    if not groups:
        return 0.0, 0.0
    if all(len(values) == 1 for values in groups.values()):
        values = [items[0] for items in groups.values()]
        return mean(values), mean(values)
    rng = random.Random(seed)
    group_names = sorted(groups)
    draws = []
    for _ in range(samples):
        resampled_group_means = []
        for name in group_names:
            values = groups[name]
            resampled_group_means.append(
                mean(values[rng.randrange(len(values))] for _ in values)
            )
        draws.append(mean(resampled_group_means))
    draws.sort()
    low_index = max(0, int(0.025 * samples))
    high_index = min(samples - 1, int(0.975 * samples))
    return draws[low_index], draws[high_index]


def _aggregate_results(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(POLICIES):
        policy_rows = [row for row in rows if row["policy"] == policy]
        p95_values = [float(row["p95_latency_ms"]) for row in policy_rows]
        latency_values = [float(row["mean_latency_ms"]) for row in policy_rows]
        p95_by_regime: dict[str, list[float]] = {}
        latency_by_regime: dict[str, list[float]] = {}
        for row in policy_rows:
            regime = str(row["regime"])
            p95_by_regime.setdefault(regime, []).append(float(row["p95_latency_ms"]))
            latency_by_regime.setdefault(regime, []).append(float(row["mean_latency_ms"]))
        p95_low, p95_high = _stratified_bootstrap_ci(p95_by_regime, seed + policy_index * 13 + 1)
        latency_low, latency_high = _stratified_bootstrap_ci(
            latency_by_regime, seed + policy_index * 13 + 2
        )
        aggregate.append({
            "policy": policy,
            "scenarios": len(policy_rows),
            "mean_p95_latency_ms": mean(p95_values),
            "p95_ci95_low_ms": p95_low,
            "p95_ci95_high_ms": p95_high,
            "mean_latency_ms": mean(latency_values),
            "mean_latency_ci95_low_ms": latency_low,
            "mean_latency_ci95_high_ms": latency_high,
            "mean_drop_rate_percent": mean(float(row["drop_rate_percent"]) for row in policy_rows),
            "mean_throughput_mb_s": mean(float(row["throughput_mb_s"]) for row in policy_rows),
        })
    return aggregate


def _paired_comparisons(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), {})[str(row["policy"])] = row

    comparisons: list[dict[str, Any]] = []
    metrics = ("p95_latency_ms", "mean_latency_ms", "throughput_mb_s", "drop_rate_percent")
    for ai_index, ai_policy in enumerate(AI_POLICIES):
        for baseline_index, baseline in enumerate(("shortest_queue", "least_delay", "persistence_delay")):
            for metric_index, metric in enumerate(metrics):
                differences_by_regime: dict[str, list[float]] = {}
                for scenario_rows in by_scenario.values():
                    if ai_policy in scenario_rows and baseline in scenario_rows:
                        regime = str(scenario_rows[ai_policy]["regime"])
                        differences_by_regime.setdefault(regime, []).append(
                            float(scenario_rows[ai_policy][metric]) - float(scenario_rows[baseline][metric])
                        )
                differences = [value for values in differences_by_regime.values() for value in values]
                low, high = _stratified_bootstrap_ci(
                    differences_by_regime,
                    seed + ai_index * 1000 + baseline_index * 100 + metric_index,
                )
                comparisons.append({
                    "ai_policy": ai_policy,
                    "baseline": baseline,
                    "metric": metric,
                    "paired_scenarios": len(differences),
                    "regimes": len(differences_by_regime),
                    "mean_ai_minus_baseline": mean(differences) if differences else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                })
    return comparisons


def _make_plot(path: Path, aggregate: list[dict[str, Any]]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = path.parent / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [str(row["policy"]) for row in aggregate]
    colors = [POLICY_COLORS[label] for label in labels]
    positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    p95 = [float(row["mean_p95_latency_ms"]) for row in aggregate]
    p95_low = [max(0.0, p95[i] - float(aggregate[i]["p95_ci95_low_ms"])) for i in positions]
    p95_high = [max(0.0, float(aggregate[i]["p95_ci95_high_ms"]) - p95[i]) for i in positions]
    axes[0].bar(positions, p95, color=colors)
    axes[0].errorbar(positions, p95, yerr=[p95_low, p95_high], fmt="none", ecolor="#222222", capsize=4)
    axes[0].set_title("Mean p95 latency with 95% CI")
    axes[0].set_ylabel("milliseconds (lower is better)")

    latency = [float(row["mean_latency_ms"]) for row in aggregate]
    latency_low = [max(0.0, latency[i] - float(aggregate[i]["mean_latency_ci95_low_ms"])) for i in positions]
    latency_high = [max(0.0, float(aggregate[i]["mean_latency_ci95_high_ms"]) - latency[i]) for i in positions]
    axes[1].bar(positions, latency, color=colors)
    axes[1].errorbar(positions, latency, yerr=[latency_low, latency_high], fmt="none", ecolor="#222222", capsize=4)
    axes[1].set_title("Mean flow latency with 95% CI")
    axes[1].set_ylabel("milliseconds (lower is better)")

    drops = [float(row["mean_drop_rate_percent"]) for row in aggregate]
    axes[2].bar(positions, drops, color=colors)
    axes[2].set_title("Mean flow drop rate")
    axes[2].set_ylabel("percent (lower is better)")

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Repeated held-out traffic scenarios (error bars: bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def run_pipeline(
    *, flows: int, train_scenarios: int, test_repetitions: int, seed: int, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "flows_per_scenario": flows,
        "training_scenarios": train_scenarios,
        "test_repetitions_per_regime": test_repetitions,
        "test_regimes": ["light", "balanced", "short_bursts", "large_flows", "overloaded"],
        "policies": list(POLICIES),
    }
    (output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generating training examples from {train_scenarios} synthetic scenarios...")
    train_rows: list[dict[str, Any]] = []
    capacity_train_rows: list[dict[str, Any]] = []
    for scenario_id, scenario in _training_scenarios(train_scenarios, flows, seed):
        _, rows, capacity_rows = SchedulerSimulator(scenario).run(
            "random", collect_training_rows=True, collect_capacity_rows=True
        )
        for row in rows:
            row["scenario_id"] = scenario_id
        for row in capacity_rows:
            row["scenario_id"] = scenario_id
        train_rows.extend(rows)
        capacity_train_rows.extend(capacity_rows)
        print(f"  {scenario_id}: {len(rows):,} accepted randomized examples")
    if len(train_rows) < 100:
        raise RuntimeError("too few training examples; increase --flows or --train-scenarios")

    training_csv = output_dir / "training_examples.csv"
    _write_training_rows(training_csv, train_rows)
    models, validation_rows = train_latency_models(train_rows, seed)
    for name, model in models.items():
        save_model(model, output_dir / f"latency_model_{name}.joblib")

    capacity_training_csv = output_dir / "capacity_training_examples.csv"
    _write_dict_rows(
        capacity_training_csv,
        capacity_train_rows,
        ["scenario_id", *FEATURE_NAMES, "target_capacity_state"],
    )
    capacity_models, capacity_validation_rows, capacity_confusion_rows = train_capacity_models(
        capacity_train_rows, seed
    )
    for name, model in capacity_models.items():
        save_model(model, output_dir / f"capacity_model_{name}.joblib")
    validation_csv = output_dir / "model_validation.csv"
    _write_dict_rows(validation_csv, validation_rows, list(validation_rows[0].keys()))
    (output_dir / "model_validation.json").write_text(
        json.dumps(validation_rows, indent=2) + "\n", encoding="utf-8"
    )
    capacity_validation_csv = output_dir / "capacity_model_validation.csv"
    _write_dict_rows(
        capacity_validation_csv,
        capacity_validation_rows,
        list(capacity_validation_rows[0].keys()),
    )
    (output_dir / "capacity_model_validation.json").write_text(
        json.dumps(capacity_validation_rows, indent=2) + "\n", encoding="utf-8"
    )
    confusion_csv = output_dir / "capacity_state_confusion.csv"
    _write_dict_rows(confusion_csv, capacity_confusion_rows, list(capacity_confusion_rows[0].keys()))
    importance_rows = [
        {"feature": name, "importance": value}
        for name, value in feature_importances(models["random_forest"])
    ]
    importance_csv = output_dir / "feature_importance_random_forest.csv"
    _write_dict_rows(importance_csv, importance_rows, ["feature", "importance"])
    capacity_importance_rows = [
        {"feature": name, "importance": value}
        for name, value in feature_importances(capacity_models["random_forest"])
    ]
    capacity_importance_csv = output_dir / "capacity_feature_importance_random_forest.csv"
    _write_dict_rows(
        capacity_importance_csv, capacity_importance_rows, ["feature", "importance"]
    )

    print("\nScenario-level validation (whole scenarios held out):")
    for row in validation_rows:
        print(
            f"  {row['model']:24s} MAE={float(row['validation_mae_ms']):9.3f} ms  "
            f"R²={float(row['validation_r2']):6.3f}"
        )
    print("\nCapacity-state forecast validation (state is measured at estimated service start):")
    for row in capacity_validation_rows:
        print(
            f"  {row['model']:24s} macro-F1={float(row['macro_f1']):.3f}  "
            f"balanced-accuracy={float(row['balanced_accuracy']):.3f}  "
            f"multiplier-MAE={float(row['multiplier_mae']):.3f}"
        )
    print("Random Forest feature importance (top 5):")
    for row in importance_rows[:5]:
        print(f"  {row['feature']}: {float(row['importance']):.3f}")

    evaluation_rows: list[dict[str, Any]] = []
    capacity_test_rows: list[dict[str, Any]] = []
    scenarios = _evaluation_scenarios(flows, test_repetitions, seed)
    print(
        f"\nEvaluating {len(scenarios)} held-out scenarios, "
        f"{len(POLICIES)} policies each..."
    )
    for scenario_id, regime, scenario in scenarios:
        _, _, capacity_eval_examples = SchedulerSimulator(scenario).run(
            "random", collect_capacity_rows=True
        )
        for row in capacity_eval_examples:
            row["scenario_id"] = scenario_id
        for metrics in evaluate_capacity_models(capacity_models, capacity_eval_examples):
            capacity_test_rows.append({"scenario_id": scenario_id, "regime": regime, **metrics})

        for policy in POLICIES:
            model_name = POLICY_MODELS.get(policy)
            simulator = SchedulerSimulator(scenario)
            capacity_model_name = CAPACITY_POLICY_MODELS.get(policy)
            selected_model = (
                models[model_name] if model_name else
                capacity_models[capacity_model_name] if capacity_model_name else
                None
            )
            results, _, _ = simulator.run(policy, model=selected_model)
            summary = summarize(policy, results, simulator.path_bytes)
            row: dict[str, Any] = {
                "scenario_id": scenario_id,
                "regime": regime,
                **summary.__dict__,
            }
            row["path_bytes"] = ";".join(str(value) for value in summary.path_bytes)
            evaluation_rows.append(row)
        if scenario_id.endswith("_rep_00"):
            print(f"  completed {regime} regime")

    result_fields = [
        "scenario_id", "regime", "policy", "flows_offered", "flows_admitted", "flows_dropped",
        "drop_rate_percent", "mean_latency_ms", "p95_latency_ms", "throughput_mb_s", "path_bytes",
    ]
    results_csv = output_dir / "evaluation_results.csv"
    _write_dict_rows(results_csv, evaluation_rows, result_fields)
    capacity_test_csv = output_dir / "capacity_forecast_test_results.csv"
    _write_dict_rows(capacity_test_csv, capacity_test_rows, list(capacity_test_rows[0].keys()))
    capacity_test_summary = []
    for model_name in ("random_forest", "hist_gradient_boosting", "persistence_last_observed"):
        model_rows = [row for row in capacity_test_rows if row["model"] == model_name]
        capacity_test_summary.append({
            "model": model_name,
            "scenarios": len(model_rows),
            "mean_accuracy": mean(float(row["accuracy"]) for row in model_rows),
            "mean_balanced_accuracy": mean(float(row["balanced_accuracy"]) for row in model_rows),
            "mean_macro_f1": mean(float(row["macro_f1"]) for row in model_rows),
            "mean_multiplier_mae": mean(float(row["multiplier_mae"]) for row in model_rows),
        })
    capacity_test_summary_csv = output_dir / "capacity_forecast_test_summary.csv"
    _write_dict_rows(
        capacity_test_summary_csv, capacity_test_summary, list(capacity_test_summary[0].keys())
    )

    aggregate = _aggregate_results(evaluation_rows, seed)
    aggregate_csv = output_dir / "policy_summary.csv"
    _write_dict_rows(aggregate_csv, aggregate, list(aggregate[0].keys()))
    paired = _paired_comparisons(evaluation_rows, seed)
    paired_csv = output_dir / "paired_comparisons.csv"
    _write_dict_rows(paired_csv, paired, list(paired[0].keys()))
    chart_created = _make_plot(output_dir / "policy_comparison.png", aggregate)

    print("\nAverage across repeated held-out scenarios (95% bootstrap CIs):")
    print(f"{'policy':26s} {'p95 ms':>12s} {'mean ms':>12s} {'drops %':>10s} {'MB/s':>10s}")
    for row in aggregate:
        print(
            f"{row['policy']:26s} {row['mean_p95_latency_ms']:12.3f} "
            f"{row['mean_latency_ms']:12.3f} {row['mean_drop_rate_percent']:10.3f} "
            f"{row['mean_throughput_mb_s']:10.3f}"
        )

    print("\nAI minus baseline paired differences (negative latency favors AI):")
    for row in paired:
        if row["metric"] == "p95_latency_ms":
            print(
                f"  {row['ai_policy']} vs {row['baseline']}: "
                f"{float(row['mean_ai_minus_baseline']):+.3f} ms "
                f"(95% CI {float(row['ci95_low']):+.3f}, {float(row['ci95_high']):+.3f})"
            )

    print("\nCapacity forecast quality on held-out scenarios:")
    for row in capacity_test_summary:
        print(
            f"  {row['model']:24s} macro-F1={float(row['mean_macro_f1']):.3f}  "
            f"balanced-accuracy={float(row['mean_balanced_accuracy']):.3f}  "
            f"multiplier-MAE={float(row['mean_multiplier_mae']):.3f}"
        )

    print("\nArtifacts:")
    print(f"  training examples: {training_csv}")
    print(f"  capacity examples:{capacity_training_csv}")
    print(f"  validation models: {validation_csv}")
    print(f"  capacity validation:{capacity_validation_csv}")
    print(f"  scenario results:  {results_csv}")
    print(f"  capacity test metrics: {capacity_test_csv}")
    print(f"  capacity test summary: {capacity_test_summary_csv}")
    print(f"  policy summary:    {aggregate_csv}")
    print(f"  paired comparison: {paired_csv}")
    print(f"  RF importances:    {importance_csv}")
    print(f"  capacity importances:{capacity_importance_csv}")
    if chart_created:
        print(f"  comparison chart:  {output_dir / 'policy_comparison.png'}")


def rebuild_reports(output_dir: Path, seed: int) -> None:
    """Recompute aggregate tables and charts without repeating simulations."""
    results_path = output_dir / "evaluation_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"no evaluation results found at {results_path}")
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    numeric_fields = (
        "flows_offered", "flows_admitted", "flows_dropped", "drop_rate_percent",
        "mean_latency_ms", "p95_latency_ms", "throughput_mb_s",
    )
    for row in rows:
        for field_name in numeric_fields:
            row[field_name] = float(row[field_name])

    metadata_path = output_dir / "experiment_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        flows = int(metadata["flows_per_scenario"])
        repetitions = int(metadata["test_repetitions_per_regime"])
        seed = int(metadata["seed"])
    else:
        flows = int(rows[0]["flows_offered"])
        regime_scenario_counts = {
            str(regime): len({str(row["scenario_id"]) for row in rows if str(row["regime"]) == str(regime)})
            for regime in {str(row["regime"]) for row in rows}
        }
        repetitions = max(regime_scenario_counts.values())
        metadata = {
            "seed": seed,
            "flows_per_scenario": flows,
            "training_scenarios": 0,
            "test_repetitions_per_regime": repetitions,
            "test_regimes": sorted(regime_scenario_counts),
            "policies": sorted({str(row["policy"]) for row in rows}),
        }

    existing_policies = {str(row["policy"]) for row in rows}
    if "persistence_delay" not in existing_policies:
        for scenario_id, regime, scenario in _evaluation_scenarios(flows, repetitions, seed):
            simulator = SchedulerSimulator(scenario)
            results, _, _ = simulator.run("persistence_delay")
            summary = summarize("persistence_delay", results, simulator.path_bytes)
            result_row: dict[str, Any] = {
                "scenario_id": scenario_id,
                "regime": regime,
                **summary.__dict__,
            }
            result_row["path_bytes"] = ";".join(str(value) for value in summary.path_bytes)
            rows.append(result_row)
        _write_dict_rows(results_path, rows, [
            "scenario_id", "regime", "policy", "flows_offered", "flows_admitted", "flows_dropped",
            "drop_rate_percent", "mean_latency_ms", "p95_latency_ms", "throughput_mb_s", "path_bytes",
        ])
    metadata["policies"] = sorted({str(row["policy"]) for row in rows})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    aggregate = _aggregate_results(rows, seed)
    _write_dict_rows(output_dir / "policy_summary.csv", aggregate, list(aggregate[0].keys()))
    paired = _paired_comparisons(rows, seed)
    _write_dict_rows(output_dir / "paired_comparisons.csv", paired, list(paired[0].keys()))
    _make_plot(output_dir / "policy_comparison.png", aggregate)

    training_path = output_dir / "capacity_training_examples.csv"
    if training_path.exists():
        with training_path.open(newline="", encoding="utf-8") as handle:
            capacity_training_rows: list[dict[str, Any]] = list(csv.DictReader(handle))
        for row in capacity_training_rows:
            for name in FEATURE_NAMES:
                row[name] = float(row[name])
            row["target_capacity_state"] = int(float(row["target_capacity_state"]))
        persistence_validation, persistence_confusion = evaluate_persistence_validation(
            capacity_training_rows, seed
        )
        validation_path = output_dir / "capacity_model_validation.csv"
        with validation_path.open(newline="", encoding="utf-8") as handle:
            validation_rows = [
                row for row in csv.DictReader(handle)
                if row.get("model") != "persistence_last_observed"
            ]
        validation_rows.append(persistence_validation)
        _write_dict_rows(validation_path, validation_rows, list(validation_rows[0].keys()))
        (output_dir / "capacity_model_validation.json").write_text(
            json.dumps(validation_rows, indent=2) + "\n", encoding="utf-8"
        )

        confusion_path = output_dir / "capacity_state_confusion.csv"
        with confusion_path.open(newline="", encoding="utf-8") as handle:
            confusion_rows = [
                row for row in csv.DictReader(handle)
                if row.get("model") != "persistence_last_observed"
            ]
        confusion_rows.extend(persistence_confusion)
        _write_dict_rows(confusion_path, confusion_rows, list(confusion_rows[0].keys()))

        metadata_path = output_dir / "experiment_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            flows = int(metadata["flows_per_scenario"])
            repetitions = int(metadata["test_repetitions_per_regime"])
            seed = int(metadata["seed"])
        else:
            flows = int(rows[0]["flows_offered"])
            repetitions = sum(1 for row in rows if str(row["regime"]) == str(rows[0]["regime"])) // len(POLICIES)
            metadata = {
                "seed": seed,
                "flows_per_scenario": flows,
                "training_scenarios": len({row["scenario_id"] for row in capacity_training_rows}),
                "test_repetitions_per_regime": repetitions,
                "test_regimes": sorted({str(row["regime"]) for row in rows}),
                "policies": sorted({str(row["policy"]) for row in rows}),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        capacity_models = {
            name: load_model(output_dir / f"capacity_model_{name}.joblib")
            for name in ("random_forest", "hist_gradient_boosting")
        }
        capacity_importance_rows = [
            {"feature": name, "importance": value}
            for name, value in feature_importances(capacity_models["random_forest"])
        ]
        _write_dict_rows(
            output_dir / "capacity_feature_importance_random_forest.csv",
            capacity_importance_rows,
            ["feature", "importance"],
        )
        test_capacity_rows: list[dict[str, Any]] = []
        for scenario_id, regime, scenario in _evaluation_scenarios(flows, repetitions, seed):
            _, _, examples = SchedulerSimulator(scenario).run("random", collect_capacity_rows=True)
            for metrics in evaluate_capacity_models(capacity_models, examples):
                test_capacity_rows.append({"scenario_id": scenario_id, "regime": regime, **metrics})
        test_metrics_path = output_dir / "capacity_forecast_test_results.csv"
        _write_dict_rows(test_metrics_path, test_capacity_rows, list(test_capacity_rows[0].keys()))
        test_summary = []
        for name in ("random_forest", "hist_gradient_boosting", "persistence_last_observed"):
            model_rows = [row for row in test_capacity_rows if row["model"] == name]
            test_summary.append({
                "model": name,
                "scenarios": len(model_rows),
                "mean_accuracy": mean(float(row["accuracy"]) for row in model_rows),
                "mean_balanced_accuracy": mean(float(row["balanced_accuracy"]) for row in model_rows),
                "mean_macro_f1": mean(float(row["macro_f1"]) for row in model_rows),
                "mean_multiplier_mae": mean(float(row["multiplier_mae"]) for row in model_rows),
            })
        _write_dict_rows(
            output_dir / "capacity_forecast_test_summary.csv",
            test_summary,
            list(test_summary[0].keys()),
        )

    print(f"Refreshed stratified summaries from {len(rows)} scenario-policy rows in {results_path}.")
