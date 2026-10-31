"""Training and persistence for candidate flow-latency models."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit

from .simulator import CAPACITY_MULTIPLIERS, FEATURE_NAMES


MODEL_NAMES = ("random_forest", "hist_gradient_boosting")
CAPACITY_MODEL_NAMES = MODEL_NAMES


@dataclass
class LatencyModel:
    name: str
    estimator: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict(self, feature_rows: Sequence[Sequence[float]]) -> np.ndarray:
        values = self.estimator.predict(np.asarray(feature_rows, dtype=float))
        return np.maximum(0.0, np.expm1(values))


@dataclass
class CapacityForecastModel:
    name: str
    estimator: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict_multiplier(self, feature_rows: Sequence[Sequence[float]]) -> np.ndarray:
        probabilities = self.estimator.predict_proba(np.asarray(feature_rows, dtype=float))
        state_multipliers = np.asarray(CAPACITY_MULTIPLIERS, dtype=float)
        class_multipliers = state_multipliers[self.estimator.classes_.astype(int)]
        return np.asarray(probabilities @ class_multipliers, dtype=float)


def _capacity_metrics_from_predictions(
    model_name: str,
    targets: np.ndarray,
    predicted_states: np.ndarray,
    expected_multipliers: np.ndarray,
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    state_multipliers = np.asarray(CAPACITY_MULTIPLIERS, dtype=float)
    actual_multipliers = state_multipliers[targets.astype(int)]
    confusion = confusion_matrix(targets, predicted_states, labels=np.arange(len(state_multipliers)))
    supported_classes = confusion.sum(axis=1) > 0
    per_class_recall = np.diag(confusion)[supported_classes] / confusion.sum(axis=1)[supported_classes]
    metrics: dict[str, float | str] = {
        "model": model_name,
        "accuracy": float(accuracy_score(targets, predicted_states)),
        "balanced_accuracy": float(np.mean(per_class_recall)) if len(per_class_recall) else 0.0,
        "macro_f1": float(f1_score(
            targets, predicted_states, labels=np.arange(len(state_multipliers)),
            average="macro", zero_division=0,
        )),
        "multiplier_mae": float(mean_absolute_error(actual_multipliers, expected_multipliers)),
        "rows": float(len(targets)),
    }
    confusion_rows: list[dict[str, float | str]] = []
    for actual_state, actual_multiplier in enumerate(state_multipliers):
        for predicted_state, predicted_multiplier in enumerate(state_multipliers):
            confusion_rows.append({
                "model": model_name,
                "actual_multiplier": float(actual_multiplier),
                "predicted_multiplier": float(predicted_multiplier),
                "count": float(confusion[actual_state, predicted_state]),
            })
    return metrics, confusion_rows


def _capacity_metrics(
    model_name: str, estimator: Any, features: np.ndarray, targets: np.ndarray
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    probabilities = estimator.predict_proba(features)
    classes = estimator.classes_.astype(int)
    predicted_states = classes[np.argmax(probabilities, axis=1)]
    state_multipliers = np.asarray(CAPACITY_MULTIPLIERS, dtype=float)
    expected_multipliers = probabilities @ state_multipliers[classes]
    return _capacity_metrics_from_predictions(
        model_name, targets, predicted_states, expected_multipliers
    )


def _persistence_metrics(features: np.ndarray, targets: np.ndarray) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    state_multipliers = np.asarray(CAPACITY_MULTIPLIERS, dtype=float)
    lag_index = FEATURE_NAMES.index("capacity_lag1_multiplier")
    last_observed = features[:, lag_index]
    predicted_states = np.abs(last_observed[:, None] - state_multipliers[None, :]).argmin(axis=1)
    expected_multipliers = state_multipliers[predicted_states]
    return _capacity_metrics_from_predictions(
        "persistence_last_observed", targets, predicted_states, expected_multipliers
    )


def _new_estimator(name: str, seed: int) -> Any:
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=160,
            max_features=0.9,
            min_samples_leaf=4,
            random_state=seed,
            # Prediction happens once per arriving flow. Parallel startup per
            # tiny three-path batch costs more than it saves, especially on Mac.
            n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def _new_capacity_estimator(name: str, seed: int) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_features=0.9,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"unknown capacity model: {name}")


def train_latency_models(
    rows: list[dict[str, Any]], seed: int
) -> tuple[dict[str, LatencyModel], list[dict[str, float | str]]]:
    if len(rows) < 100:
        raise ValueError(f"need at least 100 accepted training examples; got {len(rows)}")

    features = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    targets = np.asarray([row["target_latency_ms"] for row in rows], dtype=float)
    groups = np.asarray([row["scenario_id"] for row in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, validation_indices = next(splitter.split(features, targets, groups))
    validation_metrics: list[dict[str, float | str]] = []
    final_models: dict[str, LatencyModel] = {}

    for name in MODEL_NAMES:
        validation_estimator = _new_estimator(name, seed)
        validation_estimator.fit(features[train_indices], np.log1p(targets[train_indices]))
        predictions = np.maximum(
            0.0, np.expm1(validation_estimator.predict(features[validation_indices]))
        )
        validation_metrics.append({
            "model": name,
            "validation_mae_ms": float(mean_absolute_error(targets[validation_indices], predictions)),
            "validation_r2": float(r2_score(targets[validation_indices], predictions)),
            "training_rows": float(len(train_indices)),
            "validation_rows": float(len(validation_indices)),
            "training_scenarios": float(len(set(groups[train_indices]))),
            "validation_scenarios": float(len(set(groups[validation_indices]))),
        })

        final_estimator = _new_estimator(name, seed)
        final_estimator.fit(features, np.log1p(targets))
        final_models[name] = LatencyModel(name=name, estimator=final_estimator)

    return final_models, validation_metrics


def train_capacity_models(
    rows: list[dict[str, Any]], seed: int
) -> tuple[
    dict[str, CapacityForecastModel],
    list[dict[str, float | str]],
    list[dict[str, float | str]],
]:
    if len(rows) < 100:
        raise ValueError(f"need at least 100 candidate-path examples; got {len(rows)}")

    features = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    targets = np.asarray([row["target_capacity_state"] for row in rows], dtype=int)
    groups = np.asarray([row["scenario_id"] for row in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, validation_indices = next(splitter.split(features, targets, groups))
    validation_metrics: list[dict[str, float | str]] = []
    confusion_rows: list[dict[str, float | str]] = []
    final_models: dict[str, CapacityForecastModel] = {}

    for name in CAPACITY_MODEL_NAMES:
        validation_estimator = _new_capacity_estimator(name, seed)
        validation_estimator.fit(features[train_indices], targets[train_indices])
        metrics, confusion = _capacity_metrics(
            name, validation_estimator, features[validation_indices], targets[validation_indices]
        )
        metrics = {
            key: value if key == "model" else float(value)
            for key, value in metrics.items()
        }
        validation_metrics.append({
            **metrics,
            "training_rows": float(len(train_indices)),
            "validation_rows": float(len(validation_indices)),
            "training_scenarios": float(len(set(groups[train_indices]))),
            "validation_scenarios": float(len(set(groups[validation_indices]))),
        })
        confusion_rows.extend(confusion)

        final_estimator = _new_capacity_estimator(name, seed)
        final_estimator.fit(features, targets)
        final_models[name] = CapacityForecastModel(name=name, estimator=final_estimator)

    persistence, persistence_confusion = _persistence_metrics(
        features[validation_indices], targets[validation_indices]
    )
    validation_metrics.append({
        **persistence,
        "training_rows": float(len(train_indices)),
        "validation_rows": float(len(validation_indices)),
        "training_scenarios": float(len(set(groups[train_indices]))),
        "validation_scenarios": float(len(set(groups[validation_indices]))),
    })
    confusion_rows.extend(persistence_confusion)

    return final_models, validation_metrics, confusion_rows


def evaluate_capacity_models(
    models: dict[str, CapacityForecastModel], rows: list[dict[str, Any]]
) -> list[dict[str, float | str]]:
    features = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    targets = np.asarray([row["target_capacity_state"] for row in rows], dtype=int)
    metrics_rows = []
    for name, model in models.items():
        metrics, _ = _capacity_metrics(name, model.estimator, features, targets)
        metrics_rows.append(metrics)
    persistence, _ = _persistence_metrics(features, targets)
    metrics_rows.append(persistence)
    return metrics_rows


def evaluate_persistence_validation(
    rows: list[dict[str, Any]], seed: int
) -> tuple[dict[str, float | str], list[dict[str, float | str]]]:
    features = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    targets = np.asarray([row["target_capacity_state"] for row in rows], dtype=int)
    groups = np.asarray([row["scenario_id"] for row in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, validation_indices = next(splitter.split(features, targets, groups))
    metrics, confusion = _persistence_metrics(features[validation_indices], targets[validation_indices])
    metrics.update({
        "training_rows": float(len(train_indices)),
        "validation_rows": float(len(validation_indices)),
        "training_scenarios": float(len(set(groups[train_indices]))),
        "validation_scenarios": float(len(set(groups[validation_indices]))),
    })
    return metrics, confusion


def save_model(model: LatencyModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    loaded = joblib.load(path)
    if not isinstance(loaded, (LatencyModel, CapacityForecastModel)):
        raise TypeError(f"model at {path} is not a traffic-scheduler model")
    return loaded


def feature_importances(model: Any) -> list[tuple[str, float]]:
    if not hasattr(model.estimator, "feature_importances_"):
        return []
    pairs = zip(model.feature_names, model.estimator.feature_importances_)
    return sorted(((name, float(value)) for name, value in pairs), key=lambda item: item[1], reverse=True)
