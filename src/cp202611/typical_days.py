from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class TypicalDayResult:
    selected_days: list[int]
    weights: dict[int, int]
    features: pd.DataFrame
    metrics: dict[str, float]


def build_daily_features(time_series: pd.DataFrame) -> pd.DataFrame:
    """Build daily feature vectors for representative-day selection."""
    frame = time_series.copy()
    frame["day"] = frame["hour"] // 24
    daily = frame.groupby("day").agg(
        outdoor_mean_c=("outdoor_temperature_c", "mean"),
        outdoor_min_c=("outdoor_temperature_c", "min"),
        price_mean=("electricity_price_multiplier", "mean"),
        price_max=("electricity_price_multiplier", "max"),
        carbon_mean=("grid_carbon_factor_t_per_mwh", "mean"),
        grid_limit_min_mw=("grid_import_limit_mw", "min"),
    )
    return daily.reset_index()


def select_peak_preserving_kmedoids(
    time_series: pd.DataFrame,
    n_typical_days: int,
    feature_weights: dict[str, float] | None = None,
) -> TypicalDayResult:
    """Select real days using weighted features and nearest-to-centroid medoids.

    This is a pragmatic K-medoids variant: K-means is used only to form weighted
    clusters; the final representative is the real day nearest to each centroid.
    Extreme cold, peak price, and tight grid-limit days are forced into the set.
    """
    if n_typical_days < 3:
        raise ValueError("n_typical_days must be at least 3 to preserve cold, price, and grid-limit extremes")
    features = build_daily_features(time_series)
    if len(features) <= n_typical_days:
        selected = features["day"].astype(int).tolist()
        return TypicalDayResult(
            selected_days=selected,
            weights={d: 1 for d in selected},
            features=features,
            metrics=_quality_metrics(time_series, selected),
        )

    extreme_days = {
        int(features.loc[features["outdoor_min_c"].idxmin(), "day"]),
        int(features.loc[features["price_max"].idxmax(), "day"]),
        int(features.loc[features["grid_limit_min_mw"].idxmin(), "day"]),
    }
    remaining_slots = max(0, n_typical_days - len(extreme_days))

    feature_cols = [c for c in features.columns if c != "day"]
    weights = feature_weights or {
        "outdoor_mean_c": 1.0,
        "outdoor_min_c": 2.0,
        "price_mean": 0.7,
        "price_max": 0.7,
        "carbon_mean": 0.4,
        "grid_limit_min_mw": 1.0,
    }
    x = features[feature_cols].astype(float).to_numpy()
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    weight_vec = np.array([weights.get(col, 1.0) for col in feature_cols], dtype=float)
    x_scaled = ((x - mean) / std) * weight_vec

    candidate_mask = ~features["day"].isin(extreme_days)
    candidates = features[candidate_mask].reset_index(drop=True)
    x_candidates = x_scaled[candidate_mask.to_numpy()]
    selected = set(extreme_days)

    if remaining_slots > 0 and len(candidates) > 0:
        k = min(remaining_slots, len(candidates))
        km = KMeans(n_clusters=k, random_state=202611, n_init=20)
        labels = km.fit_predict(x_candidates)
        for cluster in range(k):
            idxs = np.where(labels == cluster)[0]
            centroid = km.cluster_centers_[cluster]
            medoid_local = idxs[np.argmin(np.linalg.norm(x_candidates[idxs] - centroid, axis=1))]
            selected.add(int(candidates.loc[medoid_local, "day"]))

    selected_sorted = sorted(selected)
    return rebuild_typical_day_result(time_series, selected_sorted, feature_weights)


def rebuild_typical_day_result(
    time_series: pd.DataFrame,
    selected_days: list[int],
    feature_weights: dict[str, float] | None = None,
) -> TypicalDayResult:
    """Recompute nearest-medoid weights for an explicit set of selected real days."""
    features = build_daily_features(time_series)
    available_days = set(features["day"].astype(int).tolist())
    missing = sorted(set(selected_days) - available_days)
    if missing:
        raise ValueError(f"selected days are outside the scenario horizon: {missing}")

    selected_sorted = sorted(set(int(day) for day in selected_days))
    feature_cols = [c for c in features.columns if c != "day"]
    weights = feature_weights or {
        "outdoor_mean_c": 1.0,
        "outdoor_min_c": 2.0,
        "price_mean": 0.7,
        "price_max": 0.7,
        "carbon_mean": 0.4,
        "grid_limit_min_mw": 1.0,
    }
    x = features[feature_cols].astype(float).to_numpy()
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    weight_vec = np.array([weights.get(col, 1.0) for col in feature_cols], dtype=float)
    x_scaled = ((x - mean) / std) * weight_vec
    assignment_days = _assign_weights_to_selected(features, x_scaled, selected_sorted)
    return TypicalDayResult(
        selected_days=selected_sorted,
        weights=assignment_days,
        features=features,
        metrics=_quality_metrics(time_series, selected_sorted, assignment_days),
    )


def _assign_weights_to_selected(features: pd.DataFrame, x_scaled: np.ndarray, selected_days: list[int]) -> dict[int, int]:
    selected_idx = [int(features.index[features["day"] == d][0]) for d in selected_days]
    weights = {d: 1 for d in selected_days}
    for i in range(len(features)):
        current_day = int(features.loc[i, "day"])
        if current_day in weights:
            continue
        nearest = min(selected_idx, key=lambda j: float(np.linalg.norm(x_scaled[i] - x_scaled[j])))
        selected_day = int(features.loc[nearest, "day"])
        weights[selected_day] += 1
    return weights


def _quality_metrics(
    time_series: pd.DataFrame,
    selected_days: list[int],
    weights: dict[int, int] | None = None,
) -> dict[str, float]:
    frame = time_series.copy()
    frame["day"] = frame["hour"] // 24
    selected = frame[frame["day"].isin(selected_days)]
    if weights:
        weighted = []
        for day in selected_days:
            day_frame = selected[selected["day"] == day].copy()
            for _ in range(weights.get(day, 0)):
                weighted.append(day_frame)
        representative = pd.concat(weighted, ignore_index=True) if weighted else selected
    else:
        representative = selected

    return {
        "selected_day_count": float(len(selected_days)),
        "full_min_temperature_c": float(frame["outdoor_temperature_c"].min()),
        "selected_min_temperature_c": float(selected["outdoor_temperature_c"].min()),
        "full_max_price_multiplier": float(frame["electricity_price_multiplier"].max()),
        "selected_max_price_multiplier": float(selected["electricity_price_multiplier"].max()),
        "mean_temperature_error_c": float(
            abs(frame["outdoor_temperature_c"].mean() - representative["outdoor_temperature_c"].mean())
        ),
        "mean_price_error": float(
            abs(frame["electricity_price_multiplier"].mean() - representative["electricity_price_multiplier"].mean())
        ),
    }
