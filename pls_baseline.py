"""Jointly optimize spectral preprocessing and PLS latent variables.

The fixed test split is never used for model selection. For every dataset,
five-fold cross-validation on the stored training split evaluates the same nine
preprocessing options used by ``temp/pls_baseline2.py`` and 1--20 latent
variables. Preprocessing that learns a reference (MSC) is fitted independently
inside each CV training fold. The selected preprocessing and LV count are then
fitted once on the full training split and evaluated on the held-out test split.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.data_loading import discover_datasets, load_dataset


CV_FOLDS = 5
CV_RANDOM_STATE = 12345
MAX_LV = 20
PREPROCESSING_VARIANTS = (
    "RAW",
    "SNV",
    "MSC",
    "SG_1st_der_w9",
    "SG_2nd_der_w9",
    "SNV_then_SG_1st_der_w9",
    "MSC_then_SG_1st_der_w9",
    "SNV_then_SG_2nd_der_w9",
    "MSC_then_SG_2nd_der_w9",
)


def _dataset_sort_key(name: str):
    try:
        prefix = int(str(name).split("-", 1)[0])
    except Exception:
        prefix = 10**9
    return prefix, str(name)


def _resolve_datasets(datasets_root: Path, dataset_filter: list[str] | None = None):
    dataset_config = discover_datasets(datasets_root)
    dataset_names = sorted(dataset_config, key=_dataset_sort_key)
    if not dataset_names:
        raise RuntimeError(f"No datasets found under {datasets_root}")
    if dataset_filter is not None:
        unknown = set(dataset_filter) - set(dataset_names)
        if unknown:
            raise ValueError(
                f"Unknown dataset(s): {sorted(unknown)}.\nAvailable: {dataset_names}"
            )
        dataset_names = [name for name in dataset_names if name in dataset_filter]
    return dataset_config, dataset_names


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.ravel(y_true), np.ravel(y_pred))))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(np.ravel(y_true), np.ravel(y_pred)))


def snv_transform(X: np.ndarray) -> np.ndarray:
    """Apply standard normal variate independently to each spectrum."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def msc_transform(X_reference: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply multiplicative scatter correction using a training-only reference."""
    reference = X_reference.mean(axis=0)
    reference_mean = float(reference.mean())
    centered_reference = reference - reference_mean
    denominator = float(centered_reference @ centered_reference)
    if denominator <= np.finfo(float).eps:
        return X.copy()
    spectrum_means = X.mean(axis=1)
    slopes = ((X - spectrum_means[:, None]) @ centered_reference) / denominator
    tiny = np.abs(slopes) <= np.finfo(float).eps
    slopes[tiny] = np.where(slopes[tiny] < 0, -1e-8, 1e-8)
    intercepts = spectrum_means - slopes * reference_mean
    return (X - intercepts[:, None]) / slopes[:, None]


def _savgol_with_fallback(
    X: np.ndarray,
    *,
    derivative: int,
    base_window: int = 9,
    base_polyorder: int = 2,
) -> np.ndarray:
    """Apply the reference script's window-9 Savitzky--Golay derivative safely."""
    n_features = int(X.shape[1])
    if n_features <= derivative:
        raise ValueError(
            f"Savitzky-Golay derivative {derivative} needs more than "
            f"{derivative} spectral features; got {n_features}."
        )
    window = min(base_window, n_features)
    if window % 2 == 0:
        window -= 1
    polyorder = max(derivative, min(base_polyorder, window - 1))
    if window <= polyorder:
        candidates = [w for w in range(polyorder + 1, n_features + 1) if w % 2 == 1]
        if not candidates:
            raise ValueError(
                f"No valid Savitzky-Golay window for {n_features} features."
            )
        window = candidates[0]
    return savgol_filter(
        X,
        window_length=window,
        polyorder=polyorder,
        deriv=derivative,
        axis=1,
    )


def apply_preprocessing(
    preprocessing: str,
    X_fit: np.ndarray,
    X_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a preprocessing on ``X_fit`` and transform both supplied arrays."""
    if preprocessing == "RAW":
        return X_fit.copy(), X_other.copy()
    if preprocessing == "SNV":
        return snv_transform(X_fit), snv_transform(X_other)
    if preprocessing == "MSC":
        return msc_transform(X_fit, X_fit), msc_transform(X_fit, X_other)

    scatter, derivative = (
        preprocessing.split("_then_")
        if "_then_" in preprocessing
        else (None, preprocessing)
    )
    if scatter == "SNV":
        X_fit_processed, X_other_processed = snv_transform(X_fit), snv_transform(X_other)
    elif scatter == "MSC":
        X_fit_processed = msc_transform(X_fit, X_fit)
        X_other_processed = msc_transform(X_fit, X_other)
    elif scatter is None:
        X_fit_processed, X_other_processed = X_fit, X_other
    else:
        raise ValueError(f"Unsupported scatter correction: {scatter}")

    derivative_order = {"SG_1st_der_w9": 1, "SG_2nd_der_w9": 2}.get(derivative)
    if derivative_order is None:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")
    return (
        _savgol_with_fallback(X_fit_processed, derivative=derivative_order),
        _savgol_with_fallback(X_other_processed, derivative=derivative_order),
    )


def _cv_splits(n_samples: int, n_splits: int, random_state: int):
    if n_samples < 2:
        raise ValueError("Need at least two training samples for CV.")
    n_splits = min(int(n_splits), n_samples)
    if n_splits < 2:
        raise ValueError("Need at least two CV folds.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.arange(n_samples)))


def evaluate_preprocessing_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    preprocessing: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    max_lv: int,
) -> list[dict]:
    """Return pooled out-of-fold metrics for every valid LV count."""
    prepared_folds = []
    max_candidate = min(int(max_lv), int(X_train.shape[1]))
    for train_idx, val_idx in splits:
        X_fold_train, X_fold_val = apply_preprocessing(
            preprocessing,
            X_train[train_idx],
            X_train[val_idx],
        )
        max_candidate = min(
            max_candidate,
            int(X_fold_train.shape[1]),
            max(1, len(train_idx) - 1),
        )
        prepared_folds.append((train_idx, val_idx, X_fold_train, X_fold_val))
    if max_candidate < 1:
        raise ValueError("No valid latent-variable count for this dataset.")

    rows = []
    for lv in range(1, max_candidate + 1):
        predictions = np.empty(len(y_train), dtype=np.float64)
        for train_idx, val_idx, X_fold_train, X_fold_val in prepared_folds:
            # Internal scaling is fitted using this fold's training rows only.
            model = PLSRegression(n_components=lv, scale=True)
            model.fit(X_fold_train, y_train[train_idx])
            predictions[val_idx] = np.asarray(model.predict(X_fold_val)).reshape(-1)
        rows.append({
            "preprocessing": preprocessing,
            "LV": lv,
            "RMSE-CV": _rmse(y_train, predictions),
            "R2-CV": _r2(y_train, predictions),
        })
    return rows


def select_preprocessing_and_lv_via_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_splits: int = CV_FOLDS,
    random_state: int = CV_RANDOM_STATE,
    max_lv: int = MAX_LV,
) -> tuple[dict, list[dict]]:
    """Jointly choose preprocessing and LV count by training-only CV RMSE."""
    splits = _cv_splits(len(y_train), n_splits, random_state)
    search_rows = []
    for preprocessing in PREPROCESSING_VARIANTS:
        candidates = evaluate_preprocessing_cv(
            X_train,
            y_train,
            preprocessing=preprocessing,
            splits=splits,
            max_lv=max_lv,
        )
        search_rows.extend(candidates)
        best_for_preprocessing = min(candidates, key=lambda row: (row["RMSE-CV"], row["LV"]))
        print(
            f"  {preprocessing:<28} best LV={best_for_preprocessing['LV']:>2} | "
            f"RMSE-CV={best_for_preprocessing['RMSE-CV']:.4f}"
        )

    preprocessing_order = {name: index for index, name in enumerate(PREPROCESSING_VARIANTS)}
    best = min(
        search_rows,
        key=lambda row: (
            row["RMSE-CV"],
            preprocessing_order[row["preprocessing"]],
            row["LV"],
        ),
    )
    return best, search_rows


def fit_evaluate_pls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    preprocessing: str,
    lv: int,
) -> dict:
    X_train_processed, X_test_processed = apply_preprocessing(
        preprocessing, X_train, X_test
    )
    components = max(
        1,
        min(int(lv), int(X_train_processed.shape[1]), max(1, len(X_train_processed) - 1)),
    )
    model = PLSRegression(n_components=components, scale=True)
    model.fit(X_train_processed, y_train)
    train_predictions = np.asarray(model.predict(X_train_processed)).reshape(-1)
    test_predictions = np.asarray(model.predict(X_test_processed)).reshape(-1)
    return {
        "Train RMSE": _rmse(y_train, train_predictions),
        "Train R2": _r2(y_train, train_predictions),
        "RMSE test": _rmse(y_test, test_predictions),
        "R2 test": _r2(y_test, test_predictions),
    }


def _output_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint preprocessing and latent-variable optimization for PLS."
    )
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="Dataset name(s) to benchmark. Default: all datasets.",
    )
    parser.add_argument("--max-lv", type=int, default=MAX_LV)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--output", default="pls_baseline_benchmark.cvs")
    parser.add_argument(
        "--search-output",
        default="pls_preprocessing_search.csv",
        help="Detailed CV results for every preprocessing/LV candidate.",
    )
    args = parser.parse_args()
    if args.max_lv < 1:
        parser.error("--max-lv must be at least 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    repo_root = Path(__file__).resolve().parent
    datasets_root = repo_root / "datasets"
    dataset_config, dataset_names = _resolve_datasets(datasets_root, args.dataset)
    print(f"Datasets to run ({len(dataset_names)}): {dataset_names}")
    print(f"Preprocessing variants ({len(PREPROCESSING_VARIANTS)}): {PREPROCESSING_VARIANTS}")

    summary_rows: list[dict] = []
    detailed_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        X_train, y_train, X_test, y_test, _ = load_dataset(
            dataset_name,
            dataset_config=dataset_config,
            datasets_root=datasets_root,
        )
        X_train = np.asarray(X_train, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)

        optimization_started = time.perf_counter()
        best, search_rows = select_preprocessing_and_lv_via_cv(
            X_train,
            y_train,
            n_splits=args.cv_folds,
            random_state=CV_RANDOM_STATE,
            max_lv=args.max_lv,
        )
        optimization_time_s = time.perf_counter() - optimization_started
        detailed_rows.extend({"dataset": dataset_name, **row} for row in search_rows)

        final_fit_started = time.perf_counter()
        final_metrics = fit_evaluate_pls(
            X_train,
            y_train,
            X_test,
            y_test,
            preprocessing=str(best["preprocessing"]),
            lv=int(best["LV"]),
        )
        final_fit_time_s = time.perf_counter() - final_fit_started
        summary_rows.append({
            "dataset": dataset_name,
            "preprocessing": best["preprocessing"],
            "LV": int(best["LV"]),
            "RMSE-CV": float(best["RMSE-CV"]),
            "R2-CV": float(best["R2-CV"]),
            **final_metrics,
            "optimization_time_s": float(optimization_time_s),
            "final_fit_time_s": float(final_fit_time_s),
            "total_time_s": float(optimization_time_s + final_fit_time_s),
        })
        print(
            f"Selected {best['preprocessing']} with LV={best['LV']} | "
            f"RMSE-CV={best['RMSE-CV']:.4f} | "
            f"RMSE test={final_metrics['RMSE test']:.4f} | "
            f"R2 test={final_metrics['R2 test']:.4f} | "
            f"optimization={optimization_time_s:.2f}s | final fit={final_fit_time_s:.2f}s"
        )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detailed_rows)
    if not summary.empty:
        summary = summary.sort_values(
            "dataset", key=lambda column: column.map(_dataset_sort_key)
        ).reset_index(drop=True)
    if not details.empty:
        details = details.assign(
            _dataset_order=details["dataset"].map(lambda name: _dataset_sort_key(name)[0]),
            _preprocessing_order=details["preprocessing"].map(
                {name: index for index, name in enumerate(PREPROCESSING_VARIANTS)}
            ),
        ).sort_values(["_dataset_order", "_preprocessing_order", "LV"]).drop(
            columns=["_dataset_order", "_preprocessing_order"]
        ).reset_index(drop=True)

    output_path = _output_path(repo_root, args.output)
    search_output_path = _output_path(repo_root, args.search_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    search_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    details.to_csv(search_output_path, index=False)
    print("\n" + "=" * 80)
    print(f"Saved optimized PLS summary: {output_path}")
    print(f"Saved full CV search: {search_output_path}")


if __name__ == "__main__":
    main()
