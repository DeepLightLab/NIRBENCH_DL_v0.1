"""
Data loading utilities for the NIR benchmark datasets.

The loader is intentionally defensive: by default it tries to infer which
columns are spectral features (numeric wavelength names) and which are
targets (non-numeric column names or a user-specified override). Each dataset
folder contains one or more train/test CSV files; discover_datasets() builds a
configuration dictionary that can be customised if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Default root assumes this repository layout: repo_root/datasets
DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"


@dataclass
class DatasetFiles:
    train_files: List[Path]
    test_files: List[Path]
    target_columns: Optional[Sequence[str]] = None
    input_type: str = "csv"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["train_files"] = [str(p) for p in self.train_files]
        data["test_files"] = [str(p) for p in self.test_files]
        return data


def _is_numeric_column(name: str) -> bool:
    try:
        float(name)
        return True
    except (TypeError, ValueError):
        return False


def discover_datasets(datasets_root: Path = DATASETS_ROOT) -> Dict[str, DatasetFiles]:
    """
    Inspect the datasets root and return a mapping from dataset name to files.
    The heuristic looks for CSV files matching data_train*.csv and data_test*.csv.
    """
    # Walk the datasets directory and collect train/test CSVs per dataset.
    datasets: Dict[str, DatasetFiles] = {}
    for folder in sorted(datasets_root.iterdir()):
        if not folder.is_dir():
            continue
        train_files = sorted(folder.glob("data_train*.csv"))
        test_files = sorted(folder.glob("data_test*.csv"))
        if not train_files or not test_files:
            continue
        datasets[folder.name] = DatasetFiles(
            train_files=train_files, test_files=test_files
        )
    return datasets


def save_dataset_config(config: Dict[str, DatasetFiles], path: Path) -> None:
    payload = {name: entry.to_dict() for name, entry in config.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _split_features_and_targets(
    df: pd.DataFrame,
    target_columns: Optional[Sequence[str]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Identify spectral feature columns and target columns and return numpy arrays.
    Assumptions for the unified datasets:
    - An optional leading ID column may exist (often non-numeric) and is discarded.
    - Last column holds the target (unless target_columns overrides).
    - Middle columns are spectral features (wavelengths as column names).
    """
    df = df.copy()
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Drop first column when it is an ID-like column (non-numeric name).
    # Many datasets already store the sample index as "Unnamed: 0", which is
    # removed above; in those cases the first remaining column is a wavelength
    # feature and must be kept.
    if len(df.columns) >= 2 and not _is_numeric_column(df.columns[0]):
        df = df.iloc[:, 1:]

    if target_columns:
        targets = list(target_columns)
        feature_cols_current = [c for c in df.columns if c not in targets]
    else:
        targets = [df.columns[-1]]
        feature_cols_current = list(df.columns[:-1])

    feature_vals_current: List[Optional[float]] = []
    for idx, c in enumerate(feature_cols_current):
        try:
            feature_vals_current.append(float(c))
        except (TypeError, ValueError):
            feature_vals_current.append(None)

    if feature_order is not None:
        selected_cols: List[str] = []
        selected_vals: List[float] = []
        tol = 1e-6
        for target_val in feature_order:
            match_idx = None
            for idx, val in enumerate(feature_vals_current):
                if val is None:
                    continue
                if abs(val - float(target_val)) <= tol * max(1.0, abs(float(target_val))):
                    match_idx = idx
                    break
            if match_idx is None:
                raise ValueError(
                    f"Missing expected feature column ~ {target_val} (tolerance {tol}). "
                    f"Available sample: {feature_cols_current[:5]}..."
                )
            selected_cols.append(feature_cols_current[match_idx])
            selected_vals.append(float(feature_vals_current[match_idx]))
        feature_cols = selected_cols
        wavelengths = np.array(selected_vals, dtype=float)
        feature_vals = selected_vals
    else:
        feature_cols = feature_cols_current
        wavelengths_vals = []
        for idx, val in enumerate(feature_vals_current):
            wavelengths_vals.append(val if val is not None else float(idx))
        wavelengths = np.array(wavelengths_vals, dtype=float)
        feature_vals = wavelengths.tolist()

    X = df[feature_cols].to_numpy(dtype=float)

    targets_df = df[targets].copy()
    for col in targets_df.columns:
        try:
            targets_df[col] = pd.to_numeric(targets_df[col])
        except ValueError:
            targets_df[col], _ = pd.factorize(targets_df[col])
    y = targets_df.to_numpy(dtype=float)
    return X, y, wavelengths, feature_vals


def load_file(
    path: Path,
    target_columns: Optional[Sequence[str]] = None,
    feature_order: Optional[Sequence[str]] = None,
):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        # Fallback for files with non-UTF8 encoding
        df = pd.read_csv(path, encoding="latin1")
    return _split_features_and_targets(df, target_columns, feature_order)


def build_global_wavelength_grid(
    dataset_names: Iterable[str],
    dataset_config: Optional[Dict[str, DatasetFiles]] = None,
    datasets_root: Path = DATASETS_ROOT,
    step_nm: float = 2.0,
) -> np.ndarray:
    # Inspect wavelength ranges across datasets and build a common grid.
    mins, maxs = [], []
    datasets = dataset_config or discover_datasets(datasets_root)
    for name in dataset_names:
        entry = datasets[name]
        # pick first file to read wavelength vector
        sample_file = entry.train_files[0]
        _, _, wl, _ = load_file(sample_file, entry.target_columns)
        if wl is None or len(wl) == 0:
            continue
        mins.append(wl.min())
        maxs.append(wl.max())
    if not mins:
        raise ValueError("No wavelength information available.")
    start = min(mins)
    end = max(maxs)
    return np.arange(start, end + step_nm, step_nm)


def interpolate_to_global_grid(
    X: np.ndarray, wavelengths: np.ndarray, global_grid: np.ndarray
) -> np.ndarray:
    # Linear interpolation of spectra onto the shared wavelength grid.
    f = interp1d(wavelengths, X, kind="linear", axis=1, fill_value="extrapolate")
    return f(global_grid)


def load_dataset(
    dataset_name: str,
    dataset_config: Optional[Dict[str, DatasetFiles]] = None,
    datasets_root: Path = DATASETS_ROOT,
    global_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load train/test splits for a dataset.

    Returns
    -------
    X_train, y_train, X_test, y_test, wavelengths
    """
    config = dataset_config or discover_datasets(datasets_root)
    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in config.")

    entry = config[dataset_name]
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    wavelengths = None
    feature_order: Optional[List[str]] = None

    for path in entry.train_files:
        # Load and optionally interpolate each train split.
        X, y, wl, cols = load_file(path, entry.target_columns, feature_order)
        if feature_order is None:
            feature_order = cols
        if global_grid is not None and wl is not None:
            X = interpolate_to_global_grid(X, wl, global_grid)
            wl = global_grid
        X_train_list.append(X)
        y_train_list.append(y)
        wavelengths = wl

    for path in entry.test_files:
        # Load and optionally interpolate each test split.
        X, y, wl, _ = load_file(path, entry.target_columns, feature_order)
        if global_grid is not None and wl is not None:
            X = interpolate_to_global_grid(X, wl, global_grid)
        X_test_list.append(X)
        y_test_list.append(y)

    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.vstack(y_train_list)
    y_test = np.vstack(y_test_list)

    def _clean_nan_rows(X: np.ndarray, y: np.ndarray):
        # Drop rows that contain NaNs in either features or targets.
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y).any(axis=1))
        return X[mask], y[mask]

    X_train, y_train = _clean_nan_rows(X_train, y_train)
    X_test, y_test = _clean_nan_rows(X_test, y_test)

    # Simplify to 1D targets when possible
    if y_train.shape[1] == 1:
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

    return X_train, y_train, X_test, y_test, wavelengths


def list_dataset_names(datasets_root: Path = DATASETS_ROOT) -> List[str]:
    datasets = discover_datasets(datasets_root)

    def _train_size(entry: DatasetFiles) -> int:
        total = 0
        for p in entry.train_files:
            try:
                total += int(pd.read_csv(p, usecols=[0]).shape[0])
            except Exception:
                total += 0
        return total

    ordered = sorted(datasets.items(), key=lambda kv: _train_size(kv[1]))
    return [name for name, _ in ordered]

# Minimal usage example:
# from src.data_loading import list_dataset_names, load_dataset
# names = list_dataset_names()
# X_train, y_train, X_test, y_test, wavelengths = load_dataset(names[0])
