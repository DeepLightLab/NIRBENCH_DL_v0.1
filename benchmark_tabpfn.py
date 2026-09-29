"""Benchmark the pretrained TabPFN 3.5 regressor on the 30 NIRBENCH datasets.

TabPFN is pretrained, so no epoch selection or cross-validation is needed.
Ten fits use the full training split and distinct seeds; the held-out test
split is used only for evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import tempfile
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nirbench_tabpfn_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.model_loading import ModelSource, get_cache_dir
from tqdm.auto import tqdm

from src.data_loading import discover_datasets, load_dataset


ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = ROOT / "datasets"
RESULTS_ROOT = ROOT / "results" / "TabPFN35"
BASE_SEED = 12345
DEFAULT_RUNS = 10
SUMMARY_COLUMNS = [
    "dataset_name",
    "train_rmse_mean", "train_rmse_std", "train_R2_mean", "train_R2_std",
    "test_rmse_mean", "test_rmse_std", "test_R2_mean", "test_R2_std",
    "avg_wall_time_s",
]


def dataset_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("-", 1)[0]), name
    except ValueError:
        return 10**9, name


def checkpoint_path(user_path: Path | None) -> Path:
    path = user_path if user_path is not None else (
        get_cache_dir() / ModelSource.get_v3_5().default_filename
    )
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"TabPFN 3.5 checkpoint is missing: {path}\n"
            "Pass --model-path /path/to/tabpfn-v3.5-*.safetensors. "
            "The Python package alone does not include the weights."
        )
    if "v3.5" not in path.name or "fast" in path.name or path.suffix != ".safetensors":
        raise ValueError(f"Expected the standard TabPFN 3.5 safetensors checkpoint, got: {path}")
    return path


def scale_pair(
    X_fit: np.ndarray, X_query: np.ndarray, y_fit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit both scalers on the current training partition only."""
    x_scaler = StandardScaler().fit(X_fit)
    y_scaler = StandardScaler().fit(y_fit.reshape(-1, 1))
    return (
        x_scaler.transform(X_fit).astype(np.float32),
        x_scaler.transform(X_query).astype(np.float32),
        y_scaler.transform(y_fit.reshape(-1, 1)).ravel().astype(np.float32),
        y_scaler,
    )


def make_regressor(args: argparse.Namespace, model_path: Path, seed: int) -> TabPFNRegressor:
    # The explicit version prevents a future package default from silently
    # changing the checkpoint used by this benchmark.
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5,
        model_path=str(model_path),
        device=args.device,
        n_estimators=args.n_estimators,
        fit_mode=args.fit_mode,
        random_state=seed,
    )


def predict_original(model: TabPFNRegressor, X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    scaled = np.asarray(model.predict(X, output_type="mean"), dtype=np.float64).reshape(-1, 1)
    pred = scaler.inverse_transform(scaled).ravel()
    if not np.isfinite(pred).all():
        raise ValueError("TabPFN returned non-finite predictions")
    return pred


def scores(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    return float(np.sqrt(mean_squared_error(y, pred))), float(r2_score(y, pred))


def release_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_dataset(
    name: str,
    config: dict,
    args: argparse.Namespace,
    model_path: Path,
) -> dict:
    X_train, y_train, X_test, y_test, _ = load_dataset(
        name, dataset_config=config, datasets_root=DATASETS_ROOT
    )
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)
    if len(y_train) < 2 or len(y_test) < 2:
        raise ValueError(f"{name}: insufficient samples for fitting or test R2")
    if not all(np.isfinite(a).all() for a in (X_train, X_test, y_train, y_test)):
        raise ValueError(f"{name}: non-finite values in loaded data")

    tqdm.write(f"{name}: train={X_train.shape}, test={X_test.shape}")
    X_fit, X_query, y_fit, y_scaler = scale_pair(X_train, X_test, y_train)
    run_rows = []
    last_pred = None
    with tqdm(total=args.runs, desc=f"Runs: {name}", unit="run", leave=False, position=1) as run_bar:
        for run in range(args.runs):
            seed = BASE_SEED + run
            model = make_regressor(args, model_path, seed)
            start = time.perf_counter()
            model.fit(X_fit, y_fit)
            pred = predict_original(model, X_query, y_scaler)
            elapsed = time.perf_counter() - start
            rmse, r2 = scores(y_test, pred)
            run_rows.append({
                "run": run + 1, "seed": seed,
                "test_rmse": rmse, "test_R2": r2, "wall_time_s": elapsed,
            })
            last_pred = pred
            run_bar.set_postfix_str(f"RMSE={rmse:.4f}, R2={r2:.4f}", refresh=False)
            run_bar.update(1)
            del model
            release_model()

    runs_df = pd.DataFrame(run_rows)
    # Predicting rows that are themselves in TabPFN's context would leak their
    # labels. Leave the DL-style in-sample train metric columns empty.
    row = dict.fromkeys(SUMMARY_COLUMNS, np.nan)
    row.update({
        "dataset_name": name,
        "test_rmse_mean": float(runs_df.test_rmse.mean()),
        "test_rmse_std": float(runs_df.test_rmse.std(ddof=1)),
        "test_R2_mean": float(runs_df.test_R2.mean()),
        "test_R2_std": float(runs_df.test_R2.std(ddof=1)),
        "avg_wall_time_s": float(runs_df.wall_time_s.mean()),
    })

    stem = RESULTS_ROOT / name
    runs_df.to_csv(f"{stem}_runs.csv", index=False)
    pd.DataFrame({"split": "test", "y_true": y_test, "y_pred": last_pred}).to_csv(
        f"{stem}_preds.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(np.min(y_test), np.min(last_pred)))
    hi = float(max(np.max(y_test), np.max(last_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.scatter(y_test, last_pred, facecolors="none", edgecolors="tab:red", s=35)
    ax.set(xlabel="Measured", ylabel="Predicted", title=f"TabPFN35 - {name}")
    ax.text(0.03, 0.97, f"Test R² = {run_rows[-1]['test_R2']:.3f}\n"
            f"Test RMSE = {run_rows[-1]['test_rmse']:.3f}",
            transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(f"{stem}_plot.png", dpi=150)
    plt.close(fig)
    pd.DataFrame([row]).to_csv(f"{stem}_metrics.csv", index=False)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", help="Dataset folder name(s); default: all 30")
    parser.add_argument("--model-path", type=Path, help="Local TabPFN 3.5 .safetensors checkpoint")
    parser.add_argument("--device", default="auto", help="TabPFN device (default: auto)")
    parser.add_argument("--n-estimators", default="auto", type=lambda s: s if s == "auto" else int(s),
                        help="TabPFN ensemble size (default: auto)")
    parser.add_argument("--fit-mode", default="fit_preprocessors",
                        choices=["fit_preprocessors", "low_memory", "fit_with_cache", "batched"])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="Final full-training fits with distinct seeds (default: 10)")
    parser.add_argument("--overwrite", action="store_true", help="Rerun selected datasets")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 to report a run-to-run standard deviation")
    if isinstance(args.n_estimators, int) and args.n_estimators < 1:
        parser.error("--n-estimators must be positive")

    config = discover_datasets(DATASETS_ROOT)
    names = sorted(config, key=dataset_sort_key)
    if args.dataset:
        unknown = set(args.dataset) - set(config)
        if unknown:
            parser.error(f"Unknown dataset(s): {sorted(unknown)}")
        names = [name for name in names if name in args.dataset]
    if not names:
        parser.error("No compatible datasets found")
    try:
        path = checkpoint_path(args.model_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    protocol = {
        "model": "TabPFNRegressor", "weights_version": "v3.5",
        "tabpfn_package_version": version("tabpfn"), "checkpoint": str(path),
        "device": args.device, "n_estimators": args.n_estimators,
        "fit_mode": args.fit_mode, "runs": args.runs,
        "base_seed": BASE_SEED, "validation": "none",
        "target_prediction": "mean", "scaling": "StandardScaler fit on the full training split",
    }
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_ROOT / "TabPFN35_summary.csv"
    existing = {}
    if summary_path.exists():
        for row in pd.read_csv(summary_path).to_dict("records"):
            old_name = str(row["dataset_name"])
            old_path = RESULTS_ROOT / f"{old_name}_protocol.json"
            if (not old_path.exists() or
                    json.loads(old_path.read_text(encoding="utf-8")) != protocol):
                if old_name not in names or not args.overwrite:
                    raise RuntimeError(
                        f"{old_name}: existing summary uses a different protocol. "
                        "Use --overwrite with all affected datasets."
                    )
                continue
            existing[old_name] = row
    with tqdm(total=len(names), desc="Datasets", unit="dataset", position=0) as dataset_bar:
        for name in names:
            stem = RESULTS_ROOT / name
            protocol_path = Path(f"{stem}_protocol.json")
            outputs = [Path(f"{stem}{suffix}") for suffix in
                       ("_metrics.csv", "_preds.csv", "_plot.png", "_runs.csv")]
            if not args.overwrite and protocol_path.exists():
                old_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                if old_protocol != protocol:
                    raise RuntimeError(f"{name}: saved protocol differs; use --overwrite to rerun")
                if all(p.is_file() for p in outputs):
                    existing[name] = pd.read_csv(outputs[0]).iloc[0].to_dict()
                    tqdm.write(f"{name}: complete, skipping")
                    dataset_bar.update(1)
                    continue
            existing[name] = run_dataset(name, config, args, path)
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
            pd.DataFrame([existing[key] for key in sorted(existing, key=dataset_sort_key)]).to_csv(
                summary_path, index=False
            )
            tqdm.write(f"Saved {name}; summary: {summary_path}")
            dataset_bar.update(1)
    if existing:
        pd.DataFrame([existing[key] for key in sorted(existing, key=dataset_sort_key)]).to_csv(
            summary_path, index=False
        )


if __name__ == "__main__":
    main()
