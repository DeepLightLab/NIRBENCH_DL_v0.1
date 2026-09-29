## Import main libraries
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "nirbench_matplotlib"))
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Override externally to select another GPU
import argparse
import importlib.util
from pathlib import Path
from sys import stdout
import random
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ModuleNotFoundError:  # optional dependency
    sns = None
import pandas as pd
import scipy.io as sio
from scipy import stats
from scipy.signal import savgol_filter
from tqdm import tqdm            # IMPORT THE CALLABLE
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score , KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    concatenate,
    LeakyReLU, 
    BatchNormalization,
    Dropout)
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


def _configure_tensorflow_runtime() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("TensorFlow: no GPU detected, using CPU.")
        return "/CPU:0"

    # With CUDA_VISIBLE_DEVICES=0 there should be a single visible GPU (index 0).
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as exc:
        print(f"TensorFlow: could not set memory growth: {exc}")

    print(f"TensorFlow: using GPU device {gpus[0].name} (/GPU:0)")
    return "/GPU:0"


##----------------------------- HELPER FUNCTIONS FOR BENCHMARKING MODELS -----------------------------##
################## SET RANDOM SEEDS FOR REPRODUCIBILITY ##################
def reproducible_comp():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(12345)
    random.seed(12345)
    tf.random.set_seed(12345)
DEVICE_NAME = _configure_tensorflow_runtime()
reproducible_comp()

MAX_EPOCHS = 600
CV_FOLDS = 5
CV_RANDOM_STATE = 12345
# When True, skips datasets that already have all expected output files.
# For this benchmark update we want to re-run everything, so the default is False.
RESUME_EXISTING_OUTPUTS = os.environ.get("NIRBENCH_RESUME", "0") == "1"
N_FINAL_RUNS = 10


def _set_seeds(seed: int):
    """Reset all random seeds for reproducible weight initialisation."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def _run_final_training_multi(
    *,
    build_compile_fn,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_scaler,
    y_train_scaled: np.ndarray,
    selected_epochs: int,
    batch_size: int = 16,
    extra_fit_kwargs_fn=None,
) -> dict:
    """
    Train the model N_FINAL_RUNS times with different seeds.
    Returns dict with summary_row (mean/std metrics + avg wall time),
    last_history, last_y_train_pred, last_y_test_pred.
    """
    run_metrics: list[dict] = []
    run_wall_times: list[float] = []
    last_history = None
    last_y_train_pred = None
    last_y_test_pred = None

    for run_i in range(N_FINAL_RUNS):
        _set_seeds(CV_RANDOM_STATE + run_i)
        tf.keras.backend.clear_session()
        extra_kwargs = extra_fit_kwargs_fn() if extra_fit_kwargs_fn else {}

        t0 = time.perf_counter()
        with tf.device(DEVICE_NAME):
            model = build_compile_fn()
            history = model.fit(
                X_train_scaled,
                y_train_scaled,
                epochs=selected_epochs,
                batch_size=batch_size,
                verbose=0,
                **extra_kwargs,
            )
            y_train_pred_scaled = np.asarray(
                model.predict(X_train_scaled, verbose=0)
            ).reshape(-1)
            y_test_pred_scaled = np.asarray(
                model.predict(X_test_scaled, verbose=0)
            ).reshape(-1)
        wall_time = time.perf_counter() - t0

        y_train_pred = (
            y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1))
            .astype(np.float32)
            .reshape(-1)
        )
        y_test_pred = (
            y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
            .astype(np.float32)
            .reshape(-1)
        )

        run_metrics.append({
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "train_R2": float(r2_score(y_train, y_train_pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "test_R2": float(r2_score(y_test, y_test_pred)),
        })
        run_wall_times.append(wall_time)
        last_history = history
        last_y_train_pred = y_train_pred
        last_y_test_pred = y_test_pred

        print(
            f"  Run {run_i + 1}/{N_FINAL_RUNS}: "
            f"test_rmse={run_metrics[-1]['test_rmse']:.4f}, "
            f"test_R2={run_metrics[-1]['test_R2']:.4f}, "
            f"wall={wall_time:.1f}s"
        )

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    metrics_df = pd.DataFrame(run_metrics)
    summary_row = {
        "train_rmse_mean": float(metrics_df["train_rmse"].mean()),
        "train_rmse_std": float(metrics_df["train_rmse"].std()),
        "train_R2_mean": float(metrics_df["train_R2"].mean()),
        "train_R2_std": float(metrics_df["train_R2"].std()),
        "test_rmse_mean": float(metrics_df["test_rmse"].mean()),
        "test_rmse_std": float(metrics_df["test_rmse"].std()),
        "test_R2_mean": float(metrics_df["test_R2"].mean()),
        "test_R2_std": float(metrics_df["test_R2"].std()),
        "avg_wall_time_s": float(np.mean(run_wall_times)),
    }

    return {
        "summary_row": summary_row,
        "last_history": last_history,
        "last_y_train_pred": last_y_train_pred,
        "last_y_test_pred": last_y_test_pred,
    }


def _best_epoch_from_history(history) -> int:
    val_losses = history.history.get("val_loss")
    if not val_losses:
        raise ValueError("Expected 'val_loss' in History to select best epoch via CV.")
    return int(np.argmin(val_losses) + 1)


def select_epochs_via_cv(
    *,
    n_samples: int,
    fit_fold_fn,
    n_splits: int = CV_FOLDS,
    random_state: int = CV_RANDOM_STATE,
) -> tuple[int, list[int]]:
    """
    Strategy (1): 5-fold CV on the training set, record best_epoch per fold (argmin val_loss),
    then select median(best_epoch) and use it as the fixed epoch budget for the final fit on the
    full training set (no validation split, no early stopping).
    """
    if n_samples < 2:
        return 1, [1]
    if n_samples < n_splits:
        n_splits = max(2, n_samples)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_epochs: list[int] = []
    for train_idx, val_idx in kf.split(np.arange(n_samples)):
        best_epochs.append(int(fit_fold_fn(train_idx, val_idx)))

    selected = int(np.median(best_epochs))
    selected = max(1, min(MAX_EPOCHS, selected))
    return selected, best_epochs

def make_early_stopping_callback():
    # Benchmark-wide default (requested): val_loss monitored, patience=100, restore best weights.
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, restore_best_weights=True
    )

def standardize_target(y_train: np.ndarray, y_test: np.ndarray):
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = (
        y_scaler.transform(y_train.reshape(-1, 1)).astype(np.float32).reshape(-1)
    )
    y_test_scaled = (
        y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32).reshape(-1)
    )
    return y_scaler, y_train_scaled, y_test_scaled

def save_loss_plot(history, *, title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(history.history.get("loss", [])) + 1)
    plt.plot(epochs, history.history.get("loss", []), label="train")
    if "val_loss" in history.history:
        plt.plot(epochs, history.history["val_loss"], label="val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

################## Function to standardize data column-wise ##################
def standardize_column(X_train, X_test):
    ## We train the scaler on the full train set and apply it to the other datasets
    scaler = StandardScaler().fit(X_train)
    # scaler = MinMaxScaler().fit(X_train)
    ## for columns we fit the scaler to the train set and apply it to the test set
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return [X_train_scaled, X_test_scaled]

################## Function to compute metrics and make prediction plots using train and test data 
def plot_prediction2(Y_train, Y_test, Y_train_pred, Y_test_pred, title, savefig=False, figname=None):

    ## Compute train error scores
    score_p0 = r2_score(Y_train, Y_train_pred)
    mse_p0 = mean_squared_error(Y_train, Y_train_pred)
    rmse_p0 = np.sqrt(mse_p0)

    ## Compute test error scores
    score_p2 = r2_score(Y_test, Y_test_pred)
    mse_p2 = mean_squared_error(Y_test, Y_test_pred)
    rmse_p2 = np.sqrt(mse_p2)

    print('ERROR METRICS: \t TRAIN  \t\t TEST')
    print('------------------------------------------------------')
    print('R2:   \t\t %5.3f  \t\t %5.3f'  % (score_p0, score_p2 ))
    print('RMSE: \t\t %5.3f  \t\t %5.3f' % (rmse_p0, rmse_p2))

    #### Plot regression for model predicted data
    ## Get plot limits
    Y = np.concatenate([Y_train, Y_test])

    rangey = np.max(Y) - np.min(Y)
    rangex = np.max(Y) - np.min(Y)
    ## x=y line and +- 1std upper and lower bowndaries
    xy_x=np.ravel([np.min(Y)-0.1*rangex, np.max(Y)+0.1*rangex])
    xy_y=np.ravel([np.min(Y)-0.1*rangey, np.max(Y)+0.1*rangey])

    fig = plt.figure(figsize=(5,5))
    z = np.polyfit(np.ravel(Y_test), np.ravel(Y_test_pred), 1)
    print('Fit result: Y=',z[1], ' + ', z[0],' * X')
    ax = plt.subplot(aspect=1)
    ax.plot(xy_x, xy_y, 'k--', linewidth=2, label=None)
    ax.scatter(Y_train, Y_train_pred, c='gray', marker='o', s=20, alpha=0.66, label='train')
    ax.scatter(Y_test,Y_test_pred, s=40, marker='o', facecolors='None', edgecolors='r', label='test')
    # Calculate the range of x-axis based on Y_train and Y_test
    x_min = min(np.min(Y_train), np.min(Y_test))
    x_max = max(np.max(Y_train), np.max(Y_test))
    # Create an array spanning the range of x-axis
    x_range = np.linspace(x_min, x_max, num=100)
    ax.plot(x_range, z[1]+z[0]*x_range, c='blue', linewidth=2,label='linear fit')
    plt.xlim(xy_x)
    plt.ylim(xy_y)
    # ax.plot(x_range, x_range, 'k--', linewidth=1.5, label='y=x')
    plt.ylabel('Predicted')
    plt.xlabel('Measured')
    plt.title(title)
    plt.legend(loc=4, frameon=False)

    # Print the scores on the plot
    plt.text(np.min(xy_x)+0.05*rangex, np.max(xy_y)-0.1*rangey, 'R$^{2}=$ %5.2f'  % score_p2, fontsize=13)
    plt.text(np.min(xy_x)+0.05*rangex, np.max(xy_y)-0.15*rangey, 'RMSE: %5.2f' % rmse_p2, fontsize=13)
    if savefig==True:
        plt.savefig(figname, dpi=150)
        print('Figure saved')
    else:
        plt.show()
    plt.close(fig)
    return rmse_p0, rmse_p2


##-------------------------------------- LOAD DATASET AND PREPROCESSING ----------------------------------##

## standardize feature-wise using the statistics of the calibration set 
# x_train_scaled, x_test_scaled = standardize_column(x_train, x_test)

from src.data_loading import discover_datasets, load_dataset


##---------------------------------------- IMPORT DL MODELS ----------------------------------------------##
# If models is in your python path or same directory
from models.DeepSpectra import deepspectra_model
from models.IPA import ipa_model
from models.MarkSpectra import compute_mark_indices_ols_r2, markspectra_model
from models.ResidualSpectra import residualspectra_model
from models.SCNet import build_scnet
from models.Spectraformer import build_spectraformer
from models.SpectraTr import build_spectratr
from models.SpectraNet32 import build_spectranet32
from models.SpectraNet53 import build_spectranet53
from models.CNN_1D_1L_3 import build_cnn_1d_1l_3
from models.CNN_1D_1N_3 import build_cnn_1d_1n_3
from models.CNN_1D_3_1 import build_cnn_1d_3_1


def _load_inception_resnet_1d_factory():
    # File name starts with a digit, so it cannot be imported via standard Python syntax.
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "models" / "1DInceptionResnet.py"
    spec = importlib.util.spec_from_file_location("models._1DInceptionResnet", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model module from: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inception_resnet_1d_model

def _repo_and_datasets(dataset_filter: list[str] | None = None):
    repo_root = Path(__file__).resolve().parent
    datasets_root = repo_root / "datasets"
    dataset_config = discover_datasets(datasets_root)
    dataset_names = list(dataset_config.keys())
    if not dataset_names:
        raise RuntimeError(f"No datasets found in {datasets_root}")
    if dataset_filter is not None:
        unknown = set(dataset_filter) - set(dataset_names)
        if unknown:
            raise ValueError(
                f"Unknown dataset(s): {sorted(unknown)}.\n"
                f"Available: {dataset_names}"
            )
        dataset_names = [d for d in dataset_names if d in dataset_filter]
    return repo_root, datasets_root, dataset_config, dataset_names

def _write_outputs(
    *,
    model_name: str,
    model_results_dir: Path,
    dataset_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    history,
    summary_rows: list[dict],
    loss_ylabel: str,
):
    metrics_path = model_results_dir / f"{dataset_name}_metrics.csv"
    pd.DataFrame([summary_rows[-1]]).to_csv(metrics_path, index=False)

    preds_path = model_results_dir / f"{dataset_name}_preds.csv"
    preds_df = pd.DataFrame(
        {
            "split": (["train"] * len(y_train)) + (["test"] * len(y_test)),
            "y_true": np.concatenate([y_train, y_test]),
            "y_pred": np.concatenate([y_train_pred, y_test_pred]),
        }
    )
    preds_df.to_csv(preds_path, index=False)

    plot_path = model_results_dir / f"{dataset_name}_plot.png"
    plot_prediction2(
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        title=f"{model_name} - {dataset_name}",
        savefig=True,
        figname=str(plot_path),
    )

    loss_path = model_results_dir / f"{dataset_name}_loss.png"
    save_loss_plot(
        history,
        title=f"{model_name} - {dataset_name} loss",
        ylabel=loss_ylabel,
        path=loss_path,
    )
    plt.close("all")

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {loss_path}")
    print(f"Epochs ran: {int(len(history.history.get('loss', [])))}")

def _dataset_outputs_complete(model_results_dir: Path, dataset_name: str) -> bool:
    return all(
        (
            (model_results_dir / f"{dataset_name}_metrics.csv").exists(),
            (model_results_dir / f"{dataset_name}_preds.csv").exists(),
            (model_results_dir / f"{dataset_name}_plot.png").exists(),
            (model_results_dir / f"{dataset_name}_loss.png").exists(),
        )
    )


def _load_existing_metrics_row(model_results_dir: Path, dataset_name: str) -> dict | None:
    metrics_path = model_results_dir / f"{dataset_name}_metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    required = {"train_rmse_mean", "train_R2_mean", "test_rmse_mean", "test_R2_mean"}
    if not required.issubset(row.keys()):
        return None
    return {
        "dataset_name": str(row.get("dataset_name", dataset_name)),
        "train_rmse_mean": float(row["train_rmse_mean"]),
        "train_rmse_std": float(row.get("train_rmse_std", 0.0)),
        "train_R2_mean": float(row["train_R2_mean"]),
        "train_R2_std": float(row.get("train_R2_std", 0.0)),
        "test_rmse_mean": float(row["test_rmse_mean"]),
        "test_rmse_std": float(row.get("test_rmse_std", 0.0)),
        "test_R2_mean": float(row["test_R2_mean"]),
        "test_R2_std": float(row.get("test_R2_std", 0.0)),
        "avg_wall_time_s": float(row.get("avg_wall_time_s", 0.0)),
    }


def build_global_model_ranking() -> None:
    """
    Build a single cross-model score for fair comparison across heterogeneous datasets.

    Score definition (higher is better):
    1) Per dataset, rank models by test RMSE (lower RMSE = better rank).
    2) Convert rank to [0,1]: (n_models - rank) / (n_models - 1).
    3) Average across datasets and scale to 0-100.

    This keeps each dataset on equal footing regardless of absolute target scale.
    """
    repo_root = Path(__file__).resolve().parent
    results_root = repo_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    model_rows: list[pd.DataFrame] = []
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue

        summary_path = model_dir / f"{model_dir.name}_summary.csv"
        if not summary_path.exists():
            continue

        try:
            summary_df = pd.read_csv(summary_path)
        except Exception as exc:
            print(f"Skipping {summary_path}: could not read CSV ({exc})")
            continue

        required = {"dataset_name", "test_rmse_mean", "test_R2_mean"}
        if not required.issubset(summary_df.columns):
            print(f"Skipping {summary_path}: missing required columns {sorted(required)}")
            continue

        keep_cols = ["dataset_name", "test_rmse_mean", "test_R2_mean"]
        if "avg_wall_time_s" in summary_df.columns:
            keep_cols.append("avg_wall_time_s")
        slim_df = summary_df[keep_cols].copy()
        slim_df["model_name"] = model_dir.name
        model_rows.append(slim_df)

    if not model_rows:
        print("Global ranking skipped: no per-model summary files found under results/.")
        return

    all_results = pd.concat(model_rows, ignore_index=True)
    all_results["dataset_name"] = all_results["dataset_name"].astype(str)
    all_results["model_name"] = all_results["model_name"].astype(str)
    all_results["test_rmse_mean"] = pd.to_numeric(all_results["test_rmse_mean"], errors="coerce")
    all_results["test_R2_mean"] = pd.to_numeric(all_results["test_R2_mean"], errors="coerce")
    all_results = all_results.dropna(subset=["test_rmse_mean", "test_R2_mean"]).copy()

    if all_results.empty:
        print("Global ranking skipped: no valid numeric rows in summaries.")
        return

    all_results["models_in_dataset"] = (
        all_results.groupby("dataset_name")["model_name"].transform("nunique").astype(int)
    )
    all_results["rmse_rank"] = all_results.groupby("dataset_name")["test_rmse_mean"].rank(
        method="average",
        ascending=True,
    )
    all_results["rank_score"] = np.where(
        all_results["models_in_dataset"] > 1,
        (all_results["models_in_dataset"] - all_results["rmse_rank"])
        / (all_results["models_in_dataset"] - 1),
        np.nan,
    )

    best_rmse_per_dataset = all_results.groupby("dataset_name")["test_rmse_mean"].transform("min")
    all_results["rmse_vs_best"] = all_results["test_rmse_mean"] / best_rmse_per_dataset.clip(lower=1e-12)

    agg_dict = dict(
        datasets_evaluated=("dataset_name", "nunique"),
        mean_test_rmse=("test_rmse_mean", "mean"),
        median_test_rmse=("test_rmse_mean", "median"),
        mean_test_r2=("test_R2_mean", "mean"),
        median_test_r2=("test_R2_mean", "median"),
        mean_rmse_rank=("rmse_rank", "mean"),
        general_score=("rank_score", "mean"),
        mean_rmse_vs_best=("rmse_vs_best", "mean"),
    )
    if "avg_wall_time_s" in all_results.columns:
        agg_dict["avg_wall_time_s"] = ("avg_wall_time_s", "mean")

    leaderboard = (
        all_results.groupby("model_name", as_index=False)
        .agg(**agg_dict)
        .sort_values("general_score", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard["general_score"] = 100.0 * leaderboard["general_score"]

    ranking_csv_path = results_root / "global_model_ranking.csv"
    leaderboard.to_csv(ranking_csv_path, index=False)
    print(f"Saved: {ranking_csv_path}")

    details_csv_path = results_root / "global_model_ranking_details.csv"
    details_cols = [
        "dataset_name",
        "model_name",
        "test_rmse_mean",
        "test_R2_mean",
        "models_in_dataset",
        "rmse_rank",
        "rank_score",
        "rmse_vs_best",
    ]
    all_results[details_cols].to_csv(details_csv_path, index=False)
    print(f"Saved: {details_csv_path}")

    plot_df = leaderboard.dropna(subset=["general_score"]).copy()
    if plot_df.empty:
        print("Global ranking plot skipped: no comparable datasets across models.")
        return

    # Compute a PLS baseline score on the same rank-based scale, but keep the
    # official leaderboard CSV DL-only. The PLS score is used only as a
    # reference line in a secondary plot.
    pls_general_score: float | None = None
    pls_baseline_path = repo_root / "pls_baseline_benchmark.cvs"
    if pls_baseline_path.exists():
        try:
            pls_df = pd.read_csv(pls_baseline_path)
            required = {"dataset", "RMSE test", "R2 test"}
            if required.issubset(pls_df.columns):
                pls_rows = pls_df[["dataset", "RMSE test", "R2 test"]].copy()
                pls_rows = pls_rows.rename(
                    columns={
                        "dataset": "dataset_name",
                        "RMSE test": "test_rmse_mean",
                        "R2 test": "test_R2_mean",
                    }
                )
                pls_rows["model_name"] = "PLSRegression"
                pls_rows["dataset_name"] = pls_rows["dataset_name"].astype(str)
                pls_rows["test_rmse_mean"] = pd.to_numeric(
                    pls_rows["test_rmse_mean"], errors="coerce"
                )
                pls_rows["test_R2_mean"] = pd.to_numeric(
                    pls_rows["test_R2_mean"], errors="coerce"
                )
                pls_rows = pls_rows.dropna(subset=["test_rmse_mean", "test_R2_mean"])

                combined = pd.concat([all_results, pls_rows], ignore_index=True)
                combined["models_in_dataset"] = (
                    combined.groupby("dataset_name")["model_name"]
                    .transform("nunique")
                    .astype(int)
                )
                combined["rmse_rank"] = combined.groupby("dataset_name")[
                    "test_rmse_mean"
                ].rank(method="average", ascending=True)
                combined["rank_score"] = np.where(
                    combined["models_in_dataset"] > 1,
                    (combined["models_in_dataset"] - combined["rmse_rank"])
                    / (combined["models_in_dataset"] - 1),
                    np.nan,
                )

                pls_scores = combined.loc[
                    combined["model_name"] == "PLSRegression", "rank_score"
                ]
                if not pls_scores.empty:
                    pls_general_score = 100.0 * float(pls_scores.mean())
            else:
                print(
                    f"Skipping PLS overlay: {pls_baseline_path} missing required columns "
                    f"{sorted(required)}"
                )
        except Exception as exc:
            print(f"Skipping PLS overlay: could not read {pls_baseline_path} ({exc})")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["model_name"], plot_df["general_score"])
    ax.set_ylabel("General Score (0-100)")
    ax.set_xlabel("Model")
    ax.set_title("Global Model Ranking Across Datasets (RMSE Rank-Based)")
    ax.set_ylim(0.0, 100.0)
    plt.xticks(rotation=35, ha="right")
    for i, value in enumerate(plot_df["general_score"].to_numpy()):
        ax.text(i, float(value) + 1.0, f"{float(value):.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    ranking_plot_path = results_root / "global_model_ranking.png"
    fig.savefig(ranking_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {ranking_plot_path}")

    if pls_general_score is not None:
        fig_pls, ax_pls = plt.subplots(figsize=(10, 5))
        ax_pls.bar(plot_df["model_name"], plot_df["general_score"])
        ax_pls.axhline(
            pls_general_score,
            color="crimson",
            linestyle="--",
            linewidth=3.0,
            label=f"PLSRegression ({pls_general_score:.1f})",
        )
        ax_pls.set_ylabel("General Score (0-100)")
        ax_pls.set_xlabel("Model")
        ax_pls.set_title("Global Model Ranking with PLS Baseline Reference")
        ax_pls.set_ylim(0.0, 100.0)
        plt.xticks(rotation=35, ha="right")
        for i, value in enumerate(plot_df["general_score"].to_numpy()):
            ax_pls.text(
                i,
                float(value) + 1.0,
                f"{float(value):.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax_pls.legend(loc="upper right")
        plt.tight_layout()

        ranking_plot_path_2 = results_root / "global_model_ranking2.png"
        fig_pls.savefig(ranking_plot_path_2, dpi=150)
        plt.close(fig_pls)
        print(f"Saved: {ranking_plot_path_2}")

    # --- Wall-time barplot ---
    if "avg_wall_time_s" in leaderboard.columns:
        time_df = leaderboard.dropna(subset=["avg_wall_time_s"]).copy()
        if not time_df.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.bar(time_df["model_name"], time_df["avg_wall_time_s"])
            ax2.set_ylabel("Avg Training Wall Time (s)")
            ax2.set_xlabel("Model")
            ax2.set_title("Average Training Wall Time per Model (across datasets)")
            plt.xticks(rotation=35, ha="right")
            for i, value in enumerate(time_df["avg_wall_time_s"].to_numpy()):
                ax2.text(
                    i, float(value) + 0.5, f"{float(value):.1f}",
                    ha="center", va="bottom", fontsize=8,
                )
            plt.tight_layout()
            time_plot_path = results_root / "training_wall_time.png"
            fig2.savefig(time_plot_path, dpi=150)
            plt.close(fig2)
            print(f"Saved: {time_plot_path}")


##------------------------------------- BENCHMARK DL MODELS ----------------------------------------------##
def _run_standard_cnn_baseline_all_datasets(
    *,
    model_name: str,
    build_model_fn,
    dataset_filter: list[str] | None = None,
    batch_size: int = 16,
) -> None:
    """
    Benchmark a simple 1D CNN regression baseline under the shared NIRBENCH
    protocol: StandardScaler preprocessing, 5-fold CV epoch selection, MSE
    loss, Adam(1e-3), and repeated final training.
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = build_model_fn(
                    input_vector_dimension=int(X_fold_train_scaled.shape[1]),
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = build_model_fn(input_vector_dimension=int(X_train_scaled.shape[1]))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
            batch_size=batch_size,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_cnn_1d_1l_3_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """Wide-filter shallow single-convolution baseline."""
    _run_standard_cnn_baseline_all_datasets(
        model_name="1D-CNN_1L_3",
        build_model_fn=build_cnn_1d_1l_3,
        dataset_filter=dataset_filter,
        batch_size=16,
    )


def run_cnn_1d_1n_3_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """Narrow-filter shallow single-convolution baseline."""
    _run_standard_cnn_baseline_all_datasets(
        model_name="1D-CNN_1N_3",
        build_model_fn=build_cnn_1d_1n_3,
        dataset_filter=dataset_filter,
        batch_size=16,
    )


def run_cnn_1d_3_1_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """Three-block vanilla 1D CNN baseline."""
    _run_standard_cnn_baseline_all_datasets(
        model_name="1D-CNN_3_1",
        build_model_fn=build_cnn_1d_3_1,
        dataset_filter=dataset_filter,
        batch_size=16,
    )


def run_deepspectra_all_datasets(dataset_filter: list[str] | None = None) -> None:
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "DeepSpectra"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = deepspectra_model(
                    input_vector_dimension=X_fold_train_scaled.shape[1],
                    kernel_size_1=11,
                    kernel_size_2=3,
                    kernel_size_3=5,
                    stride_1=1,
                    stride_2=2,
                    hidden_number=128,
                    dropout_rate=0.14,
                    l2_regularization=0.0018,
                )

                LR = 0.00438
                LR_DECAY = 0.00019105
                lr_schedule = InverseTimeDecay(
                    initial_learning_rate=LR,
                    decay_steps=1,  # per global step
                    decay_rate=LR_DECAY,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )

                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)

        # DeepSpectra expects (batch, steps, channels)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        # Standardize the target using train statistics and train on standardized y.
        # Metrics/plots are computed on the original scale after inverse-transform.
        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = deepspectra_model(
                input_vector_dimension=X_train_scaled.shape[1],
                kernel_size_1=11,
                kernel_size_2=3,
                kernel_size_3=5,
                stride_1=1,
                stride_2=2,
                hidden_number=128,
                dropout_rate=0.14,
                l2_regularization=0.0018,
            )
            lr_schedule = InverseTimeDecay(
                initial_learning_rate=0.00438,
                decay_steps=1,
                decay_rate=0.00019105,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_ipa_all_datasets(dataset_filter: list[str] | None = None) -> None:
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "IPA"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                # Paper: ExponentialDecay schedule (lr_decay=1e-3, lr_decay_steps=1e4)
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=1e4,
                    decay_rate=1e-3,
                    staircase=False,
                )
                model = ipa_model(
                    input_vector_dimension=X_fold_train_scaled.shape[1],
                    dropout_rate=0.2,
                    l2_regularization=1e-3,
                    padding="valid",
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )

                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=1e4,
                decay_rate=1e-3,
                staircase=False,
            )
            model = ipa_model(
                input_vector_dimension=X_train_scaled.shape[1],
                dropout_rate=0.2,
                l2_regularization=1e-3,
                padding="valid",
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_inception_resnet_1d_all_datasets(dataset_filter: list[str] | None = None) -> None:
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "1DInceptionResnet"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    inception_resnet_1d_model = _load_inception_resnet_1d_factory()
    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = inception_resnet_1d_model(
                    input_vector_dimension=X_fold_train_scaled.shape[1]
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )

                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[
                        make_early_stopping_callback(),
                        keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss", factor=0.1, patience=25, min_lr=1e-6
                        ),
                    ],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = inception_resnet_1d_model(
                input_vector_dimension=X_train_scaled.shape[1]
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        def _extra_fit_kwargs():
            return dict(callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.1, patience=25, min_lr=1e-6
                ),
            ])

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
            extra_fit_kwargs_fn=_extra_fit_kwargs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_markspectra_all_datasets(dataset_filter: list[str] | None = None) -> None:
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "MarkSpectra"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            # Mark selection (paper evaluates 10%..90%); use 30% as a reasonable default.
            input_dim = int(X_fold_train_scaled.shape[1])
            mark_top_k = max(10, min(input_dim, int(round(0.30 * input_dim))))
            mark_indices, _ = compute_mark_indices_ols_r2(
                X_fold_train_scaled, y_fold_train_scaled, top_k=mark_top_k
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                LR = 0.01
                LR_DECAY = 0.001
                lr_schedule = InverseTimeDecay(
                    initial_learning_rate=LR,
                    decay_steps=1,
                    decay_rate=LR_DECAY,
                )
                model = markspectra_model(
                    input_dim=input_dim,
                    mark_indices=mark_indices,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        # Mark selection (paper evaluates 10%..90%); use 30% as a reasonable default.
        input_dim = int(X_train_scaled.shape[1])
        mark_top_k = max(10, min(input_dim, int(round(0.30 * input_dim))))
        mark_indices, _ = compute_mark_indices_ols_r2(
            X_train_scaled, y_train_scaled, top_k=mark_top_k
        )

        def _build_compile():
            lr_schedule = InverseTimeDecay(
                initial_learning_rate=0.01,
                decay_steps=1,
                decay_rate=0.001,
            )
            model = markspectra_model(
                input_dim=input_dim,
                mark_indices=mark_indices,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_residualspectra_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """
    ResidualSpectra paper defines a classification model (softmax + cross-entropy).
    Our `models/ResidualSpectra.py` is already adapted to regression with Dense(1, linear).

    Paper training details (from ResidualSpectra.pdf):
    - Optimizer: Adam
    - Learning rate: 0.001
    (Batch size / epochs are not clearly extractable from the PDF text; we use benchmark defaults.)
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "ResidualSpectra"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = residualspectra_model(
                    input_vector_dimension=X_fold_train_scaled.shape[1]
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = residualspectra_model(input_vector_dimension=X_train_scaled.shape[1])
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_scnet_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """
    SCNet is a regression model (paper: SCNet.pdf).

    Paper training details (SCNet.pdf):
    - Preprocessing: Savitzky–Golay smoothing + first derivative, then min–max normalization.
    - Targets scaled to [0,1] (paper uses a sigmoid output after scaling; in this benchmark we keep linear output).
    - Optimizer: Adam; learning rate fixed to 0.001.
    - Batch size: 128; validation rate: 0.25; epochs: 20000 with Early Stop.

    Benchmark adaptations:
    - We cap epochs at MAX_EPOCHS=600 and use EarlyStopping(patience=100) as requested.
    - Output remains Dense(1, linear) per benchmark convention (see models/SCNet.py).
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "SCNet"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = build_scnet(
                    input_dim=int(X_fold_train_scaled.shape[1]),
                    dense_units=(267,),
                    dropout_rate=0.0,
                    l2=0.0,
                    name=model_name,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = build_scnet(
                input_dim=int(X_train_scaled.shape[1]),
                dense_units=(267,),
                dropout_rate=0.0,
                l2=0.0,
                name=model_name,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_spectraformer_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """
    Spectraformer is defined in the paper for classification (cross-entropy + SGD).
    Our `models/Spectraformer.py` adapts the head to Dense(1, linear) for regression.

    Paper training details (Spectraformer.pdf):
    - Optimizer: SGD with learning rate 1e-4.
    - Epochs: 200 (fixed in paper).

    Benchmark adaptations:
    - Loss switched to MSE for regression.
    - Max epochs capped at 600 + EarlyStopping(patience=100) as requested.
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "Spectraformer"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = build_spectraformer(
                    input_vector_dimension=int(X_fold_train_scaled.shape[1]),
                    input_channels=1,
                    num_conv_blocks=4,
                    initial_filters=16,
                    conv_stride=2,
                    conv_padding="same",  # paper uses padding=1 in PyTorch; 'same' is the TF approximation
                    transformer_after_block=1,
                    transformer_num_heads=2,
                    dense_units=(256, 128),
                    dense_dropout=0.5,
                    name=model_name,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = build_spectraformer(
                input_vector_dimension=int(X_train_scaled.shape[1]),
                input_channels=1,
                num_conv_blocks=4,
                initial_filters=16,
                conv_stride=2,
                conv_padding="same",
                transformer_after_block=1,
                transformer_num_heads=2,
                dense_units=(256, 128),
                dense_dropout=0.5,
                name=model_name,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_spectratr_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """
    SpectraTr is defined in the paper for qualitative analysis (classification).
    Our `models/SpectraTr.py` adapts the head to Dense(1, linear) for regression.

    Paper training details (SpectraTr.pdf):
    - Optimizer: Adam, total epochs: 200.
    - LR schedule: halve LR if training loss doesn't decline within 10 epochs.
    - Early stop: stop if test loss doesn't drop within 30 epochs.
    - Selected hyperparameters (Table 2) depend on dataset length (e.g., patch_num=40 for 2074 points,
      patch_num=10 for 404 points).

    Benchmark adaptations:
    - Loss switched to MSE for regression.
    - Max epochs capped at 600 + EarlyStopping(patience=100) as requested.
    - Use ReduceLROnPlateau(patience=10, factor=0.5) to approximate the LR-halving rule.
    - Patch_num is chosen per dataset to keep ~50 points per patch (heuristic aligned with paper discussion).
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "SpectraTr"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    allowed_patch_nums = np.array([5, 10, 20, 25, 40, 50, 80, 100], dtype=int)

    def _choose_patch_num(input_dim: int) -> int:
        # Paper suggests ~50 points per patch; pick the closest allowed patch_num <= input_dim.
        target = max(1, int(round(input_dim / 50)))
        candidates = allowed_patch_nums[allowed_patch_nums <= input_dim]
        if candidates.size == 0:
            return int(min(allowed_patch_nums))
        return int(candidates[np.argmin(np.abs(candidates - target))])

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        input_dim = int(X_train.shape[1])
        patch_num = _choose_patch_num(input_dim)

        # Heuristic configuration based on SpectraTr Table 2 (two datasets with lengths 2074 vs 404).
        if input_dim >= 500:
            # Long spectra config (closer to Table 2 Dataset A)
            num_heads = 19
            depth = 5
            mlp_dim = 512
            l2_reg = 0.01
        else:
            # Shorter spectra config (closer to Table 2 Dataset B)
            num_heads = 12
            depth = 3
            mlp_dim = 1024
            l2_reg = 0.0

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = build_spectratr(
                    input_dim=input_dim,
                    n_channels=1,
                    patch_num=patch_num,
                    pad_to_multiple=True,
                    embed_dim=1024,
                    depth=depth,
                    num_heads=num_heads,
                    dim_head=64,
                    mlp_dim=mlp_dim,
                    dropout=0.0,
                    attn_dropout=0.0,
                    use_cls_token=True,
                    pooling="cls",
                    pos_embedding="gaussian_fixed",
                    pos_stddev=0.02,
                    head_units=(128,),
                    head_dropout=0.0,
                    l2_reg=l2_reg,
                    name=model_name,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=16,
                    verbose=0,
                    callbacks=[
                        make_early_stopping_callback(),
                        # Paper: halve LR if training loss doesn't decline within 10 epochs.
                        keras.callbacks.ReduceLROnPlateau(
                            monitor="loss", factor=0.5, patience=10, min_lr=1e-6
                        ),
                    ],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        def _build_compile():
            model = build_spectratr(
                input_dim=input_dim,
                n_channels=1,
                patch_num=patch_num,
                pad_to_multiple=True,
                embed_dim=1024,
                depth=depth,
                num_heads=num_heads,
                dim_head=64,
                mlp_dim=mlp_dim,
                dropout=0.0,
                attn_dropout=0.0,
                use_cls_token=True,
                pooling="cls",
                pos_embedding="gaussian_fixed",
                pos_stddev=0.02,
                head_units=(128,),
                head_dropout=0.0,
                l2_reg=l2_reg,
                name=model_name,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        def _extra_fit_kwargs():
            return dict(callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss", factor=0.5, patience=10, min_lr=1e-6
                ),
            ])

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
            extra_fit_kwargs_fn=_extra_fit_kwargs,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_spectranet53_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """
    SpectraNet-53 is a regression model (SpectraNet53.pdf).

    Paper training details (SpectraNet53.pdf):
    - Optimizer: Adam (default)
    - Learning rate: 1e-4 with a drop multiplier of 0.7 each epoch
    - Epochs: 10 (paper reports using 10 as a good early stop point)
    - L2 regularization (weight decay): 0.05
    - Gradient clipping: L2-norm <= 0.5
    - Dropout: 0.2
    - Preprocessing: QNV-5 (not implemented here; we keep the benchmark's StandardScaler + model BN-after-input)

    Benchmark adaptations:
    - Max epochs capped at 600 + EarlyStopping(patience=100) as requested.
    - LR schedule implemented as ExponentialDecay so that lr is multiplied by 0.7 each epoch.
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "SpectraNet53"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        batch_size = 16
        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            steps_per_epoch = int(np.ceil(len(y_fold_train_scaled) / batch_size))
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4,
                decay_steps=max(1, steps_per_epoch),
                decay_rate=0.7,
                staircase=False,
            )

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = build_spectranet53(
                    input_dim=int(X_fold_train_scaled.shape[1]),
                    input_channels=1,
                    l2_coeff=0.05,
                    dropout_rate=0.20,
                    use_parametric_gelu=True,
                    input_batchnorm=True,
                    name=model_name,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]

        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)

        steps_per_epoch = int(np.ceil(len(y_train_scaled) / batch_size))

        def _build_compile():
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4,
                decay_steps=max(1, steps_per_epoch),
                decay_rate=0.7,
                staircase=False,
            )
            model = build_spectranet53(
                input_dim=int(X_train_scaled.shape[1]),
                input_channels=1,
                l2_coeff=0.05,
                dropout_rate=0.20,
                use_parametric_gelu=True,
                input_batchnorm=True,
                name=model_name,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
            batch_size=batch_size,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


def run_spectranet32_all_datasets(dataset_filter: list[str] | None = None) -> None:
    """
    SpectraNet-32 is a regression residual CNN (SpectraNet32.pdf).

    Paper training details (SpectraNet32.pdf):
    - Optimizer: Adam (default)
    - Learning rate: 1e-3 with a drop multiplier of 0.90 each epoch
    - Epochs: 15
    - Batch size: 128
    - Kernel size: 21
    - Dropout: 0.25
    - L2 regularization (weight decay): 0.05
    - Gradient clipping by norm: 0.8
    - Preprocessing in paper: QNV + Savitzky-Golay + optional PLS-based feature selection

    Benchmark adaptations:
    - Uses the benchmark-standard StandardScaler preprocessing for X and y.
    - Uses 5-fold CV epoch selection + repeated final runs (N_FINAL_RUNS).
    - Max epochs capped at 600 + EarlyStopping(patience=100) during CV.
    """
    repo_root, datasets_root, dataset_config, dataset_names = _repo_and_datasets(dataset_filter)

    model_name = "SpectraNet32"
    model_results_dir = repo_root / "results" / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(dataset_names)} datasets. Benchmarking: {model_name}")

    summary_rows: list[dict] = []
    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")

        if RESUME_EXISTING_OUTPUTS and _dataset_outputs_complete(model_results_dir, dataset_name):
            existing = _load_existing_metrics_row(model_results_dir, dataset_name)
            if existing is not None:
                summary_rows.append(existing)
                print(f"Skipping (outputs already exist): {dataset_name}")
                continue

        X_train, y_train, X_test, y_test, wavelengths = load_dataset(
            dataset_name, dataset_config=dataset_config, datasets_root=datasets_root
        )
        print(
            f"Loaded: X_train={X_train.shape}, y_train={np.shape(y_train)}, "
            f"X_test={X_test.shape}, y_test={np.shape(y_test)}"
        )
        if wavelengths is not None:
            print(
                f"Wavelengths: n={len(wavelengths)}, "
                f"min={float(np.min(wavelengths)):.3f}, max={float(np.max(wavelengths)):.3f}"
            )

        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                f"{model_name} currently supports 1D targets; got "
                f"y_train shape {y_train.shape}, y_test shape {y_test.shape} for dataset {dataset_name}."
            )

        batch_size = 128

        def _make_lr_schedule(steps_per_epoch: int):
            return keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=max(1, steps_per_epoch),
                decay_rate=0.90,
                staircase=False,
            )

        def _fit_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> int:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            X_fold_train_scaled, X_fold_val_scaled = standardize_column(X_fold_train, X_fold_val)
            X_fold_train_scaled = X_fold_train_scaled.astype(np.float32)[..., np.newaxis]
            X_fold_val_scaled = X_fold_val_scaled.astype(np.float32)[..., np.newaxis]

            _, y_fold_train_scaled, y_fold_val_scaled = standardize_target(
                y_fold_train, y_fold_val
            )

            steps_per_epoch = int(np.ceil(len(y_fold_train_scaled) / batch_size))
            lr_schedule = _make_lr_schedule(steps_per_epoch)

            tf.keras.backend.clear_session()
            with tf.device(DEVICE_NAME):
                model = build_spectranet32(
                    input_dim=int(X_fold_train_scaled.shape[1]),
                    input_channels=1,
                    kernel_size=21,
                    l2_coeff=0.05,
                    dropout_rate=0.25,
                    use_parametric_gelu=True,
                    input_batchnorm=True,
                    name=model_name,
                )
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.8),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )
                history = model.fit(
                    X_fold_train_scaled,
                    y_fold_train_scaled,
                    validation_data=(X_fold_val_scaled, y_fold_val_scaled),
                    epochs=MAX_EPOCHS,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[make_early_stopping_callback()],
                )

            best_epoch = _best_epoch_from_history(history)
            del model
            del history
            tf.keras.backend.clear_session()
            gc.collect()
            return best_epoch

        selected_epochs, fold_epochs = select_epochs_via_cv(
            n_samples=int(len(y_train)),
            fit_fold_fn=_fit_fold,
        )
        print(f"{model_name}: CV best epochs={fold_epochs} -> selected={selected_epochs}")

        X_train_scaled, X_test_scaled = standardize_column(X_train, X_test)
        X_train_scaled = X_train_scaled.astype(np.float32)[..., np.newaxis]
        X_test_scaled = X_test_scaled.astype(np.float32)[..., np.newaxis]
        y_scaler, y_train_scaled, _ = standardize_target(y_train, y_test)
        steps_per_epoch = int(np.ceil(len(y_train_scaled) / batch_size))

        def _build_compile():
            model = build_spectranet32(
                input_dim=int(X_train_scaled.shape[1]),
                input_channels=1,
                kernel_size=21,
                l2_coeff=0.05,
                dropout_rate=0.25,
                use_parametric_gelu=True,
                input_batchnorm=True,
                name=model_name,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=_make_lr_schedule(steps_per_epoch),
                    clipnorm=0.8,
                ),
                loss="mse",
                metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
            )
            return model

        multi = _run_final_training_multi(
            build_compile_fn=_build_compile,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            y_scaler=y_scaler,
            y_train_scaled=y_train_scaled,
            selected_epochs=selected_epochs,
            batch_size=batch_size,
        )

        summary_rows.append({"dataset_name": dataset_name, **multi["summary_row"]})

        _write_outputs(
            model_name=model_name,
            model_results_dir=model_results_dir,
            dataset_name=dataset_name,
            y_train=y_train,
            y_test=y_test,
            y_train_pred=multi["last_y_train_pred"],
            y_test_pred=multi["last_y_test_pred"],
            history=multi["last_history"],
            summary_rows=summary_rows,
            loss_ylabel="Loss (MSE, standardized y)",
        )

        gc.collect()

    summary_path = model_results_dir / f"{model_name}_summary.csv"
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Summary: {summary_path}")


if __name__ == "__main__":
    # ---- Model registry (case-insensitive lookup) ----
    MODEL_REGISTRY = {
        "1dcnn_1l_3":        run_cnn_1d_1l_3_all_datasets,
        "1dcnn_1n_3":        run_cnn_1d_1n_3_all_datasets,
        "1dcnn_3_1":         run_cnn_1d_3_1_all_datasets,
        "deepspectra":       run_deepspectra_all_datasets,
        "ipa":               run_ipa_all_datasets,
        "1dinceptionresnet": run_inception_resnet_1d_all_datasets,
        "markspectra":       run_markspectra_all_datasets,
        "residualspectra":   run_residualspectra_all_datasets,
        "scnet":             run_scnet_all_datasets,
        "spectraformer":     run_spectraformer_all_datasets,
        "spectratr":         run_spectratr_all_datasets,
        "spectranet32":      run_spectranet32_all_datasets,
        "spectranet53":      run_spectranet53_all_datasets,
    }

    parser = argparse.ArgumentParser(
        description="NIRBENCH Deep Learning Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "Model to benchmark (case-insensitive). Default: 'all'.\n"
            "Available: " + ", ".join(sorted(MODEL_REGISTRY.keys()))
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Dataset name(s) to benchmark. Default: all datasets.\n"
            "Pass one or more dataset folder names, e.g.:\n"
            "  --dataset 1-Wheat_kernels_protein\n"
            "  --dataset 1-Wheat_kernels_protein 22-Mango_S1_dm"
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help=(
            "Skip model training entirely and only generate the final\n"
            "benchmark barplot from existing summary CSVs in results/."
        ),
    )
    args = parser.parse_args()

    if args.plot_only:
        print("--plot-only: skipping training, generating final benchmark plots.")
        build_global_model_ranking()
    else:
        # Resolve model selection
        model_key = args.model.strip().lower()
        if model_key == "all":
            selected_fns = list(MODEL_REGISTRY.values())
        elif model_key in MODEL_REGISTRY:
            selected_fns = [MODEL_REGISTRY[model_key]]
        else:
            parser.error(
                f"Unknown model '{args.model}'. "
                f"Choose from: all, {', '.join(sorted(MODEL_REGISTRY.keys()))}"
            )

        # Resolve dataset selection
        dataset_filter = args.dataset  # None means all

        print(f"Models to run: {[fn.__name__ for fn in selected_fns]}")
        print(f"Dataset filter: {dataset_filter or 'all'}")
        print()

        for fn in selected_fns:
            fn(dataset_filter=dataset_filter)

        build_global_model_ranking()
