"""Validate the NIRBENCH-DL source tree and fixed dataset partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from src.data_loading import discover_datasets, load_dataset

ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = ROOT / "datasets"
MANIFEST = DATASETS_ROOT / "manifest.csv"
EXPECTED_MODELS = {
    "1DInceptionResnet.py",
    "CNN_1D_1L_3.py",
    "CNN_1D_1N_3.py",
    "CNN_1D_3_1.py",
    "DeepSpectra.py",
    "IPA.py",
    "MarkSpectra.py",
    "ResidualSpectra.py",
    "SCNet.py",
    "SpectraNet32.py",
    "SpectraNet53.py",
    "SpectraTr.py",
    "Spectraformer.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip SHA-256 checks (dataset shape and finite-value checks still run).",
    )
    args = parser.parse_args()

    errors: list[str] = []
    config = discover_datasets(DATASETS_ROOT)
    names = sorted(config, key=lambda name: int(name.split("-", 1)[0]))
    if len(names) != 30:
        errors.append(f"expected 30 datasets, discovered {len(names)}")

    csv_files = sorted(DATASETS_ROOT.glob("*/data_*.csv"))
    if len(csv_files) != 60:
        errors.append(f"expected 60 train/test CSV files, found {len(csv_files)}")

    manifest_rows: dict[str, dict[str, str]] = {}
    if not MANIFEST.is_file():
        errors.append(f"missing manifest: {MANIFEST.relative_to(ROOT)}")
    else:
        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            manifest_rows = {row["path"]: row for row in csv.DictReader(stream)}
        if len(manifest_rows) != 60:
            errors.append(f"expected 60 manifest rows, found {len(manifest_rows)}")

    for index, name in enumerate(names, start=1):
        try:
            X_train, y_train, X_test, y_test, wavelengths = load_dataset(
                name, dataset_config=config, datasets_root=DATASETS_ROOT
            )
            if X_train.ndim != 2 or X_test.ndim != 2:
                raise ValueError("feature arrays are not two-dimensional")
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError("train/test spectral widths differ")
            if np.asarray(y_train).ndim != 1 or np.asarray(y_test).ndim != 1:
                raise ValueError("target is not one-dimensional")
            if len(X_train) != len(y_train) or len(X_test) != len(y_test):
                raise ValueError("feature/target row counts differ")
            if not all(
                np.isfinite(array).all()
                for array in (X_train, y_train, X_test, y_test)
            ):
                raise ValueError("loaded arrays contain non-finite values")
            if wavelengths is not None and len(wavelengths) != X_train.shape[1]:
                raise ValueError("wavelength count differs from spectral width")
            print(
                f"[{index:02d}/30] {name}: "
                f"train={X_train.shape}, test={X_test.shape}"
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if not args.skip_hashes:
        for path in csv_files:
            relative = path.relative_to(ROOT).as_posix()
            expected = manifest_rows.get(relative)
            if expected is None:
                errors.append(f"manifest has no row for {relative}")
                continue
            actual = sha256(path)
            if actual != expected["sha256"]:
                errors.append(f"SHA-256 mismatch: {relative}")

    present_models = {path.name for path in (ROOT / "models").glob("*.py")} - {"__init__.py"}
    missing_models = EXPECTED_MODELS - present_models
    extra_models = present_models - EXPECTED_MODELS
    if missing_models:
        errors.append(f"missing model files: {sorted(missing_models)}")
    if extra_models:
        errors.append(f"unexpected model files: {sorted(extra_models)}")

    for entry_point in (
        "benchmark.py",
        "benchmark_newmodel.py",
        "benchmark_tabpfn.py",
        "pls_baseline.py",
    ):
        if not (ROOT / entry_point).is_file():
            errors.append(f"missing entry point: {entry_point}")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nValidation passed: 30 datasets, 60 fixed splits, 13 model modules.")


if __name__ == "__main__":
    main()
