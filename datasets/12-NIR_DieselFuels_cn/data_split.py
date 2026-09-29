#!/usr/bin/env python3
"""Download the original source and recreate this benchmark task.

Run this file from the repository root or from its dataset directory.
The adjacent notebook contains the same workflow for interactive use.
"""

TASK_FOLDER = "12-NIR_DieselFuels_cn"
INCLUDE_INDEX = True
COMPARISON_ATOL = 1e-12
SOURCE_URL = "https://eigenvector.com/wp-content/uploads/2019/06/SWRI_Diesel_NIR_CSV.zip"


from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import io
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy.io import loadmat


def find_task_dir(folder_name):
    """Support running from either the task directory or repository root."""
    candidates = [Path.cwd(), Path.cwd() / "datasets" / folder_name]
    for candidate in candidates:
        if candidate.name == folder_name and (candidate / "download_data.md").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Run this script from {folder_name!r} or from the repository root."
    )


def download(url, destination):
    """Download a source file without retaining it in the repository."""
    request = Request(url, headers={"User-Agent": "NIRBENCH-DL/0.1 dataset preparation"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    print(f"Downloaded {destination.name} ({destination.stat().st_size:,} bytes)")


def csv_roundtrip(frame, include_index):
    """Represent a generated frame as it will appear after CSV serialization."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=include_index, lineterminator="\r\n")
    buffer.seek(0)
    return pd.read_csv(buffer)


def verify_and_write(train_frame, test_frame, include_index, atol):
    """Compare with published partitions when present, then write both CSV files."""
    results = []
    for split, generated in (("train", train_frame), ("test", test_frame)):
        output_path = TASK_DIR / f"data_{split}.csv"
        serialized = csv_roundtrip(generated, include_index)

        if output_path.exists():
            published = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(
                serialized,
                published,
                check_dtype=False,
                check_exact=False,
                rtol=0,
                atol=atol,
            )
            max_delta = np.max(
                np.abs(
                    serialized.select_dtypes(include="number").to_numpy(dtype=float)
                    - published.select_dtypes(include="number").to_numpy(dtype=float)
                )
            )
            print(
                f"{split}: matches the published partition "
                f"(shape={serialized.shape}, maximum absolute difference={max_delta:.3g})"
            )
        else:
            print(f"{split}: no published CSV was present; generated shape={serialized.shape}")

        generated.to_csv(
            output_path,
            index=include_index,
            lineterminator="\r\n",
        )
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        results.append({
            "partition": split,
            "rows": len(generated),
            "spectral_variables": generated.shape[1] - 1,
            "sha256": digest,
        })

    return pd.DataFrame(results)


TASK_DIR = find_task_dir(TASK_FOLDER)
temporary_download = tempfile.TemporaryDirectory(prefix=f"nirbench_{TASK_FOLDER}_")
SOURCE_DIR = Path(temporary_download.name)
print(f"Preparing task {TASK_FOLDER}")


from sklearn.model_selection import train_test_split

archive_path = SOURCE_DIR / "SWRI_Diesel_NIR_CSV.zip"
download(SOURCE_URL, archive_path)
with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(SOURCE_DIR)

spectra_path = next(SOURCE_DIR.rglob("diesel_spec.csv"))
properties_path = next(SOURCE_DIR.rglob("diesel_prop.csv"))

# The Eigenvector CSV files contain metadata before their tabular headers.
spectra = pd.read_csv(spectra_path, skiprows=9)
properties = pd.read_csv(properties_path, skiprows=8)
assert len(spectra) == len(properties) == 784
assert spectra.iloc[:, 1].astype(str).equals(properties.iloc[:, 1].astype(str))

all_data = spectra.iloc[:, 2:-1].copy()  # 750–1550 nm; omit empty and label columns
all_data = all_data.assign(CN=properties["CN"])

# Preserve the historical order of operations: split all rows, then remove missing CN.
train_data, test_data = train_test_split(
    all_data,
    test_size=0.20,
    random_state=123,
)
train_data = train_data.dropna(subset=["CN"])
test_data = test_data.dropna(subset=["CN"])

assert train_data.shape == (298, 402)
assert test_data.shape == (83, 402)
print("Applied split-before-filter preparation and retained 401 spectral variables.")


summary = verify_and_write(
    train_data,
    test_data,
    include_index=INCLUDE_INDEX,
    atol=COMPARISON_ATOL,
)
print(summary.to_string(index=False))
temporary_download.cleanup()
print("Temporary source files removed. Preparation complete.")
