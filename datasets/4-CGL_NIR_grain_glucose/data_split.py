#!/usr/bin/env python3
"""Download the original source and recreate this benchmark task.

Run this file from the repository root or from its dataset directory.
The adjacent notebook contains the same workflow for interactive use.
"""

TASK_FOLDER = "4-CGL_NIR_grain_glucose"
INCLUDE_INDEX = False
# The upstream MATLAB archive and the published CSV differ only in the final
# decimal places of some values (maximum observed difference below 4e-9).
COMPARISON_ATOL = 1e-8
SOURCE_URL = "https://eigenvector.com/wp-content/uploads/2021/04/CGL_nir.mat_.zip"


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


archive_path = SOURCE_DIR / "CGL_nir.zip"
download(SOURCE_URL, archive_path)
with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(SOURCE_DIR)

mat_path = next(SOURCE_DIR.rglob("CGL_nir.mat"))
source = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

wavelengths = [str(value) for value in np.asarray(source["Xcal"].axisscale[1, 0]).ravel()]
response_labels = [str(value) for value in np.asarray(source["Ycal"].label[1, 0]).ravel()]
glucose_column = next(i for i, label in enumerate(response_labels) if label.strip() == "Glucose (wt %)")
target_name = response_labels[glucose_column]

train_data = pd.DataFrame(source["Xcal"].data, columns=wavelengths)
train_data = train_data.assign(**{target_name: source["Ycal"].data[:, glucose_column]})

test_data = pd.DataFrame(source["Xtest"].data, columns=wavelengths)
test_data = test_data.assign(**{target_name: source["Ytest"].data[:, glucose_column]})

assert train_data.shape == (153, 118)
assert test_data.shape == (78, 118)
print("Selected Glucose (wt %) and preserved the supplied calibration/test partition.")


summary = verify_and_write(
    train_data,
    test_data,
    include_index=INCLUDE_INDEX,
    atol=COMPARISON_ATOL,
)
print(summary.to_string(index=False))
temporary_download.cleanup()
print("Temporary source files removed. Preparation complete.")
