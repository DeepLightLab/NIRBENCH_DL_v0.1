"""Prepare the Mango V2 grouped seasonal partitions; companion to data_split.ipynb."""

from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

# Run from the repository root or its temp folder.
SOURCE_FILENAME = "NAnderson2020MendeleyMangoNIRData.csv"
SEED = 42
TEST_GROUP_FRACTION = 0.20
SPECTRAL_COLUMNS = [str(w) for w in range(750, 1051, 3)]
MODEL_COLUMNS = SPECTRAL_COLUMNS + ["DM"]

source_candidates = [Path.cwd() / SOURCE_FILENAME,
                     Path.cwd() / "temp" / SOURCE_FILENAME]
SOURCE_PATH = next((p.resolve() for p in source_candidates if p.is_file()), None)
if SOURCE_PATH is None:
    raise FileNotFoundError(f"Place {SOURCE_FILENAME} in temp and run from temp or the repository root.")
ROOT = SOURCE_PATH.parent.parent
OUTPUT_ROOT = SOURCE_PATH.parent / "mango_v2_grouped"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_SHA256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
print(f"Source: {SOURCE_PATH.name}")
print(f"Output: {OUTPUT_ROOT}")


source = pd.read_csv(SOURCE_PATH)
required = set(MODEL_COLUMNS + ["Season"])
missing = required - set(source.columns)
if missing:
    raise ValueError(f"Missing source columns: {sorted(missing)}")
if source["Season"].isna().any() or set(source["Season"].unique()) != {1, 2, 3, 4}:
    raise ValueError("Expected exactly four populated seasons, labelled 1 through 4.")
if not np.isfinite(source[MODEL_COLUMNS].to_numpy(dtype=float)).all():
    raise ValueError("Spectra and DM must be finite; no rows are silently dropped.")

# Split by Season rather than calendar year: each season spans two calendar years.
season_frames = {
    season: source.loc[source["Season"] == season].copy()
    for season in range(1, 5)
}
print(f"Source: {len(source):,} rows; {len(SPECTRAL_COLUMNS)} retained spectral variables (750–1050 nm).")
print(source.groupby("Season").size().rename("spectra").to_string())


def group_and_split(season_source, season):
    """Keep every equal-DM group within a season wholly in one partition."""
    # Factorization uses first appearance, not target magnitude or desired scores.
    # All exact DM matches are grouped, even if they might be distinct fruits.
    labels, _ = pd.factorize(season_source["DM"], sort=False)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_GROUP_FRACTION, random_state=SEED
    )
    train_positions, test_positions = next(
        splitter.split(season_source, groups=labels)
    )
    train_groups = set(labels[train_positions])
    test_groups = set(labels[test_positions])
    assert train_groups.isdisjoint(test_groups)
    assert set(train_positions).isdisjoint(test_positions)
    assert set(train_positions) | set(test_positions) == set(range(len(season_source)))

    train = season_source.iloc[train_positions][MODEL_COLUMNS].copy()
    test = season_source.iloc[test_positions][MODEL_COLUMNS].copy()
    shared_dm = set(train["DM"]) & set(test["DM"])
    assert not shared_dm, "A repeated reference value was split across partitions."

    # Exact row checks complement the inferred fruit grouping.
    train_spectra = set(map(tuple, train[SPECTRAL_COLUMNS].to_numpy()))
    test_spectra = set(map(tuple, test[SPECTRAL_COLUMNS].to_numpy()))
    shared_spectra = len(train_spectra & test_spectra)
    assert shared_spectra == 0, "Identical spectra occur in both partitions."

    partition = np.full(len(season_source), "train", dtype=object)
    partition[test_positions] = "test"
    metadata_columns = [c for c in ["Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp", "Set", "DM"]
                        if c in season_source.columns]
    membership = season_source[metadata_columns].copy()
    membership.insert(0, "source_row", season_source.index.to_numpy())
    membership.insert(1, "group_id", [f"S{season}_G{g:05d}" for g in labels])
    membership.insert(2, "partition", partition)
    assert membership.groupby("group_id")["partition"].nunique().max() == 1

    sizes = pd.Series(labels).value_counts()
    stats = {
        "task": f"{season + 21}-Mango_S{season}_dm",
        "season": season,
        "spectra": len(season_source),
        "groups": len(sizes),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_row_fraction": len(test) / len(season_source),
        "shared_groups": len(train_groups & test_groups),
        "shared_dm_values": len(shared_dm),
        "identical_spectra_across_splits": shared_spectra,
        "seed": SEED,
        "test_group_fraction": TEST_GROUP_FRACTION,
    }
    return train, test, membership, stats


partitions = {}
summaries = []
for season, seasonal in season_frames.items():
    train, test, membership, stats = group_and_split(seasonal, season)
    partitions[season] = {"full": seasonal[MODEL_COLUMNS].copy(),
                          "train": train, "test": test, "membership": membership}
    summaries.append(stats)

summary = pd.DataFrame(summaries)
print(summary[["task", "spectra", "groups", "train_rows", "test_rows",
               "shared_groups", "shared_dm_values", "identical_spectra_across_splits"]].to_string(index=False))
print("All repeated-DM groups are confined to one partition; zero shared DM values and identical spectra.")


file_records = []
for season, data in partitions.items():
    task = f"{season + 21}-Mango_S{season}_dm"
    directory = OUTPUT_ROOT / task
    directory.mkdir(parents=True, exist_ok=True)
    # Repeated runs deterministically replace files only in this output folder.
    for split in ("full", "train", "test"):
        path = directory / f"data_{split}.csv"
        data[split].to_csv(path, index=False, lineterminator="\n")
        stored = pd.read_csv(path)
        pd.testing.assert_frame_equal(
            stored, data[split].reset_index(drop=True),
            check_dtype=False, check_exact=False, atol=1e-12, rtol=0,
        )
        file_records.append({
            "task": task, "split": split,
            "path": path.relative_to(OUTPUT_ROOT).as_posix(),
            "rows": len(stored), "spectral_features": len(SPECTRAL_COLUMNS),
            "target_column": "DM", "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    membership_path = directory / "sample_groups.csv"
    data["membership"].to_csv(membership_path, index=False)
    # Recheck the saved partitions rather than relying only on in-memory results.
    saved_train = pd.read_csv(directory / "data_train.csv")
    saved_test = pd.read_csv(directory / "data_test.csv")
    assert set(saved_train["DM"]).isdisjoint(saved_test["DM"])
    assert set(map(tuple, saved_train[SPECTRAL_COLUMNS].to_numpy())).isdisjoint(
        set(map(tuple, saved_test[SPECTRAL_COLUMNS].to_numpy()))
    )
    stored_membership = pd.read_csv(membership_path)
    assert stored_membership.groupby("group_id")["partition"].nunique().max() == 1

summary.to_csv(OUTPUT_ROOT / "split_summary.csv", index=False)
pd.DataFrame(file_records).to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
protocol = {
    "source_file": SOURCE_PATH.name, "source_sha256": SOURCE_SHA256,
    "source_rows": len(source), "seasons": [1, 2, 3, 4],
    "grouping": "All identical exact DM values within a season form one conservative inferred fruit group.",
    "identity_limit": "V2 contains no explicit fruit identifier. Equal DM is a proxy, not a verified physical identity.",
    "splitter": "sklearn.model_selection.GroupShuffleSplit",
    "random_state": SEED, "test_group_fraction": TEST_GROUP_FRACTION,
    "spectral_columns": SPECTRAL_COLUMNS, "target": "DM",
    "source_set_labels_used_for_split": False,
    "grouped_cv_required": "Use group_id from sample_groups.csv for future within-training validation folds.",
    "python": platform.python_version(), "numpy": np.__version__,
    "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
}
(OUTPUT_ROOT / "split_protocol.json").write_text(
    json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
)
print("Wrote four seasonal datasets, four train/test pairs, group membership, and audit files.")
print(f"Output folder: {OUTPUT_ROOT}")
