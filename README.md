# NIRBENCH-DL v0.1

A reproducible benchmark for regression from near-infrared and visible-near-infrared spectra. It evaluates **13 deep learning architectures**, **TabPFN-3.5**, and a jointly optimized **partial least squares (PLS)** reference on **30 fixed train/test tasks**.

This clean release contains the executable benchmark, model definitions, shared data loader, fixed dataset partitions, environment files, and provenance documentation. Generated results, old script revisions, notebooks, papers, logs, and rendered model files are intentionally excluded.

## TabPFN-3.5 Hackathon submission

NIRBENCH-DL uses the TabPFN-3.5 regressor as a pretrained foundation-model baseline for high-dimensional spectral regression. It fits TabPFN on each complete training partition and evaluates the stored test partition over ten deterministic seeds. The project is submitted as a **benchmark harness / domain application** for the [Prior Labs TabPFN-3.5 Hackathon](https://platform.priorlabs.ai/hackathon-3.5).

**Pre-existing work disclosure:** the NIR benchmark, fixed dataset splits, PLS pipeline, and deep learning architecture adaptations existed before the hackathon. The TabPFN-3.5 integration, its repeated evaluation protocol, and the direct comparison with the spectroscopy models form the hackathon extension.

## Repository contents

```text
.
├── benchmark.py             # 13 TensorFlow/Keras models
├── benchmark_tabpfn.py      # TabPFN-3.5 regressor
├── pls_baseline.py          # preprocessing-aware PLS baseline
├── validate_setup.py        # dataset and source-tree validation
├── models/                  # model definitions
├── src/data_loading.py      # shared fixed-split loader
├── datasets/                # 30 train/test task folders
├── DATASETS.md              # provenance, task mapping, and data terms
├── MODEL_REFERENCES.md      # architecture citations
├── requirements.txt
└── environment.yml
```

## Benchmark mechanics

The test files are held out from all model selection.

### Deep learning models

1. Read the stored training and test partitions.
2. Within each of five shuffled training-only folds, fit feature and target `StandardScaler` objects on the fold-training rows.
3. Train a fresh model for at most 600 epochs with early stopping and record the epoch with the lowest validation loss.
4. Select the rounded median of the five best epochs.
5. Refit ten fresh models on the complete training partition for that fixed epoch count, using seeds 12345–12354.
6. Predict the untouched test partition and report the mean and sample standard deviation of RMSE and R². Metrics are returned to the original target units.

The source architectures retain their benchmark-specific optimizer and learning-rate settings. Test labels never set the epoch count, scaler parameters, or model parameters.

### TabPFN-3.5

TabPFN is pretrained, so it has no epoch-selection phase. Feature and target scalers are fitted on the complete training partition. Ten fits use the complete training data as context and vary the TabPFN ensemble seed (12345–12354). The test partition is used only for prediction and scoring. The standard TabPFN-3.5 checkpoint is selected explicitly so a future package default cannot silently change the model.

### PLS

Five-fold cross-validation on the training partition jointly selects:

- one of nine spectral preprocessing choices: raw, SNV, MSC, Savitzky–Golay first or second derivative, and the four SNV/MSC plus derivative combinations;
- 1–20 latent variables, subject to fold size and feature count.

Learned preprocessing state, including the MSC reference, is estimated inside each fold. The selected pair is refitted once on the complete training partition and evaluated on the test partition.

## Installation

Python 3.13 is the reference environment. An NVIDIA GPU is strongly recommended for the deep learning and TabPFN runs.

### Conda

```bash
conda env create -f environment.yml
conda activate nirbench-dl-v01
```

### `venv`

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and TensorFlow wheels are platform and CUDA dependent. If the pinned GPU wheels do not match your system, install the appropriate PyTorch and TensorFlow builds first, then install the remaining packages.

## Obtain the TabPFN-3.5 weights

The checkpoint is deliberately excluded because it has its own Prior Labs license. Install `tabpfn`, sign in to Prior Labs, and accept the TabPFN-3.5 model terms when prompted on first use. The standard checkpoint is named `tabpfn-v3.5-20260909.safetensors` and is normally stored in the TabPFN cache.

You may also pass its location explicitly:

```bash
python benchmark_tabpfn.py \
  --model-path /path/to/tabpfn-v3.5-20260909.safetensors \
  --device cuda
```

See the [official TabPFN installation instructions](https://github.com/PriorLabs/TabPFN#installation--setup) and the [TabPFN-3.5 model license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE). The weights permit research and limited internal evaluation and are not covered by this repository's MIT license.

## Validate the clean checkout

```bash
python validate_setup.py
```

The validator checks the 30 expected task directories, all 60 fixed split files, target dimensionality, train/test feature alignment, finite loaded values, model source files, and the recorded SHA-256 manifest.

## Run the benchmark

Run commands from the repository root.

### All 13 deep learning architectures on all datasets

```bash
python benchmark.py --model all
```

### One architecture or selected datasets

```bash
python benchmark.py --model deepspectra
python benchmark.py --model spectranet32 --dataset 1-Wheat_kernels_protein
python benchmark.py --model ipa --dataset 1-Wheat_kernels_protein 22-Mango_S1_dm
```

Model keys are:

```text
1dcnn_1l_3, 1dcnn_1n_3, 1dcnn_3_1, deepspectra, ipa,
1dinceptionresnet, markspectra, residualspectra, scnet,
spectraformer, spectratr, spectranet32, spectranet53
```

To resume and skip datasets that already have complete outputs:

```bash
# Linux/macOS
NIRBENCH_RESUME=1 python benchmark.py --model all

# Windows PowerShell
$env:NIRBENCH_RESUME="1"
python benchmark.py --model all
```

### TabPFN-3.5 on all datasets

```bash
python benchmark_tabpfn.py --device cuda
```

The command displays dataset and run progress bars. Add `--overwrite` to replace complete saved runs, or use `--dataset ...` for a subset.

### Optimized PLS on all datasets

```bash
python pls_baseline.py
```

For a quick end-to-end check:

```bash
python pls_baseline.py --dataset 1-Wheat_kernels_protein --max-lv 3
```

## Outputs

Deep learning and TabPFN outputs are written below `results/<model>/`:

- `<model>_summary.csv`: one row per completed dataset;
- `<dataset>_metrics.csv`: mean and standard deviation of repeated-run metrics;
- `<dataset>_preds.csv`: measured and predicted values;
- `<dataset>_plot.png`: prediction or training diagnostic plot;
- TabPFN additionally writes `<dataset>_runs.csv` and `<dataset>_protocol.json`.

The PLS command writes `pls_baseline_benchmark.cvs` and `pls_preprocessing_search.csv`. The `.cvs` spelling is retained for compatibility with the current report and website pipeline.

## Reproducibility and interpretation

- Stored train/test partitions are reused by every method.
- Scaling and preprocessing parameters are learned from training data only.
- RMSE is task-specific and remains in the original target units; do not average raw RMSE across tasks with different units.
- The cross-task general score ranks models by RMSE within each dataset, maps ranks to 0–100, and averages across shared datasets.
- Runtime is comparable within the published benchmark because all saved runs were obtained on the same workstation: Intel Core i9-13900K, NVIDIA GeForce RTX 2080 Ti, Windows 11/WSL2, and TensorFlow 2.20. Timed workloads differ by method and are recorded in the output schema.
- Some task configurations share source samples, instruments, or targets. They are benchmark tasks rather than 30 statistically independent studies.

## Current result snapshot

On the completed 30-task comparison, TabPFN-3.5 has the highest cross-task rank score (**94.5/100**), followed by optimized PLS (**87.4/100**), IPA (**73.1/100**), the wide-kernel 1D CNN (**69.0/100**), and DeepSpectra (**67.1/100**). PLS records the lowest RMSE on 16 tasks, TabPFN-3.5 on 12, DeepSpectra on one, and IPA on one. TabPFN-3.5 can lead the rank score while winning fewer tasks because it stays close to the top across nearly every task; the score rewards consistency over all within-task ranks rather than counting only first places.

These values describe the v0.1 saved experiment and can be regenerated from the per-model summary files produced by the commands above.

## Data, code, and model licenses

The MIT license covers the benchmark software authored for this repository. It does not relicense the included scientific datasets, TabPFN weights, or third-party packages. Dataset attribution and source links are in [DATASETS.md](DATASETS.md); architecture references are in [MODEL_REFERENCES.md](MODEL_REFERENCES.md). Users remain responsible for the terms attached to each original dataset and to the TabPFN-3.5 checkpoint.

## Citation

Use [CITATION.cff](CITATION.cff) to cite this software, and cite the original dataset and model papers relevant to your use. For TabPFN-3.5, see the [technical report](https://arxiv.org/abs/2609.17895) and the foundational TabPFN paper: Hollmann et al., *Nature* 637, 319–326 (2025), [doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6).
