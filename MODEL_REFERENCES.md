# Model references

The benchmark adapts the following spectral architectures to single-output regression and evaluates three deliberately simple 1D CNN baselines.

| Implementation | Reference |
|---|---|
| `models/DeepSpectra.py` | Zhang et al. (2019), *DeepSpectra: An end-to-end deep learning approach for quantitative spectral analysis*, [doi:10.1016/j.aca.2019.01.002](https://doi.org/10.1016/j.aca.2019.01.002) |
| `models/IPA.py` | Haffner et al. (2025), *IPA: A deep CNN based on Inception for Petroleum Analysis*, [doi:10.1016/j.fuel.2024.133016](https://doi.org/10.1016/j.fuel.2024.133016) |
| `models/1DInceptionResnet.py` | Tan et al. (2023), *1D-inception-resnet for NIR quantitative analysis and its transferability between different spectrometers*, [doi:10.1016/j.infrared.2023.104559](https://doi.org/10.1016/j.infrared.2023.104559) |
| `models/MarkSpectra.py` | Wang et al. (2022), *Mark-Spectra*, [doi:10.1016/j.compag.2021.106624](https://doi.org/10.1016/j.compag.2021.106624) |
| `models/ResidualSpectra.py` | Wang et al. (2020), *End-to-end analysis modeling of vibrational spectroscopy based on deep learning approach*, [doi:10.1002/cem.3291](https://doi.org/10.1002/cem.3291) |
| `models/SCNet.py` | Li et al. (2023), *SCNet: A deep learning network framework for analyzing near-infrared spectroscopy using short-cut*, [doi:10.1016/j.infrared.2023.104731](https://doi.org/10.1016/j.infrared.2023.104731) |
| `models/Spectraformer.py` | Chen, Zhou and Ren (2024), *Spectraformer*, [doi:10.1039/D3RA07708J](https://doi.org/10.1039/D3RA07708J) |
| `models/SpectraTr.py` | Fu et al. (2022), *SpectraTr*, [doi:10.1142/S1793545822500213](https://doi.org/10.1142/S1793545822500213) |
| `models/SpectraNet32.py` | Martins et al. (2023), *SpectraNet-32*, [doi:10.1016/j.postharvbio.2023.112281](https://doi.org/10.1016/j.postharvbio.2023.112281) |
| `models/SpectraNet53.py` | Martins et al. (2022), *SpectraNet-53*, [doi:10.1016/j.compag.2022.106945](https://doi.org/10.1016/j.compag.2022.106945) |
| `models/CNN_1D_1L_3.py` | Simple one-layer wide-kernel benchmark baseline |
| `models/CNN_1D_1N_3.py` | Simple one-layer narrow-kernel benchmark baseline |
| `models/CNN_1D_3_1.py` | Simple three-block 1D CNN benchmark baseline |
| `benchmark_tabpfn.py` | Hollmann et al. (2025), *Accurate predictions on small data with a tabular foundation model*, [doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6); [TabPFN-3.5 technical report](https://arxiv.org/abs/2609.17895) |
| `pls_baseline.py` | Wold, Sjöström and Eriksson (2001), *PLS-regression: a basic tool of chemometrics*, [doi:10.1016/S0169-7439(01)00155-1](https://doi.org/10.1016/S0169-7439(01)00155-1) |

The papers describe the originating designs. The files in this repository contain benchmark-specific regression heads, input-shape handling, scaling, and training protocol adaptations.
