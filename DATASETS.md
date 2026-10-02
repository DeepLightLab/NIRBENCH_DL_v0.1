# Dataset provenance and fixed tasks

NIRBENCH-DL contains 30 regression task configurations derived from public research datasets and CEOT-UAlg measurements. The files are harmonized copies with fixed train/test partitions. Original splits were preserved when supplied; otherwise the curated data were split and frozen so every model receives identical rows.

Each task uses `data_train.csv` and `data_test.csv`. For tasks 2, 4, 11, 12, and 14, these files must first be recreated from their original public sources with the supplied Python scripts.[^recreate-data] Spectral columns are identified from wavelength-like headers and the last remaining column is the target. Some source exports contain a leading serialized index or sample identifier and some do not; `src/data_loading.py` removes a leading `Unnamed` or non-wavelength identifier when present.

## Sources

| Tasks | Material and target | Original source |
|---|---|---|
| 1 | Wheat kernels, protein | University of Copenhagen chemometrics datasets; Nielsen et al. (2003), [doi:10.1094/CCHEM.2003.80.3.274](https://doi.org/10.1094/CCHEM.2003.80.3.274) |
| 2 | Wheat flours, protein | D. Bertrand / INRA, distributed through [ChemProject ChemData](https://www.chemproject.org/chemdata) |
| 3 | Tecator meat, moisture | Karin Thente / Tecator AB, [StatLib Tecator dataset](https://lib.stat.cmu.edu/datasets/tecator) |
| 4 | Grain, glucose | T. Næs and T. Isaksson, [Eigenvector datasets](https://eigenvector.com/resources/data-sets/) |
| 5 | Cucurbitaceae fruit, soluble solids | Kusumiyati et al., Mendeley Data, [doi:10.17632/k55b8mvs84.2](https://doi.org/10.17632/k55b8mvs84.2) |
| 6–10 | Five tomato groups, soluble solids | Ibañez et al., Zenodo, [doi:10.5281/zenodo.10633732](https://doi.org/10.5281/zenodo.10633732) |
| 11 | Olive oils, C16:0 | N. Dupuy group / University of Aix-Marseille, distributed through [ChemProject ChemData](https://www.chemproject.org/chemdata) |
| 12 | Diesel fuels, cetane number | Scott Hutzler / Southwest Research Institute, [Eigenvector SWRI dataset](https://www.eigenvector.com/data/SWRI/index.html#csv) |
| 13 | Milk, protein | Díaz-Olivares et al., Zenodo, [doi:10.5281/zenodo.8263430](https://doi.org/10.5281/zenodo.8263430) |
| 14 | Pharmaceutical tablets, assay | IDRC 2002 shootout, spectrometer 1, [Eigenvector datasets](https://eigenvector.com/resources/data-sets/) |
| 15 | Rocha pear 2021, soluble solids | Dário Passos / CEOT–UAlg; Cruz et al. (2021), [doi:10.1016/j.postharvbio.2021.111562](https://doi.org/10.1016/j.postharvbio.2021.111562); released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 16 | Rocha pear 2019, soluble solids | Dário Passos / CEOT–UAlg; Passos et al. (2019), [doi:10.3390/s19235165](https://doi.org/10.3390/s19235165); released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 17–18, 26–29 | Barley, corn, and wheat measured with Perten instruments; moisture or protein | sensAIfood / M. Lagerholm, Zenodo, [doi:10.5281/zenodo.15838136](https://doi.org/10.5281/zenodo.15838136) |
| 19–20 | Wheat measured with Grainit/AuroraNIR; moisture or protein | sensAIfood / P. Berzaghi, Zenodo, [doi:10.5281/zenodo.15838272](https://doi.org/10.5281/zenodo.15838272) |
| 21 | Rocha pear 2021, trimmed and SNV-transformed spectra, soluble solids | Adapted from task 15 under the same [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) authorization; [doi:10.1016/j.postharvbio.2021.111562](https://doi.org/10.1016/j.postharvbio.2021.111562) |
| 22–25 | Mango harvest seasons S1–S4, dry matter | Anderson et al., Mendeley Data v2, [doi:10.17632/46htwnp833.5](https://doi.org/10.17632/46htwnp833.5) |
| 30 | Wheat measured with a FOSS NIRSYSTEM-5000, protein | sensAIfood / CRA-W, Zenodo, [doi:10.5281/zenodo.16108496](https://doi.org/10.5281/zenodo.16108496) |

## Exact task directories

```text
1-Wheat_kernels_protein
2-Wheat_flours_protein
3-Tecator_moisture
4-CGL_NIR_grain_glucose
5-Cucurbitaceae_fruit_ssc
6-NIR_tomato1_ssc
7-NIR_tomato2_ssc
8-NIR_tomato3_ssc
9-NIR_tomato4_ssc
10-NIR_tomato5_ssc
11-Olive_oils_c16
12-NIR_DieselFuels_cn
13-NIR_milk_protein
14-NIR_Pharmaceutical_Tablets_assay
15-CEOT_pear_2021_brix
16-CEOT_pears_2019_brix
17-Barley_sensAIfood_Perten_moisture
18-Barley_sensAIfood_Perten_protein
19-Wheat_sensAIfood_Grainit_moisture
20-Wheat_sensAIfood_Grainit_protein
21-CEOT_pear_2021_snv_brix
22-Mango_S1_dm
23-Mango_S2_dm
24-Mango_S3_dm
25-Mango_S4_dm
26-Corn_sensAIfood_Perten_moisture
27-Corn_sensAIfood_Perten_protein
28-Wheat_sensAIfood_Perten_moisture
29-Wheat_sensAIfood_Perten_protein
30-Wheat_sensAIfood_CRAW_protein
```

## Integrity and reuse

`datasets/manifest.csv` records the row count, loaded spectral width, target name, byte size, and SHA-256 digest of every stored partition. Run `python validate_setup.py` after cloning to verify the copy.

The repository's Apache License 2.0 applies to software and does not override dataset terms. See [DATA_LICENSES.md](DATA_LICENSES.md) for the license or permission basis, redistribution status, and recorded transformations for each task. Cite the original source for every dataset you use. The CEOT–UAlg datasets in tasks 15, 16, and 21 are released under CC BY 4.0; their rights statement is in `datasets/CEOT_DATA_LICENSE.md`. Other files remain subject to the licenses and attribution requirements published by their source repositories.

[^recreate-data]: From the repository root, run the following commands after installing `requirements.txt`. Each script downloads its authoritative source, recreates the benchmark's fixed train/test partition, and writes the two CSV files in its task directory. An equivalent `data_split.ipynb` is retained beside each script for users who prefer Jupyter.

    ```bash
    python datasets/1-Wheat_kernels_protein/data_split.py
    python datasets/2-Wheat_flours_protein/data_split.py
    python datasets/4-CGL_NIR_grain_glucose/data_split.py
    python datasets/11-Olive_oils_c16/data_split.py
    python datasets/12-NIR_DieselFuels_cn/data_split.py
    python datasets/14-NIR_Pharmaceutical_Tablets_assay/data_split.py
    ```
