# Dataset licenses and redistribution status

## Scope

The Apache License 2.0 in this repository covers the NIRBENCH-DL software authored for the project. It does not relicense scientific datasets, TabPFN model weights, or third-party packages. Each dataset remains subject to the terms set by its original rights holder.

This document records the source, license or permission basis, attribution, repository handling, and benchmark transformations for the 30 tasks. It does not replace the original license or permission notice. The original terms and any specific written redistribution permissions control; this document records them without assigning a new license.

`Source download required` means that NIRBENCH-DL does not claim a right to redistribute the source data. Users must obtain the data from the linked provider and prepare the benchmark files locally. Written redistribution permission remains pending for tasks 4, 12, and 14.

The processed partitions for task 1 may be redistributed in NIRBENCH-DL under written permission provided by Rasmus Bro of the University of Copenhagen Chemometrics Group. The original website and associated publication must be acknowledged. No standard dataset license was specified, and the repository's Apache License 2.0 does not apply to these data.

The processed partitions for tasks 2 and 11 may be redistributed in NIRBENCH-DL under written permission from Jean-Michel Roger, Team COMiC / ChemHouse, in response to a request for public redistribution of processed ChemProject datasets. The permission record is in [CHEMHOUSE_PERMISSION.md](CHEMHOUSE_PERMISSION.md). The original creators, source links, and benchmark modifications remain documented. No standard dataset license was specified, and Apache License 2.0 does not apply to these data.

Tasks 15, 16, and 21 are CEOT–UAlg measurements released by Dário Passos, the data rights holder, under CC BY 4.0. The rights statement and attribution instructions are in `datasets/CEOT_DATA_LICENSE.md`.

## Summary

| Tasks | Dataset family | Original source | License or permission basis | Repository handling |
|---|---|---|---|---|
| 1 | Wheat kernels, protein | [University of Copenhagen Chemometrics](https://ucphchemometrics.com/datasets/); Nielsen et al. (2003), [doi:10.1094/CCHEM.2003.80.3.274](https://doi.org/10.1094/CCHEM.2003.80.3.274) | Written permission from Rasmus Bro to redistribute the processed benchmark CSV files with source and publication attribution; no standard dataset license specified | Included with written permission and attribution |
| 2 | Wheat flours, protein | D. Bertrand / INRA, [ChemProject ChemData](https://www.chemproject.org/chemdata) | Written redistribution permission from Jean-Michel Roger / ChemHouse; no standard dataset license specified | Included with written permission and source attribution; see [CHEMHOUSE_PERMISSION.md](CHEMHOUSE_PERMISSION.md) |
| 3 | Tecator meat, moisture | Karin Thente / Tecator AB, [StatLib Tecator](https://lib.stat.cmu.edu/datasets/tecator) | Public domain; redistribution permitted when the complete permission note is attached | Included; see `datasets/3-Tecator_moisture/TECATOR_PERMISSION.txt` |
| 4 | CGL mixture design, glucose | Tormod Næs and Tomas Isaksson, [Eigenvector datasets](https://eigenvector.com/resources/data-sets/) | No dataset-specific redistribution license was identified | **Source download required; permission pending** |
| 5 | Cucurbitaceae fruit, soluble solids | Kusumiyati et al., [Mendeley Data](https://doi.org/10.17632/k55b8mvs84.2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with attribution; converted and partitioned as described in `DATASETS.md` |
| 6–10 | Tomato groups, soluble solids | Ibañez et al., [Zenodo](https://doi.org/10.5281/zenodo.10633732) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Included as adapted material under CC BY-SA 4.0; spreadsheet groups converted to fixed CSV tasks |
| 11 | Olive oils, C16:0 | N. Dupuy group / Aix-Marseille University, [ChemProject ChemData](https://www.chemproject.org/chemdata) | Written redistribution permission from Jean-Michel Roger / ChemHouse; no standard dataset license specified | Included with written permission and source attribution; see [CHEMHOUSE_PERMISSION.md](CHEMHOUSE_PERMISSION.md) |
| 12 | Diesel fuels, cetane number | Southwest Research Institute, distributed by [Eigenvector](https://eigenvector.com/resources/data-sets/) | No dataset-specific redistribution license was identified | **Source download required; permission pending** |
| 13 | Milk, protein | Díaz-Olivares et al., [Zenodo](https://doi.org/10.5281/zenodo.8263430) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with attribution; protein target selected, incomplete targets removed, and fixed partitions stored |
| 14 | IDRC 2002 pharmaceutical tablets, assay | International Diffuse Reflectance Conference, distributed by [Eigenvector](https://eigenvector.com/resources/data-sets/) | No dataset-specific redistribution license was identified | **Source download required; permission pending** |
| 15 | CEOT pear 2021, soluble solids | Dário Passos / CEOT–UAlg; Cruz et al. (2021), [doi:10.1016/j.postharvbio.2021.111562](https://doi.org/10.1016/j.postharvbio.2021.111562) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with authorization from the data rights holder |
| 16 | CEOT pear 2019, soluble solids | Dário Passos / CEOT–UAlg; Passos et al. (2019), [doi:10.3390/s19235165](https://doi.org/10.3390/s19235165) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with authorization from the data rights holder |
| 17–18, 26–29 | sensAIfood Perten cereals | sensAIfood / M. Lagerholm, [Zenodo](https://doi.org/10.5281/zenodo.15838136) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with attribution; target-specific rows selected and fixed partitions stored |
| 19–20 | sensAIfood Grainit wheat | sensAIfood / P. Berzaghi, [Zenodo](https://doi.org/10.5281/zenodo.15838272) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with attribution; target-specific rows selected and fixed partitions stored |
| 21 | CEOT pear 2021, trimmed/SNV spectra | Same source as task 15 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Adapted version of task 15; included under the same authorization |
| 22–25 | Mango harvest seasons S1–S4, dry matter | Anderson et al., [Mendeley Data](https://doi.org/10.17632/46htwnp833.5) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with attribution; seasons separated into tasks and fixed partitions stored |
| 30 | sensAIfood CRA-W wheat, protein | sensAIfood / CRA-W, [Zenodo](https://doi.org/10.5281/zenodo.16108496) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Included with attribution; wheat/protein subset selected and fixed partitions stored |

## Datasets included with written redistribution permission

### Task 1 — Wheat kernels, protein

- **Original dataset:** Wheat kernels — NIR/NIT single-seed wheat kernels
- **Creators/source:** University of Copenhagen Chemometrics Group; Nielsen et al. (2003)
- **Source page:** <https://ucphchemometrics.com/datasets/>
- **Official archive:** <https://sid.erda.dk/share_redirect/dLQ6VHNshw/Wheat%20kernels%20.zip>
- **Benchmark modifications:** the spectral matrix and protein reference were extracted from the MATLAB source; 100 spectral variables and the protein target were retained; the source `Calibration` partition was preserved as the 415-row benchmark training set and the source `Validation` partition as the 108-row benchmark test set; the result was converted to CSV and the target was placed in the final column.
- **Permission basis:** Rasmus Bro of the University of Copenhagen Chemometrics Group provided written permission after receiving a request that explicitly covered redistribution of the processed training and test CSV files through the public NIRBENCH-DL GitHub repository.
- **Conditions recorded in the permission:** identify where the data originate by citing the University of Copenhagen dataset website and the associated publication.
- **License status:** no standard dataset license was specified. The permission supports redistribution as part of NIRBENCH-DL and does not place the data under the repository's Apache License 2.0.
- **Repository handling:** the processed benchmark partitions may be included with the required source and publication attribution. The original archive and the reproducible preparation script remain linked for provenance.

### Task 2 — Wheat flours, protein

- **Original dataset:** 140 wheat flour spectra
- **Creators/source:** D. Bertrand / INRA, distributed through ChemProject ChemData
- **Source page:** <https://www.chemproject.org/chemdata>
- **Original files:** [`x_140farines.csv`](https://www.chemproject.org/media/data/x_140farines) and [`y_140farines.csv`](https://www.chemproject.org/media/data/y_140farines)
- **Benchmark modifications:** the spectra and reference table were joined by row; the `PROTREF` protein target and 525 spectral variables were retained; non-model metadata was excluded; the target was placed in the final column; an 80/20 fixed split with random state 123 produced 112 training and 28 test rows.
- **Permission basis:** written permission from Jean-Michel Roger, Team COMiC / ChemHouse, in reply to the redistribution request dated 29 September 2026. See [CHEMHOUSE_PERMISSION.md](CHEMHOUSE_PERMISSION.md) for the request, exact reply, and recorded scope.
- **License status:** no standard dataset license was specified. The permission allows redistribution of the processed benchmark CSV files in NIRBENCH-DL; it does not place them under Apache License 2.0.
- **Attribution and modifications:** retain the original creator attribution and ChemData source links above, identify the benchmark transformations, and ask users to cite the original dataset and associated publications where available. These are the attribution commitments made in the redistribution request.
- **Repository handling:** include `data_train.csv` and `data_test.csv` with this permission and attribution record. Keep the download and preparation scripts as optional reproducibility tools.

### Task 11 — Olive oils, C16:0

- **Original dataset:** NIR spectra and chemical analyses of 187 olive oils
- **Creators/source:** N. Dupuy group / Aix-Marseille University, distributed through ChemProject ChemData
- **Source page:** <https://www.chemproject.org/chemdata>
- **Original files:** [`pir.csv`](https://www.chemproject.org/media/data/pir) and [`ags.csv`](https://www.chemproject.org/media/data/ags)
- **Benchmark modifications:** the 612-variable NIR table was matched by row with the chemical-analysis table; palmitic acid `C16:0` was selected as the target; the target was placed in the final column; an 80/20 fixed split with random state 123 produced 149 training and 38 test rows.
- **Permission basis:** written permission from Jean-Michel Roger, Team COMiC / ChemHouse, in reply to the redistribution request dated 29 September 2026. See [CHEMHOUSE_PERMISSION.md](CHEMHOUSE_PERMISSION.md) for the request, exact reply, and recorded scope.
- **License status:** no standard dataset license was specified. The permission allows redistribution of the processed benchmark CSV files in NIRBENCH-DL; it does not place them under Apache License 2.0.
- **Attribution and modifications:** retain the original creator attribution and ChemData source links above, identify the benchmark transformations, and ask users to cite the original dataset and associated publications where available. These are the attribution commitments made in the redistribution request.
- **Repository handling:** include `data_train.csv` and `data_test.csv` with this permission and attribution record. Keep the download and preparation scripts as optional reproducibility tools.

## Tasks requiring download from the original source

### Task 4 — CGL mixture design, glucose

- **Original dataset:** CGL NIR three-component mixture-design dataset
- **Creators/source:** Tormod Næs and Tomas Isaksson, distributed by Eigenvector Research
- **Source page:** <https://eigenvector.com/resources/data-sets/>
- **Official archive:** <https://eigenvector.com/wp-content/uploads/2021/04/CGL_nir.mat_.zip>
- **Benchmark modifications:** `Xcal` and `Xtest` supplied the 117 spectral variables; the glucose column was selected from `Ycal` and `Ytest`; the original calibration/test division was preserved as 153 training and 78 test rows; the arrays were converted to CSV and the glucose target was placed in the final column.
- **Redistribution status:** permission to redistribute the processed CSV copies has been requested. Until permission is granted, obtain the original archive and prepare the task locally using `datasets/4-CGL_NIR_grain_glucose/download_data.md`.

The benchmark directory retains its historical name, `4-CGL_NIR_grain_glucose`, but the source details identify this as a casein/glucose/lactate mixture-design dataset rather than a grain dataset.

### Task 12 — Diesel fuels, cetane number

- **Original dataset:** SWRI diesel-fuel NIR spectra and properties
- **Creators/source:** Southwest Research Institute; distributed by Eigenvector Research; the source notes U.S. Army project sponsorship
- **Source page:** <https://eigenvector.com/resources/data-sets/>
- **Official CSV archive:** <https://eigenvector.com/wp-content/uploads/2019/06/SWRI_Diesel_NIR_CSV.zip>
- **Benchmark modifications:** the raw spectral and property tables were aligned by row; cetane number (`CN`) was selected as the target; all 784 source rows were split 80/20 with random state 123; rows lacking `CN` were then removed separately from each partition; 401 spectral variables were retained; the target was placed in the final column; the resulting benchmark partitions contain 298 training and 83 test rows.
- **Redistribution status:** permission to redistribute the processed CSV copies has been requested. Until permission is granted, obtain the original archive and prepare the task locally using `datasets/12-NIR_DieselFuels_cn/download_data.md`.

### Task 14 — IDRC pharmaceutical tablets, assay

- **Original dataset:** IDRC 2002 NIR pharmaceutical-tablet “Shootout”
- **Creators/source:** International Diffuse Reflectance Conference; converted and distributed by Eigenvector Research
- **Source page:** <https://eigenvector.com/resources/data-sets/>
- **Official archive:** <https://eigenvector.com/wp-content/uploads/2019/06/nir_shootout_2002.mat_.zip>
- **Benchmark modifications:** spectra from spectrometer 1 and the active-ingredient assay target were selected; 650 spectral variables were retained; the source calibration and source test observations were combined into the 615-row benchmark training partition; the 40-row source validation partition was retained as the benchmark test partition; the assay reference was rounded to its reported one-decimal precision; the arrays were converted to CSV and the assay target was placed in the final column.
- **Redistribution status:** permission to redistribute the processed CSV copies has been requested. Until permission is granted, obtain the original archive and prepare the task locally using `datasets/14-NIR_Pharmaceutical_Tablets_assay/download_data.md`.

## CEOT–UAlg datasets released by the data rights holder

The dataset files for tasks 15, 16, and 21 are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The full rights statement, requested attribution, publication citations, scope, and disclaimer are in `datasets/CEOT_DATA_LICENSE.md`.

### Task 15 — CEOT pear 2021, soluble solids

- **Original dataset:** CEOT–UAlg 2021 Rocha pear VIS–NIR measurements
- **Rights holder and licensor:** Dário Passos
- **Institution:** Centre for Electronics, Optoelectronics and Telecommunications, University of Algarve (CEOT–UAlg)
- **Associated publication:** Cruz et al. (2021), [doi:10.1016/j.postharvbio.2021.111562](https://doi.org/10.1016/j.postharvbio.2021.111562)
- **Benchmark modifications:** the soluble-solids response was retained as `brix`; 874 spectral variables spanning 500.458–1102.362 nm were retained; the fixed benchmark partitions contain 3,204 training and 804 test rows; the target was placed in the final column.

### Task 16 — CEOT pear 2019, soluble solids

- **Original dataset:** CEOT–UAlg 2019 Rocha pear VIS–SWNIR measurements
- **Rights holder and licensor:** Dário Passos
- **Institution:** Centre for Electronics, Optoelectronics and Telecommunications, University of Algarve (CEOT–UAlg)
- **Associated publication:** Passos et al. (2019), [doi:10.3390/s19235165](https://doi.org/10.3390/s19235165)
- **Benchmark modifications:** the soluble-solids response was retained as `brix`; 875 spectral variables spanning 499.605–1102.362 nm were retained; the fixed benchmark partitions contain 2,640 training and 660 test rows; the target was placed in the final column.

### Task 21 — CEOT pear 2021, trimmed and SNV-transformed spectra

- **Original dataset:** the same CEOT–UAlg 2021 measurements used for task 15
- **Rights holder and licensor:** Dário Passos
- **Associated publication:** Cruz et al. (2021), [doi:10.1016/j.postharvbio.2021.111562](https://doi.org/10.1016/j.postharvbio.2021.111562)
- **Benchmark modifications:** the same samples, `brix` targets, and train/test assignments as task 15 were retained; the wavelength range was trimmed to 750.520–1049.456 nm; standard normal variate preprocessing was applied to the spectra; 459 spectral variables remain; the fixed partitions contain 3,204 training and 804 test rows.

## Attribution and modification requirements

For material under CC BY 4.0, downstream users must credit the creators, link the license, link the source when practicable, and indicate that NIRBENCH-DL converted and reorganized the data. For the tomato material under CC BY-SA 4.0, the same attribution requirements apply and adapted copies must remain under CC BY-SA 4.0 or a compatible license.

For the ChemProject datasets in tasks 2 and 11, preserve the written-permission record, original creator attribution, source links, and transformation descriptions. Their permission is separate from the repository software license.

For Tecator, redistribution requires the complete original permission note to remain attached. The repository copy is stored in `datasets/3-Tecator_moisture/TECATOR_PERMISSION.txt`.

See `DATASETS.md` for the exact task-directory mapping and `datasets/manifest.csv` for the current partition sizes and checksums.
