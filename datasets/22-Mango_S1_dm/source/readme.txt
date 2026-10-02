These scripts prepare the `NAnderson2020MendeleyMangoNIRData.csv` mango data (v2 in Mendeley website)
into 4 different datasets for tasks 22, 23, 24 and 25 (different harvest seasons). The script splits
the full dataset by season, and then does a 80%/20% train/test split taking into account that samples
are measured 2 or more times. No same samples end in the train and test sets. No data leakage.

DP