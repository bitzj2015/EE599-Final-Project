# Preprocessed AOL dataset:
* Download here: https://drive.google.com/open?id=1-yhZy-SWEBN8xpRQprastvWyg-1Ns-7p.

# Usage
* `extract_keyword.py`: Extract keywords related to each interest category via topic modeling.
* `data_read.py`: Read original AOL data, transform text dataset into json file where the key is userID and the value if time and query list, and remove user with less than 100 queries. This script will generate `dataset.json` and `dataset_clean.json`.
* `data_filter.py`: Filter user queries by keyword, and classifiy queries into three classes (cancer-0, pregnant-1, other-2). This script will generate `dataset_clean_by_category.json` and `dataset_clean_by_category_demo.json`.
* `data_sample.py`: Sampling data from queries for training.
