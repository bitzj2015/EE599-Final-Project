# Preprocessed AOL dataset:
* dataset.json (https://drive.google.com/open?id=1oG1_p8WqdqlLaLStAMBq88dixkIQxcd0)
* dataset_clean.json (https://drive.google.com/open?id=1s9E6IDBxd7vhn26i6JrBAWEIox0sKQyv)
* dataset_clean_by_category.json (https://drive.google.com/open?id=12bArCJwSCQprq-vZ5sqIm0RZ4L_-4m0r)
* dataset_clean_by_category_demo.json (https://drive.google.com/open?id=1kbJxGwEngbF8kLPz0HZsmIQ4jyBZ8Si3)

# Usage
* `extract_keyword.py`: Extract keywords related to each interest category via topic modeling.
* `data_read.py`: Read original AOL data, transform text dataset into json file where the key is userID and the value if time and query list, and remove user with less than 100 queries. This script will generate `dataset.json` and `dataset_clean.json`.
* `data_filter.py`: Filter user queries by keyword, and classifiy queries into three classes (cancer-0, pregnant-1, other-2). This script will generate `dataset_clean_by_category.json` and `dataset_clean_by_category_demo.json`.
* `data_batch.py`: Generate data batch for training.
