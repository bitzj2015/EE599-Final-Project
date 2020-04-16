#!/bin/sh
echo "[INFO] Reading data from origin AOL dataset folder ......"
python data_read.py --phase=0 --root_path="/Users/usczj/Downloads/AOL-user-ct-collection"
echo "[INFO] Generating dataset.json ......"
echo "[INFO] Filtering dataset.json by keywords ......"
python data_filter.py --phase=0
python data_filter.py --phase=1
echo "[INFO] Sampling dataset.json by keywords ......"
python data_sample.py 