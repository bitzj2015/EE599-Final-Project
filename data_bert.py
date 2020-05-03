import json
import h5py
from tqdm import tqdm
import numpy as np
from bert_serving.client import BertClient
bc = BertClient()

hf = h5py.File('dataset.h5', 'w')

with open("./data/dataset_batch_v3_.json", "r") as json_file:
    dataset = json.load(json_file)

for category in dataset.keys():
    if category == "max_seq_len":
        continue
    data_cate = dataset[category]
    query_list = []
    data_list = []
    for query in tqdm(data_cate):
        query_str = ""
        for word in query:
            if word == "<POS>":
                query_str += ","
            elif word == "<SOS>" or word == "<EOS>":
                continue
            else:
                if query_str == "":
                    query_str = word
                else:
                    query_str = query_str + " " + word
        query_list.append(query_str)
        if len(query_list) == 100:
            data = bc.encode(query_list)
            query_list = []
            data_list.append(data)
    hf.create_dataset(category, data=np.concatenate(data_list, axis=0))
     