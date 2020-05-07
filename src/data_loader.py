import json
from copy import deepcopy
import numpy as np
import random
random.seed(0)
from utils import MyDataset
from torch.utils.data import DataLoader

def LoadData(data_path="../local/dataset_batch.json", 
             word2id_path="../local/word_map.json",
             train_split=0.8,
             BATCH_SIZE=64):
    # import data from json file
    with open(word2id_path, "r") as json_file:
        word_map = json.load(json_file)
    with open(data_path, "r") as json_file:
        dataset = json.load(json_file)
    index_map = {}
    for key in word_map.keys():
        index_map[word_map[key]] = key
    MAX_SEQ_LEN = dataset["max_seq_len"]
    VOCAB_SIZE = word_map["*"] + 1 # Note: the last word is for PAD

    # generate data list
    input_data = []
    label_data = []
    count = {}
    for category in dataset.keys():
        if category == "max_seq_len":
            continue
        input_data += deepcopy(dataset[category])
        label_data += [int(category)] * len(dataset[category])
        count[category] = len(dataset[category])
    print("[INFO] Complete loading data, with # of {}".format(count))
    rd_index = np.arange(len(input_data))
    random.shuffle(rd_index)
    print(rd_index)
    input_data = [input_data[i] for i in rd_index]
    label_data = [label_data[i] for i in rd_index]
    total_size = len(input_data)
    train_size = int(total_size * train_split)

    # generate training dataset and dataloader
    train_dataset = MyDataset(input_data=input_data[:train_size],
                            label_data=label_data[:train_size],
                            word_map=word_map,
                            max_len=MAX_SEQ_LEN)
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    # generate testing dataset and dataloader
    test_dataset = MyDataset(input_data=input_data[train_size:],
                            label_data=label_data[train_size:],
                            word_map=word_map,
                            max_len=MAX_SEQ_LEN)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

    return train_loader, test_loader, MAX_SEQ_LEN, VOCAB_SIZE, index_map