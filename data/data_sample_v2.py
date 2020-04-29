import json
import random
random.seed(0)
import numpy as np
from copy import deepcopy
import argparse
from tqdm import tqdm
import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('stopwords')

parser = English()
en_stop = set(nltk.corpus.stopwords.words('english'))
en_words = set(nltk.corpus.words.words())


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def tokenize(text):
    filter_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.like_url:
            continue
        elif token.orth_.startswith('@'):
            continue
        else:
            token = token.lower_
            filter_tokens.append(token)
    return filter_tokens

def prepare_text(text):
    tokens = tokenize(text)
    tokens = [get_lemma(token) for token in tokens]
    tokens = [token for token in tokens if token in en_words]
    return tokens


# with open("dataset_clean_by_category_sample.json", "r") as json_file:
#     data = json.load(json_file)

# max_count = 0
# dataset = {}
# count = {"0": 0, "1": 0, "2": 0}
# for category in data.keys():
#     print("Processing category: {}......".format(category))
#     data_by_category = data[category]
#     dataset[category] = {}
#     for user in tqdm(data_by_category.keys()):
#         if user not in dataset[category]:
#             dataset[category][user] = {}
#         user_data = data_by_category[user]
#         if len(user_data["time"]) != len(user_data["query"]):
#             print("Error!")
#         data_by_day = {}
#         for index in range(len(user_data["time"])):
#             time = user_data["time"][index]
#             query = user_data["query"][index]
#             day = time.split(' ')[0]
#             if day not in data_by_day:
#                 data_by_day[day] = {"time":[], "query":[], "flag": 0}
#             if user_data["category"][index] == int(category):
#                 data_by_day[day]["flag"] = 1
#             data_by_day[day]["time"].append(time)
#             data_by_day[day]["query"].append(query)
#         if category == "2":
#             if len(data_by_day.keys()) >= 2:
#                 day_list = random.sample(list(data_by_day.keys()), 2)
#             # else:
#             #     day_list = random.sample(list(data_by_day.keys()), 1)
#                 for day in day_list:
#                     dataset[category][user][day] = deepcopy(data_by_day[day])
#                     count[category] += 1
#         else:
#             for day in data_by_day.keys():
#                 if data_by_day[day]["flag"] == 1:
#                     dataset[category][user][day] = deepcopy(data_by_day[day])
#                     count[category] += 1

# print("[INFO] # of samples in each category: ", count)
# with open("dataset_final_v2.json", "w") as json_file:
#     json.dump(dataset, json_file)
with open("dataset_final_v2.json", "r") as json_file:
    dataset = json.load(json_file)
word_map = {"<SOS>": 0, "<EOS>": 1, "<POS>": 2, "*": 3}
data_batch = {}
max_seq_len = 0
for category in dataset.keys():
    data_batch[category] = {}
    data_by_category = dataset[category]
    for user in tqdm(data_by_category.keys()):
        tokens = []
        len_tokens = 0
        for day in data_by_category[user].keys():
            for query in data_by_category[user][day]["query"]:
                query_token = prepare_text(query)
                if len(query_token) > 0:
                    tokens += deepcopy(query_token + ["<POS>"])
                    len_tokens += len(query_token)
            if len_tokens >= 10 and len_tokens <= 50:
                data_batch[category][user + "-" + day] = deepcopy(["<SOS>"] + tokens[0:-1] + ["<EOS>"])
                if max_seq_len < len(tokens) + 1:
                    max_seq_len = len(tokens) + 1

data_sample = {}
data_sample["max_seq_len"] = max_seq_len
print("[INFO]: Maximal data sequence length: ", data_sample["max_seq_len"])
for category in data_batch.keys():
    print("[INFO] # of samples in category {}: ".format(category), len(data_batch[category].keys()))
    data_sample[category] = []
    userID = random.sample(list(data_batch[category].keys()), 4000)
    for user in userID:
        for token in data_batch[category][user]:
            if token not in word_map:
                word_map[token] = word_map["*"]
                word_map["*"] += 1
        data_sample[category].append(deepcopy(data_batch[category][user]))
print("[INFO]: Data batch sample: ", data_sample["0"][0:2])
print("[INFO]: Data batch sample: ", data_sample["1"][0:2])
print("[INFO]: Data batch sample: ", data_sample["2"][0:2])
print("[INFO]: # of words: ", len(word_map.keys()))

with open("dataset_batch_v2.json", "w") as json_file:
    json.dump(data_sample, json_file)
with open("word_map_v2.json", "w") as json_file:
    json.dump(word_map, json_file)