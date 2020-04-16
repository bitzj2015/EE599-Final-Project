import json
import random
import numpy as np
from copy import deepcopy
import argparse
from tqdm import tqdm

with open("dataset_clean_by_category_sample.json", "r") as json_file:
    data = json.load(json_file)

max_count = 0
dataset = {}
count = {"0": 0, "1": 0, "2": 0}
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
#             else:
#                 day_list = random.sample(list(data_by_day.keys()), 1)
#             for day in day_list:
#                 dataset[category][user][day] = deepcopy(data_by_day[day])
#                 count[category] += 1
#         else:
#             for day in data_by_day.keys():
#                 if data_by_day[day]["flag"] == 1:
#                     dataset[category][user][day] = deepcopy(data_by_day[day])
#                     count[category] += 1
for category in data.keys():
    print("Processing category: {}......".format(category))
    data_by_category = data[category]
    dataset[category] = {}
    for user in tqdm(data_by_category.keys()):
        if user not in dataset[category]:
            dataset[category][user] = {"time": [], "query": []}
        user_data = data_by_category[user]
        if len(user_data["time"]) != len(user_data["query"]):
            print("Error!")
        if category != "2":
            for index in range(len(user_data["time"])):
                time = user_data["time"][index]
                query = user_data["query"][index]
                if user_data["category"][index] == int(category):
                    dataset[category][user]["time"].append(time)
                    dataset[category][user]["query"].append(query)
                    count[category] += 1
        else:
            num_query = 3 + int(random.random() < 0.5)
            start_pos = random.randint(0, len(user_data["time"]) - num_query - 1)
            dataset[category][user]["time"] = deepcopy(user_data["time"][start_pos: start_pos + num_query])
            dataset[category][user]["query"] = deepcopy(user_data["query"][start_pos: start_pos + num_query])
            count[category] += num_query
print("# of samples in each category: ", count)
with open("dataset_final.json", "w") as json_file:
    json.dump(dataset, json_file)

