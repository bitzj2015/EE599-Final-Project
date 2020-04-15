import os
import json
import time
from copy import deepcopy
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--phase', help="0 or 1", type=int)
parser.add_argument('--root_path', help="root path", default="/Users/usczj/Downloads/AOL-user-ct-collection", type=str)
args = parser.parse_args()

T = time.time()
data = {}
root_path = args.root_path
domains = [".com", ".org", ".net", ".int", ".edu", ".gov", ".mil"]
def filter_url(query):
    for domain in domains:
        if domain in query:
            return False
    return True

if args.phase == 0:
    for file in os.listdir(root_path):
        if file.startswith("user-ct-test"):
            print("[INFO] Processing {}".format(file))
            fp = open(root_path + "/" + file, 'r')
            lines = fp.readlines()
            for i in tqdm(range(len(lines) - 1)):
                log = lines[i+1].split("\t")
                userID = log[0]
                query = log[1]
                query_time = log[2]
                # query_time = log[2].split(" ")
                # date = query_time[0].split("-")
                # Time = query_time[1].split(":")
                if userID not in data.keys():
                    data[userID] = {"time":[], "query":[]}
                # data[userID]["time"].append({"year": date[0], "month": date[1], "day": date[2], "hour": Time[0], "min": Time[1], "sec": Time[2]})
                if len(data[userID]["query"]) == 0:
                    if filter_url(query):
                        data[userID]["time"].append(query_time)
                        data[userID]["query"].append(query)

                elif data[userID]["query"][-1] != query:
                    if filter_url(query):
                        data[userID]["time"].append(query_time)
                        data[userID]["query"].append(query)

    print("# of users: ", len(list(data.keys())))
    with open("dataset.json", "w") as json_file:
        json.dump(data, json_file)
else:
    # len query > 100: 35,368
    with open("dataset.json", "r") as json_file:
        data = json.load(json_file)
    clean_data = {}
    count = 0
    count_ = 0
    for user in tqdm(data.keys()):
        query = data[user]["query"]
        if len(query) >= 100:
            count += 1
            clean_data[user] = {}
            clean_data[user]["time"] = deepcopy(data[user]["time"])
            clean_data[user]["query"] = deepcopy(data[user]["query"])
        else:
            count_ += 1
    with open("dataset_clean.json", "w") as json_file:
        json.dump(clean_data, json_file)
    print(time.time() - T, count, count_)