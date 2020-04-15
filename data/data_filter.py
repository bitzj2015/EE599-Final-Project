import json
from tqdm import tqdm

with open("dataset_clean.json", "r") as json_file:
    data = json.load(json_file)

filter_tag = {"0": ["cancer", "tumor", "carcinoma", "sarcoma", "leukemia", "lymphoma"], \
              "1": ["baby", "pregnancy", "pregnant", "fertile", "childbirth", "breast feeding"]}

filter_data = {"0": {}, "1": {}, "2": {}}
num_user = {"0": 0, "1": 0, "2": 0}
# count= 0
for user in tqdm(data.keys()):
    # count += 1
    # if count > 1000:
    #     break
    tmp = {"time": [], "query": [], "category": []}
    flag_category = {"0": -1, "1": -1, "2": -1}
    for index in range(len(data[user]["query"])):
        time = data[user]["time"][index]
        query = data[user]["query"][index]
        flag = 2
        for tag in filter_tag["0"]:
            if tag in query:
                flag = 0
                flag_category["0"] = 1
                break
        for tag in filter_tag["1"]:
            if tag in query:
                flag = 1
                flag_category["1"] = 1
                break
        tmp["time"].append(time)
        tmp["query"].append(query)
        tmp["category"].append(flag)
    if flag_category["0"] == 1:
        filter_data["0"][user] = tmp
        num_user["0"] += 1
    elif flag_category["1"] == 1:
        filter_data["1"][user] = tmp
        num_user["1"] += 1
    else:
        filter_data["2"][user] = tmp
        num_user["2"] += 1

print("# of user in cancer category", num_user["0"])
print("# of user in pregrant category", num_user["1"])
print("# of user in other category", num_user["2"])

with open("dataset_clean_by_category.json", "w") as json_file:
    json.dump(filter_data, json_file)
        
        
