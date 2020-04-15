import json
with open("dataset_clean_by_category_sample.json", "r") as json_file:
    data = json.load(json_file)

for category in data.keys():
    print("Processing category: {}......".format(category))
    data_by_category = data[category]
    for user in data_by_category.keys():

