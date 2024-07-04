import os
import json
import argparse
import lm_utils
import metrics
import random
import torch.nn
import retrieve_utils
from datetime import datetime
from tqdm import tqdm

def feature_extraction(user):
    user_name = user["name"]
    if len(user_name.strip()) == len(user_name):
        user_name = user_name + " "
    follower_count = user["public_metrics"]["followers_count"]
    following_count = user["public_metrics"]["following_count"]
    tweet_count = user["public_metrics"]["tweet_count"]
    verified = str(user["verified"]).strip()
    active_years = None
    try:
        active_years = 2023 - int(user["created_at"][:4])
    except:
        active_years = 2023 - int(user["created_at"][-5:])
    # print(user["created_at"][-5:])
    if active_years == 0:
        active_years = "less than 1 year"
    elif active_years == 1:
        active_years = "1 year"
    else:
        active_years = str(active_years) + " years"

    return "Username: " + user_name + " " + "Follower count: " + str(follower_count) + " " + "Following count: " + str(following_count) + " " + "Tweet count: " + str(tweet_count) + " " + "Verified: " + str(verified) + " " + "Active years: " + active_years + "\n"

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "mistral", "llama2_70b", "chatgpt"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"
    argParser.add_argument("-t", "--textual", help="path to the textual-alterted dataset") # "Twibot-20_text_attribute_mistral_5"
    argParser.add_argument("-s", "--structure", help="path to the structural-alterted dataset") # "Twibot-20_neighbor_add_mistral_5"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    textual = args.textual
    structure = args.structure

    os.system("mkdir -p data/" + dataset + "_both_combine_" + model_name)

    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_both_combine_" + model_name + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_both_combine_" + model_name + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # alter node_new.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    # alter edge_new.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    # textual-altered dataset

    f = open("data/" + textual + "/node_new.json", "r")
    node_altered = json.load(f)
    f.close()

    # structural-altered dataset

    f = open("data/" + structure + "/edge_new.json", "r")
    edge_altered = json.load(f)
    f.close()

    # user id list
    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            if label[id] == "bot":
                node[id] = node_altered[id]
                edge[id] = edge_altered[id]

    with open("data/" + dataset + "_both_combine_" + model_name + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)

    with open("data/" + dataset + "_both_combine_" + model_name + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)