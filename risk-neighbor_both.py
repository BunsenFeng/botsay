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
    argParser.add_argument("-n", "--num", default=5, help="number of followings to consider to remove") # 5

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)

    os.system("mkdir -p data/" + dataset + "_neighbor_both_" + model_name + "_" + str(num))

    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_neighbor_both_" + model_name + "_" + str(num) + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_neighbor_both_" + model_name + "_" + str(num) + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # copy-paste node_new.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    with open("data/" + dataset + "_neighbor_both_" + model_name + "_" + str(num) + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)

    # edge_new merging

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge_original = json.load(f)
    f.close()

    f = open("data/" + dataset + "_neighbor_add_" + model_name + "_" + str(num) + "/edge_new.json", "r")
    edge_add = json.load(f)
    f.close()

    f = open("data/" + dataset + "_neighbor_remove_" + model_name + "_" + str(num) + "/edge_new.json", "r")
    edge_remove = json.load(f)
    f.close()

    for key in edge_remove.keys():

        if not key[0] == "u":
            continue
        newly_added = []
        for e in edge_add[key]:
            if not e in edge_original[key]:
                newly_added.append(e)
        edge_remove[key] = edge_remove[key] + newly_added

    with open("data/" + dataset + "_neighbor_both_" + model_name + "_" + str(num) + "/edge_new.json", "w") as outfile:
        json.dump(edge_remove, outfile)
