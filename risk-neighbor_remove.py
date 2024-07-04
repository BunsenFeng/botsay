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
    try:
        if len(user_name.strip()) == len(user_name):
            user_name = user_name + " "
    except:
        pass
    follower_count = user["public_metrics"]["followers_count"]
    following_count = user["public_metrics"]["following_count"]
    tweet_count = user["public_metrics"]["tweet_count"]
    verified = str(user["verified"]).strip()
    active_years = None
    try:
        active_years = 2023 - int(user["created_at"][:4])
    except:
        try:
            active_years = 2023 - int(user["created_at"][-5:])
        except:
            active_years = 0
    # print(user["created_at"][-5:])
    if active_years == 0:
        active_years = "less than 1 year"
    elif active_years == 1:
        active_years = "1 year"
    else:
        active_years = str(active_years) + " years"

    try:
        return "Username: " + user_name + " " + "Follower count: " + str(follower_count) + " " + "Following count: " + str(following_count) + " " + "Tweet count: " + str(tweet_count) + " " + "Verified: " + str(verified) + " " + "Active years: " + active_years + "\n"
    except:
        return "user information parsing failed"


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "mistral", "llama2_70b", "chatgpt"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"
    argParser.add_argument("-n", "--num", default=5, help="number of followings to consider to remove") # 5

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)

    lm_utils.llm_init(model_name)

    os.system("mkdir -p data/" + dataset + "_neighbor_remove_" + model_name + "_" + str(num))

    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_neighbor_remove_" + model_name + "_" + str(num) + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_neighbor_remove_" + model_name + "_" + str(num) + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)
    
    # copy-paste node_new.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    with open("data/" + dataset + "_neighbor_remove_" + model_name + "_" + str(num) + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)

    # alter the edges of bot users in edge.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    # user id list
    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            if label[id] == "bot":

                try:
                    # remove following
                    following_list = []
                    for e in edge[id]:
                        if e[0] == "follow" and e[1][0] == "u":
                            following_list.append(e[1])
                    if len(following_list) == 0:
                        continue
                    if len(following_list) > num:
                        following_list = random.sample(following_list, num)

                    prompt = "Below is a target Twitter bot and five potential users to unfollow. Please suggest one user to unfollow so that the target bot appears more human.\n\n"

                    prompt += "Target Bot:\n"
                    prompt += feature_extraction(node[id])
                    prompt += "Description: " + node[id]["description"]
                    prompt += "\n\n"
                    
                    prompt += "Potential users to unfollow:\n\n"

                    for j in range(len(following_list)):
                        prompt += "user " + str(j + 1) + ":\n"
                        prompt += feature_extraction(node[following_list[j]])
                        prompt += "Description: " + node[following_list[j]]["description"]
                        prompt += "\n\n"

                    prompt += "Please select one user to unfollow (1-" + str(len(following_list)) + "):"
                    response = lm_utils.llm_response(prompt, model_name, temperature = 1).split("\n")[0]
                    found = False
                    for option in range(len(following_list)):
                        if str(option + 1) in response:
                            edge[id].remove(["follow", following_list[option]])
                            found = True
                            print("removed!")
                            break
                    if not found:
                        # print("Error: no option selected")
                        pass
                except:
                    print("error in removing following, skipping this user")

    with open("data/" + dataset + "_neighbor_remove_" + model_name + "_" + str(num) + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)
