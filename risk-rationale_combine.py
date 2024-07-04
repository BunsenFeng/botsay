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

    lm_utils.llm_init(model_name)

    os.system("mkdir -p data/" + dataset + "_rationale_combine_" + model_name)

    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_rationale_combine_" + model_name + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_rationale_combine_" + model_name + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # alter node_new.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    # alter edge_new.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    # load node_altered from textual-altered dataset

    f = open("data/" + textual + "/node_new.json", "r")
    node_altered = json.load(f)
    f.close()

    # load edge_altered from structural-altered dataset

    f = open("data/" + structure + "/edge_new.json", "r")
    edge_altered = json.load(f)
    f.close()

    # user id list
    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            if label[id] == "bot":

                user_information_blob = feature_extraction(node[id]) + "\n" + "Description: " + node[id]["description"] + "\n\n"

                # neighbors of the user
                prompt = ""
                followers = []
                followings = []
                if id in edge.keys():
                    for e in edge[id]:
                        if e[0] == "followers" or e[0] == "friend" and e[1] in label.keys():
                            followers.append(e[1])
                        elif e[0] == "following" or e[0] == "follow" and e[1] in label.keys():
                            followings.append(e[1])

                if len(followers) > 8:
                    followers = random.sample(followers, 8)
                if len(followings) > 8:
                    followings = random.sample(followings, 8)
                random.shuffle(followers)
                random.shuffle(followings)

                if len(followers) > 0:
                    prompt += "These users follow the target user:\n\n"
                    for id in followers:
                        user_this = node[id]
                        prompt += feature_extraction(user_this)
                        prompt += "Description: " + user_this["description"].replace("\n", " ") + "\n"
                else:
                    prompt += "No user follows the target user.\n\n"
                
                if len(followings) > 0:
                    prompt += "The target user follows these users:\n\n"
                    for id in followings:
                        user_this = node[id]
                        prompt += feature_extraction(user_this)
                        prompt += "Description: " + user_this["description"].replace("\n", " ") + "\n"
                        prompt += "Label: " + label[id] + "\n\n"
                else:
                    prompt += "The target user follows no user.\n\n"

                user_information_blob += prompt

                prompt = "Twitter's bot detection models take into account various user attributes, such as the use of default avatars, location, length of self-introduction, and more. They also analyze the user's tweet history, as well as the users they follow and are followed by, in order to determine whether the account is a bot. Furthermore, certain detection methods focus on the posting behavior of users under specific tags, aiming to identify groups with highly similar posting patterns. Additionally, there are approaches that consider the social network formed by a user, utilizing graph theory methods for detection."
                prompt += "\n\n"
                prompt += "Please evaluate why the target user is a bot: does the description or follower/following list of the target user look suspicious?"
                prompt += "\n\n"
                prompt += "Target User:\n\n"
                prompt += user_information_blob
                prompt += "Description or follower/following list, which is more suspicious?\nA. Description B. Follower/Following List C. Both are suspicious\nAnswer:"
                response = lm_utils.llm_response(prompt, model_name, temperature = 1).split("\n")[0]
                if "A" in response:
                    node[id]["description"] = node_altered[id]["description"]
                elif "B" in response:
                    edge[id] = edge_altered[id]
                elif "C" in response:
                    node[id]["description"] = node_altered[id]["description"]
                    edge[id] = edge_altered[id]
                
    with open("data/" + dataset + "_rationale_combine_" + model_name + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)
    with open("data/" + dataset + "_rationale_combine_" + model_name + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)