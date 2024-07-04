import os
import json
import argparse
import lm_utils
import metrics
import random
import torch.nn
import retrieve_utils
import random
import openai
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import pipeline

def init_similarity():
    global feature_extractor
    checkpoint = "roberta-base"
    feature_extractor = pipeline("feature-extraction",framework="pt",model=checkpoint,device=-1)

def get_similarity(reference, candidates):
    similarity_scores = []
    reference = feature_extractor(reference,return_tensors = "pt")[0].numpy().mean(axis=0)
    for candidate in candidates:
        candidate = feature_extractor(candidate,return_tensors = "pt")[0].numpy().mean(axis=0)
        similarity_scores.append(np.dot(reference,candidate)/(np.linalg.norm(reference)*np.linalg.norm(candidate)))

    return similarity_scores

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


def instruction_tune_instance(user, approach): #mode: generate or evaluate
    thing = {"messages": []}

    # message 1: system
    if approach == "metadata":
        thing["messages"].append({"role": "system", "content": "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of several labeled examples. You should output the label first and explanation after."})
    elif approach == "description":
        thing["messages"].append({"role": "system", "content": "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of the user's self-written description. You should output the label first and explanation after."})
    elif approach == "descandmeta":
        thing["messages"].append({"role": "system", "content": "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of the user's self-written description and metadata. You should output the label first and explanation after."})
    elif approach == "tweet":
        thing["messages"].append({"role": "system", "content": "The following task focuses on evaluating whether a Twitter user is a bot or human with their tweets and a few labeled examples. You should output the label first and explanation after."})
    elif approach == "structure":
        thing["messages"].append({"role": "system", "content": "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of the user's followers and followings and their labels. You should output the label first and explanation after."})

    # message 2: user (or in-context examplars and target user)
    if approach == "metadata":
        prompt = ""
        num_human = 0
        num_bot = 0
        while True:
            if num == 0:
                break
            random_id = random.choice(train_ids)
            if label[random_id] == "human":
                num_human += 1
            else:
                num_bot += 1
            if num_human > num / 2:
                num_human -= 1
                continue
            if num_bot > num / 2:
                num_bot -= 1
                continue
            random_user = node[random_id]
            prompt += feature_extraction(random_user)
            prompt += "Label: " + label[random_id] + "\n\n"
            if num_human + num_bot == num:
                break
        assert num_human == num_bot == num / 2
        
        # target user
        prompt += feature_extraction(user)
        prompt += "Label:"

        thing["messages"].append({"role": "user", "content": prompt})

    elif approach == "description":
        prompt = ""
        if not num > 0:
            top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids = retrieve_utils.retrieve(user["description"], num)
            for i in range(num):
                prompt += "Description: " + top_k_user_infos[i]["description"].replace("\n", " ") + "\n"
                prompt += "Label: " + top_k_user_labels[i] + "\n\n"

        # target user
        prompt += "Description: " + user["description"].replace("\n", " ") + "\n"
        prompt += "Label:"

        thing["messages"].append({"role": "user", "content": prompt})

    elif approach == "descandmeta":
        prompt = ""
        if num > 0:
            # in-context examplars
            top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids = retrieve_utils.retrieve(user["description"], num)
            for i in range(num):
                prompt += feature_extraction(top_k_user_infos[i])
                prompt += "Description: " + top_k_user_infos[i]["description"].replace("\n", " ") + "\n"
                prompt += "Label: " + top_k_user_labels[i] + "\n\n"
        
        # target user
        prompt += feature_extraction(user)
        prompt += "Description: " + user["description"].replace("\n", " ") + "\n"
        prompt += "Label:"

        thing["messages"].append({"role": "user", "content": prompt})
    
    elif approach == "tweet":
        prompt = ""
        tweets = []
        try:
            for tweet_id in edge[id]:
                if tweet_id[0] == "post":
                    tweets.append(node[tweet_id[1]]["text"])
            if len(tweets) > tweet_num:
                random.shuffle(tweets)
                tweets = tweets[:tweet_num]
        except:
            pass

        if len(tweets) == 0:
            thing["messages"].append({"role": "user", "content": "This user has no tweets."})
        
        for tweet in tweets:
            # in-context examplars
            top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids = retrieve_utils.retrieve(tweet, num)
            for i in range(num):
                prompt = prompt + "Tweet: " + top_k_texts[i].replace("\n", " ") + "\n"
                prompt += "Label: " + top_k_user_labels[i] + "\n\n"
            
            # target user
            prompt += "Tweet: " + tweet.replace("\n", " ") + "\n"
            prompt += "Label:"
            break
            
        thing["messages"].append({"role": "user", "content": prompt})
    
    
    elif approach == "structure":
        id = user["id"]
        prompt = ""

        # neighbors of the user
        followers = []
        followings = []
        if id in edge.keys():
            for e in edge[id]:
                if e[0] == "followers" or e[0] == "friend" and e[1] in label.keys():
                    followers.append(e[1])
                elif e[0] == "following" or e[0] == "follow" and e[1] in label.keys():
                    followings.append(e[1])

        if structure_type == "random":
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
                    prompt += "Label: " + label[id] + "\n\n"
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

            prompt += "Target user:\n\n"
            prompt += feature_extraction(user)
            prompt += "Description: " + user["description"].replace("\n", " ") + "\n"
            prompt += "Label:"

            thing["messages"].append({"role": "user", "content": prompt})
        
        elif structure_type == "attention":
            if len(followers) > 8:
                followers = random.sample(followers, 8)
            if len(followings) > 8:
                followings = random.sample(followings, 8)

            if len(followers) > 0:
                # obtain similarity scores between user and followers
                follower_descriptions = [node[u]["description"] for u in followers]
                try:
                    similarity_scores = get_similarity(user["description"],follower_descriptions)
                except:
                    similarity_scores = [0] * len(followers)
                follower_descriptions = [follower_descriptions[i] for i in np.argsort(similarity_scores)[::-1]]
                followers = [followers[i] for i in np.argsort(similarity_scores)[::-1]]
                prompt += "These users follow the target user, from most related to least related:\n\n"
                for i in range(len(followers)):
                    user_this = node[followers[i]]
                    prompt += feature_extraction(user_this)
                    prompt += "Description: " + follower_descriptions[i].replace("\n", " ") + "\n"
                    prompt += "Label: " + label[followers[i]] + "\n\n"
            else:
                prompt += "No user follows the target user.\n\n"

            if len(followings) > 0:
                # obtain similarity scores between user and followings
                following_descriptions = [node[u]["description"] for u in followings]
                similarity_scores = get_similarity(user["description"],following_descriptions)
                following_descriptions = [following_descriptions[i] for i in np.argsort(similarity_scores)[::-1]]
                followings = [followings[i] for i in np.argsort(similarity_scores)[::-1]]
                prompt += "The target user follows these users, from most related to least related:\n\n"
                for i in range(len(followings)):
                    user_this = node[followings[i]]
                    prompt += feature_extraction(user_this)
                    prompt += "Description: " + following_descriptions[i].replace("\n", " ") + "\n"
                    prompt += "Label: " + label[followings[i]] + "\n\n"
            else:
                prompt += "The target user follows no user.\n\n"

            prompt += "Target user:\n\n"
            prompt += feature_extraction(user)
            prompt += "Description: " + user["description"].replace("\n", " ") + "\n"
            prompt += "Label:"

            thing["messages"].append({"role": "user", "content": prompt})

        

    # message 3: assistant
    thing["messages"].append({"role": "assistant", "content": label[user["id"]]})

    return thing

global data
global train_ids
global dev_ids
global test_ids
global label
global node
global edge
global num

data = None
train_ids = None
dev_ids = None
test_ids = None
label = None
node = None
edge = None

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"
    argParser.add_argument("-a", "--approach", help="which approach") # "metadata", "description", "descandmeta", "tweet", "structure"
    argParser.add_argument("-n", "--num", default=16, help="number of in-context examplars") # 16
    argParser.add_argument("-t", "--tweet", default = 5, help="how many tweets to consider for one user, max") # 5
    argParser.add_argument("-s", "--structure_type", default = "random", help = "random or attention, type for the structure approach")

    args = argParser.parse_args()
    dataset = args.dataset
    approach = args.approach
    num = int(args.num)
    phase = "generate"
    tweet_num = int(args.tweet)
    structure_type = args.structure_type

    if structure_type == "attention":
        init_similarity()

    preds = []
    golds = []
    gold_mapping = {"human": 0, "bot": 1}

    f = open("data/" + dataset + "/split_new.json", "r")
    data = json.load(f)
    f.close()
    train_ids = data["train"]
    dev_ids = data["dev"]
    test_ids = data["test"]

    # test_ids = test_ids[:20]

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    if approach == "tweet":
        retrieve_utils.init_retrieval("tweet")
    elif approach == "description" or approach == "descandmeta":
        retrieve_utils.init_retrieval("description")

    if phase == "generate":
        if len(dev_ids) > 1000:
            dev_ids = dev_ids[:1000]
        
        texts = []
        for id in tqdm(dev_ids):
            user = node[id]
            temp = instruction_tune_instance(user, approach)
            # print(temp["messages"][0]["content"])
            # print(temp["messages"][1]["content"])
            # print(temp["messages"][2]["content"])

            new_temp = {"messages": [{"role": "user", "content": temp["messages"][0]["content"] + "\n" + temp["messages"][1]["content"]}, {"role": "assistant", "content": temp["messages"][2]["content"]}]}
            texts.append(new_temp)
        
        with open("corpus/" + dataset + "-" + approach + "-instruction-tuning.jsonl", "w") as f:
            for text in texts:
                f.write(json.dumps(text) + "\n")
