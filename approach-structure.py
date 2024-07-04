import json
import argparse
import lm_utils
import metrics
import random
import torch.nn
import retrieve_utils
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


if __name__ == "__main__":

    feature_extractor = None

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "mistral", "llama2_70b", "chatgpt"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"
    # argParser.add_argument("-n", "--num", help="number of in-context examplars") # 16
    argParser.add_argument("-t", "--type", default="random", help = "random or attention") # random or attention
    argParser.add_argument("-p", "--prob", default = "False", help = "whether to output probabilities") # "True", "False"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    # num = int(args.num)
    type = args.type
    prob = args.prob

    if prob == "True":
        probs = []

    if type == "attention":
        init_similarity()

    lm_utils.llm_init(model_name)

    preds = []
    golds = []
    gold_mapping = {"human": 0, "bot": 1}

    f = open("data/" + dataset + "/split_new.json", "r")
    data = json.load(f)
    f.close()
    train_ids = data["train"]
    dev_ids = data["dev"]
    test_ids = data["test"]

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    # sanity check
    # test_ids = test_ids[:20]

    # retrieve_utils.init_retrieval("description")

    no_follower = 0
    no_following = 0

    for id in tqdm(test_ids):
        user = node[id]
        # print(user)

        golds.append(gold_mapping[label[id]])

        prompt =  "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of the user's followers and followings and their labels. You should output the label first and explanation after.\n\n"

        # neighbors of the user
        followers = []
        followings = []
        if id in edge.keys():
            for e in edge[id]:
                if e[0] == "followers" or e[0] == "friend" and e[1] in label.keys():
                    followers.append(e[1])
                elif e[0] == "following" or e[0] == "follow" and e[1] in label.keys():
                    followings.append(e[1])

        if type == "random":
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
                no_follower += 1
            
            if len(followings) > 0:
                prompt += "The target user follows these users:\n\n"
                for id in followings:
                    user_this = node[id]
                    prompt += feature_extraction(user_this)
                    prompt += "Description: " + user_this["description"].replace("\n", " ") + "\n"
                    prompt += "Label: " + label[id] + "\n\n"
            else:
                prompt += "The target user follows no user.\n\n"
                no_following += 1

            prompt += "Target user:\n\n"
            prompt += feature_extraction(user)
            prompt += "Description: " + user["description"].replace("\n", " ") + "\n"
            prompt += "Label:"
        
        elif type == "attention":

            if len(followers) > 8:
                followers = random.sample(followers, 8)
            if len(followings) > 8:
                followings = random.sample(followings, 8)

            if len(followers) > 0:
                # obtain similarity scores between user and followers
                follower_descriptions = [node[u]["description"] for u in followers]
                similarity_scores = get_similarity(user["description"], follower_descriptions)
                # sort followers by similarity scores
                followers = [x for _,x in sorted(zip(similarity_scores,followers),reverse=True)]

                prompt += "These users follow the target user, from most related to least related:\n\n"
                for id in followers:
                    user_this = node[id]
                    prompt += feature_extraction(user_this)
                    prompt += "Description: " + user_this["description"].replace("\n", " ") + "\n"
                    prompt += "Label: " + label[id] + "\n\n"
            else:
                prompt += "No user follows the target user.\n\n"
                no_follower += 1
            
            if len(followings) > 0:
                # obtain similarity scores between user and followings
                following_descriptions = [node[u]["description"] for u in followings]
                similarity_scores = get_similarity(user["description"], following_descriptions)
                # sort followings by similarity scores
                followings = [x for _,x in sorted(zip(similarity_scores,followings),reverse=True)]

                prompt += "The target user follows these users, from most related to least related:\n\n"
                for id in followings:
                    user_this = node[id]
                    prompt += feature_extraction(user_this)
                    prompt += "Description: " + user_this["description"].replace("\n", " ") + "\n"
                    prompt += "Label: " + label[id] + "\n\n"
            else:
                prompt += "The target user follows no user.\n\n"
                no_following += 1

            prompt += "Target user:\n\n"
            prompt += feature_extraction(user)
            prompt += "Description: " + user["description"].replace("\n", " ") + "\n"
            prompt += "Label:"

        if prob == "False":
            response = lm_utils.llm_response(prompt, model_name, probs=False)
        elif prob == "True":
            response, token_probs = lm_utils.llm_response(prompt, model_name, probs=True)
            real_prob = None
            for key in token_probs.keys():
                if "human" == key.lower().strip():
                    probs.append(1-token_probs[key])
                    real_prob = 1-token_probs[key]
                    print(probs[-1])
                    break
                elif "bot" == key.lower().strip():
                    probs.append(token_probs[key])
                    real_prob = token_probs[key]
                    print(probs[-1])
                    break
            if real_prob == None:
                print("Error: no human or bot in token_probs.keys()")
                probs.append(lm_utils.answer_parsing(response)) # 100% confidence
                
        preds.append(lm_utils.answer_parsing(response))

    print("------------------")
    print("Approach: structure")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Structure Type:", type)
    print(metrics.compute_metrics(preds, golds))
    print("------------------")

    # save preds to preds/
    to_save = {"accuracy": metrics.compute_metrics(preds, golds)["accuracy"], "f1": metrics.compute_metrics(preds, golds)["f1"], "preds": preds, "golds": golds}
    if prob == "True":
        to_save["probs"] = probs
        with open("probs/structure_" + dataset + "_" + model_name + "_" + type + "_" + str(datetime.now()) + ".json", "w") as f:
            json.dump(to_save, f, indent=4)