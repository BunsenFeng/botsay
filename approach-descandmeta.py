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
    argParser.add_argument("-n", "--num", default=16, help="number of in-context examplars") # 16
    argParser.add_argument("-p", "--prob", default = "False", help = "whether to output probabilities") # "True", "False"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)
    prob = args.prob

    if prob == "True":
        probs = []

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

    # # sanity check
    # test_ids = test_ids[:20]

    retrieve_utils.init_retrieval("description")

    for id in tqdm(test_ids):
        user = node[id]
        # print(user)

        golds.append(gold_mapping[label[id]])

        prompt =  "The following task focuses on evaluating whether a Twitter user is a bot or human with the help of the user's self-written description and metadata. You should output the label first and explanation after.\n\n"

        # in-context examplars
        if num > 0:
            top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids = retrieve_utils.retrieve(user["description"], num)
            for i in range(num):
                prompt += feature_extraction(top_k_user_infos[i])
                prompt += "Description: " + top_k_user_infos[i]["description"].replace("\n", " ") + "\n"
                prompt += "Label: " + top_k_user_labels[i] + "\n\n"
        
        # target user
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
                probs.append(lm_utils.answer_parsing(response)) # 100% confidence when error

        preds.append(lm_utils.answer_parsing(response))

    print("------------------")
    print("Approach: descandmeta")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Number of in-context examplars:", num)
    print(metrics.compute_metrics(preds, golds))
    print("------------------")

    # save preds to preds/
    to_save = {"accuracy": metrics.compute_metrics(preds, golds)["accuracy"], "f1": metrics.compute_metrics(preds, golds)["f1"], "preds": preds, "golds": golds}
    if prob == "True":
        to_save["probs"] = probs
        with open("probs/descandmeta_" + dataset + "_" + model_name + "_" + str(num) + "_" + str(datetime.now()) + ".json", "w") as f:
            json.dump(to_save, f, indent=4)