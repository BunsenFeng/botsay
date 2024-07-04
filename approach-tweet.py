import json
import argparse
import lm_utils
import metrics
import random
import torch.nn
import retrieve_utils
import random
from datetime import datetime
from tqdm import tqdm

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "mistral", "llama2_70b", "chatgpt"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"
    argParser.add_argument("-n", "--num", default=16, help="number of in-context examplars") # 16
    argParser.add_argument("-t", "--tweet", default=5, help="how many tweets to consider for one user, max") # 5
    argParser.add_argument("-p", "--prob", default = "False", help = "whether to output probabilities") # "True", "False"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)
    tweet_num = int(args.tweet)
    prob = args.prob

    lm_utils.llm_init(model_name)

    if prob:
        probs = []

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

    retrieve_utils.init_retrieval("tweet")

    # sanity check
    # test_ids = test_ids[:20]

    for id in tqdm(test_ids):
        user = node[id]
        # print(user)

        golds.append(gold_mapping[label[id]])

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
            preds.append(random.randint(0, 1))
            continue
        
        prompt =  "The following task focuses on evaluating whether a Twitter user is a bot or human with their tweets and a few labeled examples. You should output the label first and explanation after.\n\n"


        tweet_level_preds = []
        for tweet in tweets:
            # in-context examplars
            top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids = retrieve_utils.retrieve(tweet, num)
            for i in range(num):
                prompt_now = prompt + "Tweet: " + top_k_texts[i].replace("\n", " ") + "\n"
                prompt_now += "Label: " + top_k_user_labels[i] + "\n\n"
            
            # target user
            prompt_now += "Tweet: " + tweet.replace("\n", " ") + "\n"
            prompt_now += "Label:"

            response = lm_utils.llm_response(prompt_now, model_name, probs=False)

            tweet_level_preds.append(lm_utils.answer_parsing(response))
        
        # append majority in tweet_level_preds to preds
        if tweet_level_preds.count(0) > tweet_level_preds.count(1):
            preds.append(0)
        else:
            preds.append(1)
        
        if prob:
            probs.append(tweet_level_preds.count(1) / (tweet_level_preds.count(1) + tweet_level_preds.count(0)))

    print("------------------")
    print("Approach: tweet")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Number of in-context examplars:", num)
    print("Number of tweets per user:", tweet_num)
    print(metrics.compute_metrics(preds, golds))
    print("------------------")

    # save preds to preds/
    to_save = {"accuracy": metrics.compute_metrics(preds, golds)["accuracy"], "f1": metrics.compute_metrics(preds, golds)["f1"], "preds": preds, "golds": golds}
    if prob == "True":
        to_save["probs"] = probs
        with open("probs/tweet_" + dataset + "_" + model_name + "_" + str(num) + "_" + str(tweet_num) + "_" + str(datetime.now()) + ".json", "w") as f:
            json.dump(to_save, f, indent=4)