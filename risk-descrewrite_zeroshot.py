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

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "mistral", "llama2_70b", "chatgpt"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset

    lm_utils.llm_init(model_name)

    os.system("mkdir -p data/" + dataset + "_descrewrite_zeroshot_" + model_name)

    # copy-paste edge_new.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    with open("data/" + dataset + "_descrewrite_zeroshot_" + model_name + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)
    
    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_descrewrite_zeroshot_" + model_name + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_descrewrite_zeroshot_" + model_name + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # alter the descriptions of bot users in node.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            try:
                if label[id] == "bot" and len(node[id]["description"]) > 10:
                    prompt = "Please rewrite the description of this bot account to sound like a genuine user: " + node[id]["description"] + "\nNew Description:"
                    new_description = lm_utils.llm_response(prompt, model_name, temperature=1)
                    # print(node[id]["description"])
                    # print("------------------")
                    # print(new_description)
                    node[id]["description"] = new_description
            except:
                continue
    
    with open("data/" + dataset + "_descrewrite_zeroshot_" + model_name + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)

            