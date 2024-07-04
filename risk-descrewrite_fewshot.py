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
    argParser.add_argument("-n", "--num", default=5, help="number of in-context examplars") # 5

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)

    lm_utils.llm_init(model_name)

    os.system("mkdir -p data/" + dataset + "_descrewrite_fewshot_" + model_name + "_" + str(num))

    # copy-paste edge_new.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    with open("data/" + dataset + "_descrewrite_fewshot_" + model_name + "_" + str(num) + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)

    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_descrewrite_fewshot_" + model_name + "_" + str(num) + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_descrewrite_fewshot_" + model_name + "_" + str(num) + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # alter the descriptions of bot users in node.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    # user id list
    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            if label[id] == "bot" and len(node[id]["description"]) > 10:
                sample_human_descriptions = []
                while len(sample_human_descriptions) < num:
                    random_id = random.choice(user_id_list)
                    if random_id[0] == "u" and label[random_id] == "human" and len(node[random_id]["description"]) > 10:
                        sample_human_descriptions.append(node[random_id]["description"])
                prompt = "Please rewrite the description of a target bot account to sound like a genuine user, based on the following examples of genuine user descriptions:\n" + "\n".join(sample_human_descriptions) + "\n\nOriginal Description:\n" + node[id]["description"] + "\nNew Description:"
                new_description = lm_utils.llm_response(prompt, model_name, temperature = 1)
                # print(node[id]["description"])
                # print("------------------")
                # print(new_description)
                node[id]["description"] = new_description
    
    with open("data/" + dataset + "_descrewrite_fewshot_" + model_name + "_" + str(num) + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)