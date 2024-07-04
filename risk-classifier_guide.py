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
import numpy as np

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "mistral", "llama2_70b", "chatgpt"
    argParser.add_argument("-d", "--dataset", help="which dataset") # "Twibot-20", "Twibot-22"
    argParser.add_argument("-n", "--num", default=5, help="number of iterations and examples") # 5
    argParser.add_argument("-r", "--record", default = "False", help="whether to keep records of intermediate classifier scores") # "True" or "False"

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)

    if args.record == "True":
        records = []
        for i in range(num):
            records.append([])

    lm_utils.llm_init(model_name)

    os.system("mkdir -p data/" + dataset + "_classifier_guide_" + model_name + "_" + str(num))

    # # copy-paste edge_new.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    with open("data/" + dataset + "_classifier_guide_" + model_name + "_" + str(num) + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)

    # # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_classifier_guide_" + model_name + "_" + str(num) + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_classifier_guide_" + model_name + "_" + str(num) + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # alter the descriptions of bot users in node.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    # load description-based bot classifier
    lm_utils.mlm_text_classifier(None, None, None, None)

    # user id list
    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            if label[id] == "bot" and len(node[id]["description"]) > 10:
                current_description = node[id]["description"]
                prompt = "Below is a description of a Twitter user and its variants, paired with their score predicted by a bot classifier. The score is between 0 and 1, with 0 being human and 1 being bot. Please rewrite the description to make the user appear more human.\n\n"
                for i in range(num):
                    prompt += "Description: " + current_description + "\n"
                    score = lm_utils.text_classifier_inference(current_description)
                    if args.record == "True":
                        records[i].append(score)
                    prompt += "Score: " + str(score) + "\n\n"
                    prompt_temp = prompt + "New Description:"
                    current_description = lm_utils.llm_response(prompt_temp, model_name, temperature = 1).split("\n")[0]
                # print(node[id]["description"])
                # print("------------------")
                # print(current_description)
                node[id]["description"] = current_description

    with open("data/" + dataset + "_classifier_guide_" + model_name + "_" + str(num) + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)

    if args.record == "True":
        record_dict = {}
        for i in range(num):
            record_dict[i] = {"mean": np.mean(records[i]), "std": np.std(records[i]), "raw": records[i]}

        with open("records/" + dataset + "_classifier_guide_" + model_name + "_" + str(num) + "_records.json", "w") as outfile:
            json.dump(record_dict, outfile)