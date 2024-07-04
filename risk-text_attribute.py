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
    argParser.add_argument("-n", "--num", default=5, help="number of (positive, negative) examples for text attribute extraction") # 5

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    num = int(args.num)

    lm_utils.llm_init(model_name)

    os.system("mkdir -p data/" + dataset + "_text_attribute_" + model_name + "_" + str(num))

    # copy-paste edge_new.json

    f = open("data/" + dataset + "/edge_new.json", "r")
    edge = json.load(f)
    f.close()

    with open("data/" + dataset + "_text_attribute_" + model_name + "_" + str(num) + "/edge_new.json", "w") as outfile:
        json.dump(edge, outfile)

    # copy-paste split_new.json

    f = open("data/" + dataset + "/split_new.json", "r")
    split = json.load(f)
    f.close()

    with open("data/" + dataset + "_text_attribute_" + model_name + "_" + str(num) + "/split_new.json", "w") as outfile:
        json.dump(split, outfile)

    # copy-paste label_new.json

    f = open("data/" + dataset + "/label_new.json", "r")
    label = json.load(f)
    f.close()

    with open("data/" + dataset + "_text_attribute_" + model_name + "_" + str(num) + "/label_new.json", "w") as outfile:
        json.dump(label, outfile)

    # alter the descriptions of bot users in node.json

    f = open("data/" + dataset + "/node_new.json", "r")
    node = json.load(f)
    f.close()

    # init description retrieval

    retrieve_utils.init_retrieval("description")

    # user id list
    user_id_list = split["test"]

    for id in tqdm(user_id_list):

        if id[0] == "u":
            if label[id] == "bot" and len(node[id]["description"]) > 10:

                top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids = retrieve_utils.retrieve(node[id]["description"], num * 5)
                human_texts = []
                bot_texts = []
                for i in range(len(top_k_texts)):
                    if top_k_user_labels[i] == "human":
                        human_texts.append(top_k_texts[i])
                    elif top_k_user_labels[i] == "bot":
                        bot_texts.append(top_k_texts[i])
                
                while len(human_texts) < num:
                    random_id = random.choice(split["train"])
                    if random_id[0] == "u" and label[random_id] == "human" and len(node[random_id]["description"]) > 10:
                        human_texts.append(node[random_id]["description"])
                
                while len(bot_texts) < num:
                    random_id = random.choice(split["train"])
                    if random_id[0] == "u" and label[random_id] == "bot" and len(node[random_id]["description"]) > 10:
                        bot_texts.append(node[random_id]["description"])

                human_texts = random.choice(human_texts)[:num]
                bot_texts = random.choice(bot_texts)[:num]

                # extract text attribute
                prompt = "Bot descriptions:\n"
                for text in bot_texts:
                    prompt += text + "\n"

                prompt += "\nHuman descriptions:\n"
                for text in human_texts:
                    prompt += text + "\n"

                prompt += "Compare and give the key distinct feature of human's descriptions:"

                text_attribute = lm_utils.llm_response(prompt, model_name, temperature = 1).split("\n")[0]

                # rewrite with extracted text attribute

                prompt = text_attribute + "\nBased on the description, paraphrase this to human description:\nBot: " + node[id]["description"] + "\nHuman:"
                new_description = lm_utils.llm_response(prompt, model_name, temperature = 1).split("\n")[0]

                # print(node[id]["description"])
                # print("------------------")
                # print(new_description)

                node[id]["description"] = new_description

    with open("data/" + dataset + "_text_attribute_" + model_name + "_" + str(num) + "/node_new.json", "w") as outfile:
        json.dump(node, outfile)