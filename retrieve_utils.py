import json
import random
from rank_bm25 import BM25Okapi

bm25 = None

def init_bm25(corpus):
    global bm25
    corpus = [doc.lower() for doc in corpus]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

def get_bm25_top_k_indices(query, top_k=5):
    global bm25
    tokenized_query = query.lower().split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    # find out indices of top k scores
    top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i])[-top_k:]
    # return top k scores
    return top_k_indices

description_corpus = []
tweet_corpus = []
id_list = [] # Twibot-20 only
node = None
label = None
with open("data/Twibot-20/node_new.json", "r") as f:
    node = json.load(f)
with open("data/Twibot-20/label_new.json", "r") as f:
    label = json.load(f)

def init_retrieval(text_type):
    # dataset: 20
    # text_type: description or tweet
    global description_corpus, tweet_corpus, node
    train_user_set = []
    with open("data/Twibot-20/split_new.json", "r") as f:
        split = json.load(f)
        train_user_set = split["train"]
    if text_type == "description":
        for key in node.keys():
            if not key[0] == "u":
                continue
            if not key in train_user_set:
                continue
            description_corpus.append(node[key]["description"])
            id_list.append(key)
    elif text_type == "tweet":
        edge = None
        with open("data/Twibot-20/edge_new.json", "r") as f:
            edge = json.load(f)
        for key in edge.keys():
            # continue by probability
            if random.random() > 0.1:
                continue
            if not key[0] == "u":
                continue
            for thing in edge[key]:
                if thing[1][0] == "t" and thing[0] == "post" and key in train_user_set:
                    tweet_corpus.append(node[thing[1]]["text"])
                    id_list.append(key)

    if text_type == "description":
        description_corpus = list(set(description_corpus))
        description_corpus = [doc.lower() for doc in description_corpus]
        init_bm25(description_corpus)
    elif text_type == "tweet":
        tweet_corpus = list(set(tweet_corpus))
        tweet_corpus = [doc.lower() for doc in tweet_corpus]
        init_bm25(tweet_corpus)

def retrieve(query, top_k=5):
    global description_corpus, tweet_corpus, id_list
    top_k_indices = get_bm25_top_k_indices(query, top_k)
    top_k_texts = []
    if len(description_corpus) > 0:
        top_k_texts = [description_corpus[i] for i in top_k_indices]
    else:
        top_k_texts = [tweet_corpus[i] for i in top_k_indices]
    top_k_ids = [id_list[i] for i in top_k_indices]
    top_k_user_infos = []
    for id in top_k_ids:
        top_k_user_infos.append(node[id])
    top_k_user_labels = []
    for id in top_k_ids:
        top_k_user_labels.append(label[id])
    return top_k_texts, top_k_user_infos, top_k_user_labels, top_k_ids

# init_retrieval("tweet")
# a, b, c, d = retrieve("I am a student", 5)
# print(a)
# print("------------------")
# print(b)
# print("------------------")
# print(c)
# print("------------------")
# print(d)