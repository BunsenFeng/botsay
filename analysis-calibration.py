import json
import os

files = os.listdir("probs/")
for file in files:

    # # load Twibot-20 split and label
    # f = open("data/Twibot-20/split_new.json", "r")
    # split = json.load(f)
    # f.close()
    # test_ids = split["test"]

    # f = open("data/Twibot-20/label_new.json", "r")
    # label = json.load(f)
    # f.close()
    # label_mapping = {"human": 0, "bot": 1}
    # labels = [label[id] for id in test_ids]
    # labels = [label_mapping[l] for l in labels]

    f = open("probs/" + file, "r")
    log = json.load(f)
    f.close()

    preds = log["preds"]
    probs = log["probs"]
    labels = log["golds"]
    assert len(preds) == len(probs) == len(labels)

    bucket_interval = 0.1
    bucket_count = [0] * int(1 / bucket_interval)
    bucket_correct = [0] * int(1 / bucket_interval)
    bucket_total = [0] * int(1 / bucket_interval)

    for i in range(len(preds)):
        bucket = int((probs[i]-1e-4) / bucket_interval)
        bucket_count[bucket] += 1
        if preds[i] == labels[i]:
            bucket_correct[bucket] += 1
        bucket_total[bucket] += abs(probs[i] - 0.5) + 0.5

    ece = 0
    for i in range(len(bucket_count)):
        if bucket_count[i] != 0:
            acc = bucket_correct[i] / bucket_count[i]
            conf = bucket_total[i] / bucket_count[i]
            ece += abs(acc - conf) * bucket_count[i] / len(preds)

    for i in range(len(bucket_count)):
        if bucket_count[i] == 0:
            bucket_correct[i] = -1
            bucket_count[i] = 1 # so that the bucket will display -1, to indicate its empty

    print(file)
    print([bucket_correct[i] / bucket_count[i] for i in range(len(bucket_count))])
    print(ece)
    print("--------------------")