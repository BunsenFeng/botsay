import numpy as np
import sklearn.metrics as metrics

def compute_metrics(preds, labels):
    # compute accuracy, f1-score (binary)
    # preds: a list of [0,1]s representing the predictions
    # labels: a list of [0,1]s representing the ground truth
    # returns: a dictionary of metrics

    assert len(preds) == len(labels)

    accuracy = metrics.accuracy_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
