import numpy as np


def acc_sb(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        numerator = len(set_true.intersection(set_pred))
        denominator = len(set_true.union(set_pred))
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_acc = 1
        else:
            tmp_acc = numerator / float(denominator)
        acc_list.append(tmp_acc)
    return np.mean(acc_list)

def prec_sb(y_true, y_pred):
    prec_list = []
    for i in range(y_true.shape[0]):
        TP = len(set(np.where(y_true[i])[0]).intersection(set(np.where(y_pred[i])[0])))
        denominator = len(set(np.where(y_pred[i])[0]))
        tmp_prec = 0.0 if denominator == 0 else TP / float(denominator)
        prec_list.append(tmp_prec)
    return np.mean(prec_list)

def recall_sb(y_true, y_pred):
    recall_list = []
    for i in range(y_true.shape[0]):
        TP = len(set(np.where(y_true[i])[0]).intersection(set(np.where(y_pred[i])[0])))
        denominator = len(set(np.where(y_true[i])[0]))
        tmp_recall = 0.0 if denominator == 0 else TP / float(denominator)
        recall_list.append(tmp_recall)
    return np.mean(recall_list)

def F1_sb(y_true, y_pred):
    F1_list = []
    for i in range(y_true.shape[0]):
        TP = len(set(np.where(y_true[i])[0]).intersection(set(np.where(y_pred[i])[0])))
        denominator_prec = len(set(np.where(y_pred[i])[0]))
        denominator_recall = len(set(np.where(y_true[i])[0]))
        tmp_F1 = 0.0 if (denominator_prec == 0) or (denominator_recall) == 0 \
            else 2*TP / float(denominator_prec + denominator_recall)
        F1_list.append(tmp_F1)
    return np.mean(F1_list)


def acc_lb(y_true, y_pred):
    acc_list = []
    for j in range(y_true.shape[1]):
        set_true = set(np.where(y_true[:,j])[0])
        set_pred = set(np.where(y_pred[:,j])[0])
        numerator = len(set_true.intersection(set_pred))
        denominator = len(set_true.union(set_pred))
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_acc = 1
        else:
            tmp_acc = numerator / float(denominator)
        acc_list.append(tmp_acc)
    return np.mean(acc_list)

def prec_lb(y_true, y_pred):
    prec_list = []
    for j in range(y_true.shape[1]):
        TP = len(set(np.where(y_true[:,j])[0]).intersection(set(np.where(y_pred[:,j])[0])))
        denominator = len(set(np.where(y_pred[:,j])[0]))
        tmp_prec = 0.0 if denominator == 0 else TP / float(denominator)
        prec_list.append(tmp_prec)
    return np.mean(prec_list)

def recall_lb(y_true, y_pred):
    recall_list = []
    for j in range(y_true.shape[1]):
        TP = len(set(np.where(y_true[:,j])[0]).intersection(set(np.where(y_pred[:,j])[0])))
        denominator = len(set(np.where(y_true[:,j])[0]))
        tmp_recall = 0.0 if denominator == 0 else TP / float(denominator)
        recall_list.append(tmp_recall)
    return np.mean(recall_list)

def F1_lb(y_true, y_pred):
    F1_list = []
    for j in range(y_true.shape[1]):
        TP = len(set(np.where(y_true[:,j])[0]).intersection(set(np.where(y_pred[:,j])[0])))
        denominator_prec = len(set(np.where(y_pred[:,j])[0]))
        denominator_recall = len(set(np.where(y_true[:,j])[0]))
        tmp_F1 = 0.0 if (denominator_prec == 0) or (denominator_recall) == 0 \
            else 2*TP / float(denominator_prec + denominator_recall)
        F1_list.append(tmp_F1)
    return np.mean(F1_list)

