import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN
from sklearn.metrics import f1_score
from metrics import *

import warnings
warnings.filterwarnings("ignore")


def classifier(model='OneVsRest', estimator='LogisticRegression', k=3):
    if model == 'BinaryRelevance':
        if estimator == 'LogisticRegression':
            base = LogisticRegression()
        elif estimator == 'SVM':
            base = SVC()
        elif estimator == 'GaussianNB':
            base = GaussianNB()
        clf = BinaryRelevance(base)
    elif model == 'OneVsRest':
        if estimator == 'LogisticRegression':
            base = LogisticRegression(solver='liblinear')
        elif estimator == 'SVM':
            base = SVC()
        elif estimator == 'GaussianNB':
            base = GaussianNB()
        clf = OneVsRestClassifier(base)
    elif model == 'MLkNN':
        clf = MLkNN(k)

    return clf


def main(params):
    if params.text_feature:
        text = np.load(params.text_feature)
        print(text.shape)
    if params.image_feature:
        image = np.load(params.image_feature)
        print(image.shape)
    if params.text_feature and params.image_feature:
        X = np.concatenate((text, image), axis=1)
    elif not params.text_feature:
        X = image
    else:
        X = text
    df = pd.read_csv(params.df)
    y = np.array(df[df.columns[-11:]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=595)

    clf = eval(params.model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.toarray()
    print(params.model)
    print(f1_score(y_test, y_pred, average="micro"))
    print('Sample-based Accuracy:', acc_sb(y_test, y_pred))
    print('Sample-based Precision:', prec_sb(y_test, y_pred))
    print('Sample-based Recall:', recall_sb(y_test, y_pred))
    print('Sample-based F1-score:', F1_sb(y_test, y_pred))
    print('Label-based Accuracy:', acc_lb(y_test, y_pred))
    print('Label-based Precision:', prec_lb(y_test, y_pred))
    print('Label-based Recall:', recall_lb(y_test, y_pred))
    print('Label-based F1-score:', F1_lb(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training")

    parser.add_argument("--text_feature", default=None)
    parser.add_argument("--image_feature", default=None)
    parser.add_argument("--df", type=str, default='movies.csv')
    parser.add_argument("--model", type=str, default='OneVsRest')
    parser.add_argument("--estimator", type=str, default='LogisticRegression')
    parser.add_argument("--k", type=int, default=3)

    params, unknown = parser.parse_known_args()
    main(params)