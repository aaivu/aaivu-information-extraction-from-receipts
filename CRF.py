import random
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from sklearn_crfsuite.metrics import flat_classification_report
import pandas as pd
import numpy as np
from itertools import chain
import collections
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import json

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

from CRF_ReceiptGetter import ReceiptGetter



def generate_divider(receipts,percentage):
    return int((len(receipts)/100)*percentage)


def receipt2feature(receipt,i):
    features = {
        'bias' : 1.0,
        'word' :receipt[i][1],
        'pos': receipt[i][2],
        'Nalp': receipt[i][3],
        'Nnum': receipt[i][4],
        'Nspec': receipt[i][5],
        'length': receipt[i][6],
        'Ndot': receipt[i][7],
        'Ncomma': receipt[i][8],
        'Ncolons': receipt[i][9],

    }
    if i > 0:
        features.update({
            '-1:word' :receipt[i-1][1],
            '-1:pos': receipt[i-1][2],
            '-1:Nalp': receipt[i-1][3],
            '-1:Nnum': receipt[i-1][4],
            '-1:Nspec': receipt[i-1][5],
            '-1:length': receipt[i-1][6],
            '-1:Ndot': receipt[i-1][7],
            '-1:Ncomma': receipt[i-1][8],
            '-1:Ncolons': receipt[i-1][9],
        })
    else:
        features['BOR'] = True
    if i < len(receipt)-1:
        features.update({
            '+1:word' :receipt[i+1][1],
            '+1:pos': receipt[i+1][2],
            '+1:Nalp': receipt[i+1][3],
            '+1:Nnum': receipt[i+1][4],
            '+1:Nspec': receipt[i+1][5],
            '+1:length': receipt[i+1][6],
            '+1:Ndot': receipt[i+1][7],
            '+1:Ncomma': receipt[i+1][8],
            '+1:Ncolons': receipt[i+1][9],
        })
    else:
        features['EOR'] = True
    return features
def receipt2features(receipt):
    return [receipt2feature(receipt, i) for i in range(len(receipt))]

def receipt2labels(receipt):
    label_list=list()
    for i in range(len(receipt)):
        label_list.append(receipt[i][10])

    return label_list

def JSON_parser(test_x,pred_y,labels):
    print(len(pred_y))
    print(len(test_x))
    results_list=list()
    for i in range(len(pred_y)):
        result_receipt_list=list()
        receipt=test_x[i]
        result=pred_y[i]
        for j in range (len(result)):
            result_receipt_list.append([receipt[j]['word'],result[j]])
        results_list.append(result_receipt_list)
    dict_list=list()
    for receipt in results_list:
        result_dict = collections.defaultdict(list)
        for word in receipt:
            if word[1] in result_dict.keys():
                result_dict[word[1]]=str(result_dict[word[1]])+" "+word[0]
            else:
                result_dict[word[1]] =word[0]
        dict_list.append(result_dict)
    with open('JSON_Result/result.json', 'a') as f:
        for index in range (len(dict_list)):
            json.dump(dict_list[i],f)


data = pd.read_csv("Annotated/all_featured.csv", encoding="latin1")
data = data.fillna(method="ffill")

getter = ReceiptGetter(data)
receipts = getter.receipts

random.shuffle(receipts)
divider = generate_divider(receipts,80)
# train_set=receipts[:divider]
test_set = receipts[divider:]
train_set=receipts

train_x=[receipt2features(s) for s in train_set]
train_y=[receipt2labels(s) for s in train_set]

test_x=[receipt2features(s) for s in test_set]
test_y=[receipt2labels(s) for s in test_set]
# print(train_x[0])
# print(train_y[0])
#
# print(test_x[0])
# print(test_y[0])


crf = CRF(algorithm='lbfgs',
c1=0.1,
c2=0.1,
max_iterations=100,
all_possible_transitions=False)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

crf.fit(train_x, train_y)
labels=list(crf.classes_)
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)


# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=50,
#                         scoring=f1_scorer)
# rs.fit(train_x, train_y)
#
# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))


y_pred = crf.predict(test_x)
#
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    test_y, y_pred, labels=sorted_labels, digits=3
))

print ("Accuracy")
print(metrics.flat_accuracy_score(test_y,y_pred))

# JSON_parser(test_x,y_pred,labels)

# test_data = pd.read_csv("annai.csv", encoding="latin1")
# test_data = test_data.fillna(method="ffill")
#
# getter = ReceiptGetter(test_data)
# receipts = getter.receipts
# print(receipts)
# test_x=[receipt2features(s) for s in receipts]
#
# y_pred = crf.predict(test_x)
#
# JSON_parser(test_x,y_pred,labels)
