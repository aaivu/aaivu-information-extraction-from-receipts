import os
import numpy as np
from nltk import word_tokenize, pos_tag
import nltk
import csv
import string
import collections as ct
import pandas as pd
from sklearn import tree
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
from sklearn.metrics import make_scorer, classification_report
import json
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

from CRF_ReceiptGetter import ReceiptGetter

from CRF_ReceiptGetter import ReceiptGetter
import warnings
import matplotlib.pyplot as plt

plt.style.use('ggplot')
warnings.filterwarnings("ignore")

def pos_tagger(receipt_number,file_path,file_index):
    file_object = open(file_path, "r",encoding = "ISO-8859-1")
    data_string = file_object.read().replace("\n", " ")
    # data_string = data_string.replace("  ", " ")
    file_object.close()
    # data_array = np.array(data_string.split(" "))
    # data_array = data_array[data_array != '']
    # print(data_array)


    pos_tag_list=pos_tag(word_tokenize(data_string))
    # with open('mega.csv', 'a') as csvFile:
    #     for item in pos_tag_list:
    #         data=(file_index,item[0],item[1])
    #         writer = csv.writer(csvFile)
    #         writer.writerow(data)
    #     print(file_index)
    #
    # csvFile.close()
    # print(pos_tag_list)
    return pos_tag_list

def text_of_element(text):
    return str(text)

def Nalp(text):
    text=str(text)
    alp_str="QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
    count=0
    for i in text:
        if alp_str.__contains__(i):
           count=count+1
    return count

def Nnum(text):
    num_str="1234567890"
    count=0
    text=(str(text))
    for i in text:
        if num_str.__contains__(i):
            count=count+1
    return count

def Nspec(text):
    text=(str(text))
    special_chars = string.punctuation
    return(sum(v for k, v in ct.Counter(text).items() if k in special_chars))

def length_of_element(text):
    return (len(str(text)))
def Ndot(text):
    return (str(text).count("."))
def Ncommas(text):
    return (str(text).count(","))
def Ncolons(text):
    return (str(text).count(":"))

def feature_extractor (index,text,postag,tag,number):

    feature_row = ({
        'receipt_number':number,
        'receipt_index':index,
        'word':text,
        'pos':postag,
        'Nalp':Nalp(text),
        'Nnum':Nnum(text),
        'Nspec':Nspec(text),
        'length':length_of_element(text),
        'Ndot':Ndot(text),
        'Ncomma':Ncommas(text),
        'Ncolons':Ncolons(text),
        'tag':tag,
        })
    return feature_row


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
        # 'Nalp': receipt[i][2],
        # 'Nnum': receipt[i][3],
        # 'Nspec': receipt[i][4],
        # 'length': receipt[i][5],
        # 'Ndot': receipt[i][6],
        # 'Ncomma': receipt[i][7],
        # 'Ncolons': receipt[i][8],

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
            # 'Nalp': receipt[i][2],
            # 'Nnum': receipt[i][3],
            # 'Nspec': receipt[i][4],
            # 'length': receipt[i][5],
            # 'Ndot': receipt[i][6],
            # 'Ncomma': receipt[i][7],
            # 'Ncolons': receipt[i][8],
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
            # 'Nalp': receipt[i][2],
            # 'Nnum': receipt[i][3],
            # 'Nspec': receipt[i][4],
            # 'length': receipt[i][5],
            # 'Ndot': receipt[i][6],
            # 'Ncomma': receipt[i][7],
            # 'Ncolons': receipt[i][8],
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
        # label_list.append(receipt[i][9])
    return label_list

def JSON_parser(test_x,pred_y,labels):
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



def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = str(line).replace("      "," ").replace("     "," ").replace("  "," ").split(" ")
        row_data=strip_list_noempty(row_data)
        row['class'] = row_data[0]
        row['precision'] = (row_data[1])
        row['recall'] = (row_data[2])
        row['f1_score'] = (row_data[3])
        row['support'] = (row_data[4])
        report_data.append(row)
        print(row_data)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(r'/home/thumilan/Desktop/LSTM-sample/Annotated/Results/no_pos_data_A.csv',index=False)






receipt_dir = r'/home/thumilan/Desktop/LSTM-sample/CompleteBatchProcess/GoogleOCROutput'
text_list= os.listdir(receipt_dir)
i=0

for img in text_list:
    feature_receipt={}
    file_name = text_list[i]
    file_index=file_name[6:-12]
    feature_receipt_list=list()
    pos_tag_list=pos_tagger(file_index,receipt_dir+'/'+file_name,file_index)
    for element in pos_tag_list:
        feature_row=feature_extractor(file_index,element[0],element[1],"SHOP_NAME",1)
        feature_receipt_list.append(feature_row)
    with open('test.csv', 'a') as csvFile:
        for item in feature_receipt_list:
            data=(item['receipt_number'],item['receipt_index'],item['word'],item['pos'],item['Nalp'],item['Nnum'],item['Nspec'],item['length'],item['Ndot'],item['Ncomma'],item['Ncolons'],item['tag'])
            writer = csv.writer(csvFile)
            writer.writerow(data)
    csvFile.close()
    i += 1
#



data = pd.read_csv("Annotated/TreeData/data_C.csv", encoding="latin1")
data = data.fillna(method="ffill")

getter = ReceiptGetter(data)
receipts = getter.receipts

random.shuffle(receipts)
divider = generate_divider(receipts,80)
train_set=receipts[:divider]
test_set = receipts[divider:]
# train_set=receipts

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
max_iterations=25,
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



# # Feature Dimention result priting
# report = metrics.flat_classification_report(
#     test_y, y_pred, labels=sorted_labels, digits=3
# )
# classification_report_csv(report)







test_data = pd.read_csv("test.csv", encoding="latin1")
test_data = test_data.fillna(method="ffill")
#
getter = ReceiptGetter(test_data)
receipts = getter.receipts
# print(receipts)
test_x=[receipt2features(s) for s in receipts]

y_pred = crf.predict(test_x)

JSON_parser(test_x,y_pred,labels)
