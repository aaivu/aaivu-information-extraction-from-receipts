import os
import random

import numpy as np
from nltk import word_tokenize, pos_tag
import nltk
import csv
import string
import collections as ct
import pandas as pd

from CRF_ReceiptGetter import ReceiptGetter


annai_data_count = 0
mega_data_count = 0
north_data_count = 50

# annai_data=pd.read_csv("Annotated/annai_featured.csv")
# annai_getter = ReceiptGetter(annai_data)
# annai_receipts = annai_getter.receipts
# random.shuffle(annai_receipts)
# annai_receipts=annai_receipts[0:annai_data_count]

north_data=pd.read_csv("Annotated/north_featured.csv")
north_getter = ReceiptGetter(north_data)
north_receipts = north_getter.receipts
random.shuffle(north_receipts)
sub_north_receipts=north_receipts[0:north_data_count]
print(len(sub_north_receipts))


mega_data=pd.read_csv("Annotated/mega_featured.csv")
mega_getter = ReceiptGetter(mega_data)
mega_receipts = mega_getter.receipts
random.shuffle(mega_receipts)
sub_mega_receipts=mega_receipts[0:mega_data_count]
print(len(sub_mega_receipts))

i=0
with open('/home/thumilan/Desktop/LSTM-sample/Annotated/TreeData/data_not_numbered.csv', 'a') as csvFile:
    # for item in annai_receipts:
    #     for data in item:
    #         # data = (item[0], item[1], item[2], item[3], item[4],
    #         #         item[5], item[6], item[7], item[8], item[9], item[10])
    #         writer = csv.writer(csvFile)
    #         writer.writerow(data)
    for item in sub_mega_receipts:
        for data in item:
            # data = (item[0], item[1], item[2], item[3], item[4],
            #         item[5], item[6], item[7], item[8], item[9], item[10])
            writer = csv.writer(csvFile)
            writer.writerow(data)
        i=i+1
        print(i)
    for item in sub_north_receipts:
        for data in item:
            # data = (item[0], item[1], item[2], item[3], item[4],
            #         item[5], item[6], item[7], item[8], item[9], item[10])
            writer = csv.writer(csvFile)
            writer.writerow(data)
        i = i + 1
        print(i)
csvFile.close()
