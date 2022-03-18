import os
import numpy as np
from nltk import word_tokenize, pos_tag
import nltk
import csv
import string
import collections as ct
import pandas as pd

from CRF_ReceiptGetter import ReceiptGetter


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


# annai_dir = r'/home/thumilan/Desktop/LSTM-sample/Annai_NonRead'
#
#
# text_list= os.listdir(annai_dir)
#
# i=0
#
# for img in text_list:
#     feature_receipt={}
#     file_name = text_list[i]
#     file_index=file_name[6:-6]
#     print(file_index)
#     feature_receipt_list=list()
#     pos_tag_list=pos_tagger(file_index,annai_dir+'/'+file_name,file_index)
#     for element in pos_tag_list:
#         feature_row=feature_extractor(file_index,element[0],element[1])
#         feature_receipt_list.append(feature_row)
#     with open('annai.csv', 'a') as csvFile:
#         for item in feature_receipt_list:
#             data=(item['receipt_index'],item['word'],item['pos'],item['Nalp'],item['Nnum'],item['Nspec'],item['length'],item['Ndot'],item['Ncomma'],item['Ncolons'])
#             writer = csv.writer(csvFile)
#             writer.writerow(data)
#         print(file_index)
#
#     csvFile.close()
#     i += 1
# print (i)


legacy_data=pd.read_csv("Annotated/north_index.csv")
getter = ReceiptGetter(legacy_data)
old_receipts = getter.receipts
feature_receipt_list=list()
for receipt in (getter.grouped):
    for tuble in receipt:
         feature_row=feature_extractor(tuble[0],tuble[1],tuble[2],tuble[3],tuble[4])
         feature_receipt_list.append(feature_row)
with open('north_featured.csv', 'a') as csvFile:
        for item in feature_receipt_list:
            data=(item['receipt_number'],item['receipt_index'],item['word'],item['pos'],item['Nalp'],item['Nnum'],item['Nspec'],item['length'],item['Ndot'],item['Ncomma'],item['Ncolons'],item['tag'])
            writer = csv.writer(csvFile)
            writer.writerow(data)
csvFile.close()
