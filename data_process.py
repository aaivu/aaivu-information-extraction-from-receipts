import pandas as pd
import numpy as np
import csv
from pandas import DataFrame

data = pd.read_csv("/home/thumilan/Desktop/LSTM-sample/Annotated/TreeData/data_demo_not_numbered.csv", )
data = data.fillna(method="ffill")
index_series = pd.Series([])
# sample_array=np.zeros(shape=(102,3))
index=1
count=0
receipt_number=data["receipt_number"][0]
for i in range(len(data)):
    if (data["receipt_number"][i])==receipt_number:
        index_series[i]=index
        # index=index+1

    # else:
    #     sample_array[count][0]=receipt_number
    #     sample_array[count][1]=index
    #     index=1
    #     count=count+1
    #     receipt_number = data["receipt_number"][i]
    else:
        index=index+1
        receipt_number=data["receipt_number"][i]
        index_series[i] = index
# print(sample_array)
data.insert(0, "receipt_index", index_series)

df = DataFrame(data)
export_csv=df.to_csv(r'/home/thumilan/Desktop/LSTM-sample/Annotated/TreeData/data_demo.csv',index = None, header=True)


# dataframe= DataFrame(new_data)
# tag_series = pd.Series([])
# receipt_number_list=new_data["receipt_number"]
# receipt_number_list = list(dict.fromkeys(receipt_number_list))
# print(receipt_number_list)
# index=0
# index_receipt=0
# receipt_number = receipt_number_list[index_receipt]
# for i in range(len(dataframe)):
#     # if (legacy_data["receipt_number"][i]) == receipt_number:
#     #     tag_series[index]=legacy_data["tag"][i]
#     #     index=index+1
#     print(legacy_data.loc[(legacy_data['receipt_number']==dataframe['receipt_number'][i]) & (legacy_data['word']==dataframe['word'][i])]['receipt_number'])
#     break


# new_data.insert(10,"tag",tag_series)
# df=DataFrame(new_data)
# export_csv=df.to_csv(r'annai_featured.csv',index=None, header=True)
