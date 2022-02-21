import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from FeatureTransformer import FeatureTransformer
from MemoryTagger import MemoryTagger
from ReceiptGetter_notused import ReceiptGetter
from sklearn.pipeline import Pipeline



data = pd.read_csv("Annotated/mega_index.csv", encoding="latin1")
data = data.fillna(method="ffill")

#Family,RB,SHOP_NAME,99999,

getter = ReceiptGetter(data)

sent, pos, tag = getter.get_next()

tagger = MemoryTagger()
tagger.fit(sent, tag)

words = data["word"].values.tolist()
tags = data["tag"].values.tolist()

# In Memory Usage only

# pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
# report = classification_report(y_pred=pred, y_true=tags)
# print(report)

# Random Forest Classifirer

# pred = cross_val_predict(Pipeline([("feature_map", FeatureTransformer()),
# ("clf", RandomForestClassifier(n_estimators=20, n_jobs=3))]),
# X=data, y=tags, cv=5)
# report = classification_report(y_pred=pred, y_true=tags)
# print(report)






