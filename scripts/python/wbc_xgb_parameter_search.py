from pathlib import Path
import os
import pickle

import numpy
import pandas
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline


# LOAD DATA

df = pq.read_table("/home/maximl/scratch/data/wbc/results/scip/202201141724/features.parquet").to_pandas()
df["meta_group"] = df["meta_group"].astype(int)
df["meta_part"] = df["meta_part"].astype(int)
df = df[numpy.load("/home/maximl/scratch/data/wbc/results/scip/202201141724/columns.npy", allow_pickle=True)]
index = numpy.load("/home/maximl/scratch/data/wbc/results/scip/202201141724/index.npy", allow_pickle=True)
df = df.loc[index]

df = df[~df["meta_bbox_minr"].isna()]
df = df[df["meta_label"] != "unknown"]

glcms = set(df.filter(regex=".*glcm.*(BF1|BF2|SSC)$").columns)
combined_glcms = set(df.filter(regex=".*combined.*glcm.*(BF1|BF2|SSC)$").columns)
noncomb_glcms = glcms - combined_glcms
df = df.drop(columns=noncomb_glcms)
df.shape

df = df.set_index(["meta_object_number", "meta_part", "meta_group", "meta_fix"])

# PREP CLASSIFICATION INPUT

neuts = df[df["meta_label"] == "CD15 + Neutrophils"].sample(n=110000).index
df = df.drop(labels=neuts)

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
Xs = df.filter(regex="(BF1|BF2|SSC)$")

# SPLIT DATA

Xs_train, Xs_test, y_train, y_test =  train_test_split(Xs, y, test_size=0.3, random_state=0)

# PARAMETER SEARCH

model = make_pipeline(
    RandomOverSampler(sampling_strategy="not majority", random_state=0),
    XGBClassifier(
        booster="gbtree",  
        objective="multi:softmax", 
        eval_metric="merror",
        tree_method="gpu_hist",
        use_label_encoder=False
    )
)

scoring = ('balanced_accuracy', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro')
grid = RandomizedSearchCV(
    model,
    {
        "xgbclassifier__max_depth": [6, 5, 4, 3, 2],
        "xgbclassifier__learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        "xgbclassifier__n_estimators": [100, 200, 400, 600, 800, 1000],
        "xgbclassifier__subsample": [0.25, 0.5, 0.75, 1],
        "xgbclassifier__colsample_bytree": [0.25, 0.5, 0.75, 1]
    },
    n_iter=1000,
    refit=False,
    n_jobs=30,
    cv=3,
    scoring=scoring,
    verbose=2,
    return_train_score=True
).fit(
    Xs_train,
    y_train
)

# STORE RESULTS

with open("grid.pickle", "wb") as fh:
    pickle.dump(grid, fh)
