from pathlib import Path
import os
import pickle

import numpy
import pandas
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV


# LOAD DATA

df = pq.read_table("/home/maximl/scratch/data/ehv/results/scip/202201311209_Inf/features.parquet").to_pandas()
df["meta_group"] = df["meta_group"].astype(int)
df["meta_replicate"] = df["meta_replicate"].astype(int)
df = df[numpy.load("/home/maximl/scratch/data/ehv/results/scip/202201311209_Inf/columns.npy", allow_pickle=True)]
index = numpy.load("/home/maximl/scratch/data/ehv/results/scip/202201311209_Inf/index.npy", allow_pickle=True)
df = df.loc[index]

df = df[df["meta_label"] != "unknown"]
df = df.set_index(["meta_type", "meta_object_number", "meta_replicate", "meta_group"])
df["meta_label"] = pandas.Categorical(df["meta_label"], categories=["mcp-_psba+", "mcp+_psba+", "mcp+_psba-", "mcp-_psba-"], ordered=True)

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df.loc["Inf"]["meta_label"])
y = enc.transform(df.loc["Inf"]["meta_label"])

# selection of the generic channel features for SCIP
to_keep = df.filter(regex=".*(BF1|BF2|DAPI|SSC)$").columns
Xs = df.loc["Inf"][to_keep]
Xs.shape

# SPLIT DATA

Xs_train, Xs_test, y_train, y_test =  train_test_split(Xs, y, test_size=0.1, random_state=0)

print(Xs_train.shape)

# PARAMETER SEARCH

model = XGBClassifier(
    booster="gbtree",
    objective="multi:softmax",
    eval_metric="merror",
    tree_method="gpu_hist",
    use_label_encoder=False
)

grid = HalvingRandomSearchCV(
    estimator=model,
    param_distributions={
        "max_depth": [6, 5, 4, 3, 2],
        "learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        # "n_estimators": [100, 200, 400, 600, 800, 1000],
        "subsample": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },
    factor=3,
    resource='n_estimators',
    n_candidates=2000,
    max_resources=1000,
    min_resources='exhaust',
    refit=False,
    n_jobs=30,
    cv=3,
    scoring='balanced_accuracy',
    verbose=3,
    return_train_score=True,
    random_state=0
).fit(
    Xs_train,
    y_train
)

# STORE RESULTS

with open("grid.pickle", "wb") as fh:
    pickle.dump(grid, fh)
