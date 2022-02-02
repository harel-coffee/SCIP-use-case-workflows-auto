from pathlib import Path
import os
import pickle

import numpy
import pandas
import pyarrow.parquet as pq

from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


# LOAD DATA

df = pq.read_table("/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_Inf/features.parquet").to_pandas()
df["meta_group"] = df["meta_group"].astype(int)
df["meta_replicate"] = df["meta_replicate"].astype(int)
df = df[numpy.load("/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_Inf/columns.npy", allow_pickle=True)]
index = numpy.load("/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/202201311209_Inf/index.npy", allow_pickle=True)
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

# PARAMETER SEARCH

model = make_pipeline(
    StandardScaler(),
    SVC(cache_size=4096, class_weight="balanced", random_state=0)
)

grid = HalvingGridSearchCV(
    estimator=model,
    param_grid=[{
        "svc__C": [0.1, 0.5, 1, 1.5, 2, 2.5],
        "svc__gamma": ['scale', 1/100, 1/10, 1],
        "svc__kernel": ["rbf", "sigmoid"]
    }, {
        "svc__C": [0.1, 0.5, 1, 1.5, 2, 2.5],
        "svc__gamma": ['scale', 1/100, 1/10, 1],
        "svc__degree": [2, 3, 4, 5],
        "svc__kernel": ["poly"]
    }],
    factor=3,
    resource='n_samples',
    max_resources='auto',
    min_resources=1000,
    aggressive_elimination=True,
    refit=False,
    n_jobs=10,
    cv=3,
    scoring='balanced_accuracy',
    verbose=3,
    return_train_score=True,
    random_state=0
).fit(
    Xs_train.fillna(0),
    y_train
)

# STORE RESULTS

with open("grid.pickle", "wb") as fh:
    pickle.dump(grid, fh)
