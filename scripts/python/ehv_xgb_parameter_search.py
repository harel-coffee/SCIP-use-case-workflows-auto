from pathlib import Path
import os
import pickle

import numpy
import pandas
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# LOAD DATA

data_dir = Path(os.environ["HOME"]) / "scratch/data/ehv"

df = pq.read_table(data_dir / f"results/scip/202112021107_dapi/features.parquet").to_pandas()
df["meta_group"] = df["meta_group"].astype(int)
df["meta_replicate"] = df["meta_replicate"].astype(int)
df = df[numpy.load(data_dir / "results/scip/202112021107_dapi/columns.npy", allow_pickle=True)]

df = df[~df["meta_bbox_minr"].isna()]
df = df.drop(columns=df.filter(regex="BF2$"))
df = df[df["meta_label"] != "unknown"]

df = df.set_index(["meta_object_number", "meta_replicate", "meta_group", "meta_type"])

df["meta_label"] = pandas.Categorical(df["meta_label"], categories=["mcp-_psba+", "mcp+_psba+", "mcp+_psba-", "mcp-_psba-"], ordered=True)

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df.loc[:, :, :, "Inf"]["meta_label"])
y = enc.transform(df.loc[:, :, :, "Inf"]["meta_label"])

# selection of the generic channel features for SCIP
to_keep = df.filter(regex=".*scip.*(BF1|DAPI|SSC)$").columns
Xs = df.loc[:, :, :, "Inf"][to_keep]
Xs.shape

# SPLIT DATA

Xs_train, Xs_test, y_train, y_test =  train_test_split(Xs, y, test_size=0.5, random_state=0)

# PARAMETER SEARCH

model = XGBClassifier(
    booster="gbtree",
    objective="multi:softmax",
    eval_metric="merror",
    tree_method="gpu_hist",
    use_label_encoder=False
)

scoring = ('balanced_accuracy', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro')
grid = RandomizedSearchCV(
    model,
    {
        "max_depth": [6, 5, 4, 3, 2],
        "learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        "n_estimators": [100, 200, 400, 600, 800, 1000],
        "subsample": [0.25, 0.5, 0.75, 1],
        "colsample_bytree": [0.25, 0.5, 0.75, 1]
    },
    n_iter=1500,
    refit=False,
    n_jobs=50,
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