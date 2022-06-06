from pathlib import Path
import pickle

import numpy
import pyarrow.parquet as pq
import pandas

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


# LOAD DATA

data_dir = Path("/home/maximl/scratch/data/vsc/datasets/wbc/results/scip/202204271347/")

df = pq.read_table(data_dir / f"features.parquet").to_pandas()

df["meta_group"]= pandas.Categorical(df["meta_group"].astype(int), ordered=True)
df["meta_part"]= pandas.Categorical(df["meta_part"].astype(int), ordered=True)

df = df.set_index(["meta_group", "meta_part", "meta_fix", "meta_object_number"])

# df = df.fillna(0)

df = df[numpy.load(
    data_dir / "indices/columns.npy", allow_pickle=True)]
df = df.loc[numpy.load(
    data_dir / "indices/index.npy", allow_pickle=True)]

df["meta_label"] = pandas.Categorical(df["meta_label"], ordered=True)
df = df[df["meta_label"] != "unknown"]

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
Xs = df.filter(regex="(BF1|BF2|SSC)$")

# SPLIT DATA

Xs_train, _, y_train, _ =  train_test_split(
    Xs, y, stratify=y, test_size=0.2, random_state=0)

# PARAMETER SEARCH

model = make_pipeline(
    RandomUnderSampler(sampling_strategy="majority", random_state=0),
    RandomOverSampler(sampling_strategy="not majority", random_state=0),
    XGBClassifier(
        booster="gbtree",
        objective="multi:softmax",
        eval_metric="merror",
        tree_method="gpu_hist",
        use_label_encoder=False,
        random_state=0
    )
)

# scoring = (
#     'balanced_accuracy',
#     'f1_macro', 'f1_micro',
#     'precision_macro', 'precision_micro',
#     'recall_macro', 'recall_micro'
# )
# grid = RandomizedSearchCV(
#     model,
#     {
#         "rfe__estimator__max_depth": [6, 5, 4, 3, 2],
#         "rfe__estimator__learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
#         "rfe__estimator__n_estimators": [100, 200, 400, 600, 800, 1000],
#         "rfe__estimator__subsample": [0.25, 0.5, 0.75, 1],
#         "rfe__estimator__colsample_bytree": [0.25, 0.5, 0.75, 1],
#         "rfe__n_features_to_select": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
#     },
#     n_iter=1000,
#     refit=False,
#     n_jobs=30,
#     cv=5,
#     scoring=scoring,
#     verbose=2,
#     return_train_score=True
# ).fit(
#     Xs_train,
#     y_train
# )

grid = HalvingRandomSearchCV(
    estimator=model,
    param_distributions={
        "xgbclassifier__max_depth": [6, 5, 4, 3, 2],
        "xgbclassifier__learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001],
        "xgbclassifier__subsample": numpy.linspace(start=0.1, stop=1, num=10),
        "xgbclassifier__colsample_bytree": numpy.linspace(start=0.1, stop=1, num=10),
    },
    factor=2,
    resource='xgbclassifier__n_estimators',
    n_candidates=1000,
    max_resources=3000,
    min_resources=10,
    aggressive_elimination=True,
    refit=False,
    n_jobs=8,
    cv=5,
    scoring='balanced_accuracy',
    verbose=3,
    return_train_score=True,
    random_state=0
).fit(
    Xs_train,
    y_train
)

# STORE RESULTS

with open(data_dir / "rsh/grid_n_estimators_overunder_start10.pickle", "wb") as fh:
    pickle.dump(grid, fh)
