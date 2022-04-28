from pathlib import Path
import pickle

import numpy
import pyarrow.parquet as pq

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


# LOAD DATA

data_dir = Path("/home/maximl/scratch/data/wbc/results/scip/202204271347")

df = pq.read_table(data_dir / "features.parquet").to_pandas()

df["meta_group"] = df["meta_group"].astype(int)
df["meta_part"] = df["meta_part"].astype(int)

df = df.set_index(["meta_object_number", "meta_part", "meta_group", "meta_fix"])
df = df[numpy.load(data_dir / "columns.npy", allow_pickle=True)]
index = numpy.load(data_dir / "index.npy", allow_pickle=True)
df = df.loc[index]

df = df[~df["meta_bbox_minr"].isna()]
df = df[df["meta_label"] != "unknown"]

df = df.fillna(0)

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
Xs = df.filter(regex="(BF1|BF2|SSC)$")

# SPLIT DATA

Xs_train, Xs_test, y_train, y_test =  train_test_split(
    Xs, y, stratify=y, test_size=1/3, random_state=0)

# PARAMETER SEARCH

model = make_pipeline(
    RandomUnderSampler(sampling_strategy="majority", random_state=0),
    RandomOverSampler(sampling_strategy="not majority", random_state=0),
    RFE(
        XGBClassifier(
            booster="gbtree",
            objective="multi:softmax",
            eval_metric="merror",
            tree_method="gpu_hist",
            use_label_encoder=False,
            random_state=0
        )
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
        "rfe__estimator__max_depth": [6, 5, 4, 3, 2],
        "rfe__estimator__learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        "rfe__estimator__subsample": [0.25, 0.5, 0.75, 1],
        "rfe__estimator__colsample_bytree": [0.25, 0.5, 0.75, 1],
        "rfe__n_features_to_select": [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    },
    factor=3,
    resource='n_estimators',
    n_candidates=3000,
    max_resources=4500,
    min_resources=2,
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

with open(data_dir / "rsh/grid.pickle", "wb") as fh:
    pickle.dump(grid, fh)
