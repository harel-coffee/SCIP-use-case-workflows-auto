from pathlib import Path
import pickle

import numpy
import pyarrow.parquet as pq
import pandas

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_validate, HalvingRandomSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline


# LOAD DATA

data_dir = Path()

df = pq.read_table(snakemake.input.features).to_pandas()

df["meta_group"]= pandas.Categorical(df["meta_group"].astype(int), ordered=True)
df["meta_part"]= pandas.Categorical(df["meta_part"].astype(int), ordered=True)

df = df.set_index(["meta_group", "meta_part", "meta_fix", "meta_object_number"])

# df = df.fillna(0)

df = df[numpy.load(snakemake.input.columns, allow_pickle=True)]
df = df.loc[numpy.load(snakemake.input.index, allow_pickle=True)]

df["meta_label"] = pandas.Categorical(df["meta_label"], ordered=True)
df = df[df["meta_label"] != "unknown"]

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
Xs = df.filter(regex="(BF1|BF2|SSC)$")

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
    max_resources=500,
    min_resources=10,
    aggressive_elimination=True,
    refit=True,
    n_jobs=12,
    cv=5,
    scoring='balanced_accuracy',
    verbose=3,
    return_train_score=True,
    random_state=0
)

scores = cross_validate(
    grid, Xs, y,
    scoring="balanced_accuracy",
    cv=5,
    return_train_score=True,
    return_estimator=True
)

# STORE RESULTS

with open(snakemake.output, "wb") as fh:
    pickle.dump(scores, fh)
