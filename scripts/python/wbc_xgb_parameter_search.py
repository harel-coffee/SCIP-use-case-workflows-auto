from pathlib import Path
import pickle

import numpy
import pyarrow.parquet as pq
import pandas

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_validate, HalvingRandomSearchCV, RandomizedSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

from sklearn.dummy import DummyClassifier


# LOAD DATA

data_dir = Path()

df = pq.read_table(snakemake.input.features).to_pandas()

df["meta_group"]= pandas.Categorical(df["meta_group"].astype(int), ordered=True)
df["meta_part"]= pandas.Categorical(df["meta_part"].astype(int), ordered=True)

df = df.set_index(["meta_group", "meta_part", "meta_fix", "meta_object_number"])

# df = df.fillna(0)

df = df[numpy.load(snakemake.input.columns, allow_pickle=True)]
df = df.loc[numpy.load(snakemake.input.index, allow_pickle=True)]

# drop samples not used in CytoA
# df = df.drop('late', level="meta_fix")
# df = df.drop(2, level="meta_group")

df["meta_label"] = pandas.Categorical(df["meta_label"], ordered=True)
df = df[df["meta_label"] != "unknown"]

# PREP CLASSIFICATION INPUT

enc = LabelEncoder().fit(df["meta_label"])
y = enc.transform(df["meta_label"])

# selection of the generic channel features for SCIP
Xs = df.filter(regex="(BF1|BF2|SSC)$")

if snakemake.config["dummy"] == 'true':
    model = DummyClassifier(strategy="uniform", random_state=0)
    param_distributions = {}
    resource = 'n_samples'
else:
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
    param_distributions = {
        "xgbclassifier__max_depth": [6, 5, 4, 3, 2],
        "xgbclassifier__learning_rate": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001],
        "xgbclassifier__subsample": numpy.arange(start=0.1, stop=1.1, step=.1),
        "xgbclassifier__colsample_bytree": numpy.arange(start=0.1, stop=1.1, step=.1),
        "xgbclassifier__gamma": numpy.arange(start=0, stop=31, step=2),
        "xgbclassifier__min_child_weight": numpy.arange(start=1, stop=32, step=2),
        "xgbclassifier__n_estimators": numpy.arange(start=10, stop=301, step=10)
    }
    resource = 'n_samples'

if snakemake.params["grid"] == "random":
    grid = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=1500,
        refit=True,
        n_jobs=16,
        cv=5,
        scoring='balanced_accuracy',
        verbose=2,
        return_train_score=True,
        random_state=0
        
    )
else:
    grid = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        factor=1.55,
        resource=resource,
        n_candidates=1500,
        min_resources=5000,
        aggressive_elimination=False,
        refit=True,
        n_jobs=16,
        cv=5,
        scoring='balanced_accuracy',
        verbose=2,
        return_train_score=True,
        random_state=0
    )

scores = cross_validate(
    grid, Xs, y,
    scoring=('balanced_accuracy', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro'),
    cv=5,
    return_train_score=True,
    return_estimator=True
)

# STORE RESULTS

with open(snakemake.output[0], "wb") as fh:
    pickle.dump(scores, fh)
