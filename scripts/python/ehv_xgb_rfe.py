import pickle
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

import ehv_parameter_search


data_dir = Path(
    "/home/maximl/scratch/data/ehv/results/scip/" + 
    "202202071958"
)
pattern = ".*(BF1|BF2|DAPI|SSC)$"

Xs_train, y_train = ehv_parameter_search.load(data_dir, pattern)

model = XGBClassifier(
    booster="gbtree",
    objective="multi:softmax",
    eval_metric="merror",
    tree_method="gpu_hist",
    use_label_encoder=False
)

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=model,
    step=1,
    n_jobs=5,
    verbose=3,
    cv=StratifiedKFold(5, random_state=0),
    scoring="balanced_accuracy_score",
    min_features_to_select=min_features_to_select,
).fit(Xs_train, y_train)

with open("ehv_xgb_rfe.pickle", "wb") as fh:
    pickle.dump(rfecv, fh)
