import pickle
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import ehv_parameter_search

data_dir = Path(
    "/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/" + 
    "202202071958"
)
pattern = ".*(BF1|BF2|DAPI|SSC)$"

Xs_train, y_train = ehv_parameter_search.load(data_dir, pattern)

model = make_pipeline(
    StandardScaler(),
    LinearSVC(class_weight="balanced", random_state=0, dual=False)
)

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=model,
    step=1,
    n_jobs=5,
    verbose=3,
    cv=StratifiedKFold(5),
    scoring="balanced_accuracy",
    min_features_to_select=min_features_to_select,
    importance_getter="named_steps.linearsvc.coef_"
).fit(Xs_train.fillna(0), y_train)

with open(str(data_dir / "ehv_linsvc_rfe.pickle"), "wb") as fh:
    pickle.dump(rfecv, fh)
