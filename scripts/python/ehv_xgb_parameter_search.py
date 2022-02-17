import pickle
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

import ehv_parameter_search


data_dir = Path(
    "/home/maximl/scratch/data/ehv/results/scip/" + 
    "202202071958"
)

Xs_train, y_train = ehv_parameter_search.load(data_dir)

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
        "max_depth": [7, 6, 5, 4, 3, 2],
        "learning_rate": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01],
        "subsample": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "colsample_bytree": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },
    factor=3,
    resource='n_estimators',
    n_candidates=3000,
    max_resources=4500,
    min_resources=2,
    aggressive_elimination=True,
    refit=False,
    n_jobs=8,
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

with open("grid_ehv_2.pickle", "wb") as fh:
    pickle.dump(grid, fh)
