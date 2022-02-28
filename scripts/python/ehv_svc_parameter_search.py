import pickle
from pathlib import Path

from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import ehv_parameter_search

data_dir = Path(
    "/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/" + 
    "202202071958"
)

Xs_train, y_train = ehv_parameter_search.load(data_dir)

model = make_pipeline(
    StandardScaler(),
    SVC(cache_size=4096, class_weight="balanced", random_state=0)
)

grid = HalvingGridSearchCV(
    estimator=model,
    param_grid=[{
        "svc__C": [0.1, 0.5, 1, 1.5, 2, 2.5],
        "svc__gamma": ['scale', 1/250, 1/100, 1/10, 1],
        "svc__kernel": ["rbf", "sigmoid"]
    }, {
        "svc__C": [0.1, 0.5, 1, 1.5, 2, 2.5],
        "svc__gamma": ['scale', 1/250, 1/100, 1/10, 1],
        "svc__degree": [2, 3, 4, 5],
        "svc__kernel": ["poly"]
    }],
    factor=3,
    resource='n_samples',
    max_resources='auto',
    min_resources=1000,
    aggressive_elimination=True,
    refit=False,
    n_jobs=32,
    cv=5,
    scoring='balanced_accuracy',
    verbose=3,
    return_train_score=True,
    random_state=0
).fit(
    Xs_train.fillna(0),
    y_train
)

# STORE RESULTS

with open("grid_svc.pickle", "wb") as fh:
    pickle.dump(grid, fh)
