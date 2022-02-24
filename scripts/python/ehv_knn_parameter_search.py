import pickle
from pathlib import Path

from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb

import ehv_parameter_search

data_dir = Path(
    "/data/gent/vo/000/gvo00070/vsc42015/datasets/weizmann/EhV/v2/results/scip/" + 
    "202202071958"
)

Xs_train, y_train = ehv_parameter_search.load(data_dir)

model = make_pipeline_imb(
    StandardScaler(),
    RandomOverSampler(random_state=0),
    KNeighborsClassifier(n_jobs=5)
)

grid = HalvingGridSearchCV(
    estimator=model,
    param_grid={
        "kneighborsclassifier__n_neighbors": [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "kneighborsclassifier__weights": ["uniform", "distance"],
        "kneighborsclassifier__algorithm": ["ball_tree", "kd_tree", "brute"],
        "kneighborsclassifier__p": [1, 2]
    },
    factor=3,
    resource='n_samples',
    max_resources='auto',
    min_resources=1000,
    aggressive_elimination=True,
    refit=False,
    n_jobs=32,
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

with open("grid_knn.pickle", "wb") as fh:
    pickle.dump(grid, fh)
