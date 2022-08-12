import os
import time
import pickle
import sys
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, f1_score,
    make_scorer
)
import multiprocessing
import pandas
import numpy
import pyarrow.parquet as pq
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# DATA LOADING

try:
    paths = snakemake.input.paths
    moa_path = snakemake.input.moa_path
    labels_path = snakemake.input.labels
    output = snakemake.output[0]
except NameError:
    data_root = Path("/data/gent/vo/000/gvo00070/vsc42015/datasets/BBBC021")
    data_dir = data_root / "scip"
    paths = data_dir.rglob("features.parquet")
    labels_path = data_dir.parent / "labels.parquet"
    moa_path = data_root / "BBBC021_v1_moa.csv"
    output = data_dir / "fa3.pickle"

# paths = list(paths)[:3]
df = pandas.concat([pq.read_table(p).to_pandas() for p in paths])
df = df.drop(
    columns=df.filter(regex='meta_image_.*').columns.tolist() + [
            "meta_replicate", "meta_imagenumber", "meta_tablenumber"])

df = df[~df.isna().any(axis=1)]

labels = pq.read_table(labels_path).to_pandas()
df = df.merge(
    labels,
    left_on="meta_filename",
    right_on="meta_image_filename_dapi",
    suffixes=("_", ""), how="inner"
)

# Removing interplate variation

qq_dmso = df[df["meta_moa"] == "DMSO"].groupby(
    "meta_image_metadata_plate_dapi")[df.filter(regex="feat").columns].quantile((0.01, 0.99))

dfs = []
for idx, gdf in df.groupby("meta_image_metadata_plate_dapi"):
    df_scaled = (gdf.filter(regex="feat") - qq_dmso.loc[idx, 0.01])
    df_scaled /= (qq_dmso.loc[idx, 0.99] - qq_dmso.loc[idx, 0.01])
    df_scaled = pandas.concat([df_scaled, gdf.filter(regex="meta")], axis=1)

    dfs.append(df_scaled)

df = pandas.concat(dfs)
del dfs

nancols = df.columns[df.isna().any()]
df = df.drop(columns=nancols)

logging.getLogger().info(str(df.shape))


def run_fa(df, moa, n_components):

    start = time.time()

    fa = FactorAnalysis(random_state=0, n_components=n_components)
    fa.fit(df[df["meta_compound"] == "DMSO"].filter(regex="feat").sample(n=50000))

    dfs = []
    for idx, gdf in df[df["meta_moa"] != "DMSO"].groupby(["meta_compound", "meta_concentration"]):
        tmp_df = pandas.Series(data=fa.transform(gdf.filter(regex="feat")).mean(axis=0))
        tmp_df.index = ["feat_"+str(c) for c in tmp_df.index]
        tmp_df["meta_compound"] = idx[0]
        tmp_df["meta_concentration"] = idx[1]
        dfs.append(tmp_df)

    treatment_profiles = pandas.DataFrame(dfs).set_index(["meta_compound", "meta_concentration"])
    treatment_profiles.index.names = ["compound", "concentration"]
    del dfs

    treatment_profiles = treatment_profiles.merge(
        moa, left_index=True, right_index=True).reset_index()

    # Prediction

    results = cross_validate(
        cv=LeaveOneGroupOut(),
        X=treatment_profiles.filter(regex="feat"),
        y=treatment_profiles["moa"],
        groups=treatment_profiles["compound"],
        estimator=KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        return_train_score=True,
        return_estimator=True,
        scoring=dict(
            accuracy=make_scorer(accuracy_score),
            balanced_accuracy=make_scorer(balanced_accuracy_score),
            f1_macro=make_scorer(f1_score, average="macro", zero_division=0),
            precision_macro=make_scorer(precision_score, average="macro", zero_division=0)
        )
    )

    preds = []
    true = []
    for e, (_, test_index) in zip(
        results["estimator"],
        LeaveOneGroupOut().split(
            treatment_profiles.filter(regex="feat"),
            treatment_profiles["moa"],
            treatment_profiles["compound"]
        )
    ):
        pred = e.predict(treatment_profiles.filter(regex="feat").iloc[test_index])
        preds.extend(pred)
        true.extend(treatment_profiles["moa"].iloc[test_index])

    del results["estimator"]

    logging.getLogger().info("Ready %d" % n_components)

    return n_components, true, preds, results, time.time() - start

N = 20
components = numpy.arange(start=1, stop=101, step=1)
moa = pandas.read_csv(moa_path).set_index(["compound", "concentration"])

def arg_iterator():
    for _ in range(N):
        for comp in components:
            yield (df, moa, comp)

logging.getLogger().info("Using %s processes" % os.environ["PBS_NUM_PPN"])
start = time.time()
with multiprocessing.Pool(processes=int(os.environ["PBS_NUM_PPN"])) as pool:
    results = pool.starmap(run_fa, iter(arg_iterator()))
logging.getLogger().info(time.time() - start)

with open(output, "wb") as fh:
    pickle.dump(results, fh)
