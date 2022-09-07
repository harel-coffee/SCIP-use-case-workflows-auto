import os
import time
import pickle
import sys
import multiprocessing
import logging
from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FactorAnalysis

import pandas
import numpy
import pyarrow.parquet as pq


def run_fa(df, moa, n_components):

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(process)d;%(asctime)s;%(levelname)s;%(message)s",
        datefmt='%d-%m-%Y %H:%M:%S'
    )
    logger.info("Start %d" % n_components)

    start = time.time()

    fa = FactorAnalysis(n_components=n_components)
    fa.fit(df[df["meta_compound"] == "DMSO"].filter(regex="feat").sample(n=50000))
    end_fa = time.time() - start

    grouped = df[df["meta_moa"] != "DMSO"].groupby(["meta_compound", "meta_concentration"])
    rows = numpy.empty(shape=(grouped.ngroups, n_components), dtype=float)
    meta = numpy.empty(shape=(grouped.ngroups, 2), dtype=object)
    for i, (idx, gdf) in enumerate(grouped):
        rows[i] = fa.transform(gdf.filter(regex="feat")).mean(axis=0)
        meta[i, 0] = idx[0]
        meta[i, 1] = idx[1]

    treatment_profiles = pandas.DataFrame(
        data=rows,
        index=pandas.MultiIndex.from_arrays(meta.T, names=["compound", "concentration"]),
        columns=["feat_%d" % i for i in range(n_components)],
    )
    treatment_profiles = treatment_profiles.merge(
        moa, left_index=True, right_index=True).reset_index()

    n = LeaveOneGroupOut().get_n_splits(
            treatment_profiles.filter(regex="feat"),
            treatment_profiles["moa"],
            treatment_profiles["compound"]
    )
    preds = numpy.empty(shape=(n, len(treatment_profiles)), dtype=object)
    for i, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(
            treatment_profiles.filter(regex="feat"),
            treatment_profiles["moa"],
            treatment_profiles["compound"]
    )):

        e = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        e.fit(
            X=treatment_profiles.filter(regex="feat").iloc[train_index],
            y=treatment_profiles.iloc[train_index]["moa"]
        )

        preds[i, test_index] = e.predict(treatment_profiles.filter(regex="feat").iloc[test_index])
        preds[i, train_index] = e.predict(treatment_profiles.filter(regex="feat").iloc[train_index])

    end = time.time() - start
    logger.info("Ready %d in %.2f s (fa: %.2f s)" % (n_components, end, end_fa))

    return n_components, preds, end

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(process)d;%(asctime)s;%(levelname)s;%(message)s",
        datefmt='%d-%m-%Y %H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    logger.info("test")

    multiprocessing.set_start_method("spawn")

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
        output = data_dir / "fa3_norandomstate.pickle"

    # paths = list(paths)[:2]
    df = pandas.concat([pq.read_table(p).to_pandas() for p in paths])
    logger.info(str(df.shape))
    df = df.drop(
        columns=df.filter(regex='meta_image_.*').columns.tolist() + [
                "meta_replicate", "meta_imagenumber", "meta_tablenumber"])
    logger.info(str(df.shape))

    df = df[~df.isna().any(axis=1)]
    logger.info(str(df.shape))

    labels = pq.read_table(labels_path).to_pandas()
    df = df.merge(
        labels,
        left_on="meta_filename",
        right_on="meta_image_filename_dapi",
        suffixes=("_", ""), how="inner"
    )
    logger.info(str(df.shape))

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
    logger.info(str(df.shape))

    N = 20
    components = numpy.arange(start=1, stop=101, step=1)
    moa = pandas.read_csv(moa_path).set_index(["compound", "concentration"])

    def arg_iterator():
        for _ in range(N):
            for comp in components:
                yield (df, moa, comp)

    logger.info("Using %s processes" % os.environ["PBS_NUM_PPN"])
    start = time.time()
    with multiprocessing.Pool(processes=int(os.environ["PBS_NUM_PPN"])) as pool:
        results = pool.starmap(run_fa, iter(arg_iterator()))
    logger.info(time.time() - start)

    with open(output, "wb") as fh:
        pickle.dump(results, fh)
