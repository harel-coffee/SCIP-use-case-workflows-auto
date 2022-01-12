# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/core/1a_load.ipynb (unless otherwise specified).

__all__ = ['load_raw_ideas_tree', 'check_should_load', 'load_raw_ideas_fcs', 'load_raw_ideas_dir',
           'load_raw_ideas_dir_dask', 'clean_column_names', 'remove_unwanted_features', 'tag_columns', 'add_merged_col']

# Cell
# export

import pandas
import os
import numpy
import seaborn
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
from importlib import reload
from ehv import core
from joblib import load, dump
from pathlib import Path
import uuid
import re
import scipy

from ehv import load as e_load, core

plt.rcParams['figure.facecolor'] = 'white'

numpy.random.seed(42)

# Cell
import fcsparser
import dask.dataframe as dd
from dask.delayed import delayed

# Cell

def load_raw_ideas_tree(tree_path, load_labels=False):
    logger = logging.getLogger(__name__)

    data = []
    columns = set()

    for timepoint_path in [p for p in os.listdir(tree_path) if os.path.isdir(os.path.join(tree_path, p))]:
        for replicate_path in os.listdir(os.path.join(tree_path, timepoint_path)):
            path = os.path.join(tree_path, timepoint_path, replicate_path)

            if not os.path.isfile(os.path.join(path, "focused.fcs")):
                continue

            logger.info(f"Loading dir {path}")

            meta, features = fcsparser.parse(os.path.join(path, "focused.fcs"))
            features["meta_timepoint"] = "".join(filter(str.isdigit, timepoint_path))
            features["meta_replicate"] = replicate_path

            if load_labels:
                features["meta_label"] = "unknown"
                for file in [p for p in os.listdir(path) if p.endswith(".txt")]:
                    label = os.path.splitext(file)[0]
                    object_numbers = pandas.read_csv(os.path.join(path, file), skiprows=1, delimiter="\t", index_col=0).index
                    features.loc[object_numbers, "meta_label"] = label

            logger.debug(f"Loaded dataframe with shape {features.shape}")

            if len(columns) == 0:
                columns |= set(features.columns.values.tolist())
            else:
                columns &= set(features.columns.values.tolist())

            data.append(features)

    return pandas.concat(data)[columns]

# Cell

def check_should_load(cif, load_df):
    tmp = {}
    tmp["meta_timepoint"] = int("".join(filter(str.isdigit, cif.parts[-1].split("_")[1])))
    tmp["meta_replicate"] = "R"+cif.parts[-1].split("_")[0][1]
    tmp["meta_group"] = cif.parts[-2]

    sel = load_df[
        (load_df["meta_timepoint"] == tmp["meta_timepoint"]) &
        (load_df["meta_replicate"] == tmp["meta_replicate"]) &
        (load_df["meta_group"] == tmp["meta_group"])
    ]
    return len(sel) != 0

def load_raw_ideas_fcs(cif, feature_dir, feature_postfix, label_dir):

    fcs = (feature_dir / cif.parts[-2] / (str(cif.stem) + "_%s" % feature_postfix)).with_suffix(".fcs")
    meta, features = fcsparser.parse(fcs)
    features["Object Number"] = features["Object Number"].astype(int)
    features = features.set_index("Object Number")
    features["meta_timepoint"] = int("".join(filter(str.isdigit, cif.parts[-1].split("_")[1])))
    features["meta_replicate"] = "R"+cif.parts[-1].split("_")[0][1]
    features["meta_group"] = cif.parts[-2]

    csv = label_dir / cif.parts[-2] / cif.with_suffix(".csv").parts[-1]
    labels = pandas.read_csv(csv).set_index("Object Number")
    labels.columns = ["meta_label_"+c for c in labels.columns]

    return features.join(labels, how="inner").reset_index()

def load_raw_ideas_dir(path: Path, feature_dir: Path, feature_postfix: Path, label_dir: Path, load_df:pandas.DataFrame=None, glob: str="*.cif"):
    logger = logging.getLogger(__name__)

    path = Path(path)
    dfs = []
    for cif in path.rglob(glob):
        logger.info(cif)
        if (load_df is None) or check_should_load(cif, load_df):
            features = load_raw_ideas_fcs(cif, feature_dir, feature_postfix, label_dir)
            dfs.append(features)

    return pandas.concat(dfs)

# Cell
def load_raw_ideas_dir_dask(path: Path, feature_dir: Path, feature_postfix: Path, label_dir: Path, load_df: pandas.DataFrame, glob: str = "*.cif"):
    logger = logging.getLogger(__name__)

    path = Path(path)
    dfs = []
    for cif in path.rglob(glob):
        if (load_df is None) or check_should_load(cif, load_df):
            dfs.append(delayed(load_raw_ideas_fcs)(cif, feature_dir, feature_postfix, label_dir))

    return dd.from_delayed(dfs)

# Cell

def clean_column_names(df):
    df.columns = df.columns.map(lambda c: c.lower().replace(" ", "_"))
    return df

# Cell
def remove_unwanted_features(df):
    todrop = df.filter(regex="(?i).*(uncompensated|raw|bkgd|saturation).*").columns

    return df.drop(columns=todrop)

# Cell

def tag_columns(df):
    df = df.copy()

    columns = [c for c in df.columns]
    system_cols = ["flow_speed", "time", "object_number"]
    for c in system_cols:
        if c in columns:
            columns[columns.index(c)] = "meta_system_"+c

    cat_reg = r".*count.*"
    for c in columns:
        if not "meta_" in c:
            if re.match(c, cat_reg):
                columns[columns.index(c)] = "feat_cat_"+c
            else:
                columns[columns.index(c)] = "feat_cont_"+c

    df.columns = columns
    return df

# Cell

def add_merged_col(df, cols):
    df["meta_id"] = df[cols].astype(str).agg(''.join, axis=1)
    return df