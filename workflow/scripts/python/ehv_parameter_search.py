import numpy
from pandas.api.types import CategoricalDtype
import pyarrow.parquet as pq

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load(data_dir, pattern):

    df = pq.read_table(data_dir / "features.parquet").to_pandas()

    cat_type = CategoricalDtype(
        categories=sorted(df["meta_group"].astype(int).unique()), ordered=True)
    df["meta_group"] = df["meta_group"].astype(int).astype(cat_type)

    df["meta_replicate"] = df["meta_replicate"].astype(int)

    df = df.set_index(
        ["meta_type", "meta_object_number", "meta_replicate", "meta_suffix", "meta_group"])

    df = df[numpy.load(data_dir / "columns.npy", allow_pickle=True)]
    df = df.loc[numpy.load(data_dir / "index.npy", allow_pickle=True)]

    df = df[df["meta_label"] != "unknown"]
    label_cat_type = CategoricalDtype(
        categories=["mcp-_psba+", "mcp+_psba+", "mcp+_psba-", "mcp-_psba-"], ordered=True)
    df["meta_label"] = df["meta_label"].astype(label_cat_type)

    # PREP CLASSIFICATION INPUT

    enc = LabelEncoder()
    enc.classes_ = df.loc["Inf"]["meta_label"].cat.categories.values
    y = enc.transform(df.loc["Inf"]["meta_label"])

    # selection of the generic channel features for SCIP
    to_keep = df.filter(regex=pattern).filter(regex="feat").columns
    Xs = df.loc["Inf"][to_keep]
    Xs.shape

    # SPLIT DATA

    Xs_train, _, y_train, _ =  train_test_split(Xs, y, test_size=0.1, random_state=0)

    return Xs_train, y_train
