import pandas
import numpy
import panel
import hvplot.pandas  # noqa
import hvplot
from holoviews.streams import Selection1D
import holoviews
import pyarrow.parquet as pq
from pathlib import Path
from pandas.api.types import CategoricalDtype
import holoviews.operation.datashader as hd
from joblib import dump, load


data_dir = Path("/vsc/datasets/weizmann/EhV/v2/results/scip/202202071958/")

def load():

    if (data_dir / "dump.joblib").exists():
        return load(data_dir / "dump.joblib")

    else:
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
        df["meta_label"] = pandas.Categorical(
            df["meta_label"], 
            categories=["mcp-_psba+", "mcp+_psba+", "mcp+_psba-", "mcp-_psba-"], 
            ordered=True
        )

        sel1 = ~(
            (df.index.get_level_values("meta_group").isin([6,7,8,9])) & 
            (df.index.get_level_values("meta_type") == "Ctrl")
        )
        df = df[sel1]

        dimred = numpy.load(data_dir / "embedding/umap.npy")
        df["dim_1"] = dimred[:, 0]
        df["dim_2"] = dimred[:, 1]

        dump(df.reset_index(), data_dir / "dump.joblib")

        return df.reset_index()


def scatter(value):
    points = holoviews.Points(df, kdims=["dim_1", "dim_2"])
    return points.opts(color=value)


def hist(index, col="feat_sum_DAPI", by="meta_type"):
    if len(index) == 0:
        index = df.index
    return df.loc[index].hvplot.hist(col, by=by)


df = load()

autocomplete = panel.widgets.AutocompleteInput(
    name='Scatter hue', options=df.columns.tolist(),
    placeholder='Start typing here', value="meta_label")

scat = hd.rasterize(scatter()
    # holoviews.DynamicMap(
    #     scatter, streams=[autocomplete.param.value]
    # )
).opts(tools=["lasso_select"])

sel = Selection1D(source=scat)

h = holoviews.DynamicMap(hist, streams=[sel])

panel.Column(
    panel.Row(autocomplete),
    panel.Row(scat + h, width_policy="max"),
width_policy="max", height_policy="max").servable()
