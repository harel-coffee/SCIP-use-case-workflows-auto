from os import link
import pandas
import numpy
import panel
import hvplot.pandas  # noqa
import hvplot
import holoviews
import pyarrow.parquet as pq
from pathlib import Path
from pandas.api.types import CategoricalDtype
import holoviews.operation.datashader as hd
from holoviews.selection import link_selections
import datashader
import time


data_dir = Path("/vsc/datasets/weizmann/EhV/v2/results/scip/202202071958/")


def load_fake():
    n = 1000

    df = pandas.DataFrame({
        "index": numpy.arange(n),
        "meta_label": numpy.random.choice(["c1", "c2", "c3"], size=n),
        "meta_type": numpy.random.choice(["ctrl", "inf"], size=n),
        "feat_area": numpy.random.normal(loc=0, scale=2, size=n),
        "dim_1": numpy.random.normal(loc=-1, scale=1.5, size=n),
        "dim_2": numpy.random.normal(loc=2, scale=1, size=n)
    })

    return df


def load():
    from joblib import dump, load

    if (data_dir / "dump.joblib").exists():
        df = load(data_dir / "dump.joblib")

    else:
        df = pq.read_table(data_dir / "features.parquet").to_pandas()

        cat_type = CategoricalDtype(
            categories=sorted(df["meta_group"].astype(int).unique()), ordered=True)
        df["meta_group"] = df["meta_group"].astype(int)#.astype(cat_type)
        df["meta_replicate"] = df["meta_replicate"].astype(int)

        df = df.set_index(
            ["meta_type", "meta_object_number", "meta_replicate", "meta_suffix", "meta_group"])

        df = df[numpy.load(data_dir / "columns.npy", allow_pickle=True)]
        df = df.loc[numpy.load(data_dir / "index.npy", allow_pickle=True)]

        df = df[df["meta_label"] != "unknown"]
        # df["meta_label"] = pandas.Categorical(
        #     df["meta_label"],
        #     categories=["mcp-_psba+", "mcp+_psba+", "mcp+_psba-", "mcp-_psba-"],
        #     ordered=True
        # )

        sel1 = ~(
            (df.index.get_level_values("meta_group").isin([6,7,8,9])) &
            (df.index.get_level_values("meta_type") == "Ctrl")
        )
        df = df[sel1]

        dimred = numpy.load(data_dir / "embedding/umap.npy")
        df["dim_1"] = dimred[:, 0]
        df["dim_2"] = dimred[:, 1]

        df = df.reset_index()

        dump(df, data_dir / "dump.joblib")

    return df


def scatter(value):
    points = holoviews.Points(df, kdims=["dim_1", "dim_2"])
    return points.opts(color=value)

start = time.time()
df = load()
# df = load_fake()
print("Data loading (%ds)" % (time.time() - start))

autocomplete = panel.widgets.AutocompleteInput(
    name='Scatter hue', options=df.columns.tolist(),
    placeholder='Start typing here', value="meta_label")

start = time.time()
scat = holoviews.DynamicMap(
    scatter, streams=[autocomplete.param.value]
)
datashaded = hd.rasterize(scat, aggregator=datashader.count())
print("Dimred scatter (%ds)" % (time.time() - start))

start = time.time()
bar1 = holoviews.Bars(df, kdims=["meta_type"]).aggregate(function=len)
bar2 = holoviews.Bars(df, kdims=["meta_label"]).aggregate(function=len)
print("Bar plots (%ds)" % (time.time() - start))

link = link_selections.instance()

panel.Column(
    panel.Row(panel.Column(autocomplete, link(datashaded)), panel.Column(link(bar1), link(bar2))),
).servable()