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


data_dir = Path("/vsc/datasets/weizmann/EhV/v2/results/scip/202202071958/")


def load_fake():
    n = 1000

    df = pandas.DataFrame({
        "index": numpy.arange(n),
        "meta_label": numpy.random.choice([0,1,2,3], size=n),
        "meta_type": numpy.random.choice([0,1], size=n),
        "feat_area": numpy.random.normal(loc=0, scale=2, size=n),
        "dim_1": numpy.random.normal(loc=-1, scale=1.5, size=n),
        "dim_2": numpy.random.normal(loc=2, scale=1, size=n)
    })

    return df


def load():
    from joblib import dump, load

    if (data_dir / "dump.joblib").exists():
        return load(data_dir / "dump.joblib")

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

        dump(df.reset_index(), data_dir / "dump.joblib")

        return df.reset_index()


def scatter(value):
    points = holoviews.Points(df, kdims=["dim_1", "dim_2"])
    return points.opts(color=value)

df = load()
# df = load_fake()

autocomplete = panel.widgets.AutocompleteInput(
    name='Scatter hue', options=df.columns.tolist(),
    placeholder='Start typing here', value="meta_label")

scat = holoviews.DynamicMap(
    scatter, streams=[autocomplete.param.value]
)
datashaded = hd.datashade(scat, aggregator=datashader.count())
spreaded = hd.dynspread(datashaded, threshold=0.50, how='over')

bar1 = df.hvplot.hist("meta_label")
bar2 = df.hvplot.hist("meta_type")

link = link_selections.instance()

panel.Column(
    panel.Row(panel.Column(autocomplete, link(spreaded)), panel.Column(link(bar1), link(bar2))),
).servable()
