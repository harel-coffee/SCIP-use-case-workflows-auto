# AUTOGENERATED! DO NOT EDIT! File to edit: ../workflow/notebooks/core/00_core.ipynb.

# %% auto 0
__all__ = ['load_from_sqlite_db', 'add_gating_table', 'add_gate', 'do_umap', 'SelectFromCollection', 'color_dimred', 'plot_gate',
           'plot_gate_zarr', 'plot_gate_zarr_channels', 'plot_gate_czi']

# %% ../workflow/notebooks/core/00_core.ipynb 4
from .common import *

# %% ../workflow/notebooks/core/00_core.ipynb 5
import sys
import yaml
import sqlite3
import math
import tifffile
import zarr
from aicsimageio import AICSImage

from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

from scip.masking import remove_regions_touching_border

import multiprocessing
import math

# %% ../workflow/notebooks/core/00_core.ipynb 7
def load_from_sqlite_db(path, gate=None):
    with sqlite3.connect(path) as con:

        if gate is None:
            df = pandas.read_sql_query("""
                SELECT * from data
            """, con)
        else:
            df = pandas.read_sql_query("""
                SELECT * from data
                INNER JOIN gates ON data.meta_id = gates.meta_id AND data.meta_file = gates.meta_file
                WHERE %s = 1
            """ % gate, con)

    df = df.loc[:,~df.columns.duplicated()]
    df["meta_replicate"] = df["meta_file"].apply(lambda a: a.split("_")[0])
    df["meta_timepoint"] = df["meta_file"].apply(lambda a: int(a.split("_")[1][1:]))
    df["meta_fiji"] = df["meta_id"]*2+1

    return df

def add_gating_table(path):
    with sqlite3.connect(path) as con:
        return con.execute(f"""
        CREATE TABLE IF NOT EXISTS gates (
            meta_id BIGINT NOT NULL,
            meta_file VARCHAR NOT NULL,
            PRIMARY KEY (meta_id,meta_file),
            FOREIGN KEY (meta_id,meta_file) REFERENCES data(meta_id,meta_file)
        )""")

def add_gate(path, name, df):
    with sqlite3.connect(path) as con:
        if "meta_"+name not in [i[1] for i in con.execute('PRAGMA table_info(gates)')]:
            con.execute("ALTER TABLE gates ADD COLUMN meta_%s INTEGER DEFAULT 0" % name)
        return con.executemany(f"""
            INSERT INTO gates (meta_id, meta_file, meta_%s) VALUES (:meta_id, :meta_file, 1)
            ON CONFLICT (meta_id, meta_file)
            DO UPDATE SET meta_%s = 1
        """ % (name, name), df[["meta_id", "meta_file"]].to_dict(orient="records"))

# %% ../workflow/notebooks/core/00_core.ipynb 13
def do_umap(name, data, **umap_args):
    projector = umap.UMAP(**umap_args)
    projection = projector.fit_transform(data)
    projection = pandas.DataFrame(projection, columns=["dim_%d" % i for i in range(1, projector.get_params()["n_components"]+1)])
    dump(projection, "data/umap/%s.dat" % name)

# %% ../workflow/notebooks/core/00_core.ipynb 14
from matplotlib.widgets import PolygonSelector
import matplotlib.path

class SelectFromCollection:
    def __init__(self, ax, collection):
        self.canvas = ax.figure.canvas
        self.collection = collection

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.lasso = PolygonSelector(ax, onselect=self.onselect)
        self.populations = []

    def onselect(self, verts):
        path = matplotlib.path.Path(verts)
        self.populations.append(numpy.nonzero(path.contains_points(self.xys))[0])
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()

# %% ../workflow/notebooks/core/00_core.ipynb 15
def color_dimred(dimred, feat):
    fig, ax = plt.subplots(dpi=150)
    norm = Normalize(vmin=feat.quantile(0.01), vmax=feat.quantile(0.99))
    seaborn.scatterplot(x=dimred[:, 0], y=dimred[:, 1], s=0.5, alpha=0.5, edgecolors="none", palette="viridis", hue_norm=norm, legend=None, hue=feat, ax=ax)
    ax.set_axis_off()
    ax.set_title(feat.name)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, shrink=0.7)
    return ax

# %% ../workflow/notebooks/core/00_core.ipynb 17
def plot_gate(sel, df, maxn=200, sort=None):
    df = df.loc[sel]

    if len(df) > maxn:
        df = df.sample(n=maxn)

    if sort is not None:
        df = df.sort_values(by=sort)

    fig, axes = plt.subplots(ncols=10, nrows=int(math.ceil(len(df) / 10)), dpi=150)
    axes = axes.ravel()
    for (idx, r), ax in zip(df.iterrows(), axes[:len(df)]):
        pixels = tifffile.imread(r["meta_1"], key=0)
        minr, minc, maxr, maxc = int(r["meta_bbox_minr"]), int(r["meta_bbox_minc"]), int(r["meta_bbox_maxr"]), int(r["meta_bbox_maxc"])

        ax.imshow(pixels[minr:maxr, minc:maxc])
        ax.set_axis_off()
    for ax in axes[len(df):]:
        ax.set_axis_off()

# %% ../workflow/notebooks/core/00_core.ipynb 18
def plot_gate_zarr(sel, df, mask, maxn=200, sort=None, channel=0, bbox=True):
    df = df.loc[sel]
    
    if len(df) > maxn:
        df = df.sample(n=maxn)
        
    if sort is not None:
        df = df.sort_values(by=sort)
    
    nrows = int(math.ceil(len(df) / 10))
    ncols = min(10, len(df))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, dpi=150, figsize=(ncols*3, nrows*3))
    axes = axes.ravel()
    i = 0
    for path, gdf in df.groupby("meta_path"):
        z = zarr.open(path, mode="r")
        for (idx, r) in gdf.iterrows():
            ax = axes[i]
            pixels = z[r["meta_zarr_idx"]]
            pixels = pixels.reshape(z.attrs["shape"][r["meta_zarr_idx"]])[channel]
            
            if bbox:
                minr, minc, maxr, maxc = int(r[f"meta_{mask}_bbox_minr"]), int(r[f"meta_{mask}_bbox_minc"]), int(r[f"meta_{mask}_bbox_maxr"]), int(r[f"meta_{mask}_bbox_maxc"])
                ax.imshow(pixels[minr:maxr, minc:maxc])
            else:
                ax.imshow(pixels)
            ax.set_axis_off()
            
            i+=1
    for ax in axes[len(df):]:
        ax.set_axis_off()

# %% ../workflow/notebooks/core/00_core.ipynb 19
def plot_gate_zarr_channels(selectors, df, mask, maxn=20, sort=None, show_mask=False, main_channel=3, smooth=0.75, channel_ind=[0], channel_names=["c"]):
    
    dfs = []
    for i, sel in enumerate(selectors):
        tmp_df = df[sel].copy()
    
        if len(tmp_df) > maxn:
            tmp_df = tmp_df.sample(n=maxn)

        if sort is not None:
            tmp_df = tmp_df.sort_values(by=sort)
            
        tmp_df["sel"] = i
        dfs.append(tmp_df)
    df = pandas.concat(dfs)
        
    nchannels = len(channel_ind)
    
    images = {}
    masks = {}
    values = {}
    extent = numpy.empty(shape=(nchannels, 2), dtype=float)
    extent[:, 0] = numpy.inf
    extent[:, 1] = -numpy.inf
    
    for path, gdf in df.groupby("meta_path"):
        z = zarr.open(path, mode="r")
        for (idx, r) in gdf.iterrows():
            pixels = z[r["meta_zarr_idx"]]
            pixels = pixels.reshape(z.attrs["shape"][r["meta_zarr_idx"]])[channel_ind]
            
            minr, minc, maxr, maxc = int(r[f"meta_{mask}_bbox_minr"]), int(r[f"meta_{mask}_bbox_minc"]), int(r[f"meta_{mask}_bbox_maxr"]), int(r[f"meta_{mask}_bbox_maxc"])

            images[r["sel"]] = images.get(r["sel"], []) + [pixels[:, minr:maxr, minc:maxc]]
            if show_mask:
                m = li.get_mask(dict(pixels=pixels), main_channel=main_channel, smooth=smooth)
                m = remove_regions_touching_border(m, bbox_channel_index=main_channel)
                arr = m["mask"][:, minr:maxr, minc:maxc]
            else:
                arr = numpy.full(shape=pixels.shape, dtype=bool, fill_value=True)[:, minr:maxr, minc:maxc]
            masks[r["sel"]] = masks.get(r["sel"], []) + [numpy.where(arr, numpy.nan, arr)]
            
            p =  numpy.where(arr, pixels[:, minr:maxr, minc:maxc], numpy.nan)
            extent[:, 0] = numpy.nanmin(numpy.array([extent[:, 0], numpy.nanmin(p.reshape(nchannels, -1), axis=1)]), axis=0)
            extent[:, 1] = numpy.nanmax(numpy.array([extent[:, 1], numpy.nanmax(p.reshape(nchannels, -1), axis=1)]), axis=0)

    fig = plt.figure(dpi=75, figsize=(len(channel_ind)*2.5, len(df)*0.8))
    grid = gridspec.GridSpec(1, len(selectors), figure=fig, wspace=0.1)
    cmap = plt.get_cmap('viridis')
    norms = [Normalize(vmin=a, vmax=b) for a,b in extent]
    
    gs = {
        k: grid[0, k].subgridspec(len(v), nchannels)
        for k, v in images.items()
    }
    for k, v in images.items():
        for i, image in enumerate(v):
            for j, (p, m, norm) in enumerate(zip(image, masks[k][i], norms)):
                ax = plt.Subplot(fig, gs[k][i, j])
                ax.imshow(cmap(norm(p)))
                if show_mask:
                    ax.imshow(m, alpha=0.3, cmap="Blues")
                ax.set_axis_off()
                fig.add_subplot(ax)
                if i == 0:
                    ax.set_title(channel_names[j])

# %% ../workflow/notebooks/core/00_core.ipynb 20
def plot_gate_czi(sel, df, maxn=200, sort=None, channels=[0], masks_path_col=None, extent=None):
    df = df.loc[sel]

    if len(df) > maxn:
        df = df.sample(n=maxn)

    if sort is not None:
        df = df.sort_values(by=sort)

    ncols = min(df.shape[0], 5)
    nrows = int(math.ceil(len(df) / ncols))
    fig, axes = plt.subplots(
        ncols=ncols, 
        nrows=nrows, 
        dpi=50,
        figsize = (ncols*2*len(channels), nrows*2)
    )
    axes = axes.ravel()
    i = 0
    
    compute_extent = False
    if extent is None:
        compute_extent = True

    if compute_extent:
        extent = numpy.full((df.shape[0], 2, len(channels)), dtype=float, fill_value=numpy.nan)
    pixels = []
    masks = []
    ids = []
    for path, gdf in df.groupby(["meta_path"]):
        ai = AICSImage(path, reconstruct_mosaic=False)
        for scene, gdf2 in gdf.groupby(["meta_scene"]):
            ai.set_scene(scene)
            for tile, gdf3 in gdf2.groupby(["meta_tile"]):
                print(tile, scene, end=" ")
                for (idx, r) in gdf3.iterrows():
                    ax = axes[i]

                    pixels_ = ai.get_image_data("CXY", Z=0, T=0, C=channels, M=tile)
                    minr, minc, maxr, maxc = int(r["meta_bbox_minr"]), int(r["meta_bbox_minc"]), int(r["meta_bbox_maxr"]), int(r["meta_bbox_maxc"])
                    
                    if compute_extent:
                        extent[i, 0] = pixels_[:, minr:maxr, minc:maxc].reshape(pixels_.shape[0], -1).min(axis=1)
                        extent[i, 1] = pixels_[:, minr:maxr, minc:maxc].reshape(pixels_.shape[0], -1).max(axis=1)
                    pixels.append(pixels_[:, minr:maxr, minc:maxc])
                
                    if "meta_id" in r:
                        ids.append(r.meta_id)
                    else:
                        ids.append(idx[-1])
                    
                    if masks_path_col is not None:
                        mask = numpy.load(r[masks_path_col])[:, minr:maxr, minc:maxc]
                        masks.append(mask)
                    
                    i+=1
    
    if compute_extent:
        min_ = extent[:, 0].min(axis=0)
        max_ = extent[:, 1].max(axis=0)
    else:
        min_ = extent[:, 0]
        max_ = extent[:, 1]
    
    for i, (ax, pixels_, id_) in enumerate(zip(axes, pixels, ids)):
        ax.imshow(numpy.hstack((pixels_ - min_[:, numpy.newaxis, numpy.newaxis]) / (max_ - min_)[:, numpy.newaxis, numpy.newaxis]))
        if len(masks) > 0:
            ax.imshow(numpy.hstack(numpy.where(masks[i] == id_, numpy.nan, 1)), cmap="Blues", alpha=.3)

    for ax in axes:
        ax.set_axis_off()
