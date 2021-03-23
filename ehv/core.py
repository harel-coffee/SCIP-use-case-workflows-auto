# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/00_core.ipynb (unless otherwise specified).

__all__ = ['load_config', 'load_from_sqlite_db', 'add_gating_table', 'add_gate', 'do_umap', 'SelectFromCollection']

# Cell
# export

import pandas
import os
import numpy
import seaborn
import logging
import matplotlib.pyplot as plt
from importlib import reload
from ehv import core

plt.rcParams['figure.facecolor'] = 'white'

numpy.random.seed(42)

# Cell
import sys
import yaml
import imagej
import sqlite3

# Cell

def load_config(config_file):
    with open(config_file, "r") as fh:
        config = yaml.load(fh, yaml.Loader)

    for k, v in config.items():
        setattr(sys.modules[__name__], k, v)

# Cell
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

# Cell
def do_umap(name, data, **umap_args):
    projector = umap.UMAP(**umap_args)
    projection = projector.fit_transform(data)
    projection = pandas.DataFrame(projection, columns=["dim_%d" % i for i in range(1, projector.get_params()["n_components"]+1)])
    dump(projection, "data/umap/%s.dat" % name)

# Cell
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
        print("Selection made!")

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()