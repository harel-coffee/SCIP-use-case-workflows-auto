from ehv import gmm_clustering, load as e_load, preprocessing_pipeline
from pathlib import Path
import numpy
import re
import pandas
from joblib import dump
import uuid

# data loading
df = e_load.load_raw_ideas_dir(
    Path("/data/weizmann/EhV/high_time_res"), 
    Path("/data/weizmann/EhV/weizmann-ehv-metadata/representations/ideas_features/"), 
    "ALL", 
    Path("/data/weizmann/EhV/weizmann-ehv-metadata/cell_populations/manual_gating/"),
    None, "Low/*.cif")
df = e_load.clean_column_names(df)
df = e_load.remove_unwanted_features(df)
df = e_load.tag_columns(df)

df = df[df["meta_label_coi"]]

reg = r"^meta_label_(.+)$"
label_vec = numpy.full((df.shape[0]), fill_value="unknown", dtype=object)
for col in df.filter(regex="(?i)meta_label_.*psba.*"):
    label_vec[df[col].values] = re.match(reg, col).groups(1)
    
df["meta_label"] = label_vec

df = df.reset_index(drop=True)

print("Data loaded")

# preprocessing transform
pipe = preprocessing_pipeline.make_pipeline_1()[:-1]
df = pipe.fit_transform(df)

print("Data preprocessed")

# do search
data = gmm_clustering.n_components_search(df, 12, range(2, 3), 1)
data = pandas.DataFrame(data)

print("Component search finished")

dump(data, "%s.dat" % uuid.uuid4())
