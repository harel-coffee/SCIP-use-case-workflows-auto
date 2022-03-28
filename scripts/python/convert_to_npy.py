import numpy
import pyarrow.parquet as pq
from aicsimageio import AICSImage
from pathlib import Path
import multiprocessing
import operator
import os
from skimage.transform import rescale


WIDTH=112
HEIGHT=112

TARGET_WIDTH=28
TARGET_HEIGHT=28

SCALE = WIDTH / TARGET_WIDTH

CHANNELS=[4,5,6]
NORM_FEATS = ["feat_max_Bright", "feat_max_Oblique", "feat_max_PGC"]
DATA_DIR = Path("/home/maximl/scratch/data/cd7/800/results/scip/202203221745/")


def cropND(x):

    bounding = (len(CHANNELS), WIDTH, HEIGHT)
  
    # compute possibly necessary padding widths
    padding = tuple(map(lambda a,b: abs(min(0, b-a)), bounding, x.shape))
    if sum(padding) > 0:
      
        # split padding into before and after part
        before_after = tuple(map(lambda a: (a//2, (a//2)+(a%2)), padding))
        x = numpy.pad(
            array=x,
            pad_width=before_after,
            mode='constant'
        )
        
    start = tuple(map(lambda a, da: a//2-da//2, x.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return x[slices]


def load_cells(df):

    tile = df.iloc[0].meta_tile
    P = df.index.get_level_values("meta_P")[0]
    rep = df.index.get_level_values("meta_replicate")[0]
    scene = f"P{P}-D{rep}"
    path = df.iloc[0].meta_path
    
    mask = numpy.load(DATA_DIR / f"masks/{tile}_{scene}.npy")[CHANNELS[0]]
    im = AICSImage(path, reconstruct_mosaic=False, chunk_dims=["Z", "C", "X", "Y"])
    im.set_scene(scene)
    pixels = im.get_image_data("CZXY", T=0, C=CHANNELS)
    pixels = numpy.max(pixels, axis=1)

    norm = numpy.array([df[feat].max() for feat in NORM_FEATS])

    arr = numpy.empty(
        shape=(len(df), len(CHANNELS), TARGET_WIDTH, TARGET_HEIGHT), dtype=numpy.float32)

    for i, (idx, row) in enumerate(df.iterrows()):
        bbox = int(row.meta_bbox_minr), int(row.meta_bbox_minc), int(row.meta_bbox_maxr), int(row.meta_bbox_maxc)

        tmp = numpy.where(
            mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] == idx[-1],
            pixels[:, bbox[0]:bbox[2], bbox[1]:bbox[3]],
            0
        ) / norm[:, numpy.newaxis, numpy.newaxis]
        tmp = rescale(tmp, channel_axis=0, scale=1/SCALE, preserve_range=True)

        arr[i] = cropND(tmp)

    return arr


def main():
    idx = numpy.load(DATA_DIR / "neutrophils.npy", allow_pickle=True)
    print(len(idx))
    df = pq.read_table(DATA_DIR / "features.parquet").to_pandas().set_index(
        ["meta_panel", "meta_replicate", "meta_P", "meta_id"]
    ).loc["D"]
    df = df.sort_index()
    df = df.loc[idx]
 
    df["meta_path"] = df["meta_path"].map(
        lambda a: "/home/maximl/scratch/data/cd7/800/" + os.path.basename(a))
    
    with multiprocessing.Pool(processes=30) as pool:
        futures = []
        for idx, gdf in df.groupby(["meta_replicate", "meta_P"]):
            futures.append((idx, pool.apply_async(load_cells, (gdf,))))

        results = []
        for idx, future in futures:
            results.append(future.get())
            print(idx)

    arr = numpy.concatenate(results, axis=0)
    print(arr.shape)
    numpy.save(DATA_DIR / "neutrophil_images_scale_mnist.npy", arr)


if __name__ == "__main__":
    main()
