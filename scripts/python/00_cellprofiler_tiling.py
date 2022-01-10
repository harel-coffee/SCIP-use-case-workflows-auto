#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy
import tifffile
from pathlib import Path
import operator
import pandas
import click
import zarr


# In[32]:


def padding_func(vector, iaxis_pad_width, iaxis, kwargs):
    if sum(iaxis_pad_width) > 0:
        med = numpy.median(vector[iaxis_pad_width[0]:-iaxis_pad_width[1]])
        vector[:iaxis_pad_width[0]] = med
        vector[-iaxis_pad_width[1]:] = med


# In[33]:


def cropND(img, bounding):

    # compute possibly necessary padding widths
    padding = tuple(map(lambda a,b: abs(min(0, b-a)), bounding, img.shape))
    if sum(padding) > 0:

        # split padding into before and after part
        before_after = tuple(map(lambda a: (a//2, (a//2)+(a%2)), padding))
        img = numpy.pad(
            array=img,
            pad_width=before_after,
            mode=padding_func
        )

    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


@click.command()
@click.argument("part", type=str)
def main(part):
    path = Path(f"/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/v2/images/{part}.zarr")
    images = zarr.open(path)
    tile_output = Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/v2/tiles/")

    ncols = 10
    nrows = 20
    ntiles = (len(images) // (ncols*nrows)) + ((len(images) % (ncols*nrows)) != 0)
    channels = [1, 3, 7]
    channel_indices = [0, 2, 6]
    bounding = len(channels), 48, 48

    print(f"Writing {ntiles} tiles for {len(images)} images")

    for i in range(ntiles):
        out_path = tile_output / part / f"{i}_Ch{channels[0]}.tiff"
        if out_path.exists():
            continue

        tile = numpy.zeros(shape=(bounding[0], bounding[1]*nrows, bounding[2]*ncols), dtype=float)

        start = i * (ncols * nrows)
        end = min(len(images), (i + 1) * (ncols * nrows))
        for j, idx in enumerate(range(start, end)):
            pixels = images[idx].reshape(images.attrs["shape"][idx])[channel_indices]

            row = j // ncols
            col = j % ncols
            pixels = cropND(pixels, bounding)

            tile[
                :,
                row*bounding[1]:(row+1)*bounding[1],
                col*bounding[2]:(col+1)*bounding[2]
            ] = pixels

        (tile_output /  part).mkdir(exist_ok=True)
        for j, t in enumerate(tile):
            out_path = tile_output / part / f"{i}_Ch{channels[j]}.tiff"
            tifffile.imwrite(str(out_path), t, photometric="minisblack")


# In[ ]:


if __name__ == "__main__":
    main()

