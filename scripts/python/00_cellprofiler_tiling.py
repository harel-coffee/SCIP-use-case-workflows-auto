#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy
import tifffile
from pathlib import Path
import operator
import pandas
import click


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
    images = list(Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/v2/images/").glob(f"{part}/*Ch1.*.tif"))
    tile_output = Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/v2/tiles/")

    ncols = 10
    nrows = 20
    ntiles = (len(images) // (ncols*nrows)) + ((len(images) % (ncols*nrows)) != 0)
    channels = [1, 3, 7]
    bounding = len(channels), 48, 48
    
    print(f"Writing {ntiles} tiles for {len(images)} images")

    for i in range(ntiles):
        tile = numpy.zeros(shape=(bounding[0], bounding[1]*nrows, bounding[2]*ncols), dtype=float)

        start = i * (ncols * nrows)
        end = (i + 1) * (ncols * nrows)
        for j, image in enumerate(images[start:end]):
            paths = [
                image.parent / image.name.replace("Ch1", "Ch"+str(c))
                for c in channels
            ]
          
            missing = False
            for p in paths:
                if not p.exists():
                    print(f"{str(p)} is missing")
                    missing = True
          
            if not missing:
                row = j // ncols
                col = j % ncols
                pixels = cropND(tifffile.imread(paths), bounding)

                tile[
                    :,
                    row*bounding[1]:(row+1)*bounding[1],
                    col*bounding[2]:(col+1)*bounding[2]
                ] = pixels

        (tile_output /  part).mkdir(exist_ok=True)
        for j, t in enumerate(tile):
            tifffile.imwrite(str(tile_output / part / f"{i}_Ch{channels[j]}.tiff"), t, photometric="minisblack")


# In[ ]:


if __name__ == "__main__":
    main()

