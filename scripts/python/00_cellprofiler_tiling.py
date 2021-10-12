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


# In[ ]:


def make_meta():
    data = []

    ch1 = sorted(list(Path(tile_output).glob("H[1,2]_T[7,8,9]/*Ch1.tiff")))
    ch4 = sorted(list(Path(tile_output).glob("H[1,2]_T[7,8,9]/*Ch4.tiff")))
    ch6 = sorted(list(Path(tile_output).glob("H[1,2]_T[7,8,9]/*Ch6.tiff")))
    ch7 = sorted(list(Path(tile_output).glob("H[1,2]_T[7,8,9]/*Ch7.tiff")))
    ch11 = sorted(list(Path(tile_output).glob("H[1,2]_T[7,8,9]/*Ch11.tiff")))

    for p1, p4, p6, p7, p11 in zip(ch1, ch4, ch6, ch7, ch11):
        out = {
            "Image_FileName_1": str(p1.relative_to(p1.parents[2])),
            "Image_PathName_1": "/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High",
            "Image_FileName_4": str(p4.relative_to(p4.parents[2])),
            "Image_PathName_4": "/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High",
            "Image_FileName_6": str(p6.relative_to(p6.parents[2])),
            "Image_PathName_6": "/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High",
            "Image_FileName_7": str(p7.relative_to(p7.parents[2])),
            "Image_PathName_7": "/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High",
            "Image_FileName_11": str(p11.relative_to(p11.parents[2])),
            "Image_PathName_11": "/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High"
        }
        data.append(out)

    pandas.DataFrame(data).to_csv(str(tile_output.parent / "metadata.csv"), index=False)


# In[ ]:


@click.command()
@click.argument("part", type=str)
def main(part):
    images = list(Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High/images/").glob(f"{part}/*Ch1.*.tif"))
    tile_output = Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High/tiles/")

    ncols = 10
    nrows = 20
    ntiles = (len(images) // (ncols*nrows)) + ((len(images) % (ncols*nrows)) != 0)
    channels = [1, 4, 6, 7, 11]
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

        mini = tile.reshape(tile.shape[0], -1).min(axis=1)[..., numpy.newaxis, numpy.newaxis]
        maxi = tile.reshape(tile.shape[0], -1).max(axis=1)[..., numpy.newaxis, numpy.newaxis]
        tile = (tile - mini) / (maxi - mini)
        (tile_output /  part).mkdir(exist_ok=True)
        for j, t in enumerate(tile):
            tifffile.imwrite(str(tile_output / part / f"{i}_Ch{channels[j]}.tiff"), t, photometric="minisblack")


# In[ ]:


if __name__ == "__main__":
    main()

