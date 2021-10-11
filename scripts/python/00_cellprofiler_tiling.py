#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import tifffile
from pathlib import Path
import operator
import pandas


# In[36]:


images = list(Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High/images/").glob("**/*Ch1.*.tif"))
tile_output = Path("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High/tiles/")


# In[392]:


def padding_func(vector, iaxis_pad_width, iaxis, kwargs):
    if sum(iaxis_pad_width) > 0:
        med = numpy.median(vector[iaxis_pad_width[0]:-iaxis_pad_width[1]])
        vector[:iaxis_pad_width[0]] = med
        vector[-iaxis_pad_width[1]:] = med


# In[393]:


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


# In[394]:


ncols = 10
nrows = 20
ntiles = (len(images) // (ncols*nrows)) + ((len(images) % (ncols*nrows)) != 0)
channels = [1, 4, 6, 7, 11]
bounding = len(channels), 48, 48


# In[395]:


for i in range(ntiles):
    tile = numpy.zeros(shape=(bounding[0], bounding[1]*nrows, bounding[2]*ncols), dtype=float)
    
    start = i * (ncols * nrows)
    end = (i + 1) * (ncols * nrows)
    for j, image in enumerate(images[start:end]):
        
        paths = [
            image.parent / image.name.replace("Ch1", "Ch"+str(c))
            for c in channels
        ]
        
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
    (tile_output /  image.parts[-2]).mkdir(exists_ok=True)
    for j, t in enumerate(tile):
        tifffile.imwrite(str(tile_output / image.parts[-2] / f"{i}_Ch{channels[j]}.tiff"), t, photometric="minisblack")


# In[399]:


data = []

ch1 = Path(tile_output).glob("*Ch1.tiff")
ch4 = Path(tile_output).glob("*Ch4.tiff")
ch7 = Path(tile_output).glob("*Ch7.tiff")

for p1, p2, p3 in zip(ch1, ch4, ch7):
    out = {
        "Image_FileName_1": str(p1.relative_to(p1.parents[2])),
        "Image_PathName_1": str("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High"),
        "Image_FileName_4": str(p2.relative_to(p2.parents[2])),
        "Image_PathName_4": str("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High"),
        "Image_FileName_7": str(p3.relative_to(p3.parents[2])),
        "Image_PathName_7": str("/user/gent/420/vsc42015/vsc_data_vo/datasets/weizmann/EhV/high_time_res/High")
    }
    data.append(out)

pandas.DataFrame(data).to_csv(str(tile_output.parents[3] / "metadata.csv"), index=False)


# In[ ]:




