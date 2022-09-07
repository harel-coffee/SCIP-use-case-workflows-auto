#|export

import pandas
import os
import numpy
import seaborn
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from importlib import reload
from scip_workflows import core
from pathlib import Path
import uuid
import re
import scipy

from pandas.api.types import CategoricalDtype

import pyarrow.parquet as pq

plt.rcParams['figure.facecolor'] = 'white'

numpy.random.seed(42)