import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


fn = 'C:/Users/Aryaman/PycharmProjects\L1B File/3DIMG_01AUG2022_0000_L1B_STD_V01R00.h5'
f = h5py.File(fn, 'r')

print(list(f.keys()))

TIR1_TEMP = f['IMG_TIR1_TEMP'][:]
TIR1 = f['IMG_TIR1'][:]
