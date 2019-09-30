#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:26:42 2019

@author: hduser
"""

import pandas as pd
import datetime
import glob
import datetime
import os
from datetime import datetime  
from datetime import timedelta 
from pathlib import Path

path = ''                     # use your path
all_files = glob.glob(os.path.join(path, "*.pickle"))     # advisable to use os.path.join as this makes concatenation 
all_files.sort()

for i,f in enumerate(all_files):
    df = pd.read_pickle(f)
    print("file read " + f)
    df.to_parquet(Path(f).stem + '.parquet', compression='gzip')



