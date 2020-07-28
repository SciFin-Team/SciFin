# Test for Jiro

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import timeseries.timeseries as ts

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
ts1 = ts.timeseries(df)
ts1.simple_plot()

print("Awesome! It works :).")



