# Module 7: Predict the future with autoregression

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# New imports
from functions import organize_data

# Load dataset
bikes_df = pd.read_csv('./data/bikes.csv')
bikes = bikes_df['count'].values

# Code after this
