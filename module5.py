# Module 5: Evaluating model performance

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import PolynomialRegression

# New imports
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

# Load dataset
bikes_df = pd.read_csv('./data/bikes_subsampled.csv')
temperature = bikes_df[['temperature']].values
bikes_count = bikes_df['count'].values

# Code after this
