# Module 6: Avoid overfitting with regularisation

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import PolynomialRegression
from sklearn.cross_validation import cross_val_score

# New imports
from functions import PolynomialRidge, PolynomialLasso
from sklearn.grid_search import GridSearchCV

# Load dataset
bikes_df = pd.read_csv('./data/bikes.csv')

# Code after this
