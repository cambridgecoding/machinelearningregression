# Module 4: Multiple and polynomial regression

# Previous imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# New imports
from functions import PolynomialRegression
from mpl_toolkits.mplot3d.axes3d import Axes3D
from functions import model_plot_3d
from functions import polynomial_residual

# Load dataset
bikes_df = pd.read_csv('./data/bikes_subsampled.csv')

# Code after this
