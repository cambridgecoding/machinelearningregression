from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

################################################################################
################################### MODULE 4 ###################################
##################### Multiple variables linear regression #####################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

from functions import PolynomialRegression, model_plot_3d


bikes_df = pd.read_csv('./data/bikes_subsampled.csv')

# Learning activity 1: Fit a model of 2 variables and plot the model

features = ['temperature','humidity']
X = bikes_df[features].values
y = bikes_df['count'].values
