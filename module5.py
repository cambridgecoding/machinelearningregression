from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

################################################################################
################################### MODULE 5 ###################################
############################# Model evaluation #################################
################################################################################
# To be separated in a unique file #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functions import PolynomialRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error

bikes_df = pd.read_csv('./data/bikes_subsampled.csv')
temperature = bikes_df[['temperature']].values
bikes_count = bikes_df['count'].values
