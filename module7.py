from functions import save_figure
from os.path import splitext
figname = splitext(__file__)[0]+'_'
ifig = 0

################################################################################
################################### MODULE 7 ###################################
############################# Advanced fitting methods #########################
################################################################################

#Learning activity 1: use any sklearn model

from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import NonLinearRegression, organize_data


#Learning activity 2: Custom nonlinear regression
bikes_df = pd.read_csv('./data/bikes.csv')
temperature = bikes_df[['temperature']].values
bikes = bikes_df['count'].values
