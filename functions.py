# Advanced functions
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.metrics import mean_absolute_error

import matplotlib as mpl
fontsize_dict = {'xtick.labelsize' : 18,
                'ytick.labelsize' : 18,
                'legend.fontsize' : 18,
                'axes.titlesize'  : 18,
                'axes.labelsize' : 18}
for (key,val) in fontsize_dict.iteritems():
    mpl.rcParams[key] = val

def model_plot_3d(ax, model, x1, x2):
    X1, X2 = np.meshgrid(x1, x2)
    positions = np.vstack([X1.ravel(), X2.ravel()]).T
    y = model.predict(positions)
    Y = y.reshape(X1.shape)
    ax.plot_surface(X1, X2, Y, alpha = 0.2 , cmap = 'jet',\
                    linewidth=0.5, rstride=1, cstride=1, shade=True)

def PolynomialRegression(degree = 1):
    return make_pipeline(PolynomialFeatures(degree = degree,\
                            include_bias = False), LinearRegression())

def PolynomialRidge(degree = 1, alpha = 1):
    return make_pipeline(PolynomialFeatures(degree = degree,\
                            include_bias = False), StandardScaler(), Ridge(alpha = alpha))

def PolynomialLasso(degree = 1, alpha = 1):
    return make_pipeline(PolynomialFeatures(degree = degree,\
                            include_bias = False), StandardScaler(), Lasso(alpha = alpha))

def polynomial_residual(degree, X, y):
    polynomial_regression= PolynomialRegression(degree = degree)
    polynomial_regression.fit(X, y)
    y_pred = polynomial_regression.predict(X)
    mae = mean_absolute_error(y, y_pred)
    return mae

def organize_data(to_forecast, window, horizon):
    shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast,
                                        shape=shape,
                                        strides=strides)
    y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
    return X[:-horizon], y
