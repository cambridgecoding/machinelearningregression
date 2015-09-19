#cca regression functions
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
    ax.plot_surface(X1, X2, Y, alpha = 0.2 , cmap = 'jet')

def save_figure(name, i):
    plt.savefig('../figures/'+name+str(i), bbox_inches='tight')
    plt.gcf().clear()

def PolynomialRegression(degree = 1):
    return make_pipeline(PolynomialFeatures(degree = degree), LinearRegression())

def PolynomialRidge(degree = 1, alpha = 1):
    return make_pipeline(PolynomialFeatures(degree = degree), StandardScaler(), Ridge(alpha = alpha))

def PolynomialLasso(degree = 1, alpha = 1):
    return make_pipeline(PolynomialFeatures(degree = degree), StandardScaler(), Lasso(alpha = alpha))




def polynomial_residual(degree, X, y):
    polynomial_regression= PolynomialRegression(degree = degree)
    polynomial_regression.fit(X, y)
    y_pred = polynomial_regression.predict(X)
    mae = mean_absolute_error(y, y_pred)
    return mae


def organize_data(to_forecast, window, horizon):
    """
     Input:
      to_forecast, univariate time series organized as numpy array
      window, number of items to use in the forecast window
      horizon, horizon of the forecast
     Output:
      X, a matrix where each row contains a forecast window
      y, the target values for each row of X
    """
    shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast,
                                        shape=shape,
                                        strides=strides)
    y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
    return X[:-horizon], y


class NonLinearRegression(object):

    def __init__(self, fun):
        self.fun = fun
        self._find_n_params()

    def fit(self, X, y):
        self.params = basinhopping(lambda p: self._get_residual(X,y,p), np.random.randn(self._n_params), niter = 10000, niter_success = 50)['x']

    def predict(self, X):
        return self.fun(X.flatten(), self.params)

    def get_params(self, deep=True):
        return {"fun": self.fun}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def _get_residual(self, X, y, p):
        return np.mean((self.fun(X.flatten(), p)-y)**2)

    def _find_n_params(self):
        for n_params in range(1,100)[::-1]:
            try:
                self.fun(1,np.random.rand(n_params))
            except:
                n_params += 1
                break
        self._n_params = n_params
