# pytorch
import torch
import sklearn
from sklearn import linear_model
import numpy as np

# local
import pynance

class StockRegressionPredictionPytorch(torch.nn.Module):
    def __init__(self, input_size):
        # simple regression model for now
        self.input_size = input_size
        self.linear = torch.nn.Linear(input_size, out_features=1, bias=True)
        
    def forward(self, X):
        # X : batch size x input size
        X = self.linear(X)
        return X
    
    def predict(self, X):
        self.test()
        with torch.no_grad():
            preds = self.forward(X)
        self.train()
        return preds

    def score(self, X, y): # Rsquared socre
        preds = self.predict(X)
        return pynance.model.metrics.r_squared(y, preds)


class StockRegressionPredictionSklearn:
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

    def __init__(self, input_size, random_state=0) -> None:
        self.input_size = input_size
        self.kernel = sklearn.gaussian_process.kernels.DotProduct() + (
                        sklearn.gaussian_process.kernels.WhiteKernel())
        self.gpr = sklearn.gaussian_process.GaussianProcessRegressor(
                            kernel=self.kernel,
                            random_state=random_state,
                            n_features_in_=self.input_size)

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X):
        y_mean, y_std = self.gpr.predict(X, return_std=True)
        return y_mean, y_std
    
    def score(self, X, y):
        score = self.gpr.score(X, y)
        return score

class MultipleLinearRegression:
    def __init__(self, number_of_submodels) -> None:
        self.number_of_submodels = number_of_submodels
        self.submodels = [sklearn.linear_model.LinearRegression() for k in range(number_of_submodels)]

    def fit(self, x, y):
        # x : 1 x nb features x length
        # y : nb stocks x nb targets (Open / Close etc.) x nb samples (=length of time series)
        # here : features = targets ! (in general)
        x_transpose = np.transpose(x[0]) # nb samples x nb features
        y_transpose = np.transpose(y, (2, 0, 1)) # nb samples x nb stocks x nb 
        for i, reg in enumerate(self.submodels):
            # for regression we should have : X = n_samples, n_features
            # Y = n_samples, n_targets
            reg.fit(x_transpose, y_transpose[:, i, :])

    def score(self, x, y):
        scores = []
        x_transpose = np.transpose(x[0])
        y_transpose = np.transpose(y, (2, 0, 1))
        for i, reg in enumerate(self.submodels):
            score = reg.score(x_transpose, y_transpose[:, i, :])
            scores.append(score)
        return scores
