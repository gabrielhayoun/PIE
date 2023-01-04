# pytorch
import torch
import sklearn

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
    
