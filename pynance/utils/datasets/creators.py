import abc
import pandas as pd
import torch
import pynance
from sklearn.model_selection import train_test_split
import numpy as np

class DatasetCreator(abc.ABC):
    torch_return_type = "torch"
    numpy_return_type = "numpy"
    def __init__(self, train_data=None, test_data=None) -> None:
        assert(any([train_data is not None, test_data is not None]))
        self._train_data = train_data
        self._test_data = test_data
    
    @abc.abstractmethod
    def get_train_sets(self, ratio, return_type):
        assert(return_type in [self.torch_return_type, self.numpy_return_type])
        assert(ratio > 0)
        assert(ratio <= 1)

    @abc.abstractmethod
    def get_test_set(self, return_type):
        assert(return_type in [self.torch_return_type, self.numpy_return_type])


class StockValuePredictionDatasetCreator(DatasetCreator):
    def __init__(self, train_data=None, test_data=None) -> None:
        super().__init__(train_data, test_data)
        assert(any([type(train_data) == pd.DataFrame, type(test_data) == pd.DataFrame]))

    def get_train_sets(self, ratio, return_type, window):
        super().get_train_sets(ratio, return_type)
        data = self._train_data[pynance.utils.conventions.close_name].values
        if(return_type==self.torch_return_type):
            dataset = pynance.utils.datasets.torch.SlidingWindowDataset(torch.Tensor(data), window)  
            train_length = int(len(dataset) * ratio)
            valid_length = len(dataset) - train_length
            train_set, valid_set = torch.utils.data.random_split(dataset, (train_length, valid_length))
            return train_set, valid_set
        elif(return_type==self.numpy_return_type):
            print("Numpy return type remains to be done.")
            pass
            # TODO: depending on the model used behind, we don't necessarily want to create sliding windows...
            # we may just want to return the data, not even splited I believe
            # x, y = pynance.utils.transform.get_sliding_windows(data, window)
            # (x_train, y_train), (x_test, y_test) = 

    def get_test_set(self, return_type, window):
        super().get_test_set(return_type)
        data = self._test_data[pynance.utils.conventions.close_name].values
        if(return_type==self.torch_return_type):
            dataset = pynance.utils.datasets.torch.SlidingWindowDataset(data, window)            
            return dataset
        elif(return_type==self.numpy_return_type):
            print("Numpy return type remains to be done.")

class StockValueRegressionDatasetCreator(DatasetCreator):
    def __init__(self, train_data=None, test_data=None, market=None) -> None:
        # we suppose that in this dataset, the market is the name of the column with
        # the valuation of the market
        # all other columns (except date) are supposed to be targets 
        super().__init__(train_data, test_data)
        self.market = market
        assert(any([type(train_data) == pd.DataFrame, type(test_data) == pd.DataFrame]))


    def get_train_sets(self, ratio, return_type):
        super().get_train_sets(ratio, return_type)
        market = self._train_data[self.market].values
        targets = self._train_data.loc[:,
                                    ~self._train_data.columns.isin(
                                        [self.market,
                                         pynance.utils.conventions.date_name])].values
        if(return_type==self.torch_return_type):
            market = torch.Tensor(market)
            targets = torch.Tensor(targets)
            dataset = torch.utils.data.TensorDataset(market, targets)  
            train_length = int(len(dataset) * ratio)
            valid_length = len(dataset) - train_length
            train_set, valid_set = torch.utils.data.random_split(dataset, (train_length, valid_length))
            return train_set, valid_set
        elif(return_type==self.numpy_return_type):
            X_train, X_valid, y_train, y_valid = train_test_split(market, targets, train_size=ratio)
            X_train = np.expand_dims(X_train, axis=1)
            X_valid = np.expand_dims(X_valid, axis=1)
            return X_train, X_valid, y_train, y_valid
            # maybe do something here so it returns something similar to pytorch (that we can use exactly the same way)

    def get_test_set(self, return_type):
        super().get_test_set(return_type)
        market = self._test_data[self.market].values
        targets = self._test_data.loc[:,
                                    ~self._test_data.columns.isin(
                                        [self.market,
                                         pynance.utils.conventions.date_name])].values
        if(return_type==self.torch_return_type):
            market = torch.Tensor(market)
            targets = torch.Tensor(targets)
            dataset = torch.utils.data.TensorDataset(market, targets)  
            return dataset
        elif(return_type==self.numpy_return_type):
            market = np.expand_dims(market, axis=1)
            return market, targets
