import abc
import pandas as pd
import torch
import pynance
from sklearn.model_selection import train_test_split
import numpy as np

class DatasetCreator(abc.ABC):
    torch_return_type = "torch"
    numpy_return_type = "numpy"
    def __init__(self, train_path=None, test_path=None) -> None:
        assert(any([train_path is not None, test_path is not None]))
        if(train_path is not None):
            assert(train_path.suffix==".csv")
        if(test_path is not None):
            assert(test_path.suffix==".csv")
        self._train_path = train_path
        self._test_path = test_path
    
    @abc.abstractmethod
    def get_train_sets(self, ratio, return_type):
        assert(return_type in [self.torch_return_type, self.numpy_return_type])
        assert(ratio > 0)
        assert(ratio <= 1)

    @abc.abstractmethod
    def get_test_set(self, return_type):
        assert(return_type in [self.torch_return_type, self.numpy_return_type])

    def read_csv(self):
        if(self._train_path is not None):
            self.train_df = self._read_csv(self._train_path)
        if(self._test_path is not None):
            self.test_df = self._read_csv(self._test_path)

    @abc.abstractmethod
    def _read_csv(self, path):
        pass


class StockValuePredictionDatasetCreator(DatasetCreator):
    def __init__(self, train_path=None, test_path=None) -> None:
        super().__init__(train_path, test_path)
        self.read_csv()

    def _read_csv(self, path):
        df = pd.read_csv(path,
                         parse_dates=[pynance.utils.conventions.date_name]
                        ).sort_values(by=pynance.utils.conventions.date_name)
        return df
        
    def get_train_sets(self, ratio, return_type, window):
        super().get_train_sets(ratio, return_type)
        data = self.train_df[pynance.utils.conventions.close_name].values
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
        data = self.test_df[pynance.utils.conventions.close_name].values
        if(return_type==self.torch_return_type):
            dataset = pynance.utils.datasets.torch.SlidingWindowDataset(data, window)            
            return dataset
        elif(return_type==self.numpy_return_type):
            print("Numpy return type remains to be done.")

class StockValueRegressionDatasetCreator(DatasetCreator):
    def __init__(self, train_path=None, test_path=None, market=None) -> None:
        # we suppose that in this dataset, the market is the name of the column with
        # the valuation of the market
        # all other columns (except date) are supposed to be targets 
        super().__init__(train_path, test_path)
        self.market = market
        self.read_csv()

    def _read_csv(self, path):
        df = pd.read_csv(path,
                         parse_dates=[pynance.utils.conventions.date_name]
                        ).sort_values(by=pynance.utils.conventions.date_name)
        return df

    def get_train_sets(self, ratio, return_type):
        super().get_train_sets(ratio, return_type)
        market = self.train_df[self.market].values
        targets = self.train_df.loc[:,
                                    ~self.train_df.columns.isin(
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
        market = self.test_df[self.market].values
        targets = self.test_df.loc[:,
                                    ~self.test_df.columns.isin(
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