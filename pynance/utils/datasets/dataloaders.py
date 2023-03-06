import abc
import pandas as pd
import torch
import pynance
from sklearn.model_selection import train_test_split
import numpy as np
import logging

class DataLoader(abc.ABC):
    supported_frameworks = ()
    torch_return_type = 'pytorch'
    sklearn_return_type = 'sklearn'

    def __init__(self, train_data, test_data, framework, preprocesser=None) -> None:
        assert(any([train_data is not None, test_data is not None]))
        assert(framework in self.supported_frameworks)
        self._train_data = train_data
        self._test_data = test_data
        self._framework = framework
        self._preprocesser = preprocesser
        
        self._train_set = None
        self._valid_set = None
        self._test_set = None
    
    @abc.abstractmethod
    def prepare_train_data(self): # return train and valid data
        pass

    @abc.abstractmethod
    def prepare_test_data(self): # return test data
        pass

    def get_train_data(self):
        return self._train_set
    
    def get_valid_data(self):
        return self._valid_set

    def get_test_data(self):
        return self._test_set
    
class PredictionDataLoader(DataLoader):
    supported_frameworks = ('pytorch')

    def __init__(self, train_data, test_data, framework, preprocesser, ratio, window) -> None:
        super().__init__(train_data, test_data, framework, preprocesser)
        self.window = window
        self.ratio = ratio
        self._train_set, self._valid_set = self.prepare_train_data()
        self._test_set = self.prepare_test_data()

    def prepare_train_data(self):
        if(self._train_data is None):
            return None, None
        super().prepare_train_data()
        if(self._preprocesser is not None):
            data = self._preprocesser.transform(self._train_data)
        else:
            data = self._train_data
        assert(len(data.shape) == 3) # 1 x number of features x time series length
        assert(data.shape[0] == 1)
        data = data[0]
        # data = self._train_data[pynance.utils.conventions.close_name].values
        if(self._framework==self.torch_return_type):
            dataset = pynance.utils.datasets.torch.SlidingWindowDataset(torch.Tensor(data), self.window)  
            train_length = int(len(dataset) * self.ratio)
            valid_length = len(dataset) - train_length
            train_set, valid_set = torch.utils.data.random_split(dataset, (train_length, valid_length))
            return train_set, valid_set
        elif(self._framework==self.sklearn_return_type):
            raise NotImplementedError('Sklearn is not implemented yet for PredictionDataLoader.')
            # TODO: depending on the model used behind, we don't necessarily want to create sliding windows...
            # we may just want to return the data, not even splited I believe
            # x, y = pynance.utils.transform.get_sliding_windows(data, window)
            # (x_train, y_train), (x_test, y_test) = 

    def prepare_test_data(self):
        if(self._test_data is None):
            return None
        super().prepare_test_data()
        if(self._preprocesser is not None):
            data = self._preprocesser.transform(self._test_data)
        else:
            data = self._test_data
        # data = self._test_data[pynance.utils.conventions.close_name].values
        if(self._framework==self.torch_return_type):
            dataset = pynance.utils.datasets.torch.SlidingWindowDataset(data, self.window)            
            return dataset
        elif(self._framework==self.sklearn_return_type):
            raise NotImplementedError('Sklearn is not implemented yet for PredictionDataLoader.')

class RegressionDataLoader(DataLoader):
    supported_frameworks = ('pytorch', 'sklearn')

    def __init__(self, train_data, test_data, framework, preprocesser, ratio, index) -> None:
        # we suppose that in this dataset, the market is the name of the column with
        # the valuation of the market
        # all other columns (except date) are supposed to be targets 
        super().__init__(train_data, test_data, framework, preprocesser)
        self.index = index
        assert(type(self.index) == int)
        self.ratio = ratio
        # assert(any([type(train_data) == pd.DataFrame, type(test_data) == pd.DataFrame]))
        self._train_set, self._valid_set = self.prepare_train_data()
        self._test_set = self.prepare_test_data()
    
    def prepare_train_data(self):
        if(self._train_data is None):
            return None, None
        super().prepare_train_data()
        if(self._preprocesser is not None):
            data = self._preprocesser.transform(self._train_data)
        else:
            data = self._train_data
        # market = self._train_data[self.index].values
        # targets = self._train_data.loc[:,
        #                             ~self._train_data.columns.isin(
        #                                 [self.index,
        #                                  pynance.utils.conventions.date_name])].values
        market = data[self.index]
        targets = np.delete(data, self.index, axis=0)
        assert(len(market.shape) == 2)
        assert(len(targets.shape) == 3)
        if(self._framework==self.torch_return_type):
            market = torch.Tensor(market)
            targets = torch.Tensor(targets)
            dataset = torch.utils.data.TensorDataset(market, targets)  
            train_length = int(len(dataset) * self.ratio)
            valid_length = len(dataset) - train_length
            train_set, valid_set = torch.utils.data.random_split(dataset, (train_length, valid_length))
            return train_set, valid_set
        elif(self._framework==self.sklearn_return_type):
            # targets is : number of stocks x number of features x length
            # we split according to LENGTH (in theory we should not necessarily do all of them together as they are independant)
            X_train, X_valid, y_train, y_valid = train_test_split(np.transpose(market, (1, 0)), np.transpose(targets, (2, 0, 1)), train_size=self.ratio)
            X_train = np.transpose(X_train, (1, 0))
            X_valid = np.transpose(X_valid, (1, 0))
            y_train = np.transpose(y_train, (1, 2, 0))
            y_valid = np.transpose(y_valid, (1, 2, 0))
            X_train = np.expand_dims(X_train, axis=1)
            X_valid = np.expand_dims(X_valid, axis=1)
            return (X_train, y_train), (X_valid, y_valid)

    def prepare_test_data(self):
        if(self._test_data is None):
            return None
        super().prepare_test_data()
        if(self._preprocesser is not None):
            data = self._preprocesser.transform(self._test_data)
        else:
            data = self._test_data
        market = data[self.index]
        targets = np.delete(data, self.index, axis=0)
        assert(len(market.shape) == 2)
        assert(len(targets.shape) == 3)
        # market = self._test_data[self.index].values
        # targets = self._test_data.loc[:,
        #                             ~self._test_data.columns.isin(
        #                                 [self.index,
        #                                  pynance.utils.conventions.date_name])].values
        if(self._framework==self.torch_return_type):
            market = torch.Tensor(market)
            targets = torch.Tensor(targets)
            dataset = torch.utils.data.TensorDataset(market, targets)  
            return dataset
        elif(self._framework==self.sklearn_return_type):
            market = np.expand_dims(market, axis=1)
            return (market, targets)
