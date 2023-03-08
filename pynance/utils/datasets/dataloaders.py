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

    def __init__(self, train_data, test_data, framework, preprocessor) -> None:
        assert(framework in self.supported_frameworks)
        self._train_data = train_data
        self._test_data = test_data
        self._framework = framework
        self._preprocessor = preprocessor
        
        self._train_set = None
        self._valid_set = None
        self._test_set = None
        self._infer_data = None
    
    @abc.abstractmethod
    def prepare_train_data(self): # return train and valid data
        pass

    @abc.abstractmethod
    def prepare_test_data(self): # return test data
        pass

    @abc.abstractmethod
    def prepare_infer_data(self): # return test data
        """ Create data under the hypothesis that no ground truth is ever provided.
        Intend to be used for forecasting / regression in practice.
        Should return a list of entries, such that each entry should be used directly in the network. 
        """
        pass

    def get_train_data(self):
        return self._train_set
    
    def get_valid_data(self):
        return self._valid_set

    def get_test_data(self):
        return self._test_set
    
    def get_infer_data(self):
        return self._infer_set
    
    def load_scaler(self, path):
        if(self._preprocessor is not None):
            self._preprocessor.load(path)

    def load_data(self, data, kind):
        assert(kind in ['train', 'test', 'infer'])
        if(kind=='train'):
            self._train_data = data
            self._train_set, self._valid_set = self.prepare_train_data()
        elif(kind=='test'):
            self._test_data = data
            self._test_set = self.prepare_test_data()
        elif(kind=='infer'):
            # infer data should only contain the inputs (not the targets !)
            self._infer_data = data
            self._infer_set = self.prepare_infer_data()
        else:
            raise ValueError(f'Not recognized kind: {kind}.')

    def convert_predictions_to_dict(self, predictions, keys):
        pred_dict = self._preprocessor.inverse_transform(predictions, keys=keys)
        return pred_dict

class PredictionDataLoader(DataLoader):
    supported_frameworks = ('pytorch')

    def __init__(self, train_data, test_data, framework, preprocessor, ratio, window) -> None:
        super().__init__(train_data, test_data, framework, preprocessor)
        self.window = window
        self.ratio = ratio
        self._train_set, self._valid_set, self._test_set = None, None, None
        if(self._train_data is not None):
            self._train_set, self._valid_set = self.prepare_train_data()
        if(self._test_data is not None):
            self._test_set = self.prepare_test_data()

    def prepare_train_data(self):
        data = self._prepare_data(self._train_data)
        # for now
        assert(data.shape[0] == 1) # not sure it's the right call
        data = data[0]
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
        """ Objective: for each dataframe in the dictionnary, the time serie(s) is extracted entirely.
        Ground target is created too here.

        Raises:
            NotImplementedError: raise en error for sklearn
        """
        data = self._prepare_data(self._test_data)
        # for now
        assert(data.shape[0] == 1) # not sure it's the right call
        data = data[0]
        if(self._framework==self.torch_return_type):
            dataset = pynance.utils.datasets.torch.SlidingWindowDataset(torch.Tensor(data), self.window)  
            return dataset
        elif(self._framework==self.sklearn_return_type):
            raise NotImplementedError('Sklearn is not implemented yet for PredictionDataLoader.')

    def prepare_infer_data(self):
        data = self._prepare_data(self._infer_data)
        if(self._framework==self.torch_return_type):
            data = torch.Tensor(data)
            return pynance.utils.datasets.torch.ForecastingInferenceDataset(data)
        elif(self._framework==self.sklearn_return_type):
            raise NotImplementedError('Sklearn is not implemented yet for PredictionDataLoader.')

    def _prepare_data(self, data):
        if(data is None):
            raise ValueError('Data is None.')
        assert(type(data == dict))
        if(self._preprocessor is not None):
            data = self._preprocessor.transform(data)
        else:
            data = data # TODO: should raise some kind of errors.
        assert(len(data.shape) == 3) # n x number of features x time series length
        if(data.shape[1] > data.shape[2]): # in theory: nb stocks (size of dict) x nb features x time series length 
            logging.warning(f'Shape is {data.shape} which should correspond to #stocks x #features x length time series.'
                             'There are more features than samples.')
        return data

class RegressionDataLoader(DataLoader):
    supported_frameworks = ('pytorch', 'sklearn')

    def __init__(self, train_data, test_data, framework, preprocessor, ratio, index) -> None:
        # we suppose that in this dataset, the market is the name of the column with
        # the valuation of the market
        # all other columns (except date) are supposed to be targets 
        super().__init__(train_data, test_data, framework, preprocessor)
        self.index = index
        assert(type(self.index) == int)
        self.ratio = ratio
        self._train_set, self._valid_set, self._test_set = None, None, None
        if(self._train_data is not None):
            self._train_set, self._valid_set = self.prepare_train_data()
        if(self._test_data is not None):
            self._test_set = self.prepare_test_data()
    
    def prepare_train_data(self):
        data = self._prepare_data(self._train_data)
        market = data[self.index]
        targets = np.delete(data, self.index, axis=0)
        assert(len(market.shape) == 2) # #features x time series length
        assert(len(targets.shape) == 3) # #stocks x #features x time series length
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
            # X_train = np.expand_dims(X_train, axis=1)
            # X_valid = np.expand_dims(X_valid, axis=1)
            return (X_train, y_train), (X_valid, y_valid)

    def prepare_test_data(self):
        data = self._prepare_data(self._test_data)
        market = data[self.index]
        targets = np.delete(data, self.index, axis=0)
        assert(len(market.shape) == 2) # #features x time series length
        assert(len(targets.shape) == 3) # #stocks x #features x time series length
        if(self._framework==self.torch_return_type):
            market = torch.Tensor(market)
            targets = torch.Tensor(targets)
            dataset = torch.utils.data.TensorDataset(market, targets)  
            return dataset
        elif(self._framework==self.sklearn_return_type):
            # market = np.expand_dims(market, axis=1)
            return (market, targets)

    def prepare_infer_data(self):
        data = self._prepare_data(self._infer_data) # targets is not interesting
        market = data[0] # we suppose only 1 dataframe was sent in {'market_name': df}
        if(self._framework==self.torch_return_type):
            # TODO: check this works
            market = torch.Tensor(market)
            dataset = torch.utils.data.TensorDataset(market)
            return dataset
        elif(self._framework==self.sklearn_return_type):
            # market = np.expand_dims(market, axis=1)
            assert(len(market.shape) == 2)
            return market # size is : #features x #length - that's all 
    
    def _prepare_data(self, data):
        if(data is None):
            raise ValueError('Data is None.')
        assert(type(data == dict))
        if(self._preprocessor is not None):
            data = self._preprocessor.transform(data)
        else:
            data = data # TODO: should raise some kind of errors.
        assert(len(data.shape) == 3) # n x number of features x time series length
        if(data.shape[1] > data.shape[2]): # in theory: nb stocks (size of dict) x nb features x time series length 
            logging.warning(f'Shape is {data.shape} which should correspond to #stocks x #features x length time series.'
                            'There are more features than samples.')
        return data

    def convert_predictions_to_dict(self, predictions, keys):
        all_keys = self._preprocessor.features_info[0]
        key_predicted = []
        for key in all_keys:
            if(key not in keys):
                key_predicted.append(key) 
        pred_dict = self._preprocessor.inverse_transform(predictions, keys=key_predicted)
        return pred_dict