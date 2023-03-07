import pandas as pd
import numpy as np
import sklearn
import joblib
import logging

class Preprocesser:
    supported_scalers = ('min-max', 'norm', 'None')
    def __init__(self, scale, features, filter=None) -> None:
        assert(scale in self.supported_scalers)
        self.scale = scale
        if(filter is not None):
            logging.warn('No filtering strategy is implemented yet. Ignoring not None `filter` parameter')
        self.filter = filter
        self.features = features

        self.features_info = None
        self.key_to_scaler = None
        self.filter_info = None
    
    # ---------- init ----------- #
    def init_preprocessor(self, data):
        try:
            data = dict(data)
        except Exception as e:
            raise Exception(f'Wrong type: {type(data)} - Error: {e}')
        self.data = data
        self._init_features(data)
        self._init_filter(data)
        self._init_scaler(data)

    def _init_features(self, data):
        self.features_info = (list(data.keys()), self.features)
    
    def _init_scaler(self, data):
        assert(self.features_info is not None)
        if(self.scale == 'min-max'):
            scaling_class = sklearn.preprocessing.MinMaxScaler
        elif(self.scale == 'norm'):
            scaling_class = sklearn.preprocessing.StandardScaler
        
        if(self.scale == 'None'):
            self.key_to_scaler = None
        else:
            key_to_scaler = {}
            for key, df_ in data.items():
                key_to_scaler[key] = scaling_class()
                array_to_scale = df_[self.features_info[1]].values
                key_to_scaler[key].fit(array_to_scale)
        self.key_to_scaler = key_to_scaler
    
    def _init_filter(self, data):
        pass

    # ---------- transform ------------ #
    def transform(self, data):
        try:
            data = dict(data)
        except Exception as e:
            raise Exception(f'Wrong type: {type(data)} - Error: {e}')
        data = self._transform_features(data)
        data = self._transform_filter(data)
        data = self._transform_scale(data)
        return data

    def _transform_features(self, data):
        assert(list(data.keys()) == list(self.features_info[0]))
        try:
            data_as_array = np.stack([np.transpose(df_[self.features].values) for df_ in data.values()], axis=0)
            assert(len(data_as_array.shape) == 3)
        except KeyError as e:
            raise KeyError(e)
        return data_as_array
    
    def _transform_filter(self, data):
        return data

    def _transform_scale(self, data):
        assert(type(data) == np.ndarray)
        if(self.scale == 'None'):
            return data
        else:
            assert(self.key_to_scaler is not None)
            for i in range(data.shape[0]):
                key = self.features_info[0][i]
                data[i] = np.transpose(self.key_to_scaler[key].transform(np.transpose(data[i])))
        return data

    # ------------ inverse transform --------------- #
    def inverse_transform(self, data, index=None):
        data = self._inverse_transform_scale(data)
        data =self._inverse_transform_features(data, index)
        return data

    def _inverse_transform_scale(self, data):
        assert(type(data) == np.ndarray)
        if(self.scale == 'None'):
            return data
        else:
            assert(self.key_to_scaler is not None)
            for i in range(data.shape[0]):
                key = self.features_info[0][i]
                data[i] = self.key_to_scaler[key].inverse_transform(data[i])
        return data

    def _inverse_transform_features(self, data, index=None):
        features = self.features_info[1]
        data_dict = {}
        for i, key in enumerate(self.features_info[0]):
            data_dict[key] = pd.DataFrame(np.transform(data[i]), columns=features, index=index)
        return data_dict

    # ---------- save and load info -------------- #
    def save(self, path):
        for key, scaler in self.key_to_scaler.items():            
            joblib.dump(scaler, path / f'{key}_scaler.bin', compress=True)
        joblib.dump(self.features_info, path / 'features_info.bin', compress=True)
    
    def load(self, path):
        key_to_scaler = {}
        features_info = None
        paths = path.glob('*.bin')
        for p in paths:
            if('scaler' in p.stem):
                key = p.stem.split('_')[0]
                key_to_scaler[key] = joblib.load(p)
            elif('features_info' == p.stem):
                features_info = joblib.load(p)
            else:
                print(f'{p} not corresponding to scalers or features info.')
        self.key_to_scaler = key_to_scaler
        self.features_info = features_info
        return True