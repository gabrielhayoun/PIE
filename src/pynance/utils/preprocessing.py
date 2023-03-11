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
            params = {'feature_range': (-1, 1)}
        elif(self.scale == 'norm'):
            scaling_class = sklearn.preprocessing.StandardScaler
            params = {}
        if(self.scale == 'None'):
            self.key_to_scaler = None
        else:
            key_to_scaler = {}
            for key, df_ in data.items():                                                                                                                                                                                                           
                key_to_scaler[key] = scaling_class(**params)
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
        data, keys = self._transform_features(data)
        data = self._transform_filter(data, keys)
        data = self._transform_scale(data, keys)
        return data
    
    def _transform_features(self, data):
        # assert(list(data.keys()) == list(self.features_info[0]))
        try:
            data_as_array = np.stack([np.transpose(df_[self.features].values) for df_ in data.values()], axis=0)
            assert(len(data_as_array.shape) == 3)
        except KeyError as e:
            raise KeyError(e)
        return data_as_array, list(data.keys())
    
    def _transform_filter(self, data, keys):
        return data

    def _transform_scale(self, data, keys):
        assert(type(data) == np.ndarray)
        if(self.scale == 'None'):
            return data
        else:
            assert(self.key_to_scaler is not None)
            for i in range(data.shape[0]):
                key = keys[i] # self.features_info[0][i]
                # data in the current hypothesis is n features n samples
                data[i] = np.transpose(self.key_to_scaler[key].transform(np.transpose(data[i])))
        return data
    
    def transform_df(self, df, key): # only one in this case
        data = self._transform_features_key(df, key)
        data = self._transform_filter(data, key)
        data = self._transform_scale_key(data, key)
        return data

    def _transform_features_key(self, df, key):
        assert(type(df) == pd.DataFrame)
        return np.transpose(df[self.features].values)

    def _transform_filter_key(self, data, key):
        return data

    def _transform_scale_key(self, data, key): # data: nb features x nb samples
        assert(type(data) == np.array)
        data = np.transpose(self.key_to_scaler[key].transform(np.transpose(data)))
        return data


    # ------------ inverse transform --------------- #
    def inverse_transform(self, data, keys=None, index=None):
        assert(all([key in self.features_info[0] for key in keys]))
        assert(len(keys) == data.shape[0]) # data shape should be: #stocks x #features x #length
        data = self._inverse_transform_scale(data, keys)
        data = self._inverse_transform_features(data, keys, index)
        return data

    def _inverse_transform_scale(self, data, keys):
        assert(type(data) == np.ndarray)
        if(self.scale == 'None'):
            return data
        else:
            assert(self.key_to_scaler is not None)
            for i in range(data.shape[0]):
                key = keys[i] # self.features_info[0][i]
                # size should be: nb samples x nb features
                data[i] = np.transpose(self.key_to_scaler[key].inverse_transform(np.transpose(data[i]))) # why no transpose ?
        return data

    def _inverse_transform_features(self, data, keys, index=None):
        data_dict = {}
        for i, key in enumerate(keys):
            data_dict[key] = pd.DataFrame(np.transpose(data[i]), columns=self.features, index=index)
        return data_dict
    
    def inverse_transform_df(self, df, key):
        data = self._inverse_transform_scale_key(data, key)
        df = self._inverse_transform_features_key(df, key)
        return data

    def _inverse_transform_scale_key(self, data, key): # data: nb features x nb samples
        assert(type(data) == np.array)
        data = np.transpose(self.key_to_scaler[key].transform(np.transpose(data)))
        return data

    def _inverse_transform_features_key(self, data, key, index=None):
        features = self.features_info[1]
        return pd.DataFrame(np.transpose(data), columns=features, index=index)

    
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