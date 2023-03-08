import torch

class SlidingWindowDataset(torch.utils.data.Dataset):
    # TODO: can we make it faster ? Don't remember how __getitem__ works...
    def __init__(self, data, window):
        self.data = data # number of featues x time series length
        self.window = min(window, data.shape[1]-1)

    def __getitem__(self, index):
        x = self.data[:, index: index + self.window]
        y = self.data[:, index + self.window + 1]
        return x, y

    def __len__(self):
        return self.data.shape[1] - (self.window + 1)

class ForecastingInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __getitem__(self, index):
        return self.data[index] # #features x time series length

    def __len__(self):
        return len(self.data) 
    
    def get_iterator_of_inputs(self):
        return [self.data[k] for k in range(len(self.data))]