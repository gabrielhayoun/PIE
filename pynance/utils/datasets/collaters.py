import torch

class TimeSeriesCollater:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    def __call__(self, batch):
        # arrays of structures
        x_list = []
        y_list = []
        for elem in batch:
            x, y = elem
            x_list.append(x)
            y_list.append(y)
        x_tensor = torch.stack(x_list, dim=0).to(device=self.device, dtype=self.dtype)
        y_tensor = torch.stack(y_list, dim=0).to(device=self.device, dtype=self.dtype)

        if(len(x_tensor.shape)==2):
            x_tensor = torch.unsqueeze(x_tensor, dim=2)
            y_tensor = torch.unsqueeze(y_tensor, dim=1)
        return x_tensor, y_tensor