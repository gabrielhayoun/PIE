import torch

class TFnaive(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        
        # TODO : init weights
        self.rnn = torch.nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bias=True,
                                 batch_first=False,
                                 dropout=0,
                                 proj_size=min(1, hidden_size-1))

    def forward(self, x):
        x = self.rnn(x)
        return x

    def predict(self, X, window=10):
        predicions = []
        for k in range(window):
            predicions.append(self.rnn(X))
