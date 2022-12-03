import torch

# https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN
# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?highlight=gru#torch.nn.GRU
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear

class TFnaive(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO : init weights
        # self.rnn = torch.nn.LSTM(input_size=input_size,
        #                          hidden_size=hidden_size,
        #                          num_layers=num_layers,
        #                          bias=True,
        #                          batch_first=True,
        #                          dropout=0,
        #                          proj_size=hidden_size)

        self.rnn = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=0)
        self.output_layer = torch.nn.Linear(in_features=hidden_size,
                                             out_features=input_size,
                                             bias=True)                        

    def forward(self, X, return_intermediary=False):
        assert(len(X.shape)==3)
        X, h = self.rnn(X) # size is batch size x sequence length x output_size = input_size
        X = self.output_layer(X)
        if(return_intermediary):
            return X
        return X[:, -1, :]

    def predict(self, X, window=10):
        X_base = X
        with torch.no_grad():
            X_pred_on_base = self.forward(X, return_intermediary=True)
            X_ = torch.unsqueeze(X_pred_on_base[:, -1, :], dim=1)
            X = torch.cat([X_base, X_], dim=1)
            for k in range(window):
                X_ = self.forward(X)
                X = torch.cat([X, torch.unsqueeze(X_, dim=1)], dim=1)
        return X, X_pred_on_base[:, :-1, :]

    def init_weights(self):
        pass

