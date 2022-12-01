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

    def forward(self, X, future=0):
        if(len(X.shape) == 2):
            X = torch.unsqueeze(X, dim=2)
        X, h = self.rnn(X) # size is batch size x sequence length x output_size = input_size
        X = self.output_layer(X)
        if(future > 0):
            outputs = [X]
            for k in future:
                X, h = self.rnn(X)
                X = self.output_layer(X)
                outputs.append(X)
            return self.cat(outputs, dim=0) # for now 0, but I don't know if it's best.
        return X

    def predict(self, X, window=10):
        if(len(X.shape) == 2):
            X = torch.unsqueeze(X, dim=2)
        predicions = []
        for k in range(window):
            predicions.append(self.output_layer(self.rnn(X)[0]))

    def init_weights(self):
        pass

