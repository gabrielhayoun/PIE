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
        #                          proj_size=input_size)

        self.rnn = torch.nn.Sequential(*[
                        torch.nn.GRU(input_size=input_size,
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     bias=True,
                                     batch_first=True,
                                     dropout=0),
                        torch.nn.Linear(in_features=hidden_size,
                                        out_features=input_size,
                                        bias=True)])                        
        # for GRU, the output size is the hidden size, which means we need a linear layer afterwards to project

    def forward(self, x, future=0):
        x, h = self.rnn(x) # size is batch size x sequence length x output_size = input_size
        if(future > 0):
            outputs = [x]
            for k in future:
                x, h = self.rnn(x)
                outputs.append(x)
            return self.cat(outputs, dim=0) # for now 0, but I don't know if it's best.
        return x

    def predict(self, X, window=10):
        predicions = []
        for k in range(window):
            predicions.append(self.rnn(X))

    def init_weights(self):
        pass

