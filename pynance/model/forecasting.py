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
        assert(len(X.shape)==3) # batch size x number of features x length - as returned by the collater
        # we want : batch size x sequence length x number of features (or channel size)
        X = torch.transpose(X, 1, 2) 
        X, h = self.rnn(X) # size is batch size x sequence length x output_size = input_size
        X = self.output_layer(X)
        if(return_intermediary):
            return X
        return X[:, -1, :]

    def predict(self, dataset, window):
        with torch.no_grad():
            list_preds = []
            for X in dataset:
                assert(len(X.shape) == 2) # nb feature x length
                X = torch.unsqueeze(X, dim=0)
                list_pred = []
                for k in range(window):
                    X_ = self.forward(X) # size is batch_size x number of features - batch size is 1
                    X_tmp = torch.cat([X, torch.unsqueeze(X_, dim=2)], dim=2)
                    X = X_tmp[:, :, 1:] # take last one and go again
                    list_pred.append(X_[0])
                list_preds.append(torch.stack(list_pred, axis=0))  # TODO: know what we are really supposed to return here ???
        return torch.transpose(torch.stack(list_preds, dim=0), 1, 2).cpu().numpy() # always return numpy
        # shape: number in dataset x nb features x nb samples
    
    def init_weights(self):
        pass

    def load(self, path_dir):
        state_dict = torch.load(path_dir / 'state_dict.pt')
        self.load_state_dict(state_dict)

    def save(self, saving_dir):
        torch.save(self.state_dict(), saving_dir / 'state_dict.pt')
