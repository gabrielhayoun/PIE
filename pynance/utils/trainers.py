# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import abc
from tqdm import tqdm
import logging
import pynance

class Trainer(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def train(model, train_set, valid_set):
        pass

    @abc.abstractmethod
    def evaluate(model, valid_set):
        pass

class TorchTrainer(Trainer):
    def __init__(self,
                 learning_rate,
                 epochs,
                 batch_size,
                 device,
                 dtype,
                 saving_dir,
                 collater_fn) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.saving_dir = saving_dir
        self.collater_fn = collater_fn
        self.loss_function = torch.nn.MSELoss() # TODO: modify that

    def train(self,
            model,
            train_set,
            valid_set):
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_writer = SummaryWriter(self.saving_dir / 'tb')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_loader = torch.utils.data.DataLoader(
                            train_set,
                            batch_size=self.batch_size,
                            shuffle=True,
                            collate_fn=self.collater_fn)
        valid_loader = torch.utils.data.DataLoader(
                                    valid_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    collate_fn=self.collater_fn)

        best_vloss = 1_000_000.
        loss_list = []
        vloss_list = []
        best_model_state = None 
        model_path = self.saving_dir / 'model_state_dict_epoch:None.pt'
        pbar = tqdm(range(1, self.epochs + 1))
        for epoch in pbar:
            model.train(True)
            avg_loss = self._train_one_epoch(
                                    epoch,
                                    model,
                                    train_loader,
                                    optimizer,
                                    tb_writer)
            model.train(False)

            avg_vloss = self._evaluate(model, valid_loader)

            message = "epoch:{}-train{:0.3f}-valid{:0.3f}".format(epoch, avg_loss, avg_vloss)
            pbar.set_description(message)
            # logging.log(msg=message)
    
            # Log the running loss averaged per batch
            # for both training and validation
            tb_writer.add_scalars('Training vs. Validation Loss',
                                { 'Training' : avg_loss, 'Validation' : avg_vloss},
                                epoch + 1)
            tb_writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = self.saving_dir / 'model_state_dict_epoch:{}.pt'.format(epoch)
                best_model_state = model.state_dict()

            loss_list.append(avg_loss)
            vloss_list.append(avg_vloss)
        if(best_model_state is not None):
            torch.save(best_model_state, model_path)
        self._plot_losses(loss_list, vloss_list)
        # logging.log(f'Training finished. Best model state dict saved at {model_path}.')
        return tb_writer

    def _train_one_epoch(self,
                        epoch,
                        model,
                        train_loader,
                        optimizer,
                        tb_writer):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        last_loss = running_loss / len(train_loader)
        tb_x = epoch * len(train_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        return last_loss

    def _evaluate(self, model, valid_loader):
        model.train(False)
        running_loss = 0.0
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            outputs = model(inputs)
            loss = self.loss_function(outputs, labels)
            running_loss += loss.item()
        avg_vloss = running_loss / (i + 1)
        model.train(True)
        return avg_vloss
    
    def evaluate(self, model, valid_set):
        valid_loader = torch.utils.data.DataLoader(
                                    valid_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    collate_fn=self.collater_fn)
        return self._evaluate(model, valid_loader)

    def _plot_losses(self, train_loss, valid_loss):
        fig, ax = pynance.utils.plot.plot_losses(train_loss, valid_loss)
        fig.savefig(self.saving_dir / 'losses.png', dpi=300)

class SklearnTrainer(Trainer):
    def __init__(self, saving_dir) -> None:
        super().__init__()
        self.saving_dir = saving_dir

    def train(self,
              model,
              train_set,
              valid_set):
        x_train, y_train = train_set
        model.fit(x_train, y_train)
        self.save(model)
    
    def evaluate(self, model, valid_set):
        x_valid, y_valid = valid_set
        return model.score(x_valid, y_valid)

    def save(self, model):
        # TODO: may be add an asserttion to make sure the right function is there
        saving_path = self.saving_dir / 'model'
        saving_path.mkdir(parents=True)
        model.save(saving_path)