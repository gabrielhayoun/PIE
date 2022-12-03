# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def train(epochs,
          model,
          loss_fn,
          training_loader,
          validation_loader,
          optimizer,
          saving_path,
          saving_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(saving_path / 'tb_{}_{}'.format(saving_name, timestamp))

    best_vloss = 1_000_000.

    best_model_state = None 
    for epoch in range(1, epochs + 1):
        print('EPOCH {}:'.format(epoch))

        model.train(True)
        avg_loss = train_one_epoch(epoch,
                                   training_loader,
                                   model,
                                   loss_fn,
                                   optimizer,
                                   tb_writer)

        model.train(False)
        
        avg_vloss = eval(model, loss_fn, validation_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        tb_writer.add_scalars('Training vs. Validation Loss',
                               { 'Training' : avg_loss, 'Validation' : avg_vloss },
                               epoch + 1)
        tb_writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = saving_path / 'model_state_dict_{}_{}_{}.pt'.format(saving_name, timestamp, epoch)
            best_model_state = model.state_dict()

    if(best_model_state is not None):
        torch.save(best_model_state, model_path)

    return tb_writer

def train_one_epoch(epoch,
                    training_loader,
                    model,
                    loss_fn,
                    optimizer,
                    tb_writer):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    last_loss = running_loss / len(training_loader)
    tb_x = epoch * len(training_loader) + i + 1
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    return last_loss


def eval(model, loss_fn, validation_loader):
    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    return avg_vloss