import seaborn as sns
import matplotlib.pyplot as plt

def plot_losses(train_loss, valid_loss):
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    epochs = range(1, len(train_loss)+1)
    sns.lineplot(x=epochs, y=train_loss, ax=ax, label='train')
    sns.lineplot(x=epochs, y=valid_loss, ax=ax, label='valid')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    return fig, ax
    