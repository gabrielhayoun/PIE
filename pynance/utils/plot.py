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
    
def plot_stock_values(plot_dict, saving_path):
    fig, ax = plt.subplots(figsize=(4, 3),
                           constrained_layout=True)
                        #    sharex=True, sharey=True)
    for key, (x, y) in plot_dict.items():
        sns.lineplot(x=x, y=y, label=key, ax=ax)
    ax.set_title(saving_path.stem)
    ax.tick_params(labelrotation=45)
    ax.set_xlabel('date')
    ax.set_ylabel('value')
    fig.savefig(saving_path, dpi=300)
    return fig, ax

# -------------- crypto live ----------

def plot_opportunity(zscore, ax=None):
    #Affichage du cours présent (aide à la décision)
    if(ax is None):
        fig, ax = plt.subplots(figsize=(6, 4))
    buy = zscore.copy()
    sell = zscore.copy()
    buy[zscore>-1] = -10
    sell[zscore<1] = -10
    ax.set_ylim(-2,3)
    ax.plot(zscore, label='ratio')
    ax.plot(buy, color='g', linestyle='None', marker='^', label='buy position')
    ax.plot(sell,color='r', linestyle='None', marker='^', label='sell position')
    ax.axhline(0, color='black')
    ax.axhline(1.0, color='red')
    ax.axhline(-1.0, color='green')
    return ax
