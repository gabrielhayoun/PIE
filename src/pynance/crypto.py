# local
import pynance
import matplotlib.pyplot as plt

def main(path_to_cfg):
    print('\n#------------------ CRYPTO PROCESS -----------------#\n')
    parameters = pynance.config.cfg_reader.read(path_to_cfg, kind='crypto')

    fig1, ax1 = plt.subplots(figsize=(5, 4), constrained_layout=True) 
    fig2, ax2 = plt.subplots(figsize=(5, 4), constrained_layout=True)
    fig1.suptitle(r'Past $\frac{crypto1}{crypto}$ normalised (3 last hours)', fontsize=12)
    fig2.suptitle(r'Current $\frac{crypto1}{crypto}$ normalised (20 last seconds)', fontsize=12)

    saving_dir = pynance.utils.setup.get_results_dir(parameters['saving_name'])

    exchange = parameters['exchange'] 
    crypto_ticker1 = parameters['crypto_ticker1'] 
    crypto_ticker2 = parameters['crypto_ticker2'] 

    zscore, mean, std = pynance.strategy.live.get_zscore(exchange, crypto_ticker1, crypto_ticker2)

    ax1.set_xlabel('minutes')
    pynance.utils.plot.plot_opportunity(zscore, ax=ax1)
    fig1.savefig(saving_dir/'zscore_3h.png')

    fig1.show()
    # fig2.show()
    
    count = 1
    while True:
        zscore_actuel = pynance.strategy.live.get_opportunity(
            exchange, crypto_ticker1, crypto_ticker2, mean, std)

        ax2.clear()
        ax2.set_xlabel('seconds')
        pynance.utils.plot.plot_opportunity(zscore_actuel, ax=ax2)
        fig2.savefig(saving_dir/f'zscore_20s_{count}.png')
        fig2.show()

        count += 1
        user_input = input("Press Enter to continue ('stop' to stop) ...")
        if(user_input == 'stop'):
            break

    print('\n------------------ END -----------------\n')
