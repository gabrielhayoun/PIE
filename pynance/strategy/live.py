#Y a t'il une opportunité d'achat à l'instant présent?
#Voir les dictionnaires pour choisir la plateforme d'exchange et les crypto à comparer
import numpy as np
import pynance
#TODO: can't it be use in real time ?

def get_zscore(exchange, crypto1, crypto2, timeframe='1m', limit=180):
    S1=np.asarray(pynance.data.crypto.get_crypto_data(exchange, crypto1,timeframe, limit)['Open'])
    S2=np.asarray(pynance.data.crypto.get_crypto_data(exchange, crypto2,timeframe, limit)['Open'])
    
    serie = S1/S2
    mean = serie.mean()
    std = serie.std()
    zscore = (serie - mean) / std
    return zscore, mean, std

def get_opportunity(exchange, crypto1, crypto2, mean=None, std=None):
    # usage : opportunity('bittrex','bitcoin','ethereum')
    # print("Exchange : "+exchange)
    # print("Paire : "+crypto1+"/"+crypto2)
    #On prend une valeur par minute sur les 3 dernières heures
    if(mean is None or std is None):
        zscore, mean, std = get_zscore(exchange, crypto1, crypto2)
    
    #On récupère le cours sur les 20 dernières seconde pour tenter d'y déceler une apportunité
    S1_actuel, S2_actuel=pynance.data.crypto.get_bid_ask(exchange,crypto1,crypto2,limit=20)
    # S2_actuel=pynance.data.crypto.get_bid_ask(exchange,crypto2, limit=20)[1]
    
    serie_actuelle=S1_actuel/S2_actuel
    zscore_actuel = (serie_actuelle - mean) / std
    i=0
    j=0
    k=0
    for i in range(len(S1_actuel)):
        if zscore_actuel[i] < -1:
            i+=1
        elif zscore_actuel[i] > 1:
            j+=1
        elif abs(zscore_actuel[i]) < 0.75:
            k+=1
    #Si le ratio passe plus de 5 seconde dans une zone d'interet un conseil de prise de position est proposé
    if i>5:
        print("Il y a une opportunité de gain, achat de "+crypto1+" et vente de "+crypto2)
    elif j>5:
        print("Il y a une opportunité de gain, achat de "+crypto2+" et vente de "+crypto1)
    elif k > 5:
        print("Le ratio est proche de la normale il peut être interessant de déboucler la position si il y en a une en cours")

    return zscore_actuel