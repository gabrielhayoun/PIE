#Y a t'il une opportunité d'achat à l'instant présent?
#Voir les dictionnaires pour choisir la plateforme d'exchange et les crypto à comparer
import numpy as np
import pynance
#TODO: can't it be use in real time ?
def opportunity(exchange, crypto1, crypto2):
    # usage : opportunity('bittrex','bitcoin','ethereum')
    print("Exchange : "+exchange)
    print("Paire : "+crypto1+"/"+crypto2)
    #On prend une valeur par minute sur les 3 dernières heures
    S1=np.asarray(pynance.data.crypto.get_value(exchange,crypto1,"1m",180,False)['OPEN'])
    S2=np.asarray(pynance.data.crypto.get_value(exchange,crypto2,"1m",180,False)['OPEN'])
    
    serie = S1/S2
    zscore = (serie - serie.mean()) / np.std(serie)
    
    #On récupère le cours sur les 20 dernières seconde pour tenter d'y déceler une oppourtunité
    S1_actuel=pynance.data.crypto.get_bid_ask(exchange,crypto1,False)[1]
    S2_actuel=pynance.data.crypto.get_bid_ask(exchange,crypto2,False)[1]
    
    serie_actuelle=S1_actuel/S2_actuel
    zscore_actuel = (serie_actuelle - serie.mean()) / np.std(serie)
    i=0
    j=0
    k=0
    for i in range(len(S1_actuel)):
        if zscore_actuel[i] < -1:
            i+=1
        elif zscore_actuel[i] > 1:
            j+=1
        elif abs(zscore[i]) < 0.75:
            k+=1
    #Si le ratio passe plus de 5 seconde dans une zone d'interet un conseil de prise de position est proposé
    if i>5:
        print("Il y a une opportunité de gain, achat de "+crypto1+" et vente de "+crypto2)
    elif j>5:
        print("Il y a une opportunité de gain, achat de "+crypto2+" et vente de "+crypto1)
    elif k > 5:
        print("Le ratio est proche de la normale il peut être interessant de déboucler la position si il y en a une en cours")
    
    plot_opportunity_2s(zscore_actuel)
    plot_opportunity_3H(zscore)

def plot_opportunity_2s(zscore_actuel):
    import matplotlib.pyplot as plt
    #Affichage du cours présent (aide à la décision)
    plt.figure(figsize=(12,6))
    buy = zscore_actuel.copy()
    sell = zscore_actuel.copy()
    buy[zscore_actuel>-1] = -10
    sell[zscore_actuel<1] = -10
    plt.ylim(-2,3)
    plt.plot(zscore_actuel)
    plt.plot(buy,color='g', linestyle='None', marker='^')
    plt.plot(sell,color='r', linestyle='None', marker='^')
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.title("Rapport de valeur $\dfrac{crypto1}{crypto2}$ ACTUEL (20 dernières secondes) normalisé ")
    plt.legend(['Ratio', 'Achat de la position', 'Vente de la position'])
    plt.show()
  
def plot_opportunity_3H(zscore):
    import matplotlib.pyplot as plt 
    #Affichage du cours passée sur les 3 dernières heures
    plt.figure(figsize=(12,6))
    buy = zscore.copy()
    sell = zscore.copy()
    buy[zscore>-1] = -10
    sell[zscore<1] = -10
    plt.ylim(-2,3)
    plt.plot(zscore)
    plt.plot(buy,color='g', linestyle='None', marker='^')
    plt.plot(sell,color='r', linestyle='None', marker='^')
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.title("Rapport de valeur $\dfrac{crypto1}{crypto2}$ PASSEE (3 dernières heures) noarmalisé ")
    plt.legend(['Ratio', 'Achat de la position', 'Vente de la position'])
    plt.show()