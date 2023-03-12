import numpy as np

# S1 et S2 sont les prévisions des deux actions co-intégrés étudiées et dont on a effectué une prévision
# La fonction performance permet de connaître les gains potentiels sur les simulations effectués en travaillant sur la fonction ratio des paires d'action co-intégrées
# r fait référence au risque qu'on est prêt à prendre. Par exemple pour r = 0.1, on achètera jamais plus de r*money/S1 du produit 1.
def get_performance(S1, S2, risk, money):
    assert(type(S1) == np.ndarray)
    assert(type(S2) == np.ndarray)
    ratios = S1/S2
    zscore = (ratios - ratios.mean()) / np.std(ratios) 
    # On peut à priori commencer à trader avec rien puisqu'on trade quand le ratio vaut 1 ou -1
    countS1 = 0
    countS2 = 0
    # TODO: we should be able to vectorize it completely.
    for i in range(len(ratios)):
        # Si le z-score est < -1, on achète S1 et on vend S2
        if zscore[i] < -1: 
            countS1 += money*risk/S1[i]
            countS2 -= ratios[i]*countS1
            money += -countS1*S1[i] + countS2*S2[i]
        # Si le z-score est > 1, on vend S1 et on achète S2
        elif zscore[i] > 1:
            countS1 += money*risk/S1[i]
            countS2 -= ratios[i]*countS1
            money += S1[i]*countS1 - S2[i]*countS2
        # Si le z-score est entre -0.5 et 0.5, on ferme les positions et on empoche les gains
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0          
    return float(money) # On retourne le gain potentiel 

# On veut permettre au client de choisir le nombre de pair sur lesquelles il va trader.
# On lui permet de  peut choisir la p-value minimale afin de sélectionner ses paires
# On lui permet de choisir son risque.
# En entrée : nb de pair, p value_limite, argent, dictionnaire dico1 pair/p_value, data sont les prévisions des cours, booléen risque (True si on veut trader sur les 
# paires ayant le meilleur gain potentiel et false si on veut trader sur toutes les paires ayant une p_value minimal)
# En sorti : gain total théorique de la stratégie (+ perfo de chaque pair) / graphe prévision + ordre d'achat sur les paires sélectionnées
# TODO: improve clarity (better naming) - and maybe also performance while doing so (use dataframe, vectorized operations) - on this function
def get_pairs(names):
    names.sort() # inplace I believe
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs

def get_best_action(df_coint, data_dicts, p_value_limit, money, risk, nb_pairs, sort_by_risk=True):
    df_coint_ = df_coint.copy(deep=True)
    names = list(data_dicts.keys())
    pairs = get_pairs(names)
    df_coint_ = df_coint_[df_coint_['ticker1'].isin(names) & df_coint_['ticker2'].isin(names)]
    assert len(df_coint_) == len(pairs), f'Pairs length: {len(pairs)} but {len(df_coint_)} pairs in cointegration dataframe.'
    print(f'Number of pairs to study in data: {len(df_coint_)}')
    # we can either sort by risk (~p-value) or by performance
    print(f'Selection of co-integrated pairs (p-value < {p_value_limit}).')
    df_coint_ = df_coint_[df_coint_['p-value'] < p_value_limit]

    df_coint_['performance'] = [
        get_performance(data_dicts[row['ticker1']], data_dicts[row['ticker2']], risk, money) for i, row in
        df_coint_.iterrows()]

    if sort_by_risk: 
        print('Sorting by p-value (~risk).')
        df_coint_.sort_values(by='p-value', inplace=True)
    else :    
        print('Sorting by performance.')
        df_coint_.sort_values(by='performance', inplace=True)

    print(f'Selection of best {nb_pairs} pairs.')
    df_coint_ = df_coint_.head(nb_pairs)

    print(f'Theoretical gain for {money} money (per pair).')
    print(df_coint_[['ticker1', 'ticker2', 'performance']])
    print("Total gain :", df_coint_['performance'].sum())
    return df_coint_

# def best_action_old(dico1, data, nb_pair, p_value_limite, money, risk, risque=False):
#     dico2 = {}
#     # On parcourt dico1 pour récupérer les paires dont les p_value sont inférieures à p_value_limite
#     for key, value in dico1.items():
#         if value < p_value_limite:
#             dico2[key] = value
#     if risque : 
#         dico3 = {}
#         for key, value in dico2.items():
#             dico3[key] = get_performance(data[key[0]], data[key[1]], risk, money)
#         # On trie dico3 par ordre décroissant de performance
#         dico3 = sorted(dico3.items(), key=lambda x: x[1], reverse = True)
#         # On récupère les nb_pair premières paires
#         dico3 = dico3[:nb_pair]
#         gain = []
#         for i in range(len(dico3)):
#             gain.append(get_performance(data[dico3[i][0][0]], data[dico3[i][0][1], risk, money]))
#     else :    
#         # On trie dico2 par ordre croissant de p_value
#         dico2 = sorted(dico2.items(), key=lambda x: x[1])
#         # On récupère les nb_pair premières paires
#         dico2 = dico2[:nb_pair]
#         gain = []
#         for i in range(len(dico2)):
#             gain.append(performance(data[dico2[i][0][0]], data[dico2[i][0][1]], risk, money))
#     # On affiche les gains potentiels de chaque paire
#     if risque :
#         for i in range(len(dico3)):
#             print("Gain potentiel de la paire", dico3[i][0], ":", dico3[i][1])
#             print("Gain Total théorique :", sum(gain))
#     else :
#         for i in range(len(dico2)):
#             print("Gain potentiel de la paire", dico2[i][0], ":", dico2[i][1])
#             print("Gain Total théorique :", sum(gain))
