# Projet en IA et trading
Dans le cadre du *Projet Ingénierie et Entreprise* à l'ISAE SUPAERO en dernière année.



## Installation

### Environnement conda
Modules nécessaires et installation dans un conda env :
```shell
conda create -n pienv python==3.9 pandas matplotlib yfinance pandas-datareader seaborn scikit-learn ipykernel ipympl -c conda-forge -c anaconda -y
```
Puis activer le :
```shell
conda activate pienv 
```

Ensuite installer la version de pytorch adaptée à votre système ([ici](https://pytorch.org/)) et [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) (`pip install tensorboard`).

Pour finir, installer le package `pynance` en utilisant le [Makefile](Makefile.md) depuis le dossier où se trouve le Makefile.
```shell
make install
```

### User.cfg file
Pour faciliter la gestion des chemins, un fichier, dont le chemin doit être ajouté aux variables d'environnement, est utilisé.

Créer un fichier `user.cfg` (en réalité n'importe quel nom fonctionne) et placer le où vous le souhaitez dans vos dossiers.

Il doit contenir :
```cfg
[pie]
path_to_data = votre/chemin/vers/le/dossier
path_to_results = votre/chemin/vers/le/dossier
path_to_trained_models = votre/chemin/vers/le/dossier
path_to_configuration_file = votre/chemin/vers/le/dossier
```

Pour l'instant seuls `path_to_data` et `path_to_data` vous seront utiles.

Description de chaque chemin :
- path_to_data : pointe vers le dossier contenant les fichiers *.txt* avec les noms des stocks dedans. Le format doit correspondre aux exemples fournis dans [ce dossier](data/tech_us.txt).
- path_to_results : pointe vers le dossier où seront automatiquement sauvegardés les résultats des entraînements par exemple. Rappel : sur git, on ne push que du code (sauf à utiliser Git LFS)
- path_to_trained_models : pointera vers le dossier contnant les modèles déjà entraînés. Sera utile lorsqu'on utilisera des modèles entraînés (par exemple dans la phase de stratégie)
- path_to_configuration_file : Sera utile dans un second temps pour la phase opérationnelle. Une fois que le code sera bien développé, il est plus aisé d'utiliser des fichiers de configuration pour lancer des entraînements voire des analyses.

Une fois cela fait, il est nécessaire d'ajouter le fichier aux variables d'environnement sous le nom : USERCFG.

#### Procédure Ubuntu et MacOS
Dans le fichier `.bashrc` sous votre `Home`, ajouter toute à la fin la ligne suivante :
```
export USERCFG="mon/chemin/vers/le/fichier/user.cfg"
```

Note : ctrl + h permet d'afficher les fichiers cachés (sur linux du moins). Sinon utiliser le terminal.

#### Procédure Windows 10
Voir par exemple ce [lien](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/). Attention a bien ajouter pour nom de variable `USERCFG`.

## Auteurs
- [CALOT Paul](https://www.linkedin.com/in/paul-calot-43549814b/)
- COURNUT Thomas
- [DRIF Norman](https://www.linkedin.com/in/norman-drif-85081119b/)
- [GERARD Hugo](https://www.linkedin.com/in/hugo-g%C3%A9rard-290a77241/)
- [HAYOUN Gabriel](https://www.linkedin.com/in/gabriel-hayoun/)
- [MRINI Soufiane](https://www.linkedin.com/in/soufiane-mrini-5b6375205/)
