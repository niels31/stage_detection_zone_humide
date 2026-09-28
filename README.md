# Stage SHREQ
Ce dépôt contient les codes du modèle développé pendant le stage SHREQ d'Avril 2026 à Septembre 2026 au groupe OSECC du Cerema Occitanie.  
Ce stage visait à améliorer et adapter à la Haute-Garonne la méthode REAUZOH.  
  
# Installation
le fichier /requirements.txt contient les librairies python utilisée pour le développement des scripts python de ce dépôt.  
  
# Utilisation
Le dossier /pipeline contient les scripts du modèle au format .py, directement utilisables.  
Les scripts .py y sont numérotés dans l'ordre nécessaire à leur utilisation pour la reproduction de la méthode.  
  
## script 0_bdtopo_mask.sql
Ce script est à exécuter sur une base de données PGSQL incluant les couches de la BD TOPO de l'IGN qui sont mentionnées dans les requêtes. Ce script permet de créer une couche pgsql bdtopo_mask qui est utilisée par le script 1_create_datasets.py

## script 1_create_datasets.py 
Ce script crée le jeu de données de référence. 

### entrée
Il prend en entrée le chemin vers un fichier de configuration .yaml. Le format d'un tel fichier et la signification des différents éléments à y spécifier est détaillé dans le fichier d'exemple 1_create_datasets.yaml.

### sorties
Ce script exporte, dans le dossier paramétré dans le fichier de configuration, deux dataframe pandas correspondant au jeu de données de référence des individus labellisés zones humides et à celui des individus labellisés non zones humides.

### exécution
Pour exécuter ce fichier, il faut se placer dans un environnement python vérifiant les dépendances nécessaires (cf requirements.txt) et dans le dossier /pipeline dans un terminal de commande et exécuter :  
python 1_create_datasets.py /chemin/du/fichier/de/config.yaml

## script 2_training.py
Ce script entraîne le modèle de cartographie des zones humides à partir du jeu de référence issus du script 1_create_datasets.py.

### entrées
entrées nécessaires :
- seed (entier): graine pour fixer la rng du script
- out_folder (str): dossier d'export des sorties
- zh_indivs_fp (str): chemin du fichier zh_indivs.pkl exporté par 1_create_datasets.py
- znh_indivs_fp (str): chemin du fichier znh_indivs.pkl exporté par 1_create_datasets.py

entrées optionnelles : 
- zh_group_col (str): nom de la colonne du dataframe des individus zones humides utilisé pour la gestion de l'auto-corrélation spatiale (défaut : "nom")
- znh_autocorrel_grid_size (entier): taille des carreaux utilisés pour la gestion de l'auto-corrélation spatiale des non zones humides en nombre de pixels (défaut : 500)
- train_test_frac (float): fraction entre 0 et 1 de la division entre jeu d'entraînement et jeu de test (défaut : 0.75)
- optim_n_trials (entier): nombre d'itérations de l'optimisation des hyperparams (défaut : 15)
- season_filter_graphic (bool): export optionnel de l'histogramme du pourcentage de chaque mois conservé pour l'entraînement du modèle (défaut : False)
- roc_curve (bool): export optionnel de la courbe ROC du jeu de test, avec AUC et valeur des seuils optimaux précisés dessus (défaut : False)
- model_analysis (bool): export de
   - graphique des importances par permutation et informations mutuelles de chaque CP
   - tableau latex des variables les plus corrélées avec chaque CP
   - visualisation de la relation entre chaque CP et la probabilité d'être une zone humide selon le modèle
   (défaut : False)

### sorties
- model.pkl : dictionnaire des paramètres du modèle
les attributs de ce dictionnaire sont:
    - rf : objet sklearn.ensemble.RandomForestClassifier entraîné
    - column_orders : ordre des colonnes pour formatter les dataframes sur lesquels appliquer le modèle
    - pca : objet sklearn.decomposition.PCA contenant les coordonnées des composantes principales, calculées sur le jeu d'entraînement, retenues
    - proba_seuil_optim : seuil minimal de probabilité d'être une zone humide selon rf pour classer un individu en zone humide
    - scalers : dictionnaire d'objets /lib/shreklib/torchScaler contenant les moyennes et écart-types calculés sur le jeu d'entraînement pour le centrage et réduction des variables
    - to_keep : indices des lignes du df zh_indivs.pkl retenues pour l'entraînement du modèle
- matrice_de_confusion.csv : matrice de confusion du jeu de test par classe majoritaire (pas pour chaque individu mensuel mais pour chaque pixel)
- metriques.csv : précision, rappel et f1_score de chaque classe
- train_dataset.pkl : jeu d'entraînement avec trois plis pour validation croisée (colonne "pli" du dataframe)
- test_dataset.pkl : jeu de test

### exécution
Pour exécuter ce fichier, il faut se placer dans un environnement python vérifiant les dépendances nécessaires (cf requirements.txt) et dans le dossier /pipeline dans un terminal de commande et exécuter :  
python 2_training.py seed dossier/de/sortie dossier/modele/zh_indivs.pkl dossier/modele/znh_indivs.pkl --nom --500 --0.75 --15 --False --False --False

## 3_inference.py
Ce script exporte la cartographie des zones humides potentielles à partir des images fournies en entrée et du modèle issu de 2_training.py

### entrée
Il prend en entrée le chemin vers un fichier de configuration .yaml. Le format d'un tel fichier et la signification des différents éléments à y spécifier est détaillé dans le fichier d'exemple 3_inference.yaml.

### sorties :
- carte_zones_humides_potentielles.tif : geotiff de la cartographie des zones humides potentielles
- probabilites_mensuelles.tif : geotiff des probabilités pour chacun des mois fournis en entrée. Les probabilités sont quantifiées sur les entiers de 0 à 254. 255 est réservé aux nodata.

### exécution :
Pour exécuter ce fichier, il faut se placer dans un environnement python vérifiant les dépendances nécessaires (cf requirements.txt) et dans le dossier /pipeline dans un terminal de commande et exécuter :  
python 3_inference.py /chemin/du/fichier/de/config.yaml


## Structure du dépôt
.
├───lib        # contient shreklib.py qui inclue des fonctions mineures utilisées dans différents scripts
├───pipeline        # dossier contenant les scripts .py de la chaîne de production de la cartographie des zones humides potentielles
└───pre_traitement_donnees      # dossier contenant des jupyter notebooks utilisés pour le pré-traitement des images fournies en entrée du modèle
   ├───LST_landsat
   ├───S1
   └───topo