# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: evaluation.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
# Version de la crossvalidation avec une liste de classifieurs
def crossvalidation(LC, DS, m):
    """ List[Classifieur] * tuple[array, array] * int ->  List[tuple[tuple[float,float], tuple[float,float]]]
        Hypothèse: m>0
        Par défaut, m vaut 10
    """
    print("Il y a ", len(LC), "classifieurs à comparer.")
    
    # --------------------- Création des 10 datasets
    
    # séparation des 2 classes:
    dataset = list(zip(DS[0].tolist(), DS[1]))

    positive_data = [data for data in dataset if data[1] == +1]
    negative_data = [data for data in dataset if data[1] == -1]
    
    # mélange aléatoire puis répartition dans m datasets
    
    DS_list = [[] for i in range(m)]
    
    i = 0
    while len(positive_data) > 0 or len(negative_data) > 0:
        if len(positive_data) > 0:
            random_index = np.random.randint(0,len(positive_data))
            DS_list[i%m].append(positive_data.pop(random_index))
        if len(negative_data) > 0:
            random_index = np.random.randint(0,len(negative_data))
            DS_list[i%m].append(negative_data.pop(random_index))        
        i += 1

        
    # --------------------- Phases d'apprentissage et de test pour tous les classifieurs
    
    resultats = []
    
    for i in range(len(LC)):
        
        classifieur = LC[i]
        
        # Liste des accuracy pour les phases d'apprentissage et de test
        accuracy_train = []
        accuracy_test = []

        # Pour chaque dataset, on entraîne le classifieur avec le dataset courant et on teste avec les autres
        for dataset in DS_list:

            # Dataset de test
            test_desc = np.asarray([dataset[i][0] for i in range(len(dataset))])
            test_labels = np.asarray([dataset[i][1] for i in range(len(dataset))])

            # Datasets d'apprentissage
            train_desc = []
            train_labels = []

            for other in DS_list:
                if not np.array_equal(dataset, other):
                    train_desc += [elt[0] for elt in other]
                    train_labels += [elt[1] for elt in other]
                    
            # On reformate train_desc: list -> numpy.array
            train_desc = np.asarray(train_desc)
            
            classifieur.reset()
            classifieur.train(train_desc, train_labels)
            accuracy_train.append(classifieur.accuracy(train_desc, train_labels))
            accuracy_test.append(classifieur.accuracy(test_desc, test_labels))
        
        print("\n[Info debug] Classifieur   ", i)
        print("[Info debug] liste accuracies Apprentissage:", accuracy_train)
        print("[Info debug] liste accuracies Test         :", accuracy_test)

        # On rend le résultat sous la forme de deux tuples:
        #    - 1er tuple : moyenne et écart type d'apprentissage
        #    - 2e tuple : moyenne et écart type de test.

        resultats.append(((np.mean(accuracy_train), np.std(accuracy_train)), (np.mean(accuracy_test), np.std(accuracy_test))))
    
    return resultats

# ------------------------
def leave_one_out(C, DS):
    """ Classifieur * tuple[array, array] -> float
    """
    
    # ------- Dataset
    data_desc, data_labels = DS
    n = len(data_desc)
    
    # Nombre de points marqués
    points = 0
    
    # Pour chaque dataset de DS...
    for i in range(n):
        
        # Dataset de test
        test_desc = data_desc[i]
        test_labels = data_labels[i]
        
        # Dataset d'apprentissage
        train_desc = []
        train_labels = []
        
        for j in range(n):
            if i!=j:
                train_desc.append(data_desc[j])
                train_labels.append(data_labels[j])
        
        # On entraîne le classifieur sur le dataset d'apprentissage
        C.reset()
        C.train(train_desc, train_labels)
        
        # Mise à jour du nombre de points marqués
        if C.predict(test_desc)==test_labels:
            points += 1
            
    return points/n
# ------------------------