# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de LU3IN026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(desc,labels):
    """ ndarray * ndarray -> affichage
    """
    # Ensemble des exemples de classe -1:
    negatifs = desc[labels == -1]
    # Ensemble des exemples de classe +1:
    positifs = desc[labels == +1]
    # Affichage de l'ensemble des exemples :
    plt.scatter(negatifs[:,0],negatifs[:,1],marker='o') # 'o' pour la classe -1
    plt.scatter(positifs[:,0],positifs[:,1],marker='x') # 'x' pour la classe +1
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])    
    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
        par défaut: binf vaut -1 et bsup vaut 1
    """
    descriptions = np.random.uniform(binf, bsup, (n, p))
    labels = np.asarray([-1 for i in range(n//2)] + [+1 for i in range(n//2)])
    return descriptions, labels

def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    # On tire aléatoirement tous les exemples des classes -1 et +1
    negative_desc = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    positive_desc = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    # On fusionne les deux ensemble pour obtenir descriptions
    descriptions = np.vstack((negative_desc, positive_desc))
    # On crée les labels
    labels = np.asarray([-1 for i in range(nb_points)] + [+1 for i in range(nb_points)])
    return descriptions, labels
# ------------------------ 
def create_XOR(n, var):
    negative_points_1 = np.random.multivariate_normal(np.array([0,0]), np.array([[var,0],[0,var]]), n)
    negative_points_2 = np.random.multivariate_normal(np.array([1,1]), np.array([[var,0],[0,var]]), n)
    positive_points_1 = np.random.multivariate_normal(np.array([1,0]), np.array([[var,0],[0,var]]), n)
    positive_points_2 = np.random.multivariate_normal(np.array([0,1]), np.array([[var,0],[0,var]]), n)
    
    descriptions = np.vstack((negative_points_1, negative_points_2, positive_points_1, positive_points_2))
    labels = np.asarray([-1 for i in range(2*n)] + [+1 for i in range(2*n)])
    
    return descriptions, labels

# ------------------------
def cree_dataframe(DS, L_noms, Nom_label = "label"):
    """ Dataset * List[str] * Str -> DataFrame
        Hypothèse: la liste a autant de chaînes que la description a de colonnes
    """
    # On commence par créer un dataframe avec les descriptions :
    df = pd.DataFrame(DS[0], columns= L_noms)

    # Puis on rajoute au dataframe une nouvelle colonne contenant les labels de chaque exemple :
    df[Nom_label] = DS[1]
    
    return df

# ------------------------
def categories_2_numeriques(DF,nom_col_label =''):
    """ DataFrame * str -> DataFrame
        nom_col_label est le nom de la colonne Label pour ne pas la transformer
        si vide, il n'y a pas de colonne label
        rend l'équivalent numérique de DF
    """
    dfloc = DF.copy()  # pour ne pas modifier DF
    L_new_cols = []    # pour mémoriser le nom des nouvelles colonnes créées
    Noms_cols = [nom for nom in dfloc.columns if nom != nom_col_label]
     
    for colonne in Noms_cols:
        if dfloc[colonne].dtypes != 'object':  # pour détecter un attribut non catégoriel
            L_new_cols.append(colonne)  # on garde la colonne telle quelle dans ce cas
        else:
            for v in dfloc[colonne].unique():
                nom_col = colonne + '_' + v    # nom de la nouvelle colonne à créer
                dfloc[nom_col] = 0
                dfloc.loc[dfloc[colonne] == v, nom_col] = 1
                L_new_cols.append(nom_col)
            
    return dfloc[L_new_cols]  # on rend que les valeurs numériques

# ------------------------
class AdaptateurCategoriel:
    """ Classe pour adapter un dataframe catégoriel par l'approche one-hot encoding
    """
    def __init__(self,DF,nom_col_label=''):
        """ Constructeur
            Arguments: 
                - DataFrame représentant le dataset avec des attributs catégoriels
                - str qui donne le nom de la colonne du label (que l'on ne doit pas convertir)
                  ou '' si pas de telle colonne. On mémorise ce nom car il permettra de toujours
                  savoir quelle est la colonne des labels.
        """
        self.DF = DF  # on garde le DF original  (rem: on pourrait le copier)
        self.nom_col_label = nom_col_label 
    
        # Conversion des colonnes catégorielles en numériques:
        self.DFcateg = categories_2_numeriques(DF, nom_col_label)
        
        # Pour faciliter les traitements, on crée 2 variables utiles:
        self.data_desc = self.DFcateg.values
        self.data_label = self.DF[nom_col_label].values
        
        # Dimension du dataset convertit (sera utile pour définir le classifieur)
        self.dimension = self.data_desc.shape[1]
                
    def get_dimension(self):
        """ rend la dimension du dataset dé-catégorisé 
        """
        return self.dimension
        
    def train(self,classifieur):
        """ Permet d'entrainer un classifieur sur les données dé-catégorisées 
        """        
        classifieur.train(self.data_desc, self.data_label)
    
    def accuracy(self,classifieur):
        """ Permet de calculer l'accuracy d'un classifieur sur les données
            dé-catégorisées de l'adaptateur.
            Hypothèse: le classifieur doit avoir été entrainé avant sur des données
            similaires (mêmes colonnes/valeurs)
        """
        return classifieur.accuracy(self.data_desc,self.data_label)

    def converti_categoriel(self,x):
        """ transforme un exemple donné sous la forme d'un dataframe contenant
            des attributs catégoriels en son équivalent dé-catégorisé selon le 
            DF qui a servi à créer cet adaptateur
            rend le dataframe numérisé correspondant             
        """
        # NOTE: il est possible que l'exemple donné ait plus de colonnes que le DF ayant servi pour cet adapteur
        #       dans ce cas on les ignore
        
        # Dataframe à renvoyer
        x_df = pd.DataFrame([[0 for i in range(self.dimension)]], columns = self.DFcateg.columns)
        
        # On convertit x en numérique
        x_numerique = categories_2_numeriques(x, self.nom_col_label)
        
        # On recopie les colonnes de x_numerique dans le dataframe à envoyer
        for colonne in x_numerique.columns:
            x_df[colonne] = x_numerique[colonne]
        
        # On rajoute la colonne de label
        x_df[self.nom_col_label] = x[self.nom_col_label]

        return x_df
        
    def predict(self,x,classifieur):
        """ rend la prédiction de x avec le classifieur donné
            Avant d'être classifié, x doit être converti
        """
        x_df = self.converti_categoriel(x)
        return classifieur.predict(x_df[self.DFcateg.columns].values)
    
    def crossvalidation(self, LC, m):
        """ effectue la validation croisée sur les données que contient l'adaptateur
            avec le classifieur donné
        """
        print("Il y a ", len(LC), "classifieurs à comparer.")
    
        # --------------------- Création des 10 datasets
        
        DS = (self.data_desc, self.data_label)
        
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
        
    
    def leave_one_out(self, classifieur):
        """ effectue la validation croisée sur les données que contient l'adaptateur
            avec le classifieur donné
        """
        # ------- Dataset
        n = len(self.data_desc)
        
        # Nombre de points marqués
        points = 0
        
        # Pour chaque dataset de DS...
        for i in range(n):
            
            # Dataset de test
            test_desc = self.data_desc[i]
            test_labels = self.data_labels[i]
            
            # Dataset d'apprentissage
            train_desc = []
            train_labels = []
            
            for j in range(n):
                if i!=j:
                    train_desc.append(self.data_desc[j])
                    train_labels.append(self.data_labels[j])
            
            # On entraîne le classifieur sur le dataset d'apprentissage
            classifieur.reset()
            classifieur.train(train_desc, train_labels)
            
            # Mise à jour du nombre de points marqués
            if classifieur.predict(test_desc)==test_labels:
                points += 1
                
        return points/n    
# ------------------------  