# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import math
import random as rd

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes, ensemble ils forment le dataset d'entraînement
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes, ensemble ils forment le dataset d'entraînement
        """
        # Principe: on prédit la classe de chaque exemple de desc_set et on compare avec la classe réelle donnée dans label_set
        #           - si la prédiction est bonne, on incrémente le compteur de bonnes classifications
        #           - enfin on divise par le nombre total de prédictions (i.e. la taille du dataset)
        
        accurate_cpt = 0
        
        for i in range(len(desc_set)):            
            if(self.predict(desc_set[i]) == label_set[i]):
                accurate_cpt += 1;
                
        return accurate_cpt/len(desc_set)
    
    def reset(self):
        """ réinitialise le classifieur si nécessaire avant un nouvel apprentissage
        """
        # en général, cette méthode ne fait rien :
        pass
        # dans le cas contraire, on la redéfinit dans le classifier concerné
    
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        # On utilisera self.w = np.random.rand() si l'on veut une distribution normale
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        #Pas d'apprentissage pour ce classifieur
        pass
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        rd.seed(sum(x))
        return rd.random()
    
    def predict(self, x):
        """ rend la prediction sur x (entre 0 et 9)
            x: une description
        """        
        return math.floor(self.score(x)*10)
    
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return "ClassifierLineaireRandom [{}] w={}".format(self.input_dimension, self.w)
    
# ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        # On choisit w le vecteur des poids synaptiques aléatoirement
        self.w = np.random.uniform(size = self.input_dimension)
        # On garde en mémoire ce w initial dans le cas où il faudrait réinitialiser le classifieur
        self.w_init = self.w
        self.epsilon = learning_rate
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """  
        
        #On commence à mélanger notre dataset pour un échantillonage correct par la suite
        dataset = list(zip(desc_set, label_set))
        np.random.shuffle(dataset)
        
        shuffled_desc = [data[0] for data in dataset]
        shuffled_label = [data[1] for data in dataset]
        
        for i in range(len(shuffled_desc)):
            x_i = shuffled_desc[i]
            y_i = shuffled_label[i]            
            if(self.predict(x_i) != y_i):
                self.w += self.epsilon*x_i*y_i
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.vdot(self.w, x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1
    
    def reset(self):
      """ réinitialise le classifieur si nécessaire avant un nouvel apprentissage
      """
      # les poids sont remis à leur valeurs initiales:
      self.w = self.w_init
      
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return "ClassifierPerceptron [{}] rate={} et w={}".format(self.input_dimension, self.epsilon, self.w)

# ---------------------------
class ClassifierOneVsAll(Classifier):
    """ Approche One vs All avec le perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        # Vitesse d'apprentissage
        self.epsilon = learning_rate
        
        # Liste de 10 classifieurs perceptrons: 1 pour chaque classe
        self.classifieurs = [ClassifierPerceptron(self.input_dimension, self.epsilon) for i in range(10)]
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """  
        for i in range(len(self.classifieurs)):
            # label_copy tel que toutes les classes différentes de la classe courante sont égales à -1
            label_binaire = []
            for label in label_set:
                if label == i:
                    label_binaire.append(+1)
                else:
                    label_binaire.append(-1)
                    
            self.classifieurs[i].train(desc_set, label_binaire)
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        liste_scores = []
        for i in range(50):
            for cl in self.classifieurs:
                liste_scores.append(cl.score(x))
        return liste_scores
    
    def predict(self, x):
        """ rend la prediction sur x
            x: une description
        """
        scores = self.score(x)
        return scores.index(max(scores))
    
    def reset(self):
      """ réinitialise le classifieur si nécessaire avant un nouvel apprentissage
      """
      # les poids sont remis à leur valeurs initiales:
      self.w = self.w_init
      
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return "ClassifierOneVsAll [{}] et rate={}".format(self.input_dimension, self.epsilon)
    
# ------------------------ 
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def score(self,x):
        """ rend les proportion de chaque lasse parmi les k ppv de x (valeurs réelles)
            x: une description : un ndarray
        """
        # Liste des distances entre x et les exemples du dataset
        distances = [np.linalg.norm(x-y) for y in self.desc_set]
        
        # On trie le tableau et on garde les indices triés
        k_indices = np.argsort(distances)[:self.k]
        
        # Calcul du score pour les k plus proches voisins, i.e. proportion d'exemples de chaque classe
        scores = [0 for i in range(10)]
        for i in k_indices:
            scores[self.label_set[i]] += 1
        
        return [score/self.k for score in scores]
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        scores = self.score(x)
        return scores.index(max(scores))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return "ClassifierKNN [{}] k={}".format(self.input_dimension, self.k)
# ------------------------

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)
    return valeurs[np.argmax(nb_fois)]

# ------------------------
import math

def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    k = len(P)
    
    # Cas particulier
    if(k <= 1):
        return 0.
    
    entropie = 0
    for pi in P:
        if pi > 0:
            entropie -= pi * math.log(pi, k)
    return entropie

# ------------------------
def entropie(Y):
    """ np.array[String] -> float
        labels correspond à une array list de labels (classes)
        rend l'entropie de la distribution des classes dans cet array
    """
    P = [] # liste de la distribution de probabilités
    
    classes, nb_fois = np.unique(Y, return_counts=True)
    
    # on complète P
    for i in range(len(classes)):
        P.append(nb_fois[i]/sum(nb_fois))
        
    return shannon(P)

# ------------------------
class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            #print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            #return 0
            
            ### Ce qu'on peut essayer de faire: si la valeur de l'attribut d'exemple n'est pas dans la liste des fils,
            # on peut essayer de trouver la classe majoritaire des exemples à partir de ce noeud là
            return classe_majoritaire([self.Les_fils[attribut].classifie(exemple) for attribut in self.Les_fils])
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g

# ------------------------

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset (X: exemples, Y: ensemble des classes)
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        ############################# DEBUT ########
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
                    
        for attribut in LNoms: #Pour chacun des attributs
            
            index = LNoms.index(attribut) # indice de l'attribut dans LNoms
            attribut_valeurs = np.unique([x[index] for x in X]) #liste des valeurs (sans doublon) prises par l'attribut
            
            # Liste des entropies de chaque valeur pour l'attribut courant
            entropies = []
            # Liste des probabilités de chaque valeur pour l'attribut courant
            probas_val = []
            
            for v in attribut_valeurs: #pour chaque valeur prise par l'attribut
                # on construit l'ensemble des exemples de X qui possède la valeur v ainsi que l'ensemble de leurs labels
                X_v = [i for i in range(len(X)) if X[i][index] == v]
                Y_v = np.array([Y[i] for i in X_v])
                e_v = entropie(Y_v)
                entropies.append(e_v)
                probas_val.append(len(X_v)/len(X))
                
            entropie_conditionnelle = 0
            
            # On calcule l'entropie conditionnelle de l'attribut courant
            for i in range(len(attribut_valeurs)): 
                entropie_conditionnelle += probas_val[i] * entropies[i]
            
            if entropie_conditionnelle < min_entropie:
                min_entropie = entropie_conditionnelle
                i_best = LNoms.index(attribut)
                Xbest_valeurs = attribut_valeurs
        
        ############################# FIN ########        
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
# ------------------------
    
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (seuil d'entropie pour le critère d'arrêt)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set, label_set, self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ------------------------