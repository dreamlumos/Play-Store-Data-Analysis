# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classifieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes
import numpy as np
import pandas as pd
import math
import copy
from iads import utils as ut

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
            Hypothèse: desc_set et label_set ont le même nombre de lignes
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
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # Mine 
        # correct = 0
        # total = label_set.size
        # for i in range(0, total):
        #   if self.predict(desc_set[i]) == label_set[i]:
        #       correct += 1
        # return correct/total

        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()
# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    #TODO: Classe à Compléter
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
            Hypothèse: desc_set et label_set ont le même nombre de lignes
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
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        
        distances = []
        for i in range(len(self.label_set)):
            distance = 0
            for j in range(self.input_dimension):
                distance += (self.desc_set[i][j] - x[j])**2
            distance = math.sqrt(distance)
            distances.append(distance)
        
        indices = np.argsort(distances)
        
        total = 0
        for i in indices[:self.k]:
            if self.label_set[i] == +1:
                total += 1
        return total/self.k
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x) >= 0.5:
            return +1
        else:
            return -1
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
 # ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.learning_rate = learning_rate
        self.history = history
        if history:
            self.allw = [np.copy(self.w)]

    def getW(self):
        """ Rend le vecteur de poids actuel du perceptron
        """
        return self.w

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        nbIt = 0
        nbItMax = 1

        while (nbIt < nbItMax):
            
            # shuffle des données pour le tirage aléatoire
            whole_set = np.append(desc_set, label_set[:, None], axis = 1)
            np.random.shuffle(whole_set)
            shuffled_desc_set = whole_set[:, :-1]
            shuffled_label_set = whole_set[:, -1]

            wref = np.copy(self.w)

            for i in range(shuffled_label_set.size):
                x = shuffled_desc_set[i]
                y = shuffled_label_set[i]
                
                if self.predict(x) != y:
                    self.w += self.learning_rate * y * x
                
                if self.history:
                    self.allw.append(np.copy(self.w))

            # teste la convergence (if the column that changed the most didn't change too much, we've converged)
            if np.max((wref - self.w)**2) < 1e-3:
                break

            nbIt += 1

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.sum(x*self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1

class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.learning_rate = learning_rate
        self.history = history
        if history:
            self.allw = [np.copy(self.w)]

    def getW(self):
        """ Rend le vecteur de poids actuel du perceptron
        """
        return self.w

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        nbIt = 0
        nbItMax = 1

        while (nbIt < nbItMax):
            
            # shuffle des données pour le tirage aléatoire
            whole_set = np.append(desc_set, label_set[:, None], axis = 1)
            np.random.shuffle(whole_set)
            shuffled_desc_set = whole_set[:, :-1]
            shuffled_label_set = whole_set[:, -1]

            wref = np.copy(self.w)

            for i in range(shuffled_label_set.size):
                x = shuffled_desc_set[i]
                y = shuffled_label_set[i]
                
                if y * self.score(x) < 1:
                    self.w += self.learning_rate * y * x
                
                    if self.history:
                        self.allw.append(np.copy(self.w))

            # teste la convergence (if the column that changed the most didn't change too much, we've converged)
            if np.max((wref - self.w)**2) < 1e-3:
                break

            nbIt += 1

        # print("Final w:", self.w)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.sum(x*self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1
 # ---------------------------
class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.w = np.random.uniform(-1, 1, noyau.get_output_dim())
        self.learning_rate = learning_rate
        self.kernel = noyau
        
    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """

        return np.sum(x*self.w)
                
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1   

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        nbIt = 0
        nbItMax = 1

        transformed_desc_set = self.kernel.transform(desc_set)

        while (nbIt < nbItMax):
            
            # shuffle des données pour le tirage aléatoire
            whole_set = np.append(transformed_desc_set, label_set[:, None], axis = 1)
            np.random.shuffle(whole_set)
            shuffled_desc_set = whole_set[:, :-1]
            shuffled_label_set = whole_set[:, -1]

            wref = np.copy(self.w)

            for i in range(shuffled_label_set.size):
                x = shuffled_desc_set[i]
                y = shuffled_label_set[i]
                
                if y != self.predict(x):
                    self.w += self.learning_rate * y * x

            # teste la convergence
            if np.max((wref - self.w)**2) < 1e-3:
                break

            nbIt += 1

# ------------------------ 
class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension) #initialisation aléatoire ?
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        if history:
            self.allw = [self.w]
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        
        nbIt = 0

        while (nbIt < 10):

            # shuffle des données pour le tirage aléatoire
            whole_set = np.append(desc_set, label_set[:, None], axis = 1)
            np.random.shuffle(whole_set)
            shuffled_desc_set = whole_set[:, :-1]
            shuffled_label_set = whole_set[:, -1]
            
            wref = np.copy(self.w)
            
            for i in range(shuffled_label_set.size):
                x = shuffled_desc_set[i]
                y = shuffled_label_set[i]

                gradient = np.transpose(x) * (x @ self.w - y)
                self.w -= self.learning_rate * gradient
                
                if self.history:
                    self.allw.append(np.copy(self.w))

            # teste la convergence
            if np.max((wref - self.w)**2) < 1e-3:
                break
            nbIt += 1
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.sum(x*self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0:
            return -1
        else:
            return 1
# ------------------------ 
class ClassifierMultiOAA(Classifier):
    
    def __init__(self, binaryClassifier):
        """ Constructeur de Classifier
        """
        self.binaryClassifier = binaryClassifier
        self.classifiers = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            nCl: 
            Hypothèse: desc_set et label_set ont le même nombre de lignes
            Hypothèse: les étiquettes sont définies sur [0, nCl]
        """        
        
        nbRows = len(label_set)
        etiquettes = np.unique(label_set)
        for i in range(len(etiquettes)):
            self.classifiers.append(copy.deepcopy(self.binaryClassifier))
            
            newY = np.copy(label_set)
            
            for j in range(nbRows):
                if newY[j] == i:
                    newY[j] = +1
                else:
                    newY[j] = -1
            
            #print("ETIQUETTE:", etiquettes[i])
            self.classifiers[i].train(desc_set, newY)
        
    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        scores = []
        for classif in self.classifiers:
            scores.append(classif.score(x))
        return scores
    
    def predict(self, x):
        """ rend la prediction sur x
            x: une description
        """
        scores = self.score(x)
        return np.argmax(scores)

# ------------------------ 

def kmoyennes(k, data, epsilon, iter_max):
    
    centroides = ut.initialisation(k, data)
    dictAffect = ut.affecte_cluster(data, centroides)
    inertie = ut.inertie_globale(data, dictAffect)
    
    i = 0
    while i < iter_max:
        centroides = ut.nouveaux_centroides(data, dictAffect)
        dictAffect = ut.affecte_cluster(data, centroides)
        ancienne_inertie = inertie
        inertie = ut.inertie_globale(data, dictAffect)
        
        if (abs(ancienne_inertie - inertie) < epsilon):
            break
    
    return centroides, dictAffect