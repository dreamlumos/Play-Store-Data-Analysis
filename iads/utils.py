# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


from numpy.random import default_rng
rng = default_rng()

# ------------------------ 
def plot2DSet(desc, labels):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Mine
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    plt.grid(True)
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color='red') # 'o' pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color='blue') # 'x' pour la classe +1

def plot2DSetMulticlass(X, Y):
	# Mine

	markers = ['o', 'x', '^', 's', '*']
	colors = ['red', 'blue', 'orange', 'green', 'purple', 'cyan']

	labels = np.unique(Y)
	for i in range(len(labels)):
		l = labels[i]
		data = X[Y==l]
		plt.scatter(data[:,0], data[:,1], marker=markers[i%5], color=colors[i%6])

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
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    # Mine
    data_desc = np.random.uniform(inf, sup, (n*2, p))
    data_label = np.asarray([-1 for i in range(0, n)] + [+1 for i in range(0, n)])
    return data_desc, data_label
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    # Mine
    positive = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    negative = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    data_desc = np.concatenate((negative, positive))
    data_label = np.asarray([-1 for i in range(0, nb_points)] + [+1 for i in range(0, nb_points)])
    return data_desc, data_label
# ------------------------ 
def create_XOR(n, var):
    # Mine
    covariance = np.array([[sigma,0],[0,sigma]])
    
    d1 = np.random.multivariate_normal(np.array([0,0]), covariance, n)
    d2 = np.random.multivariate_normal(np.array([0,1]), covariance, n)
    d3 = np.random.multivariate_normal(np.array([1,0]), covariance, n)
    d4 = np.random.multivariate_normal(np.array([1,1]), covariance, n)
    data_xor = np.concatenate((d1, d4, d2, d3))
    label_xor = np.asarray([-1 for i in range(0, n*2)] + [+1 for i in range(0, n*2)]) 
    
    return data_xor, label_xor
 # ------------------------ 
def crossval(X, Y, n_iterations, iteration):
	# Mine
    nbRows = len(Y)
    nbTests = len(Y) // n_iterations
    
    testI = nbTests * iteration # starting index for tests
    testJ = testI + nbTests # ending index for tests
    
    Xtest = X[testI:testJ]
    Ytest = Y[testI:testJ]
    
    Xapp = np.vstack((X[0:testI], X[testJ:]))
    Yapp = np.append(Y[0:testI], Y[testJ:])
    
    return Xapp, Yapp, Xtest, Ytest

def crossval_strat(X, Y, n_iterations, iteration):
	# Mine
    labels = np.unique(Y)
    Xsets = []
    Ysets = []
    for l in labels:
        Xsets.append(X[Y==l])
        Ysets.append(Y[Y==l])
    
    _,n = np.shape(X)
    Xtests = np.empty(shape=[0,n], dtype=int)
    Ytests = np.empty(shape=[0,1], dtype=int)
    Xapps = np.empty(shape=[0,n], dtype=int)
    Yapps = np.empty(shape=[0,1], dtype=int)
    for i in range(len(Xsets)):
        
        Xset = Xsets[i]
        Yset = Ysets[i]
        
        nbRows = len(Xset)
        nbTests = nbRows // n_iterations

        testI = nbTests * iteration # starting index for tests
        testJ = testI + nbTests # ending index for tests
 
        Xtests = np.concatenate((Xtests, Xset[testI:testJ]))
        Ytests = np.append(Ytests, Yset[testI:testJ])

        Xapp = np.vstack((Xset[0:testI], Xset[testJ:]))
        Xapps = np.concatenate((Xapps, Xapp))
        Yapp = np.append(Yset[0:testI], Yset[testJ:])
        Yapps = np.append(Yapps, Yapp)

    return Xapps, Yapps, Xtests, Ytests

# ----------------------------------------------------------------

class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")

class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.hstack((V, np.ones((len(V),1))))
        return V_proj

class KernelPoly(Kernel):
	def transform(self,V):
		""" ndarray de dim 2 -> ndarray de dim 6            
		...
		"""
		x1, x2 = np.split(V, 2)
		ones = np.ones(len(V))

		V_proj = np.concatenate((ones, x1, x2, np.multiply(x1, x1), np.multiply(x2, x2), np.multiply(x1, x2)))

		return V_proj

# ----------------------------------------------------------------

def normalisation(A):
    absA = np.abs(A)
    maxA = np.max(absA, axis=0)
    minA = np.min(absA, axis=0)
    diffA = A - minA
    return diffA / (maxA - minA)

def dist_vect(v1, v2):
    """Calcule la distance euclidienne entre deux vecteurs représentés sous forme d'arrays. """

    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist

def centroide(exemples):
    return np.mean(exemples, axis=0)

def inertie_cluster(cluster):
    c = centroide(cluster)
    return sum([(dist_vect(x, c))**2 for x in cluster])


def initialisation(k, data):
    return rng.choice(data, k, axis=0)

def plus_proche(x, centroides):
    return np.argmin([dist_vect(x, c) for c in centroides])

def affecte_cluster(data, centroides):
    
    # initialisation du dictionnaire
    dictAffect = dict()
    for i_c in range(len(centroides)):
        dictAffect.update({i_c: []})
    
    # affectation des exemples
    for i_x in range(len(data)):
        i_c = plus_proche(data[i_x], centroides)
        dictAffect[i_c].append(i_x)
    
    return dictAffect 

def nouveaux_centroides(data, dictAffect):
    
    nb_c = len(dictAffect.keys())
    centroides = []
    for i_c in range(nb_c):
        i_exemples = dictAffect[i_c]
        exemples = [data[i_x] for i_x in i_exemples]
        c = centroide(exemples)
        centroides.append(c)
    
    return np.asarray(centroides)

def inertie_globale(data, dictAffect):
    
    nb_c = len(dictAffect.keys())
    inertie_globale = 0
    for i_c in range(nb_c):
        i_exemples = dictAffect[i_c]
        exemples = [data[i_x] for i_x in i_exemples]
        inertie_globale += inertie_cluster(exemples)
    
    return inertie_globale

def affiche_resultat(data, centroides, dictAffect):
    plt.scatter(centroides[:,0], centroides[:,1], color='red', marker='x')
        
    couleurs = ['b', 'g', 'y']    
    nb_c = len(dictAffect.keys())
    for i_c in range(nb_c):
        i_exemples = dictAffect[i_c]
        exemples = np.asarray([data[i_x] for i_x in i_exemples])
        plt.scatter(exemples[:,0], exemples[:,1], color=couleurs[i_c])