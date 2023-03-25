# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:58:04 2023

@author: MASSON KEVIN
"""

#Importation des packages nécessaires
import numpy as np
import pandas as pd
import numpy.linalg as nla
import random


#DEBUT LECTURE FICHIER/CREATION MATRICE ADJACENCE
#Ouverture du fichier
df = pd.read_csv("paths_finished.tsv", sep="\t", skiprows=15, header = None)
paths = list(df[3].values)

#####Création des dictionnaires/listes####

global_list = []
for string in paths:
    global_list.extend(string.split(';'))              #Sépare notre élément à chaque rencontre de ";"


keyInd = []
global_list = dict.fromkeys(global_list)
keyInd = dict.fromkeys(global_list)
keys = list(global_list.keys())

##########################################

#######Suppression des chevrons#######

global_list.pop('<')
keyInd.pop('<')
keys.remove('<')

##########################################

global_list = dict(sorted(global_list.items())) #On trie la liste globale car les indices doivent correspondre
keyInd = dict(sorted(global_list.items()))
keys = list(global_list.keys())

#######Ajout des clés à global_list#######
                                                #On initialise chaque élément du dict à une liste et on indexe les éléments
index = 0
for key in keys:
    global_list[key] = []
    keyInd[key] = index
    index = index + 1
    
##########################################

paths = [i.split(";") for i in paths]

####Remplissage de la global_list en parcourant les chemins####

for j in range(len(paths)):
    i=0
    while(i < len(paths[j]) - 1 ):
        while (paths[j][i] == '<'):
            paths[j].remove(paths[j][i])
            paths[j].remove(paths[j][i-1])
            i = i-2
        if (paths[j][i] != '<' and paths[j][i+1] != '<' and i < (len(paths[j]) - 1) and paths[j][i+1] not in global_list[paths[j][i]]):
            global_list[paths[j][i]].append(paths[j][i+1])
        i=i+1
        
################################################################

A = np.zeros(shape=(len(global_list),len(global_list)), dtype=float)

for i in range(len(global_list)):
    for elt in global_list[keys[i]]:
        A[i][keyInd[elt]] = 1
#FIN LECTURE FICHIER/CREATION MATRICE ADJACENCE

#DEBUT IMPLEMENTATION ALGORITHMES PAGERANK
#Algorithme Matrice Stochastique
def stoch(M):
    n, p = np.shape(M)             #Dimensions de M
    for i in range(n):
        S = np.sum(M[i])
        for j in range(p):
            if S == 0:
                M[i, j] = 0
            else:
                M[i,j] = M[i,j] / S    #Normalisation de la matrice
    return np.transpose(M)

#Algorithme PageRank
def PageRank(b, M):
    
    ###########Initialisation############
    
    P = stoch(M)                                  #Détermine la transposée de la matrice de transition
    n,p = np.shape(P)                             #Dimensions de M
    
    v = np.array([[1/n for i in range(n)]]).T
    eT = np.array([[1 for i in range(n)]])
    
    
    Q = np.matrix(v@eT)
    R = np.dot(b, P) + np.dot(((1 - b)/n), Q)
    
    x = np.zeros(shape=(n,1))
    x[random.randint(0, n - 1)] = 1
    
    #####################################
  
    #######Méthode de la puissance#######
    
    eps = 1e-5                                    #Initialisation de la condition d'arrêt
    err = 1

    while(err > eps):                             #Arrete la boucle lors de la convergence du vecteur
        next_x = np.dot(R, x)
        next_x = next_x / np.sum(next_x)
        diff = np.abs(x - next_x)
        err = np.linalg.norm(diff)
        x = next_x
    
    #####################################
    
    return x

#Algorithme PageRank Personnalisé
def PageRank_perso(b, M):
    
    
    ###########Initialisation############
    P = stoch(M)                                  #Détermine la transposée de la matrice de transition
    n,p = np.shape(P)                             #Dimensions de M
    
    v = np.array([[1/n for i in range(n)]]).T
    eT = np.array([[1 for i in range(n)]])

    
    ###Personnalisation de v###
    
    s = random.randint(0, int((n-1)/50))
    compteur = 0
    CheckedList = []
  
    while compteur < s:
        ind = random.randint(0, n-1)
        if (ind not in CheckedList):
            compteur+=1
            CheckedList.append(ind)
            v[ind] = 1/s
            
    ###########################

    Q = np.matrix(v@eT)                           #Création de lien imaginaire
    R = np.dot(b, P) + np.dot((1 - b), Q) / n     #Détermine la nouvelle matrice de transition
    
    x = np.zeros(shape=(n,1))
    x[random.randint(0, n - 1)] = 1
    
    #####################################

    #######Méthode de la puissance#######
    
    eps = 1e-5                                    #Initialisation de la condition d'arrêt
    err = 1

    while(err > eps):                             #Arrete la boucle lors de la convergence du vecteur
        next_x = np.dot(R, x)
        next_x = next_x / np.sum(next_x)
        diff = np.abs(x - next_x)
        err = np.linalg.norm(diff)
        x = next_x
    
    #####################################
    
    return x

#Affichage du résultat
def ten_best(x):
    X = sorted(x, reverse=True)                                   #On trie le vecteur de manière décroissante
    five_best = X[:10]                                             #Choix arbitraire d'obtenir les 10 meilleurs résultats
    
    dic_fb = []
    for elem in five_best:                                        #Permet de récupérer les noms des liens
        index = np.array([i for i, val in enumerate(x) if val == elem])
        key = [k for k, v in keyInd.items() if v in index]
        dic_fb.append([key[0], elem])
        
    return dic_fb
#FIN IMPLEMENTATION ALGORITHMES PAGERANK

#DEBUT EXECUTABLES
x = PageRank(0.85, A.copy())
df = pd.DataFrame(ten_best(x), columns=["Page", "Score autorite"])
df.to_csv("Sortie1.tsv", index=False, sep = "\t")


x = PageRank(0.1, A.copy())
df2 = pd.DataFrame(ten_best(x), columns=["Page", "Score autorite"])
df2.to_csv("Sortie2.tsv", index=False, sep = "\t")



x = PageRank_perso(0.85, A.copy())
df3 = pd.DataFrame(ten_best(x), columns=["Page", "Score autorite"])
df3.to_csv("Sortie3.tsv", index=False, sep = "\t")


x = PageRank_perso(0.1, A.copy())
df4 = pd.DataFrame(ten_best(x), columns=["Page", "Score autorite"])
df4.to_csv("Sortie4.tsv", index=False, sep = "\t")
