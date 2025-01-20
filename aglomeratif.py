import scipy.cluster.hierarchy as shc
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time
from sklearn import cluster

# Donnees dans datanp
path = './artificial/'
databrut = arff.loadarff(open(path+"zelnik1.arff",'r'))
datanp = np.array([[x[0] ,x[1]] for x in databrut[0]])
f0 = datanp[:,0]
f1 = datanp[:,1]

methodes = ['single', 'average', 'complete', 'ward']

# On boucle sur les différentes méthodes de linkage de clustering agglomeratif
for linkage in methodes:

    best_score = -1

    # On affiche le dendogramme
    linked_mat = shc.linkage(datanp, linkage)
    plt.figure(figsize =( 10 , 10 ) )
    shc.dendrogram(linked_mat,
    orientation ='top',
    distance_sort ='descending',
    show_leaf_counts = False)
    plt.show ()

    # On determine ensuite le clustering pour différents treshold
    for threshold in np.linspace(0.1, 5, 10):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold=threshold, linkage=linkage, n_clusters=None)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_

        # On détermine à chaque itération si c'est notre meilleur résultat
        if (len(set(labels)) > 1) : # On rajoute cette condition pour empehcer une erreur à la première itération
            score = silhouette_score(datanp, labels)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_labels = labels

    print("Méthode de linkage: " + str(linkage) + ", Meilleur Threshold: " + str(best_threshold) + ", Score de Silhouette: " + str(best_score) + ", Runtime: " + str(round((tps2 - tps1) * 1000, 2)) + " ms")
    plt.figure(figsize =( 10 , 10 ) )
    plt.scatter(f0, f1, c=best_labels, s=8)
    plt.title("Méthode de linkage: " + str(linkage) + ", Meilleur Threshold: " + str(best_threshold) + ", Score de Silhouette: " + str(best_score) + ", Runtime: " + str(round((tps2 - tps1) * 1000, 2)) + " ms")
    plt.show()
