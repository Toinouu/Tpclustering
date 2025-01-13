import scipy.cluster.hierarchy as shc
from scipy.io import arff
import numpy as np
import matplotlib . pyplot as plt
from sklearn.metrics import silhouette_score
import time
from sklearn import cluster

# Donnees dans datanp
print ( "Dendrogramme ’single’ donnees initiales " )
path = './artificial/'
databrut = arff.loadarff(open(path+"xclara.arff",'r'))
datanp = np.array([[x[0] ,x[1]] for x in databrut[0]])
f0 = datanp[:,0] # tous les elements de la premiere colonne
f1 = datanp[:,1]

linkage_methods = ['single', 'average', 'complete', 'ward']

for linkage in linkage_methods:
    best_score = -1
    best_threshold = None
    best_labels = None
    k = 3
    for threshold in np.linspace(0.1, 10, 100):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(distance_threshold=threshold, linkage=linkage, n_clusters=None)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        
        if len(set(labels)) > 1:  # Avoid silhouette_score error with single cluster
            score = silhouette_score(datanp, labels)
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_labels = labels
    
    print(f"Linkage: {linkage}, Best Threshold: {best_threshold:.2f}, Silhouette Score: {best_score:.2f}, Runtime: {round((tps2 - tps1) * 1000, 2)} ms")
    plt.figure()
    plt.scatter(f0, f1, c=best_labels, s=8)
    plt.title(f"Linkage: {linkage} - Best Threshold: {best_threshold:.2f} - Silhouette Score: {best_score:.2f}")
    plt.show()
    


