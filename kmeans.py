import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from scipy.io import arff
from sklearn.metrics import silhouette_score
#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension

# f1 : valeur sur la deuxieme dimension
#
path = './dataset-rapport/'
#databrut = arff.loadarff(open(path+"x1.txt",'r'))
datanp = np.loadtxt(path + "x4.txt")

k_min = 2
k_max = 20
silhouette_scores = []
runtimes_kmeans = []

for k in range(k_min, k_max + 1):
    print("Calcul pour k = ",k,"...")

    # Temps calcul 
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++')
    model.fit(datanp)
    tps2 = time.time()
    runtimes_kmeans.append(tps2 - tps1)

    labels = model.labels_

    # Coefficient de silhouette
    silhouette_avg = silhouette_score(datanp, labels)
    silhouette_scores.append(silhouette_avg)

    print("k =", k,"Silhouette = ",round(silhouette_avg, 4),"Runtime KMeans = ",round((tps2 - tps1) * 1000, 2)," ms")


plt.figure()


# Plot du coefficient de silhouette
plt.plot(range(k_min, k_max + 1),silhouette_scores)
plt.xticks(range(k_min, k_max + 1))
plt.title("Coefficient de silhouette en fonction de k")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score de silhouette")
plt.grid(True)
plt.legend()

# Plot du temps d'exécution
plt.figure()
plt.plot(range(k_min, k_max + 1),[t * 1000 for t in runtimes_kmeans])
plt.xticks(range(k_min, k_max + 1))
plt.title("Temps de calcul")
plt.xlabel("Nombre de clusters")
plt.ylabel("Temps de calcul (ms)")
plt.legend()


f0 = datanp[:,0] # tous les elements de la premiere colonne
f1 = datanp[:,1] # tous les elements de la deuxieme colonne
model = cluster.KMeans(n_clusters=15, init='k-means++')
model.fit(datanp)
labels = model.labels_
plt.figure()
plt.scatter(f0,f1,c=labels,s=8)
plt.show()