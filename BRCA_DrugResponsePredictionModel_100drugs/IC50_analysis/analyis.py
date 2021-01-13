import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors 
from mpl_toolkits.mplot3d import Axes3D
import math

df = pd.read_csv("drug_resp.csv", header = None)
#print(df.head(10))
#print(df.shape)
dft = df.T
print(dft.shape)
print(dft.head(10))
info = df.describe()
info.to_csv("info.csv")

#dft.hist()
#dft.plot.box()
#plt.show()

##******************* cluster analysis**************************
clusters = 3 #sensitive, resistant, intermediate
kmeans = KMeans(n_clusters = clusters)
kmeans.fit(dft)
print(kmeans.labels_)
print(kmeans.inertia_)

#********************PCA******************************
pca = PCA(2)
pca.fit(dft)

pca_data = pd.DataFrame(pca.transform(dft))
print(pca_data.head())


fig = plt.figure()
data = (pca_data[0], pca_data[1])
colors = ('r', 'g')
groups = ("group1","group2")
ax = fig.add_subplot(1,1,1, facecolor = "#E6E6E6")
for data, color, group in zip(data, colors, groups):
	x = pca_data[0]	
	y = pca_data[1]
	ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
plt.title('PCA_drug_resp')
plt.legend(loc = 2)
plt.show()

'''
#Generating different colors in ascending order of their hsv values
colors = list(zip(*sorted(( 
                    tuple(mcolors.rgb_to_hsv( 
                          mcolors.to_rgba(color)[:3])), name) 
                     for name, color in dict( 
                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS 
                                                      ).items())))[1]
# number of steps to taken generate n(clusters) colors  
skips = math.floor(len(colors[5 : -5])/clusters) 
cluster_colors = colors[5 : -5 : skips]  


## PLot Data
fig = plt.figure() 
ax = fig.add_subplot(111, projection = '3d') 
ax.scatter(pca_data[0], pca_data[1], pca_data[2],  
           c = list(map(lambda label : cluster_colors[label], 
                                            kmeans.labels_))) 
   
str_labels = list(map(lambda label:'% s' % label, kmeans.labels_)) 
   
list(map(lambda data1, data2, data3, str_label: 
        ax.text(data1, data2, data3, s = str_label, size = 16.5, 
        zorder = 20, color = 'k'), pca_data[0], pca_data[1], 
        pca_data[2], str_labels)) 
   
plt.show() 
'''

