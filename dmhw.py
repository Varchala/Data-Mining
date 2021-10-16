# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:29:12 2020

@author: G.S Ramchandra
"""



import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%

data = pd.read_csv('C:/Users/G.S Ramchandra/Desktop/Varchala/GSU/Classes/DM/homework_2.csv',delimiter=';')
data.head()


#%%

"""Make a copy of the pandas dataframe such that the categorical varaibles 
align one after other. This helps in preprocessing the data easily and 
we do not want to manipulate the original data that was imported """

x = pd.DataFrame(index=range(data.shape[0]), columns=['age','balance','day','duration','campaign','pdays','previous','job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'poutcome','month'])
x = pd.DataFrame(data, dtype=None, copy=False, index = x.index, columns=x.columns )

x_columns = x.columns
x_index = x.index
y = data.iloc[:,16]
x.head()


#%%

print(x.dtypes)
x['job'] = x['job'].astype('category')
x['marital'] = x['marital'].astype('category')
x['education'] = x['education'].astype('category')
x['default'] = x['default'].astype('category')
x['housing'] = x['housing'].astype('category')
x['loan'] = x['loan'].astype('category')
x['contact'] = x['contact'].astype('category')
x['poutcome'] = x['poutcome'].astype('category')
x['month'] = x['month'].astype('category')

# data['job'] = data['job'].astype('category')

x['job'] = x['job'].cat.codes
x['marital'] = x['marital'].cat.codes
x['education'] = x['education'].cat.codes
x['default'] = x['default'].cat.codes
x['housing'] = x['housing'].cat.codes
x['loan'] = x['loan'].cat.codes
x['contact'] = x['contact'].cat.codes
x['poutcome'] = x['poutcome'].cat.codes
x['month'] = x['month'].cat.codes




#%%


x.dtypes
x.head()

#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# one_hot = pd.get_dummies(x.iloc[:,7:16])
# # Drop column B as it is now encoded
# df = df.drop('B',axis = 1)
# # Join the encoded df
# df = df.join(one_hot)

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [7,8,9,10,11,12,13,14,15])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.str)


# one_hot = OneHotEncoder()
# # x_sub = x.iloc[:,7:16]
# # one_hot.fit(x_sub)
# x.iloc[:,7:16] = one_hot.fit_transform(x.iloc[:,7:16]).toarray()
print(x.shape)


#%%

#Feature Scaling using StandardScaler	
x = StandardScaler().fit_transform(x)


#%%
pca = PCA()
x_pca = pca.fit(x).components_.T
print(x_pca)
#df_plot = pd.DataFrame(pca.fit(x).components_, columns= x_columns)
#df_plot.head()
#print(df_plot)
print(pca.explained_variance_)
#print(x_pca.shape)
#%%

from sklearn.cluster import KMeans
def computeKMeans(samples, n=2,max_iter=300):
    km = KMeans(n_clusters = n, init='k-means++',max_iter=max_iter)
    km.fit(samples)
    clusters = km.predict(samples)
    centroids = km.cluster_centers_
    inertia = km.inertia_
    return (clusters, centroids, inertia)


wcss = []
for i in range(1,11):
    clusters, centroids, inertia = computeKMeans(x, n=3, max_iter=400)
    wcss.append(inertia)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
    












