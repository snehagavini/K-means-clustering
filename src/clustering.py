import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def kmeans(file_name, num_clusters, mode):
    
    X = pd.read_csv(file_name,index_col=0)
    m = X.shape[0]
    n = X.shape[1]
    
    n_iter=100
    K = num_clusters # number of clusters
    
    Centroids=np.array([]).reshape(n,0)
    for i in range(K):
        rand=rd.randint(0,m-1)
        Centroids=np.c_[Centroids,X.iloc[rand]]

    Output={}
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(K):
        tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    C = np.argmin(EuclidianDistance,axis=1) + 1

    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X.iloc[i]]        
    for k in range(K):
        Y[k+1]=Y[k+1].T        
    for k in range(K):
        Centroids[:,k]=np.mean(Y[k+1],axis=0)
    
    for j in range(n_iter):
        #step 2.a
        EuclidianDistance=np.array([]).reshape(m,0)
        for k in range(K):
            tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1)+1
        #step 2.b
        Y={}
        for k in range(K):
            Y[k+1]=np.array([]).reshape(2,0)
        for i in range(m):
            Y[C[i]]=np.c_[Y[C[i]],X.iloc[i]]
        
        for k in range(K):
            Y[k+1]=Y[k+1].T
        
        for k in range(K):
            Centroids[:,k]=np.mean(Y[k+1],axis=0)
        if j > 0 and all(np.array_equal(Y[key], Output[key]) for key in Y):
            break
        Output = Y

    color=['red','blue','green','cyan','black','magenta','brown']
    labels=['cluster1','cluster2','cluster3','cluster4', 'cluster5', 'cluster6']

    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    
    plt.scatter(Centroids[0,:],Centroids[1,:],s=30,c='yellow',label='Centroids')
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.legend()
    plt.savefig('clustered_{}_{}.png'.format(K, mode))
    plt.clf()

    return Output, Centroids

def sum_of_squares(Output,Centroids,num_clusters):
    sum_of_squares = 0
    Centroids = Centroids.T
    for k in range(num_clusters):
        sum_of_squares += np.sum((Output[k+1] - Centroids[k,:]**2))
    return sum_of_squares

def plot_elbow(sum_of_squares,mode):
    k_array = np.arange(3,7)
    plt.plot(k_array, sum_of_squares)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of squares')
    plt.title('Elbow method to determine optimum number of clusters')
    plt.savefig("elbow_{}.png".format(mode))
    plt.clf()

def main():
    sum_of_square_mds = np.array([])
    sum_of_square_pca = np.array([])
    for i in range(3,7):
        Output, Centroids = kmeans("../HW3_MDS.csv",i, "mds")
        sum_of_square_mds = np.append(sum_of_square_mds,sum_of_squares(Output, Centroids, i))
        Output, Centroids = kmeans("../HW3_PCA.csv", i, "PCA")
        sum_of_square_pca = np.append(sum_of_square_pca,sum_of_squares(Output, Centroids, i))
    
    plot_elbow(sum_of_square_mds,"mds")
    plot_elbow(sum_of_square_pca,"pca")

if __name__ == "__main__":
    main()
