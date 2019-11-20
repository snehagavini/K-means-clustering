import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_name):
    return pd.read_csv(file_name)
    

def main():

    file_name = "../HW3.csv"
    df = read_file(file_name)
    pca_df = StandardScaler().fit_transform(df)     # Scaling
    # mean_vec = np.mean(pca_df, axis=0)              # Mean
    # cov_mat = (pca_df - mean_vec).T.dot((pca_df - mean_vec)) / (pca_df.shape[0]-1)      # Covariance matrix
    cov_mat = np.cov(pca_df.T)    
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # print('Eigenvectors \n%s' %eig_vecs)
    # print('\nEigenvalues \n%s' %eig_vals)
    for ev in eig_vecs.T:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # print('Everything ok!')    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(pca_df)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    principalDf.to_csv("../HW3_PCA.csv")
    plt.xlabel('Feature 1', fontsize = 15)
    plt.ylabel('Feature 2', fontsize = 15)
    plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
    plt.title('2 component PCA', fontsize = 20)
    plt.savefig('pca.png')

if __name__ == "__main__":
    main()