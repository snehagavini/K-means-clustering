import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main(): 
    ham_df = pd.read_csv("../HW3_ham.csv", index_col = 0)
    model = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=None, random_state=1,dissimilarity='precomputed')
    out = model.fit_transform(ham_df)
    principalDf = pd.DataFrame(data = out, columns = ['X', 'Y'])
    principalDf.to_csv("../HW3_MDS.csv")
    plt.scatter(out[:, 0], out[:, 1],label='MDS')
    plt.savefig('mds.png')

if __name__ == "__main__":
    main()