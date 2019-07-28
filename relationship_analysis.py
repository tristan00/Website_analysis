import pandas as pd

from common import get_adjacency_matrix
from sklearn.decomposition import PCA


df = get_adjacency_matrix()



class Encoder():
    def __init__(self, method = 'pca'):
        self.method = method

        if self.method == 'pca':
            self.model = PCA()


    def fit(self, df):
        self.model.fit(df)


    def transform(self,  df):
        return self.model.transform(df)



def cluster_analysis():
    pass



if __name__ == '__main__':
    df = get_adjacency_matrix()
    df
