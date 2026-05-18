import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler


class KNNGraphBuilder:
    def __init__(self,
                 k: int = 5,
                 metric: str = "euclidean",  
                 symmetrize: bool = False,
                 scaler = None  
                 ):
        self.k = k
        self.metric = metric.lower()
        self.symmetrize = symmetrize
        self.scaler = scaler  

    def build_from_file(self, data_path: str, dataset: str):

        dataset = dataset.upper()
        train = self._load_train_array(data_path, dataset)             
        train = np.nan_to_num(train)                                  
        if self.scaler is None:
            self.scaler = StandardScaler().fit(train)                   
        train_norm = self.scaler.transform(train)
        return self._knn_from_array(train_norm)

    def build_from_array(self, train_array: np.ndarray):

        if self.scaler is None:
            self.scaler = StandardScaler().fit(train_array)
        train_norm = self.scaler.transform(train_array)
        return self._knn_from_array(train_norm)


    def _knn_from_array(self, train_norm: np.ndarray):
        channel_features = train_norm.T          # [D, T]
        if self.metric == "euclidean":
            dist = euclidean_distances(channel_features)   
            np.fill_diagonal(dist, np.inf)
            neighbors = np.argsort(dist, axis=1)[:, :self.k]
        elif self.metric == "cosine":
            sim = cosine_similarity(channel_features)      
            np.fill_diagonal(sim, -np.inf)
            neighbors = np.argsort(-sim, axis=1)[:, :self.k]
        else:
            raise ValueError("metric must be 'euclidean' or 'cosine'")

        D = channel_features.shape[0]
        adj = np.zeros((D, D), dtype=np.float32)
        for i in range(D):
            adj[i, neighbors[i]] = 1.0
        if self.symmetrize:
            adj = np.maximum(adj, adj.T)

        adj += np.eye(D, dtype=np.float32)
        adj[adj==0] = -1.0

        return adj


    @staticmethod
    def _load_train_array(data_path: str, dataset: str) -> np.ndarray:
        if dataset == 'SMD' or dataset == 'SMD+':
            return np.load(os.path.join(data_path, "SMD_train.npy"))

        elif dataset == 'SMAP':
            return np.load(os.path.join(data_path, "SMAP_train.npy"))

        elif dataset == 'MSL':
            return np.load(os.path.join(data_path, "MSL_train.npy"))

        elif dataset == 'PSM':
            df = pd.read_csv(os.path.join(data_path, "train.csv"))
            return df.values[:, 1:]

        elif dataset == 'SWAT':
            return np.load(os.path.join(data_path, "SWaT_train.npy"))

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    @staticmethod
    def save(adj: np.ndarray, out_path: str):
        np.save(out_path, adj)

    @staticmethod
    def load(adj_path: str) -> np.ndarray:
        return np.load(adj_path)