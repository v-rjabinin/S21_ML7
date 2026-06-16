from sklearn.metrics import silhouette_score
from sklearn.utils import check_array
from sklearn.base import ClusterMixin

import pandas as pd
import numpy as np
import time


class ClusteringComparator:
    def __init__(self):
        cols_for_res = ["name", "distortion", "silhouette_score", "runtime", "data_dim", "set_num", "n_outliers"]
        self.results = pd.DataFrame(data=None, columns=cols_for_res)

    def run_algorithm(self, x: pd.DataFrame | np.ndarray, model_class: type[ClusterMixin], model_params: dict, alg_name: str, set_num: int) -> np.ndarray:
        self._validate_params(x, model_params, alg_name, set_num)
        x = x.values if isinstance(x, pd.DataFrame) else x

        start_time = time.perf_counter()

        model = model_class(**model_params)
        labels = model.fit_predict(x)

        end_time = time.perf_counter()

        mask, output = (labels == -1), labels

        n_outliers = np.sum(mask)
        if np.any(mask):
            mask = ~mask
            x, labels = x[mask, :], labels[mask]

        n_clust = np.max(labels) + 1
        distort = self.calculate_distortion(x, labels, n_clust)
        sil_score = -1.0 if len(np.unique(labels)) < 2 else silhouette_score(x, labels)

        self.results.loc[len(self.results)] = [alg_name, distort, sil_score, end_time - start_time, x.shape[1], set_num, n_outliers]

        return output

    def __call__(self, x: pd.DataFrame | np.ndarray, model_class: type[ClusterMixin], model_params: dict, alg_name: str, set_num: int) -> np.ndarray:
        return self.run_algorithm(x, model_class, model_params, alg_name, set_num)

    def _validate_params(self, x: pd.DataFrame | np.ndarray, model_params: dict, alg_name: str, set_num: int):
        self._validate_x(x)

        if not isinstance(model_params, dict):
            raise TypeError("'model_params' must be a dict")

        if not isinstance(alg_name, str):
            raise TypeError("'alg_name' must be a string")

        self.check_natural_num(set_num, 'set_num')

    @staticmethod
    def calculate_distortion(x: np.ndarray, outputs: np.ndarray, n_clust: int):
        distort = 0.0

        for i in range(n_clust):
            cluster_objs = x[outputs == i, :]
            if cluster_objs.shape[0] != 0:
                curr_center = np.mean(cluster_objs, axis=0)
                distort += np.sum((cluster_objs - curr_center) ** 2)

        return distort

    @staticmethod
    def check_natural_num(val: int, arg_name: str) -> None:
        if not isinstance(val, int):
            raise TypeError(f"'{arg_name}' must be an integer")
        elif val < 1:
            raise ValueError(f"'{arg_name}' must be positive")

    @staticmethod
    def _validate_x(x: pd.DataFrame | np.ndarray) -> None:
        check_array(x, input_name="x")

if __name__ == "__main__":
    print("\n\n✅The module containing ClusteringComparator class was successfully imported✅\n\n")