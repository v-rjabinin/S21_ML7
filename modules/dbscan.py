from tqdm import tqdm
from collections import deque
from additional_descriptors import PositiveInt, PositiveFloat
import numpy as np
import pandas as pd


class DBSCAN_S21:
    min_samples = PositiveInt()
    eps, p = PositiveFloat(), PositiveFloat()

    def __init__(self, eps: float, min_samples: int, p: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self.p = p

        if p < 1:
            raise ValueError("Parameter 'p' must be in [1; +inf]")

        self.func = lambda x, y: np.sum(np.abs(x - y) ** p, axis=1) ** (1 / p)
        self.labels_, self.x = None, None

    def fit(self, x: pd.DataFrame | np.ndarray, y: None = None) -> None:
        x = self._validate_x(x, "x")
        n = x.shape[0]
        labels = np.full(n, -1, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            dists = self.func(x, x[i])
            neighbors = np.where(dists <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            queue = deque(neighbors.tolist())

            while queue:
                idx = queue.popleft()
                if visited[idx]:
                    if labels[idx] == -1:
                        labels[idx] = cluster_id
                    continue

                visited[idx] = True
                labels[idx] = cluster_id

                dists = self.func(x, x[idx])
                core_neighbors = np.where(dists <= self.eps)[0]

                if len(core_neighbors) >= self.min_samples:
                    for nb in core_neighbors:
                        if not visited[nb]:
                            queue.append(nb)
                        elif labels[nb] == -1:
                            labels[nb] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        self.x = x

    def predict(self, x: pd.DataFrame | np.ndarray):
        x = self._validate_x(x, "x")
        if self.x is None:
            raise AttributeError("Before calling this method you must run .fit first")
        if x.shape[1] != self.x.shape[1]:
            raise ValueError("'x' must have the same second dimension as the x used when training algorithm")

        len_1, len_2 = np.sum(x ** 2, axis=1), np.sum(self.x ** 2, axis=1)

        final_labels = np.full(shape=x.shape[0], fill_value=-2, dtype=np.int64)
        for i in range(x.shape[0]):
            curr_obj, curr_len = x[i], len_1[i]
            similar_objs = np.where(np.isclose(curr_len, len_2, 1e-6))[0]
            for index in similar_objs:
                sim_obj = self.x[index]
                if np.allclose(sim_obj, curr_obj, 1e-6):
                    final_labels[i] = self.labels_[index]
                    break

        return final_labels

    @staticmethod
    def _validate_x(data: pd.DataFrame | np.ndarray, arg_name: str) -> np.ndarray:
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError(f"Parameter '{arg_name}' must be either pandas.DataFrame or np.ndarray")

        if data.ndim != 2:
            raise ValueError(f"Parameter '{arg_name}' must be 2-d")

        is_nan = np.any(data.isna() if isinstance(data, pd.DataFrame) else np.isnan(data))
        if is_nan:
            raise ValueError(f"Parameter '{arg_name}' contains NA-values")

        if isinstance(data, pd.DataFrame):
            is_numeric = (data.select_dtypes(include='number').shape[1] == data.shape[1])
            if not is_numeric:
                raise TypeError(f"All columns of '{arg_name}' must be numeric")

            data = data.values
        else:
            is_numeric = np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)
            if not is_numeric:
                raise TypeError(f"The dtype of '{arg_name}' must be numeric")

        return data


if __name__ == "__main__":
    print("\n\n✅The module containing DBSCAN_S21 class was successfully imported✅\n\n")