from additional_descriptors import PositiveInt
from typing import Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class KMeansS21:
    max_iter, random_state, n_init, step, n_features_in_ = PositiveInt(), PositiveInt(), PositiveInt(), PositiveInt(), PositiveInt()

    def __init__(self, n_clusters: int, init: str, n_init: int = 10, max_iter: int = 300, verbose: bool = False, tol: float = 1e-04, step: int = 10, random_state: int = 21):
        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.step = step
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

        self._cluster_centers_ = None
        self._inertia_ = np.inf
        self._n_features_in_ = None
        self.labels_ = None

    def fit_predict(self, x: pd.DataFrame | np.ndarray, y: None = None) -> np.ndarray:
        self.fit(x, y)

        if isinstance(x, pd.DataFrame):
            x = x.values

        cluster_centers = self.cluster_centers_.reshape(self.n_clusters, 1, self.n_features_in_)

        return np.argmin(np.sum((x - cluster_centers) ** 2, axis=2), axis=0)

    def fit(self, x: pd.DataFrame | np.ndarray, y: None = None):
        x = self._validate_data(x, "x")

        n_clusters = self.n_clusters
        max_iter = self.max_iter
        step = self.step
        n_init = self.n_init
        n, d = x.shape

        if n < n_clusters:
            raise ValueError("The number of clusters is greater than the number of data points")

        self.n_features_in_ = d
        if self.verbose:
            max_points = max_iter // step if not max_iter % step else max_iter // step + 1
            scores = np.zeros(shape=(n_init, max_points + 1), dtype=np.float64)
            end_points = np.zeros(shape=n_init, dtype=np.int64)

        for i in range(n_init):
            rng = np.random.default_rng(seed=(self.random_state + i))
            clusters = self._clusters_init(x, rng).reshape(n_clusters, 1, d)
            prev_clusters, score_idx = clusters.copy(), 0

            tqdm_iter = tqdm(range(max_iter), leave=True)
            for j in tqdm_iter:
                if self.verbose and j % step == 0:
                    scores[i, score_idx] = self._calculate_inertia(x, clusters)
                    score_idx += 1

                labels = np.argmin(np.sum((x - clusters) ** 2, axis=2), axis=0)
                for k in range(n_clusters):
                    mask = (labels == k)
                    if np.any(mask):
                        np.mean(x[mask], axis=0, out=clusters[k, 0])
                    else:
                        clusters[k, 0] = x[rng.integers(0, n)]

                if np.all(np.sqrt(np.sum((clusters - prev_clusters) ** 2, axis=2)) < self.tol):
                    break

                prev_clusters = clusters.copy()
                tqdm_iter.set_description(f"Trial {i + 1} | Iter [{j + 1}/{max_iter}]")

            curr_inertia = self._calculate_inertia(x, clusters)

            if self.verbose:
                scores[i, score_idx] = curr_inertia
                end_points[i] = score_idx

            if curr_inertia < self.inertia_:
                self.inertia_ = curr_inertia
                self.cluster_centers_ = clusters.reshape(n_clusters, d)

        self.labels_ = self.predict(x)

        if self.verbose:
            fig, ax = plt.subplots(n_init, 1, figsize=(24, 80))
            plt.subplots_adjust(hspace=0.4)
            for i, axis in enumerate(ax.ravel() if n_init > 1 else [ax]):
                axis.plot(np.arange(0, step * end_points[i] + 1, step), scores[i, :(end_points[i] + 1)], "--r")
                axis.set_title(f"{i}-th iter", fontsize=24)
                axis.set_xlabel("The number of iteration", fontsize=20)
                axis.set_ylabel("Inertia", fontsize=20)
                axis.tick_params(axis="both", labelsize=15)

        return self

    def predict(self, x: pd.DataFrame | np.ndarray):
        x = self._validate_data(x, "x")

        if self.cluster_centers_ is None:
            raise AttributeError("Before running this method you must start .fit to find the cluster centers")
        if self.n_features_in_ != x.shape[1]:
            raise ValueError("The dimension of the 'x' must correspond the dimension used when fitting")

        cluster_centers = self.cluster_centers_.reshape(self.n_clusters, 1, self.n_features_in_)

        return np.argmin(np.sum((x - cluster_centers) ** 2, axis=2), axis=0)

    def _clusters_init(self, x: np.ndarray, rng: np.random.Generator):
        match self.init:
            case "random":
                clusters = self._forgy_init(x, rng)
            case "median-wise":
                clusters = self._rand_partition_init(x, rng, np.median)
            case "mean-wise":
                clusters = self._rand_partition_init(x, rng, np.mean)

        return clusters

    def _forgy_init(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(x), self.n_clusters, replace=False)

        return x[idx]

    def _rand_partition_init(self, x: np.ndarray, rng: np.random.Generator, func: Callable[..., float]):
        n_clusters = self.n_clusters
        n, d = x.shape

        labels = rng.integers(n_clusters, size=n)
        clusters = np.zeros(shape=(n_clusters, d), dtype=np.float64)

        for i in range(n_clusters):
            cluster_points = x[labels == i]

            if len(cluster_points) > 0:
                clusters[i] = func(cluster_points, axis=0)
            else:
                clusters[i] = x[rng.integers(0, n)]

        return clusters

    @staticmethod
    def _calculate_inertia(x: np.ndarray, clusters: np.ndarray) -> np.floating:
        return np.sum(np.min(np.sum((x - clusters) ** 2, axis=2), axis=0))

    @staticmethod
    def _validate_data(data: pd.DataFrame | np.ndarray, arg_name: str) -> np.ndarray:
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

    @property
    def cluster_centers_(self):
        return self._cluster_centers_

    @cluster_centers_.setter
    def cluster_centers_(self, cluster_centers_):
        if not isinstance(cluster_centers_, np.ndarray):
            raise TypeError("Parameter 'cluster_centers_' must be a numpy.ndarray")
        if cluster_centers_.shape != (self.n_clusters, self.n_features_in_):
            raise ValueError("Parameter 'cluster_centers_' must have the dimension of (n_clusters, n_features_in_)")
        if not np.issubdtype(cluster_centers_.dtype, np.floating):
            raise TypeError("Parameter 'cluster_centers_' must be filled with float values")

        self._cluster_centers_ = cluster_centers_

    @property
    def inertia_(self):
        return self._inertia_

    @inertia_.setter
    def inertia_(self, inertia_):
        try:
            inertia_ = float(inertia_)
        except ValueError:
            raise TypeError(f"'inertia_' must be a float value")

        if 0.0 > inertia_:
            raise ValueError(f"'inertia_' must be in [0.0, +inf)")

        self._inertia_ = inertia_

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, tol):
        try:
            tol = float(tol)
        except ValueError:
            raise TypeError(f"'tol' must be a float value")

        if not (0.0 < tol < 1.0):
            raise ValueError(f"'tol' must be in (0.0, 1.0)")

        self._tol = tol

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, init):
        if not init in ("random", "mean-wise", "median-wise"):
            raise ValueError("'init' must be in ('random', 'mean-wise', 'median-wise')")

        self._init = init

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        try:
            n_clusters = int(n_clusters)
        except ValueError:
            raise TypeError(f"'n_clusters' must be an integer")

        if n_clusters < 2:
            raise ValueError(f"'n_clusters' must be greater than 1")

        self._n_clusters = n_clusters

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if not isinstance(verbose, bool):
            raise TypeError("Parameter 'verbose' must be a boolean value")

        self._verbose = verbose


if __name__ == "__main__":
    print("\n\n✅The module containing KMeansS21 class was successfully imported✅\n\n")