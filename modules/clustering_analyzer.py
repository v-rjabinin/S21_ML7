import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from typing import Type, Optional
from additional_descriptors import PositiveInt, Ratio


class ClusteringAnalyzer:
    n_features_in, top_features, n_clusters = PositiveInt(), PositiveInt(), PositiveInt()
    pct_low, pct_high = Ratio(), Ratio()

    def __init__(self, top_features: int = 10, pct_low: float = 0, pct_high: float = 0):
        self._n_features_in = None
        self._n_clusters = None
        self.top_features = top_features
        self.pct_low, self.pct_high = pct_low, pct_high
        self.results = pd.DataFrame(columns=["Model name", "Distortion", "Silhouette score"] + [f"Top {i + 1} feature" for i in range(top_features)])

    def __call__(self, model_class: Type, model_params: dict, model_name: str, x_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray, train_feat: list[int] | list[str], scatter_params: dict, fig_params: dict, *args, **kwargs):
        if isinstance(train_feat, list):
            flag_1 = any(not isinstance(feat, str) for feat in train_feat)
            flag_2 = any(not isinstance(feat, int) for feat in train_feat)

            if flag_1 and flag_2:
                raise TypeError("Parameter 'train_feat' must consist of either strings or integers")
        else:
            raise TypeError("Parameter 'train_feat' must be a list")

        train_feat = np.array(train_feat)

        x_train, train_cols = self._validate_x(x_train, "x_train", train_feat, flag_2)
        x_test, test_cols = self._validate_x(x_test, "x_test", train_feat, flag_2)
        y_train = self._validate_y(y_train, "y_train")
        y_test = self._validate_y(y_test, "y_test")

        train_cols = np.concatenate([train_cols, ["clusters"]])

        n_features_in = x_train.shape[1]
        self.n_features_in = n_features_in

        if n_features_in < self.top_features:
            raise ValueError("The number of features is less than 'top_features'")

        if flag_1 and (np.max(train_feat) > n_features_in or np.min(train_feat) < 0):
            raise ValueError("'train_feat' contains incorrect indices")

        if flag_2:
            idx = np.zeros_like(train_feat, dtype=np.int64)
            for i, col in enumerate(train_feat):
                idx[i] = np.argmax(train_cols == col)
            train_feat = idx

        if x_test.shape[1] != n_features_in:
            raise ValueError("Parameter 'x_test' must have the same second dimension as the 'x_train'")

        if len(y_train) != len(x_train):
            raise ValueError("Parameter 'x_train' and 'y_train' must have the same length")

        if len(y_test) != len(x_test):
            raise ValueError("Parameter 'x_test' and 'y_test' must have the same length")

        if not isinstance(model_params, dict):
            raise TypeError("Parameter 'model_params' must be a dict")

        if not isinstance(model_name, str):
            raise TypeError("Parameter 'model_name' must be a string")

        model = model_class(**model_params)
        model.fit(x_train[:, train_feat])
        outputs = model.labels_ if hasattr(model, "labels_") else model.predict(x_train[:, train_feat])

        unique_labels = np.unique(outputs)
        self.n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if self.n_clusters < 2:
            sil_score = -1.0
        else:
            sil_score = silhouette_score(x_train[:, train_feat], outputs)
        distort = self.calculate_distortion(x_train[:, train_feat], outputs)

        fig, ax = plt.subplots(1, 1, figsize=fig_params["figsize"])

        if len(train_feat) < 2:
            raise ValueError("To build scatter-plot based on the first two features from train_feat, len(train_feat) must be in [2; +inf)")

        idx_1, idx_2 = train_feat[0], train_feat[1]

        for_scatter = x_train[:, [idx_1, idx_2]]
        thlds = np.percentile(for_scatter, [100 * self.pct_low, 100 * self.pct_high], axis=0)
        feat_1, feat_2 = for_scatter[:, 0], for_scatter[:, 1]
        mask = ((feat_1 > thlds[0, 0]) & (feat_1 < thlds[1, 0])) & ((feat_2 > thlds[0, 1]) & (feat_2 < thlds[1, 1]))

        ax.scatter(feat_1[mask], feat_2[mask], alpha=scatter_params["alpha"], c=outputs[mask], cmap=scatter_params["cmap"])
        ax.set_xlabel(scatter_params["x_label"], fontsize=scatter_params["x_size"])
        ax.set_ylabel(scatter_params["y_label"], fontsize=scatter_params["y_size"])
        ax.set_title(scatter_params["title"], fontsize=scatter_params["title_fontsize"])
        ax.tick_params(axis="both", labelsize=scatter_params["labelsize"])

        x_train = np.hstack([x_train, outputs.reshape(-1, 1)])

        scaler = MinMaxScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)

        param_grid = {"alpha": np.array([0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000], dtype=np.float64)}

        estimator = Lasso(fit_intercept=True, max_iter=10_000, random_state=21)
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=(-1), cv=8)
        grid_search.fit(scaled_x_train, y_train)

        best_estimator = grid_search.best_estimator_
        coef = np.abs(best_estimator.coef_)

        idx = np.argsort(coef)[::(-1)]
        top_features = self.top_features
        import_feat = train_cols[idx[:top_features]]

        final_row = list()
        final_row.extend([model_name, distort, sil_score])
        final_row.extend([str(feat) for feat in import_feat])

        next_elem = len(self.results)
        self.results.loc[next_elem] = final_row

    def calculate_distortion(self, x: np.ndarray, outputs: np.ndarray):
        distort = 0.0

        if np.any(outputs == -1):
            x = x[outputs != -1, :]
            outputs = outputs[outputs != -1]

        for i in range(self.n_clusters):
            cluster_objs = x[outputs == i, :]
            if cluster_objs.shape[0] != 0:
                curr_center = np.mean(cluster_objs, axis=0)
                distort += np.sum((cluster_objs - curr_center) ** 2)

        return distort

    @staticmethod
    def _validate_x(data: pd.DataFrame | np.ndarray, arg_name: str, train_feat: np.ndarray, is_str: bool) -> tuple[np.ndarray, Optional[pd.Index | np.ndarray]]:
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

            cols = data.columns
            if is_str and not np.all(np.isin(train_feat, cols)):
                raise ValueError(f"Parameter '{arg_name}' must contain all strings stored in 'train_feat'")

            data = data.values
        else:
            is_numeric = np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)
            if not is_numeric:
                raise TypeError(f"The dtype of '{arg_name}' must be numeric")

            cols = np.arange(data.shape[1])

        return data, cols

    @staticmethod
    def _validate_y(data: pd.Series | np.ndarray, arg_name: str) -> np.ndarray:
        if not isinstance(data, (pd.Series, np.ndarray)):
            raise TypeError(f"Parameter '{arg_name}' must be either pandas.Series or np.ndarray")

        if data.ndim != 1:
            raise ValueError(f"Parameter '{arg_name}' must be 1-d")

        is_nan = np.any(data.isna() if isinstance(data, pd.Series) else np.isnan(data))
        if is_nan:
            raise ValueError(f"Parameter '{arg_name}' contains NA-values")

        if isinstance(data, pd.Series):
            data = data.values

        is_numeric = np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)
        if not is_numeric:
            raise TypeError(f"The dtype of '{arg_name}' must be numeric")

        return data


if __name__ == "__main__":
    print("\n\n✅The module containing ClusteringAnalyzer class was successfully imported✅\n\n")