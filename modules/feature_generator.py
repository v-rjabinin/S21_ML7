from sklearn.utils import check_array
from additional_descriptors import Ratio
import numpy as np
import pandas as pd


class FeatureGenerator:
    nonlinear_func = {"sin(x)": np.sin, "cos(x)": np.cos, "ln(x + 1)": lambda x: np.log(x + 1), "e^(x)": np.exp, "2^(x)": np.exp2,
                      "x / y": lambda x, y: x / (y + 1e-5), "x * y": lambda x, y: x * y}
    agg_func = {"count": lambda x: len(x), "max": np.max, "min": np.min, "mean": np.mean, "std": np.std, "median": np.median,
                "25%": lambda x: np.percentile(x, 25), "75%": lambda x: np.percentile(x, 75), "sum": np.sum}
    thld = Ratio()

    def __init__(self, drop_corr: bool = True, thld: float = 0.9):
        self.drop_corr = drop_corr
        self.thld = thld

    def __call__(self, x: pd.DataFrame | np.ndarray, poly: dict, aggr: dict, non_lin: dict, *args, **kwargs) -> pd.DataFrame:
        self._validate_x(x, 'x')

        self._validate_dict(poly, "poly")
        self._validate_dict(aggr, "aggr")
        self._validate_dict(non_lin, "non_lin")

        dicts = [poly, aggr, non_lin]

        self._validate_cols_to_use(x, dicts)
        self._validate_operations(dicts)

        total_len = len(poly) + len(aggr) + len(non_lin)
        outcome = np.zeros(shape=(x.shape[0], total_len), dtype=np.float64)

        is_df = isinstance(x, pd.DataFrame)
        x = x if is_df else pd.DataFrame(x, columns=[f"{i}" for i in range(x.shape[1])])

        names_1 = self._calculate_poly(x, poly, outcome, 0)
        names_2 = self._calculate_aggr(x, aggr, outcome, len(poly))
        names_3 = self._calculate_non_lin(x, non_lin, outcome, len(poly) + len(aggr))
        names = names_1 + names_2 + names_3

        outcome = pd.DataFrame(data=outcome, columns=names)
        if self.drop_corr:
            self._drop_correlated_features(outcome, self.thld)

        return outcome

    @classmethod
    def _calculate_non_lin(cls, x: pd.DataFrame, non_lin: dict, outcome: np.ndarray, shift: int) -> list[str]:
        names = list()

        for i, (key, f) in enumerate(non_lin.items()):
            is_bin, func = isinstance(key, (list, tuple)), cls.nonlinear_func[f]

            if is_bin:
                feat_1, feat_2 = (str(feat) for feat in key)
            else:
                feat_1 = str(key)

            outcome[:, shift + i] = func(x[feat_1], x[feat_2]) if is_bin else func(x[feat_1])
            names.append(f.replace("x", feat_1).replace("y", feat_2) if is_bin else f.replace("x", feat_1))

        return names

    @classmethod
    def _calculate_aggr(cls, x: pd.DataFrame, aggr: dict, outcome: np.ndarray, shift: int) -> list[str]:
        names = list()

        for i, ((calc, group), f) in enumerate(aggr.items()):
            to_calc, to_group = str(calc), str(group)
            func = cls.agg_func[f]

            outcome[:, shift + i] = x.groupby(to_group)[to_calc].transform(func)
            names.append(f + "(" + to_calc + ") by " + to_group)

        return names

    @classmethod
    def _validate_operations(cls, dicts: list[dict]) -> None:
        if not all(cls._is_float_like(value) for value in dicts[0].values()):
            raise ValueError("All values of 'poly' dict must be numeric")

        agg_func = set(cls.agg_func.keys())
        if not all(func in agg_func for func in dicts[1].values()):
            raise ValueError(f"All values of 'aggr' dict must be in {agg_func}")

        nonlinear_func = set(cls.nonlinear_func.keys())
        if not all(func in nonlinear_func for func in dicts[2].values()):
            raise ValueError(f"All values of 'non_lin' dict must be in {nonlinear_func}")

    @staticmethod
    def _drop_correlated_features(df: pd.DataFrame, threshold: float) -> None:
        corr, to_drop = df.corr().abs(), list()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        for col in range(corr.shape[1] - 1, -1, -1):
            if np.any(upper.iloc[:, col] > threshold):
                to_drop.append(df.columns[col])

        df.drop(columns=to_drop, inplace=True)

    @staticmethod
    def _calculate_poly(x: pd.DataFrame, poly: dict, outcome: np.ndarray, shift: int) -> list[str]:
        names = list()

        for i, (col, power) in enumerate(poly.items()):
            col_name = str(col)

            outcome[:, shift + i] = x[col_name] ** power
            names.append(col_name + f"**{power:.1f}")

        return names

    @staticmethod
    def _validate_dict(d: dict, arg_name: str) -> None:
        if not isinstance(d, dict):
            raise TypeError(f"Parameter {arg_name} must be a dict")

    @staticmethod
    def _is_float_like(value: int | float) -> bool:
        flag = True

        try:
            value = float(value)
        except ValueError:
            flag = False

        return flag

    @staticmethod
    def _validate_cols_to_use(x: pd.DataFrame | np.ndarray, dicts: list[dict]) -> None:
        is_df, cols_to_use = isinstance(x, pd.DataFrame), set()
        cols_in_x = x.columns.values if is_df else np.arange(x.shape[1])

        cols_to_use.update(dicts[0].keys())

        for pair in dicts[1].keys():
            if isinstance(pair, (list, tuple)):
                cols_to_use.update(pair)
            else:
                cols_to_use.add(pair)

        for pair in dicts[2].keys():
            if isinstance(pair, (list, tuple)):
                cols_to_use.update(pair)
            else:
                cols_to_use.add(pair)

        if not cols_to_use.issubset(set(cols_in_x)):
            if is_df:
                raise ValueError("The object 'x' must contain all column names passed through 'poly', 'aggr' and 'non_lin' dicts")
            else:
                raise ValueError("The object 'x' must contain indices passed through 'poly', 'aggr' and 'non_lin' dicts")

    @staticmethod
    def _validate_x(x: pd.DataFrame | np.ndarray, arg_name: str) -> None:
        check_array(x, input_name=arg_name)

    @property
    def drop_corr(self):
        return self._drop_corr

    @drop_corr.setter
    def drop_corr(self, drop_corr):
        if not isinstance(drop_corr, bool):
            raise TypeError("Parameter 'drop_corr' must be a bool")

        self._drop_corr = drop_corr

if __name__ == "__main__":
    print("\n\n✅The module containing FeatureGenerator class was successfully imported✅\n\n")