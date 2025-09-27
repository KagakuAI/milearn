import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing._data import MinMaxScaler
from milearn.preprocessing import BagMinMaxScaler
from numpy import ndarray
from typing import Any
from typing import List
from typing import Optional

class BagScaler(BaseEstimator, TransformerMixin):
    """
    Wrapper to apply scikit-learn scalers to bags of instances.

    Each bag is a 2D array of shape [n_instances, n_features]. The scaler is
    fitted on all instances from all bags and then applied individually to each bag.
    """
    def __init__(self, scaler: MinMaxScaler = None) -> None:
        """
        Initialize BagScaler.

        Args:
            scaler: scikit-learn scaler instance (e.g., MinMaxScaler, StandardScaler).
                    Defaults to MinMaxScaler() if None.
        """
        self.scaler = scaler if scaler is not None else MinMaxScaler()

    def fit(self, x: List[ndarray], y: Optional[Any] = None) -> BagMinMaxScaler:
        """
        Fit the scaler using all instances from all bags.

        Args:
            x (list of np.ndarray): list of bags, each bag is [n_instances, n_features].
            y: optional target values (passed to scaler if needed).

        Returns:
            self
        """
        all_instances = np.vstack(x)  # stack all bags for fitting
        self.scaler.fit(all_instances, y)
        return self

    def transform(self, x):
        """
        Transform each bag using the fitted scaler.

        Args:
            x (list of np.ndarray): list of bags to transform.

        Returns:
            list of np.ndarray: scaled bags
        """
        x_scaled = [self.scaler.transform(bag) for bag in x]
        return x_scaled

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the scaler and transform the bags in a single step.

        Args:
            X (list of np.ndarray): list of bags to fit and transform.
            y: optional target values.
            **fit_params: additional fit parameters.

        Returns:
            list of np.ndarray: scaled bags
        """
        return self.fit(X, y).transform(X)


class BagMinMaxScaler(BagScaler):
    """BagScaler using sklearn's MinMaxScaler."""
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(scaler=MinMaxScaler(**kwargs))


class BagStandardScaler(BagScaler):
    """BagScaler using sklearn's StandardScaler."""
    def __init__(self, **kwargs):
        super().__init__(scaler=StandardScaler(**kwargs))


class BagMaxAbsScaler(BagScaler):
    """BagScaler using sklearn's MaxAbsScaler."""
    def __init__(self, **kwargs):
        super().__init__(scaler=MaxAbsScaler(**kwargs))


class BagRobustScaler(BagScaler):
    """BagScaler using sklearn's RobustScaler."""
    def __init__(self, **kwargs):
        super().__init__(scaler=RobustScaler(**kwargs))
