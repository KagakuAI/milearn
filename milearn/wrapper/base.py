import numpy as np
from sklearn.base import BaseEstimator


def probs_to_class(probs):
    """
    Convert probability predictions to class labels.

    Handles different shapes of probability arrays:
    - 1D array: threshold at 0.5
    - 2D array with 1 or 2 columns: threshold or argmax
    - Multi-class: argmax over classes
    """
    if probs.ndim == 1:
        return (probs > 0.5).astype(int)
    elif probs.shape[1] == 1:
        return (probs[:, 0] > 0.5).astype(int)
    elif probs.shape[1] == 2:
        return (probs[:, 1] > 0.5).astype(int)
    else:
        return np.argmax(probs, axis=1)


class BagWrapper(BaseEstimator):

    VALID_POOLS = {'mean', 'max', 'min', 'extreme'}

    def __init__(self, estimator, pool='mean'):
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        if not (pool in self.VALID_POOLS or callable(pool)):
            raise ValueError(f"Pooling strategy '{pool}' is not supported.")
        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # Set during fit()

    def __repr__(self):
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f'{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling'

    def _pooling(self, bags):

        # 1. Compute bag representation
        if self.pool == 'mean':
            bag_embed = np.asarray([np.mean(bag, axis=0) for bag in bags])
            return bag_embed
        elif self.pool == 'max':
            bag_embed = np.asarray([np.max(bag, axis=0) for bag in bags])
            return bag_embed
        elif self.pool == 'min':
            bag_embed = np.asarray([np.min(bag, axis=0) for bag in bags])
            return bag_embed
        elif self.pool == 'extreme':
            bags_max = np.asarray([np.max(bag, axis=0) for bag in bags])
            bags_min = np.asarray([np.min(bag, axis=0) for bag in bags])
            bag_embed = np.concatenate((bags_max, bags_min), axis=1)
            return bag_embed

        raise RuntimeError("Unknown pooling strategy.")

    def hopt(self):
        return NotImplementedError

    def fit(self, bags, labels):

        # 1. Check if estimator is classifier
        self.is_classifier = hasattr(self.estimator, 'predict_proba')

        # 2. Compute bag embedding -> transform to single-instance dataset
        bag_embed = self._pooling(bags)

        # 3. Train estimator
        self.estimator.fit(bag_embed, labels)

        return self

    def predict_proba(self, bags):
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        bag_embed = self._pooling(bags)
        y_prob = self.estimator.predict_proba(bag_embed)
        return y_prob

    def predict(self, bags):
        if self.is_classifier:
            y_prob = self.predict_proba(bags)
            return probs_to_class(y_prob)
        else:
            bag_embed = self._pooling(bags)
            y_pred = self.estimator.predict(bag_embed)
            return y_pred


class InstanceWrapper(BaseEstimator):

    VALID_POOLS = {'mean', 'max', 'min'}

    def __init__(self, estimator, pool='mean'):
        if not hasattr(estimator, "fit") or not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
            raise ValueError("Estimator must have a 'fit' and 'predict' or 'predict_proba' method.")
        self.estimator = estimator
        self.pool = pool
        self.is_classifier = None  # Set during fit()

    def __repr__(self):
        pool_name = self.pool.__name__ if callable(self.pool) else self.pool.title()
        return f'{self.__class__.__name__}|{self.estimator.__class__.__name__}|{pool_name}Pooling'

    def _pooling(self, inst_pred):

        # 1. Compute instance predictions
        inst_pred = np.asarray(inst_pred)

        # 2. Apply pooling to instance predictions to get bag prediction
        if callable(self.pool):
            bag_pred = self.pool(inst_pred)
            return bag_pred
        elif self.pool == 'mean':
            bag_pred = np.mean(inst_pred, axis=0)
            return bag_pred
        elif self.pool == 'max':
            bag_pred = np.max(inst_pred, axis=0)
            return bag_pred
        elif self.pool == 'min':
            bag_pred = np.min(inst_pred, axis=0)
            return bag_pred
        else:
            raise ValueError(f"Pooling strategy '{self.pool}' is not recognized.")

    def hopt(self):
        return NotImplementedError

    def fit(self, bags, labels):

        # 1. Check if estimator is classifier
        self.is_classifier = hasattr(self.estimator, 'predict_proba')

        # 2. Assign each instance the same parent bag label -> transform to single-instance dataset
        bags_transformed = np.vstack(np.asarray(bags, dtype=object)).astype(np.float32)
        labels_transformed = np.hstack([np.full(len(bag), lb) for bag, lb in zip(bags, labels)])

        # 3. Train estimator
        self.estimator.fit(bags_transformed, labels_transformed)

        return self

    def predict_proba(self, bags):
        if not self.is_classifier:
            raise NotImplementedError("predict_proba is only available for classifiers.")
        y_pred = []
        for bag in bags:
            bag = bag.reshape(-1, bag.shape[-1])
            inst_pred = self.estimator.predict_proba(bag)
            bag_pred = self._pooling(inst_pred)
            y_pred.append(bag_pred)
        y_pred = np.array(y_pred)
        return y_pred

    def predict(self, bags):
        if self.is_classifier:
            y_prob = self.predict_proba(bags)
            return probs_to_class(y_prob)
        else:
            y_pred = []
            for bag in bags:
                bag = bag.reshape(-1, bag.shape[-1])
                inst_pred = self.estimator.predict(bag)
                bag_pred = self._pooling(inst_pred)
                y_pred.append(bag_pred)
            y_pred = np.asarray(y_pred)
            return y_pred
