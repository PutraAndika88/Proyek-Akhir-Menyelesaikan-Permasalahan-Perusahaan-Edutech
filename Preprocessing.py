import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataQualityChecker(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.low_ = X.quantile(0.01)
        self.high_ = X.quantile(0.99)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if X.isnull().sum().sum() > 0:
            raise ValueError("Missing value detected")
        return X.clip(self.low_, self.high_, axis=1)