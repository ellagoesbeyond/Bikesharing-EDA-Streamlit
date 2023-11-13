# Autor: Elisabth Oeljeklaus
# Date: 2023-11-07
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor as XGBoost
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from math import sqrt

# note to sucessfully run lightgbm you might need to install libomp.
# for mac users: brew install libomp (need to have homebrew installed)
# install homebrew visiting https://brew.sh/


class OutlierClipper(BaseEstimator, TransformerMixin):  # inherits functionalities for column transformer
    def __init__(self, factor=1.5): # default IQR factor==1.5 but can be altered in the params_grid
        self.factor = factor

    def get_feature_names_out(self, input_features=None):  #to measure variable importance later
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out")
        return input_features if input_features is not None else []

    # fitting of the class: mark bounds above and below 1.5IQR
    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        if self.factor != None:
          self.lower_bound_ = Q1 - self.factor * IQR
          self.upper_bound_ = Q3 + self.factor * IQR
        return self


    def transform(self, X, y=None):
        # only clip values if in the factor is not none --> we want to test in the hyper param is acutally worth it
        if self.factor != None:
          X_clipped=np.clip(X, self.lower_bound_, self.upper_bound_)
          #X_clipped=pd.DataFrame(X_clipped,columns=X.columns)
          return X_clipped
        else:
          return X
        
class RemoveRedundant(BaseEstimator, TransformerMixin):  # inherits functionalities for column transformer
    def __init__(self, redundant=0.7):
        self.redundant = redundant
        #self.to_drop=None

    def get_feature_names_out(self, input_features=None):
        if self.redundant is not None:
        # Use a list comprehension to create a list of feature names that are not redundant (because these were dropped)
          return [feature for i, feature in enumerate(input_features) if i not in self.redundant]
        else:
          return input_features

    def fit(self, X, y):

      X_df=pd.DataFrame(X)
      corr_matrix = X_df.corr().abs()
      # The matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
      to_drop = set()
      for col in upper.columns:
                  # check if in the corr matrix there is a corr above the defined threshold, which is saved in the attribute "redundant"
                  if any(upper[col] > self.redundant):
                      correlated_cols = upper.index[upper[col] > self.redundant ].tolist()
                      correlated_cols.append(col)

                      # Append the column with lower correlation to the target
                      correlations_with_target = X_df[correlated_cols].apply(lambda x: x.corr(y))
                      # keep only the columns of the correlated predicotrs that has the highest corr. with the target
                      col_to_keep = correlations_with_target.abs().idxmax()

                      correlated_cols.remove(col_to_keep) # list of redundant vars.

                      to_drop.update(correlated_cols)

      self.redundant = list(to_drop)
      #print(self.redundant)
      return self


    def transform(self, X, y=None):
        # drop columns that were marked as irrelevant during the fitting -->
        #print(self.redundant)
        if self.redundant:
          #all_indices=set(range(X.shape(1)))
          drop_indices=self.redundant
          #keep_indices=all_indices-drop_indices
          #cols_keep= [X.columns[index] for index in keep_indices]
          X_relevant=np.delete(X,drop_indices,axis=1)

          #X_relevant=pd.DataFrame(X_relevant,columns=cols_keep)
          return X_relevant
        else:
          return X
        

Polynomial_trans = ColumnTransformer(
    transformers=[
        ('quad', PolynomialFeatures(degree=2, include_bias=False), ['mnth','weekday','temp','hum']), #based on visual inspection these could be quadaritc to targets
        ('polynom', PolynomialFeatures(degree=3, include_bias=False), ['hr']) #based on visual inspection these could be polynomial maybe 3 to target
    ] ,
    remainder='passthrough'
)

